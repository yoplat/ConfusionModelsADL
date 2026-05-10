import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from .config import IMG_SIZE, PATCH_GRID, SEED, W_MB, device, worker_init_fn
from .datasets import TestDataset, preprocess
from .features import extract_multilayer_patches
from .model import SegHead


_TTA_FLIPS = [[], [-1]]  # original, horizontal flip


@torch.no_grad()
def memorybank_infer(
    model,
    paths: list[str],
    bank: torch.Tensor,
    batch_size: int = 32,
    seed: int = SEED,
) -> tuple[np.ndarray, list[str]]:
    """Score images by nearest-neighbour distance to the memory bank.

    Each patch embedding is compared to all bank entries; the patch score is
    ``1 − max_cosine_similarity``, then upsampled to IMG_SIZE × IMG_SIZE.

    Args:
        model:      Frozen DINOv2 backbone.
        paths:      Image file paths.
        bank:       (K, FEATURE_DIM) L2-normalised memory bank tensor.
        batch_size: DataLoader batch size.
        seed:       DataLoader generator seed.

    Returns:
        scores:    (N, IMG_SIZE, IMG_SIZE) float32 anomaly score array.
        filenames: Corresponding list of filenames.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        TestDataset(paths, preprocess),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    bank_gpu = bank.to(device)
    scores, fns = [], []
    for imgs, names in tqdm(loader, desc="MB infer", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        out = model.forward_features(imgs)
        patches = F.normalize(out["x_norm_patchtokens"], p=2, dim=2)
        max_sim, _ = (patches @ bank_gpu.T).max(dim=2)
        dist = (1.0 - max_sim).reshape(-1, PATCH_GRID, PATCH_GRID)
        s = F.interpolate(
            dist.unsqueeze(1),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        scores.append(s.cpu().numpy())
        fns.extend(names)
    return np.concatenate(scores, axis=0), fns


@torch.no_grad()
def seghead_infer(
    model,
    paths: list[str],
    head: SegHead,
    batch_size: int = 32,
    seed: int = SEED,
    tta: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Score images using a trained SegHead on multi-layer DINOv2 features.

    With ``tta=True``, scores are averaged over the original, horizontal flip,
    and vertical flip of each image.

    Args:
        model:      DINOv2 backbone.
        paths:      Image file paths.
        head:       Trained SegHead model.
        batch_size: DataLoader batch size.
        seed:       DataLoader generator seed.
        tta:        Enable test-time augmentation (h-flip + v-flip).

    Returns:
        scores:    (N, IMG_SIZE, IMG_SIZE) float32 anomaly score array.
        filenames: Corresponding list of filenames.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        TestDataset(paths, preprocess),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    head.eval()
    scores, fns = [], []
    for imgs, names in tqdm(loader, desc="SH infer", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        aug_scores = []
        for flip_dims in (_TTA_FLIPS if tta else [[]]):
            x = torch.flip(imgs, flip_dims) if flip_dims else imgs
            patches = extract_multilayer_patches(model, x)
            s = torch.sigmoid(head(patches)).squeeze(1)
            if flip_dims:
                s = torch.flip(s, flip_dims)
            aug_scores.append(s)
        s = torch.stack(aug_scores).mean(0).cpu().numpy()
        scores.append(s)
        fns.extend(names)
    return np.concatenate(scores, axis=0), fns


def run_inference(
    model,
    data_root: Path,
    memory_banks: dict,
    seg_heads: dict,
) -> tuple[dict, dict, dict, dict]:
    """Run memory-bank and SegHead inference on the good-train and test splits.

    Training-split scores are collected to compute per-class normalisation
    statistics in the submission step.

    Args:
        model:        DINOv2 backbone.
        data_root:    Dataset root.
        memory_banks: Per-class banks from :func:`~training.build_memory_banks`.
        seg_heads:    Per-class heads from :func:`~training.train_seg_heads`.

    Returns:
        mb_train: class_name -> (N, H, W) memory-bank scores on good images.
        mb_test:  class_name -> {filename: (H, W)} memory-bank scores on test.
        sh_train: class_name -> (N, H, W) seg-head scores on good images.
        sh_test:  class_name -> {filename: (H, W)} seg-head scores on test.
    """
    print("\nRunning inference...")
    mb_train, mb_test, sh_train, sh_test = {}, {}, {}, {}
    for class_name in sorted(seg_heads.keys()):
        good_dir = data_root / class_name / "train" / "good"
        good = [str(p) for p in sorted(good_dir.glob("*.png"))]
        test_dir = data_root / class_name / "test"
        test = [str(p) for p in sorted(test_dir.glob("*.png"))]

        if W_MB > 0:
            mb_tr, _ = memorybank_infer(model, good, memory_banks[class_name])
            mb_te, mfns = memorybank_infer(model, test, memory_banks[class_name])
            mb_train[class_name] = mb_tr
            mb_test[class_name] = dict(zip(mfns, mb_te))

        sh_tr, _ = seghead_infer(model, good, seg_heads[class_name])
        sh_te, sfns = seghead_infer(model, test, seg_heads[class_name])
        sh_train[class_name] = sh_tr
        sh_test[class_name] = dict(zip(sfns, sh_te))
        print(f"  {class_name} done")
    return mb_train, mb_test, sh_train, sh_test
