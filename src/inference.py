"""Inference utilities: SegHead scoring, 4-fold TTA, normalised ensemble."""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import BLUR_SIGMA, IMG_SIZE, NUM_WORKERS, PATCH_GRID, P_HI, P_LO, SEED, W_MB, device, worker_init_fn
from .datasets import TestDataset, preprocess
from .features import extract_multilayer_patches
from .model import SegHead


# 4-fold TTA: original, h-flip, v-flip, 90° rotation
_TTA_OPS = [
    ("orig",   lambda x: x,                            lambda x: x),
    ("flip_h", lambda x: torch.flip(x, [-1]),           lambda x: torch.flip(x, [-1])),
    ("flip_v", lambda x: torch.flip(x, [-2]),           lambda x: torch.flip(x, [-2])),
    ("rot90",  lambda x: torch.rot90(x, 1, [-2, -1]),   lambda x: torch.rot90(x, -1, [-2, -1])),
]


@torch.no_grad()
def memorybank_infer(
    model,
    paths: list[str],
    bank: torch.Tensor,
    batch_size: int = 32,
    seed: int = SEED,
) -> tuple[np.ndarray, list[str]]:
    """Score images by nearest-neighbour distance to the memory bank."""
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        TestDataset(paths, preprocess),
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
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
        del out
        max_sim, _ = (patches @ bank_gpu.T).max(dim=2)
        dist = (1.0 - max_sim).reshape(-1, PATCH_GRID, PATCH_GRID)
        s = F.interpolate(
            dist.unsqueeze(1), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
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
    """Score images using a trained SegHead with 4-fold TTA.

    With ``tta=True``, scores are averaged over original, h-flip, v-flip,
    and 90° rotation.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        TestDataset(paths, preprocess),
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    head.eval()
    scores, fns = [], []
    for imgs, names in tqdm(loader, desc="SH infer", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        aug_scores = []
        ops = _TTA_OPS if tta else [_TTA_OPS[0]]
        for _, fwd, inv in ops:
            x = fwd(imgs)
            patches = extract_multilayer_patches(model, x)
            s = torch.sigmoid(head(patches)).squeeze(1)
            aug_scores.append(inv(s))
            del patches, s
        scores.append(torch.stack(aug_scores).mean(0).cpu().numpy())
        fns.extend(names)
        del imgs, aug_scores
    return np.concatenate(scores, axis=0), fns


@torch.no_grad()
def ensemble_infer(
    model,
    paths: list[str],
    heads: list[SegHead],
    good_paths: list[str],
    tta: bool = True,
    p_lo: float = P_LO,
    p_hi: float = P_HI,
) -> tuple[np.ndarray, list[str]]:
    """Normalised ensemble inference over multiple SegHeads.

    Each head is normalised independently — its P_LO/P_HI percentiles are
    computed on good_paths, then applied before averaging.  This keeps models
    with different output scales in the same space before combining.

    Returns:
        scores: (N, H, W) float32 in [0, 1], already blurred if BLUR_SIGMA > 0.
        fns:    Corresponding filename list.
    """
    normed: list[np.ndarray] = []
    fns_out: list[str] = []

    for rank, head in enumerate(heads):
        sh_good, _ = seghead_infer(model, good_paths, head, tta=False)
        lo = np.percentile(sh_good.flatten(), p_lo)
        hi = np.percentile(sh_good.flatten(), p_hi)

        sh, fns_out = seghead_infer(model, paths, head, tta=tta)
        if BLUR_SIGMA > 0:
            sh = np.stack([gaussian_filter(s, sigma=BLUR_SIGMA) for s in sh])
        norm = np.clip((sh - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
        normed.append(norm)
        print(f"    model {rank + 1}/{len(heads)}: lo={lo:.4f}  hi={hi:.4f}")

    return np.stack(normed).mean(0), fns_out


def run_inference(
    model,
    data_root: Path,
    seg_heads_dict: dict[str, list[SegHead]],
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Run normalised ensemble inference on test images for all classes.

    Returns:
        dict class_name -> (scores, filenames) where scores are (N, H, W) in [0, 1].
    """
    print("\nRunning inference...")
    test_scores: dict[str, tuple[np.ndarray, list[str]]] = {}
    for class_name, heads in sorted(seg_heads_dict.items()):
        good_paths = [str(p) for p in sorted((data_root / class_name / "train" / "good").glob("*.png"))]
        test_paths = [str(p) for p in sorted((data_root / class_name / "test").glob("*.png"))]
        print(f"\n  {class_name}: {len(test_paths)} test | {len(heads)} model(s) | 4-fold TTA")
        scores, fns = ensemble_infer(model, test_paths, heads, good_paths, tta=True)
        test_scores[class_name] = (scores, fns)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {class_name} done")
    return test_scores
