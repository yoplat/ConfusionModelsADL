import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    CORESET_RATIO,
    EPOCHS,
    LR,
    SAMPLES_PER_EPOCH,
    SEED,
    device,
    worker_init_fn,
)
from .coreset import coreset_subsample
from .datasets import TrainSynthDataset, preprocess
from .features import extract_multilayer_patches, extract_patch_features
from .model import SegHead, bce_dice_loss


# ── Public API ────────────────────────────────────────────────────────────────


def build_memory_banks(
    model,
    data_root: Path,
    coreset_ratio: float = CORESET_RATIO,
    seed: int = SEED,
    coreset_method: str = "random",
) -> dict[str, torch.Tensor]:
    """Build per-class L2-normalised memory banks from good training images.

    Args:
        model:           Frozen DINOv2 backbone.
        data_root:       Dataset root (one subdirectory per class).
        coreset_ratio:   Fraction of patch features to keep (default 1 %).
        seed:            RNG seed passed to the coreset sampler.
        coreset_method:  ``'random'`` or ``'greedy'`` — see
                         :func:`~coreset.coreset_subsample`.

    Returns:
        dict mapping class_name -> (K, FEATURE_DIM) normalised coreset tensor.
    """
    print(
        f"Building memory banks  (method={coreset_method}, ratio={coreset_ratio * 100:.1f}%)..."
    )
    banks: dict[str, torch.Tensor] = {}
    for class_dir in sorted(data_root.iterdir()):
        good_dir = class_dir / "train" / "good"
        if not good_dir.is_dir():
            continue
        class_name = class_dir.name
        paths = [str(p) for p in sorted(good_dir.glob("*.png"))]
        feats = extract_patch_features(model, paths)
        bank = F.normalize(
            coreset_subsample(
                feats, ratio=coreset_ratio, seed=seed, method=coreset_method
            ),
            p=2,
            dim=1,
        )
        banks[class_name] = bank
        print(
            f"  {class_name}: {len(paths)} images → bank {tuple(bank.shape)} "
            f"(kept {coreset_ratio * 100:.1f}% of {len(feats)} patches)"
        )
        del feats
        gc.collect()

    return banks


def collect_anomaly_sources(
    data_root: Path,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Load annotated anomaly pairs and split into training and validation sets.

    Per anomaly type: first 3 images go to training (cut-paste / real-anomaly
    branches), last 2 go to validation (held out for evaluation).

    Args:
        data_root: Dataset root.

    Returns:
        train_sources: class_name -> list of {'image': ndarray, 'mask': ndarray}.
        val_items:     class_name -> list of {'path': str, 'mask': ndarray}.
    """
    train_sources: dict[str, list[dict]] = defaultdict(list)
    val_items: dict[str, list[dict]] = defaultdict(list)
    for class_dir in sorted(data_root.iterdir()):
        train_dir = class_dir / "train"
        gt_dir = class_dir / "ground_truth_train"
        if not (train_dir.is_dir() and gt_dir.is_dir()):
            continue
        class_name = class_dir.name
        for ano_gt_dir in sorted(gt_dir.iterdir()):
            ano_img_dir = train_dir / ano_gt_dir.name
            pairs = []
            for mask_path in sorted(ano_gt_dir.glob("*.png")):
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask.max() == 0:
                    continue
                pairs.append((ano_img_dir / mask_path.name, mask))
            for img_path, mask in pairs[:3]:
                img = np.array(Image.open(img_path).convert("RGB"))
                train_sources[class_name].append({"image": img, "mask": mask})
            for img_path, mask in pairs[3:]:
                val_items[class_name].append({"path": str(img_path), "mask": mask})
    return train_sources, val_items


def train_seg_heads(
    model,
    data_root: Path,
    anomaly_sources: dict,
    epochs: int = EPOCHS,
    seed: int = SEED,
) -> dict[str, SegHead]:
    """Train one SegHead per class using synthetic cut-paste anomaly augmentation.

    DINOv2 features are extracted with ``torch.no_grad()``; only SegHead
    parameters are updated.

    Args:
        model:           Frozen DINOv2 backbone.
        data_root:       Dataset root.
        anomaly_sources: Per-class anomaly pairs from :func:`collect_anomaly_sources`.
        epochs:          Training epochs per class.
        seed:            Base seed; each class gets ``seed + class_index``.

    Returns:
        dict mapping class_name -> trained SegHead (eval mode).
    """
    print("\nTraining multi-layer SegHeads...")
    seg_heads: dict[str, SegHead] = {}
    for cls_idx, class_name in enumerate(sorted(anomaly_sources.keys())):
        good_dir = data_root / class_name / "train" / "good"
        good_paths = [str(p) for p in sorted(good_dir.glob("*.png"))]
        sources = anomaly_sources[class_name]

        # Per-class seed so classes are independently reproducible
        cls_seed = seed + cls_idx
        g = torch.Generator()
        g.manual_seed(cls_seed)

        loader = DataLoader(
            TrainSynthDataset(
                good_paths, sources, SAMPLES_PER_EPOCH, preprocess
            ),
            batch_size=BATCH_SIZE,
            num_workers=2,
            pin_memory=True,
            generator=g,
            worker_init_fn=worker_init_fn,
        )
        head = SegHead().to(device)
        optim = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)

        for epoch in range(epochs):
            head.train()
            epoch_loss = 0.0
            for imgs, masks in loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).unsqueeze(1)
                with torch.no_grad():
                    patches = extract_multilayer_patches(model, imgs)
                loss = bce_dice_loss(head(patches), masks)
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(
                    f"  {class_name} epoch {epoch + 1:2d}/{epochs}  "
                    f"loss={epoch_loss / len(loader):.4f}"
                )

        head.eval()
        seg_heads[class_name] = head
        print(f"  {class_name} done")
    return seg_heads
