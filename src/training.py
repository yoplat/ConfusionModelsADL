import gc
import json
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


# ── Memory bank cache helpers ─────────────────────────────────────────────────


def _bank_cache_key(
    data_root: Path, coreset_ratio: float, seed: int, coreset_method: str
) -> dict:
    """Return the dict that uniquely identifies a set of memory banks."""
    return {
        "data_root": str(data_root.resolve()),
        "coreset_ratio": coreset_ratio,
        "seed": seed,
        "coreset_method": coreset_method,
    }


def _load_bank_cache(
    cache_dir: Path,
    data_root: Path,
    coreset_ratio: float,
    seed: int,
    coreset_method: str,
) -> dict[str, torch.Tensor] | None:
    """Return cached banks if they exist and were built with the same parameters.

    Returns ``None`` on any mismatch so the caller falls back to a full rebuild.
    """
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        saved = json.load(f)
    if saved != _bank_cache_key(data_root, coreset_ratio, seed, coreset_method):
        return None
    banks: dict[str, torch.Tensor] = {}
    for pt in sorted(cache_dir.glob("*.pt")):
        banks[pt.stem] = torch.load(pt, map_location="cpu")
    return banks or None


def _save_bank_cache(
    cache_dir: Path,
    banks: dict[str, torch.Tensor],
    data_root: Path,
    coreset_ratio: float,
    seed: int,
    coreset_method: str,
) -> None:
    """Persist memory banks and the parameters used to build them."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for cls, bank in banks.items():
        torch.save(bank, cache_dir / f"{cls}.pt")
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(
            _bank_cache_key(data_root, coreset_ratio, seed, coreset_method),
            f,
            indent=2,
        )


# ── Public API ────────────────────────────────────────────────────────────────


def build_memory_banks(
    model,
    data_root: Path,
    coreset_ratio: float = CORESET_RATIO,
    seed: int = SEED,
    coreset_method: str = "random",
    cache_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Build (or restore from cache) per-class L2-normalised memory banks.

    On the first run with a given ``(data_root, coreset_ratio, seed, method)``
    combination the banks are computed from scratch and saved to ``cache_dir``.
    On subsequent runs with the same parameters the banks are loaded directly,
    skipping both feature extraction and subsampling.

    Pass ``cache_dir=None`` to disable caching entirely.

    Args:
        model:           Frozen DINOv2 backbone.
        data_root:       Dataset root (one subdirectory per class).
        coreset_ratio:   Fraction of patch features to keep (default 1 %).
        seed:            RNG seed passed to the coreset sampler.
        coreset_method:  ``'random'`` or ``'greedy'`` — see
                         :func:`~coreset.coreset_subsample`.
        cache_dir:       Directory used to persist banks between runs.

    Returns:
        dict mapping class_name -> (K, FEATURE_DIM) normalised coreset tensor.
    """
    if cache_dir is not None:
        cached = _load_bank_cache(
            cache_dir, data_root, coreset_ratio, seed, coreset_method
        )
        if cached is not None:
            print(
                f"Memory banks loaded from cache ({cache_dir})  "
                f"[{', '.join(f'{k}: {tuple(v.shape)}' for k, v in cached.items())}]"
            )
            return cached
        print("Cache miss — rebuilding memory banks...")

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

    if cache_dir is not None:
        _save_bank_cache(
            cache_dir, banks, data_root, coreset_ratio, seed, coreset_method
        )
        print(f"Memory banks cached → {cache_dir}")

    return banks


def collect_anomaly_sources(data_root: Path) -> dict[str, list[dict]]:
    """Load all annotated anomaly image/mask pairs for cut-paste augmentation.

    Skips masks with no foreground pixels.

    Args:
        data_root: Dataset root.

    Returns:
        defaultdict mapping class_name -> list of {'image': ndarray, 'mask': ndarray}.
    """
    sources: dict[str, list[dict]] = defaultdict(list)
    for class_dir in sorted(data_root.iterdir()):
        train_dir = class_dir / "train"
        gt_dir = class_dir / "ground_truth_train"
        if not (train_dir.is_dir() and gt_dir.is_dir()):
            continue
        class_name = class_dir.name
        for ano_gt_dir in sorted(gt_dir.iterdir()):
            ano_img_dir = train_dir / ano_gt_dir.name
            for mask_path in sorted(ano_gt_dir.glob("*.png")):
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask.max() == 0:
                    continue
                img = np.array(
                    Image.open(ano_img_dir / mask_path.name).convert("RGB")
                )
                sources[class_name].append({"image": img, "mask": mask})
    return sources


def train_seg_heads(
    model,
    data_root: Path,
    anomaly_sources: dict,
    output_dir: Path,
    epochs: int = EPOCHS,
    seed: int = SEED,
) -> dict[str, SegHead]:
    """Train one SegHead per class using synthetic cut-paste anomaly augmentation.

    DINOv2 features are extracted with ``torch.no_grad()``; only SegHead
    parameters are updated. Trained heads are saved to ``output_dir``.

    Args:
        model:           Frozen DINOv2 backbone.
        data_root:       Dataset root.
        anomaly_sources: Per-class anomaly pairs from :func:`collect_anomaly_sources`.
        output_dir:      Directory for intermediate checkpoints.
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
        ckpt = output_dir / f"seghead_v7_{class_name}.pt"
        torch.save(head.state_dict(), ckpt)
        print(f"  {class_name} done")
    return seg_heads
