"""Training utilities: anomaly source collection, SegHead training, multi-run."""

import copy
import gc
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    CLASS_CONFIG,
    CORESET_RATIO,
    EPOCHS,
    IMG_SIZE,
    LR,
    N_RUNS,
    NUM_WORKERS,
    PATIENCE,
    SAMPLES_PER_EPOCH,
    SEED,
    TOP_K,
    device,
    worker_init_fn,
)
from .coreset import coreset_subsample
from .datasets import TrainSynthDataset, preprocess
from .features import extract_multilayer_patches, extract_patch_features
from .model import SegHead, bce_tversky_loss


# ── Public API ────────────────────────────────────────────────────────────────


def build_memory_banks(
    model,
    data_root: Path,
    coreset_ratio: float = CORESET_RATIO,
    seed: int = SEED,
    coreset_method: str = "random",
) -> dict[str, torch.Tensor]:
    """Build per-class L2-normalised memory banks from good training images."""
    print(f"Building memory banks  (method={coreset_method}, ratio={coreset_ratio * 100:.1f}%)...")
    banks: dict[str, torch.Tensor] = {}
    for class_dir in sorted(data_root.iterdir()):
        good_dir = class_dir / "train" / "good"
        if not good_dir.is_dir():
            continue
        class_name = class_dir.name
        paths = [str(p) for p in sorted(good_dir.glob("*.png"))]
        feats = extract_patch_features(model, paths)
        bank = F.normalize(
            coreset_subsample(feats, ratio=coreset_ratio, seed=seed, method=coreset_method),
            p=2,
            dim=1,
        )
        banks[class_name] = bank
        print(f"  {class_name}: {len(paths)} images → bank {tuple(bank.shape)}")
        del feats
        gc.collect()
    return banks


def collect_sources_split(
    data_root: Path,
    class_name: str,
    good_paths: list[str],
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Per-class train/val split matching the v16 notebook strategy.

    For each anomaly type, one view is chosen randomly (seed) as the val item;
    the rest go to train_sources.  All non-zero-mask images feed cutpaste_sources
    (including the val view so cut-paste diversity isn't reduced).

    val_items contains the anomaly val views + an equal number of randomly
    sampled good images (for a balanced AP signal during early stopping).

    Returns:
        train_sources:    Real anomaly images for direct supervision.
        cutpaste_sources: All non-zero-mask images as cut-paste donors.
        val_items:        Held-out items for early-stopping AP computation.
    """
    train_sources: list[dict] = []
    cutpaste_sources: list[dict] = []
    val_ano: list[dict] = []

    train_dir = data_root / class_name / "train"
    gt_dir = data_root / class_name / "ground_truth_train"
    if not (train_dir.is_dir() and gt_dir.is_dir()):
        return train_sources, cutpaste_sources, []

    rng = random.Random(seed)
    for ano_gt_dir in sorted(gt_dir.iterdir()):
        ano_img_dir = train_dir / ano_gt_dir.name
        pairs: list[tuple[str, np.ndarray]] = []
        for mask_path in sorted(ano_gt_dir.glob("*.png")):
            mask = np.array(Image.open(mask_path).convert("L"))
            if mask.max() == 0:
                continue
            pairs.append((str(ano_img_dir / mask_path.name), mask))

        if not pairs:
            continue

        if len(pairs) == 1:
            img_path, mask = pairs[0]
            img = np.array(Image.open(img_path).convert("RGB"))
            entry = {"image": img, "mask": mask, "path": img_path}
            train_sources.append(entry)
            cutpaste_sources.append(entry)
            continue

        val_idx = rng.randint(0, len(pairs) - 1)
        val_path, val_mask = pairs[val_idx]
        val_img = np.array(Image.open(val_path).convert("RGB"))
        val_ano.append({"path": val_path, "mask": val_mask, "is_good": False})
        cutpaste_sources.append({"image": val_img, "mask": val_mask, "path": val_path})

        for i, (img_path, mask) in enumerate(pairs):
            if i == val_idx:
                continue
            img = np.array(Image.open(img_path).convert("RGB"))
            entry = {"image": img, "mask": mask, "path": img_path}
            train_sources.append(entry)
            cutpaste_sources.append(entry)

    n_good_val = len(val_ano)
    rng_np = np.random.RandomState(seed)
    good_val_idx = rng_np.choice(len(good_paths), size=min(n_good_val, len(good_paths)), replace=False)
    val_items = val_ano.copy()
    for idx in good_val_idx:
        val_items.append({"path": good_paths[idx], "mask": np.zeros((1, 1), dtype=np.uint8), "is_good": True})

    print(
        f"  {class_name}[{seed}]: train={len(train_sources)} cp={len(cutpaste_sources)} "
        f"val_ano={len(val_ano)} val_good={n_good_val}"
    )
    return train_sources, cutpaste_sources, val_items


def collect_anomaly_sources(
    data_root: Path,
    seed: int = SEED,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Backwards-compatible wrapper used by ensemble.py.

    Returns the old-style (train_sources dict, val_items dict) using
    collect_sources_split with a fixed seed.  val_items contains anomaly images
    only (good images stripped out).
    """
    train_dict: dict[str, list[dict]] = {}
    val_dict: dict[str, list[dict]] = {}
    for class_dir in sorted(data_root.iterdir()):
        if not (class_dir / "ground_truth_train").is_dir():
            continue
        class_name = class_dir.name
        good_paths = [str(p) for p in sorted((class_dir / "train" / "good").glob("*.png"))]
        train_src, _, val_items = collect_sources_split(data_root, class_name, good_paths, seed)
        train_dict[class_name] = train_src
        val_dict[class_name] = [it for it in val_items if not it.get("is_good", False)]
    return train_dict, val_dict


@torch.no_grad()
def _val_pixel_ap(model, head: SegHead, val_items: list[dict]) -> float:
    """Pixel-level AP on val_items (anomaly images + balanced good images)."""
    head.eval()
    all_preds: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []

    for item in val_items:
        img = preprocess(Image.open(item["path"]).convert("RGB")).unsqueeze(0).to(device)
        patches = extract_multilayer_patches(model, img)
        score = torch.sigmoid(head(patches)).squeeze().cpu().numpy()
        del img, patches

        if item.get("is_good", False):
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            mask = (item["mask"] > 0).astype(np.uint8)
            if mask.shape != (IMG_SIZE, IMG_SIZE):
                mask = (
                    np.array(
                        Image.fromarray(mask * 255).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
                    ) > 127
                ).astype(np.uint8)

        all_preds.append(score.flatten())
        all_gt.append(mask.flatten())

    if device.type == "cuda":
        torch.cuda.empty_cache()

    gt = np.concatenate(all_gt)
    if gt.max() == 0:
        return 0.0
    return float(average_precision_score(gt, np.concatenate(all_preds)))


def _train_one(
    model,
    data_root: Path,
    class_name: str,
    train_sources: list[dict],
    cp_sources: list[dict],
    val_items: list[dict],
    l1_lambda: float = 0.0,
    p_good: float = 0.40,
    p_real: float = 0.30,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    seed: int = SEED,
) -> tuple[SegHead, float]:
    """Train one SegHead for one class for one run.

    Returns:
        head:    Trained SegHead (eval mode, best weights restored).
        best_ap: Best val pixel-AP achieved during training.
    """
    good_paths = [str(p) for p in sorted((data_root / class_name / "train" / "good").glob("*.png"))]
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        TrainSynthDataset(good_paths, train_sources, cp_sources, SAMPLES_PER_EPOCH, preprocess, p_good, p_real),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    head = SegHead().to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=LR * 0.01)

    best_ap = -1.0
    best_state: dict | None = None
    no_improve = 0
    stopped_at = epochs

    ckpt_dir = data_root.parent / "output" / "checkpoints" / class_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        head.train()
        epoch_loss = 0.0
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).unsqueeze(1)
            optim.zero_grad(set_to_none=True)
            with torch.no_grad():
                patches = extract_multilayer_patches(model, imgs)
            loss = bce_tversky_loss(head(patches), masks, l1_lambda=l1_lambda)
            loss.backward()
            optim.step()
            scheduler.step()
            epoch_loss += loss.item()
            del imgs, masks, patches, loss

        avg_loss = epoch_loss / len(loader)
        val_ap = _val_pixel_ap(model, head, val_items) if val_items else 0.0
        head.train()

        improved = val_ap > best_ap
        if improved:
            best_ap = val_ap
            no_improve = 0
            del best_state
            best_state = copy.deepcopy(head.state_dict())
            torch.save({"state_dict": head.state_dict(), "val_ap": best_ap, "epoch": epoch}, ckpt_dir / f"best_seed{seed}.pt")
        else:
            no_improve += 1

        bar = "#" * int(val_ap * 20) + "." * (20 - int(val_ap * 20))
        star = "*" if improved else " "
        warn = f" >{no_improve}/{patience}" if no_improve > 0 else ""
        print(f"    ep {epoch + 1:>4}  loss={avg_loss:.4f}  val_ap={val_ap:.4f} [{bar}]  best={best_ap:.4f}{star}{warn}")

        if no_improve >= patience:
            stopped_at = epoch + 1
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()

    del best_state, loader, optim, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"    -> best_val_ap={best_ap:.4f} (ep.{stopped_at})")
    return head, best_ap


def train_all_classes(
    model,
    data_root: Path,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    seed: int = SEED,
) -> dict[str, list[SegHead]]:
    """Train N_RUNS SegHeads per class and return the TOP_K best ones.

    Per-class hyperparameters (l1, p_good, p_real) are read from CLASS_CONFIG.
    Each run uses a different seed for the val split so runs see diverse data.

    Returns:
        dict mapping class_name -> list of TOP_K SegHeads (eval mode).
    """
    print(f"\nTraining SegHeads  (N_RUNS={N_RUNS}, TOP_K={TOP_K})...")
    best_models: dict[str, list[SegHead]] = {}

    for class_dir in sorted(data_root.iterdir()):
        if not (class_dir / "ground_truth_train").is_dir():
            continue
        class_name = class_dir.name
        cfg = CLASS_CONFIG.get(class_name, {"l1": 0.0, "p_good": 0.40, "p_real": 0.30})
        l1, p_good, p_real = cfg["l1"], cfg["p_good"], cfg["p_real"]

        good_paths = [str(p) for p in sorted((class_dir / "train" / "good").glob("*.png"))]

        print(f"\n{'=' * 56}")
        cp_pct = (1 - p_good - p_real) * 100
        print(f"  {class_name}  L1={l1} | {p_good * 100:.0f}%g/{p_real * 100:.0f}%r/{cp_pct:.0f}%cp")
        print(f"{'=' * 56}")

        run_results: list[tuple[float, SegHead]] = []
        for run_idx in range(N_RUNS):
            run_seed = seed + run_idx * 100 + hash(class_name) % 1000
            print(f"\n  Run {run_idx + 1}/{N_RUNS}  [seed={run_seed}]")
            train_src, cp_src, val_items = collect_sources_split(
                data_root, class_name, good_paths, seed=run_seed
            )
            if not train_src and not cp_src:
                print(f"  {class_name}: no anomaly sources, skipping")
                break
            head, best_ap = _train_one(
                model, data_root, class_name,
                train_src, cp_src, val_items,
                l1_lambda=l1, p_good=p_good, p_real=p_real,
                epochs=epochs, patience=patience, seed=run_seed,
            )
            run_results.append((best_ap, head))

        if not run_results:
            continue

        run_results.sort(key=lambda x: x[0], reverse=True)
        best_models[class_name] = [h for _, h in run_results[:TOP_K]]

        top_str = " | ".join([f"#{i + 1} ap={ap:.4f}" for i, (ap, _) in enumerate(run_results[:TOP_K])])
        print(f"\n  {class_name} TOP_{TOP_K}: {top_str}")

    print("\nTraining complete!")
    return best_models


# Kept for backwards-compat (used by old train_seg_heads callers if any)
def train_seg_heads(
    model,
    data_root: Path,
    anomaly_sources: dict,
    val_sources: dict,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    seed: int = SEED,
) -> dict[str, SegHead]:
    """Single-run training, one head per class.  Kept for ensemble.py compat."""
    print("\nTraining SegHeads (single-run mode)...")
    seg_heads: dict[str, SegHead] = {}
    for cls_idx, class_name in enumerate(sorted(anomaly_sources.keys())):
        good_paths = [str(p) for p in sorted((data_root / class_name / "train" / "good").glob("*.png"))]
        sources = anomaly_sources[class_name]
        cls_seed = seed + cls_idx
        val_items = [
            dict(it, is_good=False) for it in val_sources.get(class_name, [])
        ]
        head, _ = _train_one(
            model, data_root, class_name,
            sources, sources, val_items,
            epochs=epochs, patience=patience, seed=cls_seed,
        )
        seg_heads[class_name] = head
    return seg_heads
