"""
Evaluation utilities: metrics on held-out validation data + visualisations.

2 out of 5 images per anomaly type are reserved as a validation set and never
seen during SegHead training, giving an unbiased proxy for generalisation.
"""

import json
from pathlib import Path
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from .config import BLUR_SIGMA, IMG_SIZE, P_HI, P_LO, W_MB
from .inference import ensemble_infer, memorybank_infer, seghead_infer


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_good_paths(
    data_root: Path,
    class_name: str,
    max_good: int | None = 150,
    seed: int = 0,
) -> list[str]:
    """Return a (capped) list of good-image paths for one class.

    Args:
        data_root:  Dataset root.
        class_name: Class subdirectory name.
        max_good:   Cap on the number of good images (randomly sampled).
                    Pass ``None`` to use all.
        seed:       RNG seed for subsampling.

    Returns:
        List of good-image path strings (at most max_good).
    """
    all_good = sorted((data_root / class_name / "train" / "good").glob("*.png"))
    if max_good is not None and len(all_good) > max_good:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_good), size=max_good, replace=False)
        all_good = [all_good[i] for i in sorted(idx)]
    return [str(p) for p in all_good]


def _resize_mask(mask: np.ndarray) -> np.ndarray:
    """Resize a ground-truth mask to IMG_SIZE×IMG_SIZE and binarise."""
    return (
        np.array(
            Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        )
        > 0
    ).astype(np.uint8)


# ── Plot helpers ──────────────────────────────────────────────────────────────


def _save_roc_curve(
    gt_flat: np.ndarray,
    score_dict: dict[str, np.ndarray],
    class_name: str,
    save_path: Path,
) -> dict[str, float]:
    """Plot multi-branch pixel-level ROC curves and return AUC values.

    Args:
        gt_flat:    1-D binary ground-truth array.
        score_dict: Mapping branch-name -> 1-D float score array.
        class_name: Used in the plot title.
        save_path:  Output PNG path.

    Returns:
        dict mapping branch-name -> AUROC.
    """
    aucs = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(gt_flat, scores)
        auc = float(roc_auc_score(gt_flat, scores))
        aucs[label] = auc
        ax.plot(fpr, tpr, label=f"{label}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"Pixel-level ROC — {class_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return aucs


def _save_score_dist(
    good_scores: np.ndarray,
    ano_scores: np.ndarray,
    class_name: str,
    save_path: Path,
) -> None:
    """Histogram of per-image max scores for good vs. anomaly images.

    Args:
        good_scores: 1-D array of max-pixel scores for good images.
        ano_scores:  1-D array of max-pixel scores for anomaly images.
        class_name:  Used in the plot title.
        save_path:   Output PNG path.
    """
    def _safe_hist(ax, scores, bins, **kwargs):
        if len(scores) > 0 and np.ptp(scores) > 0:
            ax.hist(scores, bins=min(bins, len(scores)), **kwargs)
        elif len(scores) > 0:
            ax.axvline(float(scores[0]), lw=2, **{k: v for k, v in kwargs.items() if k != "alpha"})

    fig, ax = plt.subplots(figsize=(8, 5))
    _safe_hist(ax, good_scores, 30, alpha=0.6, color="steelblue", label="Good")
    _safe_hist(ax, ano_scores,  30, alpha=0.6, color="crimson",   label="Anomaly")
    ax.set_xlabel("Max pixel score (ensemble)")
    ax.set_ylabel("Count")
    ax.set_title(f"Image-level score distribution — {class_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_heatmaps(
    items: list[dict],
    ens_scores: np.ndarray,
    class_name: str,
    save_path: Path,
    n: int = 4,
    vis_scores: np.ndarray | None = None,
) -> None:
    """Grid of (original | GT mask | raw score | normalised score | overlay) for anomaly examples.

    Args:
        items:      List of {'path', 'mask'} anomaly dicts (sorted by score).
        ens_scores: (N_anomaly, H, W) raw ensemble score array matching items.
        class_name: Used in the figure title.
        save_path:  Output PNG path.
        n:          Number of examples to show.
        vis_scores: (N_anomaly, H, W) normalised scores; if provided a second
                    score column is shown alongside the raw one.
    """
    n = min(n, len(items))
    if n == 0:
        return

    order = np.argsort([s.max() for s in ens_scores])[::-1][:n]

    cols = (
        ["Original", "GT Mask", "Raw Score", "Norm Score", "Overlay"]
        if vis_scores is not None
        else ["Original", "GT Mask", "Score Map", "Overlay"]
    )
    fig, axes = plt.subplots(
        n, len(cols), figsize=(3.5 * len(cols), 3.5 * n), squeeze=False
    )
    for col, title in enumerate(cols):
        axes[0, col].set_title(title, fontsize=10)

    for row, idx in enumerate(order):
        path = items[idx]["path"]
        mask = _resize_mask(items[idx]["mask"])
        score = ens_scores[idx]

        img = np.array(
            Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        )

        axes[row, 0].imshow(img)
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(score, cmap="hot", vmin=0, vmax=score.max() + 1e-8)
        if vis_scores is not None:
            vs = vis_scores[idx]
            axes[row, 3].imshow(vs, cmap="hot", vmin=0, vmax=1)
            axes[row, 4].imshow(img)
            axes[row, 4].imshow(vs, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        else:
            axes[row, 3].imshow(img)
            axes[row, 3].imshow(
                score, cmap="hot", alpha=0.55, vmin=0, vmax=score.max() + 1e-8
            )
        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(f"Top anomaly heatmaps — {class_name}", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _collect_zeromask_paths(data_root: Path, class_name: str) -> list[str]:
    """Return paths to anomaly images whose GT mask is entirely black."""
    paths = []
    gt_dir = data_root / class_name / "ground_truth_train"
    train_dir = data_root / class_name / "train"
    if not gt_dir.is_dir():
        return paths
    for ano_gt_dir in sorted(gt_dir.iterdir()):
        ano_img_dir = train_dir / ano_gt_dir.name
        for mask_path in sorted(ano_gt_dir.glob("*.png")):
            if np.array(Image.open(mask_path).convert("L")).max() == 0:
                img_path = ano_img_dir / mask_path.name
                if img_path.exists():
                    paths.append(str(img_path))
    return paths


def _save_score_heatmaps(
    paths: list[str],
    scores: np.ndarray,
    title: str,
    save_path: Path,
    n: int = 4,
) -> None:
    """Grid of (original | score map | overlay) sorted by highest score, no GT mask."""
    n = min(n, len(paths))
    if n == 0:
        return
    order = np.argsort([s.max() for s in scores])[::-1][:n]
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.5 * n), squeeze=False)
    for col, t in enumerate(["Original", "Score Map", "Overlay"]):
        axes[0, col].set_title(t, fontsize=10)
    for row, idx in enumerate(order):
        score = scores[idx]
        img = np.array(
            Image.open(paths[idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        )
        axes[row, 0].imshow(img)
        axes[row, 1].imshow(score, cmap="hot", vmin=0, vmax=1)
        axes[row, 2].imshow(img)
        axes[row, 2].imshow(score, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Per-class evaluation ──────────────────────────────────────────────────────


def _evaluate_class(
    model,
    data_root: Path,
    class_name: str,
    memory_banks: dict,
    seg_heads: dict,
    val_items: list[dict],
    plot_dir: Path,
    n_vis: int = 4,
    max_good: int | None = 150,
    seed: int = 0,
) -> dict:
    """Compute pixel- and image-level metrics for one class; save plots.

    Args:
        model:        Frozen DINOv2 backbone.
        data_root:    Dataset root.
        class_name:   Class to evaluate.
        memory_banks: Per-class memory banks (may be empty if W_MB == 0).
        seg_heads:    Per-class SegHead models.
        val_items:    Held-out anomaly items: list of {'path': str, 'mask': ndarray}.
        plot_dir:     Directory where PNG files are saved.
        n_vis:        Number of anomaly examples to visualise.
        max_good:     Cap on good images for evaluation (None = all).
        seed:         RNG seed for good-image subsampling.

    Returns:
        Metrics dict with keys: pixel_auroc_{mb,sh,ens}, image_auroc_ens,
        avg_precision_ens, n_good, n_anomaly.
    """
    W_SH = 1.0 - W_MB
    good_paths = _load_good_paths(data_root, class_name, max_good, seed)
    if not val_items or not good_paths:
        return {}

    ano_paths = [it["path"] for it in val_items]
    all_paths = good_paths + ano_paths
    n_good = len(good_paths)

    # ── Raw inference scores ──────────────────────────────────────────────────
    heads = seg_heads[class_name]
    if isinstance(heads, list):
        # Multi-head: use normalised ensemble on good_paths for normalisation reference
        sh_scores, _ = ensemble_infer(model, all_paths, heads, good_paths)
    else:
        sh_scores, _ = seghead_infer(model, all_paths, heads)
    if W_MB > 0:
        mb_scores, _ = memorybank_infer(
            model, all_paths, memory_banks[class_name]
        )
        ens_scores = W_MB * mb_scores + W_SH * sh_scores
    else:
        mb_scores = None
        ens_scores = sh_scores.copy()
    if BLUR_SIGMA > 0:
        ens_scores = np.stack(
            [gaussian_filter(s, sigma=BLUR_SIGMA) for s in ens_scores]
        )

    # ── Ground-truth pixel masks ──────────────────────────────────────────────
    zero = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    gt_masks = [zero] * n_good + [_resize_mask(it["mask"]) for it in val_items]
    gt_flat = np.stack(gt_masks).flatten().astype(int)

    if gt_flat.min() == gt_flat.max():
        return {}

    # ── Pixel-level AUROC ─────────────────────────────────────────────────────
    score_dict = {
        "SegHead": sh_scores.flatten(),
        "Ensemble": ens_scores.flatten(),
    }
    if mb_scores is not None:
        score_dict["Memory Bank"] = mb_scores.flatten()
    aucs = _save_roc_curve(
        gt_flat, score_dict, class_name, plot_dir / f"{class_name}_roc.png"
    )

    # ── Image-level AUROC ─────────────────────────────────────────────────────
    image_scores = ens_scores.reshape(len(all_paths), -1).max(axis=1)
    image_labels = [0] * n_good + [1] * len(val_items)
    image_auroc = float(roc_auc_score(image_labels, image_scores))
    avg_prec = float(average_precision_score(image_labels, image_scores))

    # ── Score distribution plot ───────────────────────────────────────────────
    _save_score_dist(
        image_scores[:n_good],
        image_scores[n_good:],
        class_name,
        plot_dir / f"{class_name}_dist.png",
    )

    # ── Heatmap visualisations (normalised to match submission) ──────────────
    _lo = np.percentile(sh_scores[:n_good].flatten(), P_LO)
    _hi = np.percentile(sh_scores[:n_good].flatten(), P_HI)
    vis_scores = np.clip((ens_scores - _lo) / (_hi - _lo + 1e-8), 0, 1).astype(
        np.float32
    )
    _save_heatmaps(
        val_items,
        ens_scores[n_good:],
        class_name,
        plot_dir / f"{class_name}_heatmaps.png",
        n=n_vis,
        vis_scores=vis_scores[n_good:],
    )

    # Good images — highest scorers are the worst false positives
    _save_score_heatmaps(
        good_paths,
        vis_scores[:n_good],
        f"Top false positives (good images) — {class_name}",
        plot_dir / f"{class_name}_good_heatmaps.png",
        n=n_vis,
    )

    # Anomaly images with no pixel annotation (all-black mask)
    zm_paths = _collect_zeromask_paths(data_root, class_name)
    if zm_paths:
        if isinstance(heads, list):
            zm_sh, _ = ensemble_infer(model, zm_paths, heads, good_paths)
        else:
            zm_sh, _ = seghead_infer(model, zm_paths, heads)
        if BLUR_SIGMA > 0:
            zm_sh = np.stack(
                [gaussian_filter(s, sigma=BLUR_SIGMA) for s in zm_sh]
            )
        zm_vis = np.clip((zm_sh - _lo) / (_hi - _lo + 1e-8), 0, 1).astype(
            np.float32
        )
        _save_score_heatmaps(
            zm_paths,
            zm_vis,
            f"Zero-mask anomalies (no pixel GT) — {class_name}",
            plot_dir / f"{class_name}_zeromask_heatmaps.png",
            n=n_vis,
        )

    metrics = {
        "pixel_auroc_sh": aucs["SegHead"],
        "pixel_auroc_ens": aucs["Ensemble"],
        "image_auroc_ens": image_auroc,
        "avg_precision_ens": avg_prec,
        "n_good": n_good,
        "n_anomaly": len(val_items),
    }
    if mb_scores is not None:
        metrics["pixel_auroc_mb"] = aucs["Memory Bank"]
    return metrics


# ── Top-level entry point ─────────────────────────────────────────────────────


def evaluate_all(
    model,
    data_root: Path,
    memory_banks: dict,
    seg_heads: dict,
    val_sources: dict[str, list[dict]],
    run_dir: Path,
    n_vis: int = 4,
    max_good: int | None = 150,
    seed: int = 0,
) -> dict:
    """Evaluate all classes on held-out validation anomalies and print a summary.

    Results (metrics + plots) are saved under ``run_dir/evaluation/``.

    Args:
        model:        Frozen DINOv2 backbone.
        data_root:    Dataset root.
        memory_banks: Per-class memory banks (may be empty if W_MB == 0).
        seg_heads:    Per-class SegHead models.
        val_sources:  Held-out anomaly items per class from collect_anomaly_sources.
        run_dir:      Timestamped run directory.
        n_vis:        Anomaly heatmap examples per class.
        max_good:     Max good images per class for evaluation (None = all).
        seed:         RNG seed for good-image subsampling.

    Returns:
        Nested dict ``{class_name: {metric: value}}``.
    """
    plot_dir = run_dir / "evaluation" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    print("\nEvaluating on held-out validation anomalies...")
    for class_name in tqdm(sorted(seg_heads.keys()), desc="Evaluating classes"):
        m = _evaluate_class(
            model,
            data_root,
            class_name,
            memory_banks,
            seg_heads,
            val_sources.get(class_name, []),
            plot_dir,
            n_vis=n_vis,
            max_good=max_good,
            seed=seed,
        )
        if m:
            all_metrics[class_name] = m

    # ── Print summary table ───────────────────────────────────────────────────
    use_mb = W_MB > 0
    header = (
        f"{'Class':<12}"
        + (f" {'Px-AUC MB':>10}" if use_mb else "")
        + f" {'Px-AUC SH':>10} {'Px-AUC Ens':>11} {'Img-AUC':>8} {'AvgPrec':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for cls, m in all_metrics.items():
        row = f"{cls:<12}"
        if use_mb:
            row += f" {m.get('pixel_auroc_mb', 0):>10.3f}"
        row += (
            f" {m['pixel_auroc_sh']:>10.3f}"
            f" {m['pixel_auroc_ens']:>11.3f}"
            f" {m['image_auroc_ens']:>8.3f}"
            f" {m['avg_precision_ens']:>8.3f}"
        )
        print(row)

    if all_metrics:
        keys = [
            "pixel_auroc_sh",
            "pixel_auroc_ens",
            "image_auroc_ens",
            "avg_precision_ens",
        ]
        if use_mb:
            keys = ["pixel_auroc_mb"] + keys
        means = {
            k: float(np.mean([m[k] for m in all_metrics.values() if k in m]))
            for k in keys
        }
        print("-" * len(header))
        row = f"{'MEAN':<12}"
        if use_mb:
            row += f" {means.get('pixel_auroc_mb', 0):>10.3f}"
        row += (
            f" {means['pixel_auroc_sh']:>10.3f}"
            f" {means['pixel_auroc_ens']:>11.3f}"
            f" {means['image_auroc_ens']:>8.3f}"
            f" {means['avg_precision_ens']:>8.3f}"
        )
        print(row)
        all_metrics["_mean"] = means

    # ── Persist metrics ───────────────────────────────────────────────────────
    metrics_path = run_dir / "evaluation" / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Plots saved to   {plot_dir}/")

    return all_metrics
