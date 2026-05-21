"""
Ensemble multiple saved runs into a single submission.

Usage examples:
  # Average the three best runs by val avg_precision_ens:
  python ensemble.py --top 3

  # Average specific runs:
  python ensemble.py --runs 20260510_221607 20260511_213752

  # Majority vote (threshold each map at 0.5, vote across runs):
  python ensemble.py --top 3 --method vote
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import warnings
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore", message="xFormers is not available")

from src.checkpoint import load_run
from src.config import BLUR_SIGMA, IMG_SIZE, P_HI, P_LO, SEED, W_MB, device
from src.evaluation import (
    _collect_zeromask_paths,
    _resize_mask,
    _save_heatmaps,
    _save_score_heatmaps,
)
from src.inference import ensemble_infer, seghead_infer
from src.submission import float_matrix_to_q8rle
from src.training import collect_anomaly_sources

DATA_ROOT = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / "output"
RUNS_DIR = OUTPUT_DIR / "runs"


def _infer_multilayer_dim(run_dir: Path) -> int:
    """Read in_dim from the first checkpoint in a run without loading the full model."""
    pts = sorted((run_dir / "seg_heads").glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt files in {run_dir / 'seg_heads'}")
    state = torch.load(pts[0], map_location="cpu", weights_only=True)
    return int(state["conv1.weight"].shape[1])


_BACKBONE_FOR_DIM = {
    1152: "dinov2_vits14_reg",  # ViT-S/14 @ 224px, 3 layers × 384
    2304: "dinov2_vitb14_reg",  # ViT-B/14 @ 224px, 3 layers × 768
}


def _get_top_runs(n: int) -> list[Path]:
    """Return the top-N run dirs ranked by mean val avg_precision_ens."""
    scored = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        metrics_path = run_dir / "evaluation" / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        ap = metrics.get("_mean", {}).get("avg_precision_ens", 0.0)
        scored.append((ap, run_dir))
    scored.sort(reverse=True)
    selected = [rd for _, rd in scored[:n]]
    for ap, rd in scored[:n]:
        print(f"  {rd.name}  avg_precision_ens={ap:.4f}")
    return selected


def _load_config(run_dir: Path) -> dict:
    p = run_dir / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _sample_good_paths(
    cls_dir: Path, max_good: int, seed: int = 0
) -> list[str]:
    all_paths = sorted((cls_dir / "train" / "good").glob("*.png"))
    if max_good and len(all_paths) > max_good:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_paths), size=max_good, replace=False)
        all_paths = [all_paths[i] for i in sorted(idx)]
    return [str(p) for p in all_paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble saved runs into one submission"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--runs",
        nargs="+",
        metavar="RUN",
        help="Run timestamps (or full paths) to ensemble",
    )
    group.add_argument(
        "--top",
        type=int,
        metavar="N",
        help="Auto-select top N runs by val avg_precision_ens",
    )
    parser.add_argument(
        "--method",
        choices=["average", "vote"],
        default="average",
        help="Combination method (default: average)",
    )
    parser.add_argument(
        "--vote-threshold",
        type=float,
        default=0.5,
        help="Per-map threshold used before majority vote (default: 0.5)",
    )
    parser.add_argument(
        "--n-vis",
        type=int,
        default=4,
        help="Anomaly examples per class in heatmaps (default: 4)",
    )
    parser.add_argument(
        "--max-good",
        type=int,
        default=150,
        help="Good images per class used for AP computation (default: 150)",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    # ── Resolve run directories ───────────────────────────────────────────────
    if args.top:
        print(f"Selecting top {args.top} runs:")
        run_dirs = _get_top_runs(args.top)
    else:
        run_dirs = []
        for r in args.runs:
            p = Path(r)
            run_dirs.append(p if p.is_absolute() else RUNS_DIR / r)
        for rd in run_dirs:
            print(f"  {rd.name}")

    if not run_dirs:
        raise SystemExit("No valid run directories found.")

    # ── Detect backbone from checkpoints; verify all runs match ───────────────
    dims = {rd: _infer_multilayer_dim(rd) for rd in run_dirs}
    unique_dims = set(dims.values())
    if len(unique_dims) > 1:
        detail = "\n".join(f"  {rd.name}: in_dim={d}" for rd, d in dims.items())
        raise SystemExit(
            f"Runs have incompatible SegHead input dimensions:\n{detail}\n"
            "Only runs trained with the same backbone can be ensembled."
        )
    multilayer_dim = unique_dims.pop()
    backbone_name = _BACKBONE_FOR_DIM.get(multilayer_dim)
    if backbone_name is None:
        raise SystemExit(f"Unknown multilayer_dim={multilayer_dim}; add it to _BACKBONE_FOR_DIM.")

    # ── Load DINOv2 backbone (shared across all runs) ─────────────────────────
    print(f"\nLoading {backbone_name} (in_dim={multilayer_dim})...")
    dinov2 = torch.hub.load(
        "facebookresearch/dinov2", backbone_name, verbose=False
    )
    dinov2 = dinov2.to(device).eval()
    for p in dinov2.parameters():
        p.requires_grad = False

    # ── Load val items (GT masks for heatmaps + AP) ───────────────────────────
    _, val_sources = collect_anomaly_sources(args.data_root)

    # ── Inference per run ─────────────────────────────────────────────────────
    # norm_test[cls] = list of (N_test, H, W) normalised arrays, one per run
    # norm_val[cls]  = list of (N_val,  H, W) normalised arrays, one per run
    # good_acc[cls]  = running sum of normalised good-image scores (averaged at end)
    norm_test: dict[str, list[np.ndarray]] = {}
    norm_val: dict[str, list[np.ndarray]] = {}
    good_acc: dict[str, np.ndarray] = {}
    good_count: dict[str, int] = {}
    zeromask_acc: dict[str, np.ndarray] = {}
    zeromask_count: dict[str, int] = {}
    good_paths_per_cls: dict[str, list[str]] = {}
    zeromask_paths_per_cls: dict[str, list[str]] = {}
    test_filenames: dict[str, list[str]] = {}
    class_names: list[str] | None = None

    for run_dir in run_dirs:
        _, seg_heads = load_run(run_dir)
        if class_names is None:
            class_names = sorted(seg_heads.keys())

        print(f"\n{run_dir.name}")
        for cls in class_names:
            # All good images used as normalisation reference for ensemble_infer
            all_good_paths = [
                str(p)
                for p in sorted(
                    (args.data_root / cls / "train" / "good").glob("*.png")
                )
            ]
            good_paths = _sample_good_paths(args.data_root / cls, args.max_good)
            test_paths = [
                str(p)
                for p in sorted((args.data_root / cls / "test").glob("*.png"))
            ]
            val_paths = [it["path"] for it in val_sources.get(cls, [])]
            heads = seg_heads[cls]  # now list[SegHead]

            # ensemble_infer handles per-head normalisation and blur internally
            norm_test_scores, fns = ensemble_infer(
                dinov2, test_paths, heads, all_good_paths
            )
            norm_test.setdefault(cls, []).append(norm_test_scores)
            if cls not in test_filenames:
                test_filenames[cls] = fns

            normed_good, _ = ensemble_infer(
                dinov2, good_paths, heads, all_good_paths
            )
            if cls not in good_acc:
                good_acc[cls] = normed_good.astype(np.float64)
                good_count[cls] = 1
                good_paths_per_cls[cls] = good_paths
            else:
                good_acc[cls] += normed_good
                good_count[cls] += 1

            if val_paths:
                norm_val_scores, _ = ensemble_infer(
                    dinov2, val_paths, heads, all_good_paths
                )
                norm_val.setdefault(cls, []).append(norm_val_scores)

            if cls not in zeromask_paths_per_cls:
                zeromask_paths_per_cls[cls] = _collect_zeromask_paths(
                    args.data_root, cls
                )
            zm_paths = zeromask_paths_per_cls[cls]
            if zm_paths:
                normed_zm, _ = ensemble_infer(
                    dinov2, zm_paths, heads, all_good_paths
                )
                if cls not in zeromask_acc:
                    zeromask_acc[cls] = normed_zm.astype(np.float64)
                    zeromask_count[cls] = 1
                else:
                    zeromask_acc[cls] += normed_zm
                    zeromask_count[cls] += 1

            print(f"  {cls} done")

    # ── Combine ───────────────────────────────────────────────────────────────
    print(f"\nCombining {len(run_dirs)} runs — method={args.method}")

    def _combine(arrays: list[np.ndarray]) -> np.ndarray:
        stack = np.stack(arrays, axis=0)
        if args.method == "average":
            return np.clip(stack.mean(axis=0), 0, 1)
        binary = (stack >= args.vote_threshold).astype(np.float32)
        return (binary.mean(axis=0) >= 0.5).astype(np.float32)

    # ── Create ensemble run directory ─────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / "ensembles" / f"{args.method}_{ts}"
    plot_dir = run_dir / "heatmaps"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Submission CSV ────────────────────────────────────────────────────────
    rows = []
    for cls in sorted(class_names):
        combined = _combine(norm_test[cls])
        for fn, score_map in zip(test_filenames[cls], combined):
            rows.append(
                {"ID": fn[:-4], "Label": float_matrix_to_q8rle(score_map)}
            )

    submission_path = run_dir / f"submission_{ts}.csv"
    pd.DataFrame(rows).to_csv(submission_path, index=False)
    print(f"Ensemble submission → {submission_path}")

    # ── Metrics + heatmaps on val anomaly images ──────────────────────────────

    zero_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    all_ap: dict[str, float] = {}

    header = f"{'Class':<12} {'Pixel AP':>9} {'Image AP':>9}"
    print(f"\n{header}")
    print("-" * len(header))

    for cls in sorted(class_names):
        items = val_sources.get(cls, [])
        if not items or cls not in norm_val:
            continue

        combined_val = _combine(norm_val[cls])  # (N_val, H, W)
        combined_good = np.clip(good_acc[cls] / good_count[cls], 0, 1).astype(
            np.float32
        )  # (N_good, H, W)

        # Pixel-level AP
        gt_masks = [zero_mask] * len(combined_good) + [
            _resize_mask(it["mask"]) for it in items
        ]
        gt_flat = np.stack(gt_masks).flatten()
        scores_flat = np.concatenate(
            [combined_good, combined_val], axis=0
        ).flatten()
        if gt_flat.max() > 0:
            pixel_ap = float(average_precision_score(gt_flat, scores_flat))
        else:
            pixel_ap = 0.0

        # Image-level AP
        image_scores = np.concatenate(
            [
                combined_good.reshape(len(combined_good), -1).max(axis=1),
                combined_val.reshape(len(combined_val), -1).max(axis=1),
            ]
        )
        image_labels = [0] * len(combined_good) + [1] * len(items)
        image_ap = float(average_precision_score(image_labels, image_scores))

        all_ap[cls] = pixel_ap
        print(f"{cls:<12} {pixel_ap:>9.4f} {image_ap:>9.4f}")

        _save_heatmaps(
            items,
            combined_val,
            cls,
            plot_dir / f"{cls}_heatmaps.png",
            n=args.n_vis,
        )

        _save_score_heatmaps(
            good_paths_per_cls[cls],
            combined_good,
            f"Top false positives (good images) — {cls}",
            plot_dir / f"{cls}_good_heatmaps.png",
            n=args.n_vis,
        )

        zm_paths = zeromask_paths_per_cls.get(cls, [])
        if zm_paths and cls in zeromask_acc:
            combined_zm = np.clip(
                zeromask_acc[cls] / zeromask_count[cls], 0, 1
            ).astype(np.float32)
            _save_score_heatmaps(
                zm_paths,
                combined_zm,
                f"Zero-mask anomalies (no pixel GT) — {cls}",
                plot_dir / f"{cls}_zeromask_heatmaps.png",
                n=args.n_vis,
            )

    if all_ap:
        mean_ap = float(np.mean(list(all_ap.values())))
        print("-" * len(header))
        print(f"{'MEAN':<12} {mean_ap:>9.4f}")

    print(f"\nAll outputs in {run_dir}")


if __name__ == "__main__":
    main()
