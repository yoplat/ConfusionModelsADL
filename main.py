# ================================================================
# Spacepresso v7 - LB 0.5884
# DINOv2 memory bank + multi-layer SegHead, ensemble 0.45/0.55
# ================================================================
import argparse
import gc
import shutil
import warnings
from pathlib import Path

import torch

# DINOv2 emits these on every import when xFormers isn't installed; the
# fallback to standard PyTorch attention is silent and correct.
warnings.filterwarnings("ignore", message="xFormers is not available")

from src.config import (
    CORESET_RATIO,
    EPOCHS,
    SEED,
    device,
    set_all_seeds,
)
from src.training import (
    build_memory_banks,
    collect_anomaly_sources,
    train_seg_heads,
)
from src.inference import run_inference
from src.evaluation import evaluate_all
from src.checkpoint import save_run
from src.submission import encode_submission

DATA_ROOT = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    """Parse arguments, run the full training → evaluation → inference pipeline."""
    parser = argparse.ArgumentParser(
        description="Spacepresso v7 – ADL anomaly detection"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Global RNG seed (default: %(default)s)",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=CORESET_RATIO,
        help="Fraction of patches kept in the memory bank (default: %(default)s = 1%%).",
    )
    parser.add_argument(
        "--coreset-method",
        choices=["random", "greedy"],
        default="random",
        help="Subsampling method for the memory bank: 'random' (fast) or "
        "'greedy' (k-center, better coverage but slower). Default: random.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="SegHead training epochs per class (default: %(default)s)",
    )
    parser.add_argument(
        "--n-vis",
        type=int,
        default=4,
        help="Anomaly heatmap examples to visualise per class",
    )
    parser.add_argument(
        "--eval-max-good",
        type=int,
        default=150,
        help="Max good images per class used for proxy evaluation "
        "(does not affect submission scores; pass 0 for all). Default: %(default)s.",
    )
    parser.add_argument(
        "--rebuild-banks",
        action="store_true",
        help="Ignore any existing memory bank cache and recompute from scratch",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_all_seeds(args.seed)
    print(
        f"Seed: {args.seed}  |  Device: {device}  |  "
        f"Coreset ratio: {args.coreset_ratio * 100:.1f}%  |  "
        f"Coreset method: {args.coreset_method}"
    )

    # ── Backbone ──────────────────────────────────────────────────────────────
    print("\nLoading DINOv2...")
    dinov2 = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", verbose=False
    )
    dinov2 = dinov2.to(device).eval()
    for p in dinov2.parameters():
        p.requires_grad = False

    # ── Training ──────────────────────────────────────────────────────────────
    bank_cache = None if args.rebuild_banks else args.output_dir / "bank_cache"
    memory_banks = build_memory_banks(
        dinov2,
        args.data_root,
        coreset_ratio=args.coreset_ratio,
        seed=args.seed,
        coreset_method=args.coreset_method,
        cache_dir=bank_cache,
    )
    anomaly_sources = collect_anomaly_sources(args.data_root)
    seg_heads = train_seg_heads(
        dinov2,
        args.data_root,
        anomaly_sources,
        args.output_dir,
        epochs=args.epochs,
        seed=args.seed,
    )
    torch.cuda.empty_cache()
    gc.collect()

    # ── Evaluation (proxy metrics on training data) ───────────────────────────
    eval_tmp = args.output_dir / "_eval_tmp"
    eval_max_good = args.eval_max_good if args.eval_max_good > 0 else None
    metrics = evaluate_all(
        dinov2,
        args.data_root,
        memory_banks,
        seg_heads,
        run_dir=eval_tmp,
        n_vis=args.n_vis,
        max_good=eval_max_good,
        seed=args.seed,
    )
    torch.cuda.empty_cache()

    # ── Test inference ────────────────────────────────────────────────────────
    mb_train, mb_test, sh_train, sh_test = run_inference(
        dinov2,
        args.data_root,
        memory_banks,
        seg_heads,
    )
    torch.cuda.empty_cache()

    # ── Checkpoint (timestamped, includes evaluation artefacts) ──────────────
    config_dict = {
        "seed": args.seed,
        "coreset_ratio": args.coreset_ratio,
        "epochs": args.epochs,
        "data_root": str(args.data_root),
    }
    run_dir = save_run(
        args.output_dir,
        memory_banks,
        seg_heads,
        config_dict=config_dict,
        metrics=metrics,
    )
    if eval_tmp.exists():
        shutil.copytree(eval_tmp, run_dir / "evaluation", dirs_exist_ok=True)
        shutil.rmtree(eval_tmp)

    # ── Submission ────────────────────────────────────────────────────────────
    encode_submission(mb_test, sh_test, mb_train, sh_train, run_dir)

    print(f"\nAll outputs in {run_dir}")


if __name__ == "__main__":
    main()
