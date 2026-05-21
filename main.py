"""Spacepresso v16 — DINOv2 + multi-run SegHead ensemble."""
import argparse
import gc
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", message="xFormers is not available")

from src.config import (
    BATCH_SIZE,
    BLUR_SIGMA,
    CLASS_CONFIG,
    EPOCHS,
    FEATURE_DIM,
    IMG_SIZE,
    LAYERS_TO_USE,
    LR,
    MULTILAYER_DIM,
    N_RUNS,
    PATCH_GRID,
    PATIENCE,
    P_HI,
    P_LO,
    SAMPLES_PER_EPOCH,
    SEED,
    TOP_K,
    TVERSKY_ALPHA,
    W_MB,
    device,
    set_all_seeds,
)
from src.training import collect_anomaly_sources, train_all_classes
from src.inference import run_inference
from src.evaluation import evaluate_all
from src.checkpoint import save_run
from src.submission import encode_submission

DATA_ROOT = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Spacepresso v16 – ADL anomaly detection")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--n-vis", type=int, default=4)
    parser.add_argument("--eval-max-good", type=int, default=150)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed)
    print(f"Seed: {args.seed}  |  Device: {device}  |  N_RUNS={args.n_runs}  TOP_K={args.top_k}")

    # ── Backbone ──────────────────────────────────────────────────────────────
    print("\nLoading DINOv2...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    dinov2 = dinov2.to(device).eval()
    for p in dinov2.parameters():
        p.requires_grad = False

    # ── Training ──────────────────────────────────────────────────────────────
    seg_heads_dict = train_all_classes(
        dinov2,
        args.data_root,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    torch.cuda.empty_cache()
    gc.collect()

    # ── Evaluation (proxy metrics on val split, fixed seed) ───────────────────
    _, val_sources = collect_anomaly_sources(args.data_root, seed=args.seed)
    eval_max_good = args.eval_max_good if args.eval_max_good > 0 else None
    metrics = evaluate_all(
        dinov2,
        args.data_root,
        {},                   # no memory banks
        seg_heads_dict,
        val_sources,
        run_dir=args.output_dir / "_eval_tmp",
        n_vis=args.n_vis,
        max_good=eval_max_good,
        seed=args.seed,
    )
    torch.cuda.empty_cache()

    # ── Test inference ────────────────────────────────────────────────────────
    test_scores = run_inference(dinov2, args.data_root, seg_heads_dict)
    torch.cuda.empty_cache()

    # ── Checkpoint ────────────────────────────────────────────────────────────
    config_dict = {
        "seed": args.seed,
        "img_size": IMG_SIZE,
        "patch_grid": PATCH_GRID,
        "feature_dim": FEATURE_DIM,
        "layers_to_use": LAYERS_TO_USE,
        "multilayer_dim": MULTILAYER_DIM,
        "epochs": args.epochs,
        "patience": args.patience,
        "n_runs": args.n_runs,
        "top_k": args.top_k,
        "samples_per_epoch": SAMPLES_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "tversky_alpha": TVERSKY_ALPHA,
        "w_mb": W_MB,
        "blur_sigma": BLUR_SIGMA,
        "p_lo": P_LO,
        "p_hi": P_HI,
        "class_config": CLASS_CONFIG,
        "data_root": str(args.data_root),
    }
    run_dir = save_run(args.output_dir, seg_heads_dict, config_dict, metrics=metrics)

    # Copy eval plots into the run dir
    import shutil
    eval_tmp = args.output_dir / "_eval_tmp"
    if eval_tmp.exists():
        shutil.copytree(eval_tmp, run_dir / "evaluation", dirs_exist_ok=True)
        shutil.rmtree(eval_tmp)

    # ── Submission ────────────────────────────────────────────────────────────
    encode_submission(test_scores, run_dir)

    print(f"\nAll outputs in {run_dir}")


if __name__ == "__main__":
    main()
