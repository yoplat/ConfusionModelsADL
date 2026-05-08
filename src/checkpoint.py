import json
from datetime import datetime
from pathlib import Path

import torch

from .model import SegHead
from .config import device


def save_run(
    output_dir: Path,
    memory_banks: dict[str, torch.Tensor],
    seg_heads: dict[str, SegHead],
    config_dict: dict,
    metrics: dict | None = None,
) -> Path:
    """Save a complete model run to a timestamped directory.

    Directory layout::

        output_dir/
          runs/
            YYYYMMDD_HHMMSS/
              config.json
              memory_banks/
                class_01.pt
                ...
              seg_heads/
                class_01.pt
                ...
              evaluation/
                metrics.json  (copied here for convenience, if provided)

    Args:
        output_dir:   Root output directory.
        memory_banks: Per-class normalised coreset tensors.
        seg_heads:    Per-class trained SegHead models.
        config_dict:  Hyper-parameter dict serialised to config.json.
        metrics:      Optional evaluation metrics dict.

    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / timestamp
    (run_dir / "memory_banks").mkdir(parents=True, exist_ok=True)
    (run_dir / "seg_heads").mkdir(parents=True, exist_ok=True)

    for cls, bank in memory_banks.items():
        torch.save(bank, run_dir / "memory_banks" / f"{cls}.pt")

    for cls, head in seg_heads.items():
        torch.save(head.state_dict(), run_dir / "seg_heads" / f"{cls}.pt")

    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    if metrics is not None:
        eval_dir = run_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        with open(eval_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"\nRun saved → {run_dir}")
    return run_dir


def load_run(
    run_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, SegHead]]:
    """Restore memory banks and SegHeads from a saved run directory.

    Args:
        run_dir: Path returned by :func:`save_run`.

    Returns:
        memory_banks: dict class_name -> coreset tensor (on ``device``).
        seg_heads:    dict class_name -> SegHead (eval mode, on ``device``).
    """
    memory_banks: dict[str, torch.Tensor] = {}
    for pt in sorted((run_dir / "memory_banks").glob("*.pt")):
        memory_banks[pt.stem] = torch.load(pt, map_location=device)

    seg_heads: dict[str, SegHead] = {}
    for pt in sorted((run_dir / "seg_heads").glob("*.pt")):
        head = SegHead()
        head.load_state_dict(torch.load(pt, map_location=device))
        seg_heads[pt.stem] = head.to(device).eval()

    return memory_banks, seg_heads
