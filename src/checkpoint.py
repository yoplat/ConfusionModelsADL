import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

from .config import device
from .model import SegHead


def save_run(
    output_dir: Path,
    seg_heads_dict: dict[str, list[SegHead]],
    config_dict: dict,
    metrics: dict | None = None,
) -> Path:
    """Save TOP_K SegHeads per class to a timestamped directory.

    Layout::

        output_dir/runs/YYYYMMDD_HHMMSS/
            config.json
            seg_heads/
                class_01_rank1.pt
                class_01_rank2.pt
                ...
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / timestamp
    (run_dir / "seg_heads").mkdir(parents=True, exist_ok=True)

    for cls, heads in seg_heads_dict.items():
        heads_list = heads if isinstance(heads, list) else [heads]
        for rank, head in enumerate(heads_list):
            torch.save(head.state_dict(), run_dir / "seg_heads" / f"{cls}_rank{rank + 1}.pt")

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
) -> tuple[dict, dict[str, list[SegHead]]]:
    """Load SegHeads from a saved run.

    Handles both old format (``class_01.pt``) and new format
    (``class_01_rank1.pt``).  Always returns a list of heads per class
    (old format becomes a single-element list).

    Returns:
        memory_banks: Empty dict (W_MB=0 by default; old banks ignored).
        seg_heads:    dict class_name -> list of SegHeads (eval mode, on device).
    """
    by_class: dict[str, list[Path]] = defaultdict(list)
    for pt in sorted((run_dir / "seg_heads").glob("*.pt")):
        cls = pt.stem.rsplit("_rank", 1)[0] if "_rank" in pt.stem else pt.stem
        by_class[cls].append(pt)

    seg_heads: dict[str, list[SegHead]] = {}
    for cls, paths in sorted(by_class.items()):
        heads = []
        for pt in sorted(paths):
            state = torch.load(pt, map_location=device, weights_only=True)
            # Infer in_dim from the saved weights so old (ViT-S) and new (ViT-B)
            # checkpoints both load correctly regardless of the current config.
            in_dim = state["conv1.weight"].shape[1]
            head = SegHead(in_dim=in_dim)
            head.load_state_dict(state)
            heads.append(head.to(device).eval())
        seg_heads[cls] = heads

    return {}, seg_heads
