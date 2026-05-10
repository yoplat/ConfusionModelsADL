from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .config import BLUR_SIGMA, P_HI, P_LO, W_MB


def float_matrix_to_q8rle(x: np.ndarray) -> str:
    """Encode a [0, 1] float matrix as a column-major run-length uint8 string.

    Format: ``'q8rle <H> <W> <val0> <run0> <val1> <run1> ...'``

    Values are quantised to [0, 255] (multiply by 255, round). The matrix is
    traversed column-major to match the competition scorer format.

    Args:
        x: (H, W) float32 array with values in [0, 1].

    Returns:
        Encoded string.
    """
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(
        np.uint8
    )
    h, w = q.shape
    flat = q.T.reshape(-1)
    if flat.size == 0:
        return f"q8rle {h} {w}"
    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, flat.size]
    parts = ["q8rle", str(h), str(w)]
    for v, n in zip(flat[starts], ends - starts):
        parts += [str(int(v)), str(int(n))]
    return " ".join(parts)


def encode_submission(
    mb_test: dict,
    sh_test: dict,
    mb_train: dict,
    sh_train: dict,
    run_dir: Path,
) -> Path:
    """Normalise, ensemble, and encode per-class predictions into submission files.

    Per-class normalisation clips each branch to its [0.5th, 99.9th] percentile
    range computed from training-split scores, then ensembles as a weighted sum
    (``W_MB * memory_bank + W_SH * seg_head``).

    Writes ``submission_<timestamp>.csv`` into ``run_dir``.

    Args:
        mb_test:  class_name -> {filename: (H, W) score}.
        sh_test:  class_name -> {filename: (H, W) score}.
        mb_train: class_name -> (N, H, W) training scores for normalisation.
        sh_train: class_name -> (N, H, W) training scores for normalisation.
        run_dir:  Timestamped run directory (its name is used as the timestamp).

    Returns:
        Path to the submission CSV file.
    """
    W_SH = 1.0 - W_MB
    print("\nEncoding submission...")
    rows = []
    for class_name in sorted(sh_test.keys()):
        sh_lo, sh_hi = np.percentile(
            sh_train[class_name].flatten(), [P_LO, P_HI]
        )
        for fn in sorted(sh_test[class_name].keys()):
            n_sh = np.clip(
                (sh_test[class_name][fn] - sh_lo) / (sh_hi - sh_lo + 1e-8), 0, 1
            )
            ens = W_SH * n_sh
            if W_MB > 0:
                mb_lo, mb_hi = np.percentile(
                    mb_train[class_name].flatten(), [P_LO, P_HI]
                )
                n_mb = np.clip(
                    (mb_test[class_name][fn] - mb_lo) / (mb_hi - mb_lo + 1e-8), 0, 1
                )
                ens = ens + W_MB * n_mb
            if BLUR_SIGMA > 0:
                ens = gaussian_filter(ens, sigma=BLUR_SIGMA)
            ens = np.clip(ens, 0, 1).astype(np.float32)
            rows.append({"ID": fn[:-4], "Label": float_matrix_to_q8rle(ens)})

    ts = run_dir.name
    csv_path = run_dir / f"submission_{ts}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Submission written → {csv_path}")
    return csv_path
