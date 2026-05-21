from pathlib import Path

import numpy as np
import pandas as pd


def float_matrix_to_q8rle(x: np.ndarray) -> str:
    """Encode a [0, 1] float matrix as a column-major run-length uint8 string.

    Format: ``'q8rle <H> <W> <val0> <run0> <val1> <run1> ...'``
    """
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
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
    test_scores: dict[str, tuple[np.ndarray, list[str]]],
    run_dir: Path,
) -> Path:
    """Encode pre-normalised ensemble scores into the submission CSV.

    ``test_scores`` is the direct output of ``run_inference`` — scores are
    already blurred and normalised to [0, 1] by ``ensemble_infer``, so no
    further transformation is applied.

    Args:
        test_scores: class_name -> (scores_array, filenames) from run_inference.
        run_dir:     Timestamped run directory.

    Returns:
        Path to the submission CSV file.
    """
    print("\nEncoding submission...")
    rows = []
    for class_name in sorted(test_scores.keys()):
        scores, fns = test_scores[class_name]
        for fn, score_map in zip(fns, scores):
            rows.append({"ID": fn[:-4], "Label": float_matrix_to_q8rle(score_map)})

    ts = run_dir.name
    csv_path = run_dir / f"submission_{ts}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Submission written → {csv_path}")
    return csv_path
