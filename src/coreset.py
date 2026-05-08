import torch

from .config import CORESET_RATIO, SEED, device


@torch.no_grad()
def greedy_coreset(
    features: torch.Tensor,
    n_samples: int,
    seed: int = SEED,
) -> torch.Tensor:
    """PatchCore k-center greedy coreset selection.

    Iteratively selects the point that is farthest from the currently selected
    set, producing a coreset that maximally covers the feature space. This gives
    significantly better coverage than random subsampling at the same budget.

    Time complexity is O(n * n_samples) with a constant factor of D (feature dim)
    per distance evaluation. For typical inputs (n ≈ 10k–100k, n_samples ≈ 100–1k)
    this runs in seconds on a GPU.

    Args:
        features:  (N, D) feature matrix on any device.
        n_samples: Number of points to select.
        seed:      RNG seed for the random starting point.

    Returns:
        (n_samples, D) selected feature matrix on the same device as input.
    """
    n = len(features)
    n_samples = min(n_samples, n)
    if n_samples >= n:
        return features

    # Offload distance computation to GPU when available
    feats_gpu = features.to(device)

    rng = torch.Generator()
    rng.manual_seed(seed)
    start = int(torch.randint(0, n, (1,), generator=rng))

    selected = [start]
    min_dist = torch.cdist(feats_gpu, feats_gpu[start : start + 1]).squeeze(1)

    for _ in range(n_samples - 1):
        idx = int(min_dist.argmax())
        selected.append(idx)
        d = torch.cdist(feats_gpu, feats_gpu[idx : idx + 1]).squeeze(1)
        torch.minimum(min_dist, d, out=min_dist)

    idx_t = torch.tensor(selected, dtype=torch.long)
    return features[idx_t]  # return on original (CPU) device


def random_subsample(
    features: torch.Tensor,
    n_samples: int,
    seed: int = SEED,
) -> torch.Tensor:
    """Randomly subsample ``n_samples`` rows from a feature matrix.

    Simple O(N) alternative to the greedy k-center coreset.  Coverage of the
    feature space is not guaranteed — dense regions will be over-represented —
    but it is much faster for large N.

    Args:
        features:  (N, D) feature matrix.
        n_samples: Number of rows to keep.
        seed:      NumPy-compatible RNG seed for reproducibility.

    Returns:
        (n_samples, D) subsampled feature matrix on the same device as input.
    """
    import numpy as np

    n = len(features)
    n_samples = min(n_samples, n)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=n_samples, replace=False)
    return features[torch.from_numpy(idx).long()]


def coreset_subsample(
    features: torch.Tensor,
    ratio: float = CORESET_RATIO,
    seed: int = SEED,
    method: str = "random",
) -> torch.Tensor:
    """Subsample a fraction of patch features for the memory bank.

    ``ratio`` is the fraction of patches **kept** (e.g. 0.01 keeps 1 % and
    discards 99 %).

    Args:
        features: (N, D) feature matrix.
        ratio:    Fraction of rows to keep (default 0.01 = 1 %).
        seed:     RNG seed for reproducibility.
        method:   ``'random'`` (fast, no coverage guarantee) or
                  ``'greedy'`` (k-center, maximises feature-space coverage
                  but scales as O(N × n_samples)).

    Returns:
        (max(1, N*ratio), D) subsampled feature matrix.
    """
    n_samples = max(1, int(len(features) * ratio))
    if method == "greedy":
        return greedy_coreset(features, n_samples, seed=seed)
    return random_subsample(features, n_samples, seed=seed)
