import random
import numpy as np
import torch

SEED = 42
IMG_SIZE = 224
PATCH_GRID = 16
FEATURE_DIM = 384
LAYERS_TO_USE = [5, 8, 11]  # ViT-S/14 layers 6, 9, 12
MULTILAYER_DIM = FEATURE_DIM * len(LAYERS_TO_USE)  # 1152
CORESET_RATIO = (
    0.01  # fraction of patches kept in the memory bank (greedy k-center)
)
EPOCHS = 40
SAMPLES_PER_EPOCH = 200  # number of synthetic images at each epoch
BATCH_SIZE = 16
LR = 1e-3
W_MB = 0.3  # memory-bank ensemble weight
W_SH = 0.7  # seg-head ensemble weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed: int = SEED) -> None:
    """Seed all RNGs so results are reproducible across machines."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for deterministic multi-worker loading."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
