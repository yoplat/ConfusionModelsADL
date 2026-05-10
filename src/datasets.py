import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

from .config import IMG_SIZE, P_GOOD, P_REAL
from .augmentation import cut_paste

preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


class ImageFolderDataset(Dataset):
    """Simple dataset that loads images from a flat list of file paths."""

    def __init__(self, paths: list[str], tf):
        self.paths, self.tf = paths, tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


class TestDataset(Dataset):
    """Dataset that returns (image_tensor, filename) pairs for inference."""

    def __init__(self, paths: list[str], tf):
        self.paths, self.tf = paths, tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        return self.tf(Image.open(path).convert("RGB")), Path(path).name


def _resize_mask(mask: np.ndarray) -> np.ndarray:
    """Binarise and resize a GT mask to (IMG_SIZE, IMG_SIZE)."""
    if mask.shape == (IMG_SIZE, IMG_SIZE):
        return (mask > 0).astype(np.float32)
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return (np.array(pil) > 127).astype(np.float32)


class TrainSynthDataset(Dataset):
    """30 % good / 40 % real anomaly / 30 % cut-paste synthetic anomaly.

    Real anomaly samples are training-split images with their GT masks resized
    to IMG_SIZE. Cut-paste samples paste an anomaly crop onto a good image.
    Falls back to 50/50 good/synth when no anomaly sources are available.
    """

    def __init__(
        self,
        good_paths: list[str],
        sources: list[dict],
        n: int,
        tf,
        p_good: float = P_GOOD,
        p_real: float = P_REAL,
    ):
        self.good_paths = good_paths
        self.sources = sources
        self.n = n
        self.tf = tf
        self.p_good = p_good
        self.p_real = p_real

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        r = random.random()

        if r < self.p_good or not self.sources:
            img = np.array(Image.open(random.choice(self.good_paths)).convert("RGB"))
            return self.tf(Image.fromarray(img)), torch.zeros(IMG_SIZE, IMG_SIZE)

        if r < self.p_good + self.p_real:
            src = random.choice(self.sources)
            mask = _resize_mask(src["mask"])
            return self.tf(Image.fromarray(src["image"])), torch.from_numpy(mask).float()

        good_img = np.array(Image.open(random.choice(self.good_paths)).convert("RGB"))
        src = random.choice(self.sources)
        synth, mask = cut_paste(good_img, src["image"], src["mask"])
        return self.tf(Image.fromarray(synth)), torch.from_numpy(mask).float()
