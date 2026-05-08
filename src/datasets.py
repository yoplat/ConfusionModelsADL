import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

from .config import IMG_SIZE
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


class TrainSynthDataset(Dataset):
    """Dataset that synthesises anomalies on-the-fly via cut-paste augmentation.

    With 50 % probability a random anomaly patch is pasted onto a good image;
    otherwise the good image is returned unchanged with a zero mask.
    """

    def __init__(self, good_paths: list[str], sources: list[dict], n: int, tf):
        self.good_paths = good_paths
        self.sources = sources
        self.n = n
        self.tf = tf

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        good_img = np.array(
            Image.open(random.choice(self.good_paths)).convert("RGB")
        )
        if random.random() < 0.5 and self.sources:
            src = random.choice(self.sources)
            synth, mask = cut_paste(good_img, src["image"], src["mask"])
            return self.tf(Image.fromarray(synth)), torch.from_numpy(
                mask
            ).float()
        return self.tf(Image.fromarray(good_img)), torch.zeros(
            IMG_SIZE, IMG_SIZE
        )
