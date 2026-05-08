import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MULTILAYER_DIM, PATCH_GRID, IMG_SIZE


class SegHead(nn.Module):
    """Pixel-wise anomaly segmentation head operating on concatenated ViT patch tokens.

    Takes (B, num_patches, in_dim) features, reshapes them to a PATCH_GRID×PATCH_GRID
    spatial map, then produces a (B, 1, IMG_SIZE, IMG_SIZE) logit map via three
    convolutional layers and bilinear upsampling.
    """

    def __init__(self, in_dim: int = MULTILAYER_DIM, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, 1, 1)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Map patch tokens to an upsampled anomaly logit map.

        Args:
            p: (B, num_patches, in_dim) patch feature tensor.

        Returns:
            (B, 1, IMG_SIZE, IMG_SIZE) logit tensor.
        """
        B = p.size(0)
        x = p.transpose(1, 2).reshape(B, -1, PATCH_GRID, PATCH_GRID)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return F.interpolate(
            x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined binary cross-entropy and soft Dice loss for anomaly segmentation.

    Args:
        logits: (B, 1, H, W) raw model outputs.
        target: (B, 1, H, W) float mask in [0, 1].

    Returns:
        Scalar loss tensor.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1.0) / (union + 1.0)
    return bce + dice.mean()
