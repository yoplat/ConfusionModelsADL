import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IMG_SIZE, MULTILAYER_DIM, PATCH_GRID, TVERSKY_ALPHA


class SegHead(nn.Module):
    """Pixel-wise anomaly segmentation head operating on concatenated ViT patch tokens."""

    def __init__(self, in_dim: int = MULTILAYER_DIM, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, 1, 1)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        B = p.size(0)
        x = p.transpose(1, 2).reshape(B, -1, PATCH_GRID, PATCH_GRID)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = TVERSKY_ALPHA,
    eps: float = 1e-7,
) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    pf = pred.view(pred.size(0), -1)
    tf = target.view(target.size(0), -1)
    tp = (pf * tf).sum(dim=1)
    fp = (pf * (1 - tf)).sum(dim=1)
    fn = ((1 - pf) * tf).sum(dim=1)
    t = (tp + eps) / (tp + alpha * fp + (1 - alpha) * fn + eps)
    return (1 - t).mean()


def bce_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    l1_lambda: float = 0.0,
) -> torch.Tensor:
    """BCE + Tversky with optional L1 sparsity on activations."""
    loss = F.binary_cross_entropy_with_logits(logits, target) + tversky_loss(logits, target)
    if l1_lambda > 0.0:
        loss = loss + l1_lambda * torch.sigmoid(logits).mean()
    return loss
