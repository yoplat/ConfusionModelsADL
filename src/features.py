import torch
from torch.utils.data import DataLoader

from .config import FEATURE_DIM, LAYERS_TO_USE, device, worker_init_fn
from .datasets import ImageFolderDataset, preprocess


@torch.no_grad()
def extract_multilayer_patches(
    model,
    x: torch.Tensor,
    layers: list[int] = LAYERS_TO_USE,
) -> torch.Tensor:
    """Extract and concatenate intermediate patch tokens from multiple ViT layers.

    Args:
        model:  DINOv2 ViT model.
        x:      (B, C, H, W) image batch on the model's device.
        layers: 0-indexed layer indices to sample.

    Returns:
        (B, num_patches, FEATURE_DIM * len(layers)) tensor.
    """
    intermediates = model.get_intermediate_layers(
        x, n=layers, reshape=False, return_class_token=False, norm=True
    )
    return torch.cat(intermediates, dim=2)


@torch.no_grad()
def extract_patch_features(
    model,
    paths: list[str],
    batch_size: int = 32,
) -> torch.Tensor:
    """Extract last-layer patch tokens from a list of image paths.

    Args:
        model:      DINOv2 ViT model.
        paths:      Image file paths.
        batch_size: DataLoader batch size.

    Returns:
        (N * num_patches, FEATURE_DIM) CPU float32 tensor.
    """
    g = torch.Generator()
    g.manual_seed(0)
    loader = DataLoader(
        ImageFolderDataset(paths, preprocess),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    feats = []
    for batch in loader:
        out = model.forward_features(batch.to(device, non_blocking=True))
        feats.append(out["x_norm_patchtokens"].reshape(-1, FEATURE_DIM).cpu())
    return torch.cat(feats, dim=0)
