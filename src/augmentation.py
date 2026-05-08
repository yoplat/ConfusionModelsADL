import random
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .config import IMG_SIZE


def get_object_mask(img: np.ndarray, threshold: int = 30) -> np.ndarray:
    """Return a boolean foreground mask based on per-pixel mean brightness.

    Used to bias paste locations toward the product surface rather than the
    background.

    Args:
        img: (H, W, 3) or (H, W) uint8 image array.
        threshold: Mean pixel value above which a pixel is foreground.

    Returns:
        (H, W) bool array.
    """
    return (img.mean(axis=2) if img.ndim == 3 else img) > threshold


def maybe_subcrop_large(
    crop_img: np.ndarray,
    crop_mask: np.ndarray,
    max_ratio: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subcrop a patch whose foreground fraction exceeds max_ratio.

    Prevents unrealistically large synthesised defects by selecting a smaller
    sub-region centred on the anomaly area.

    Args:
        crop_img:  (h, w, 3) uint8 anomaly crop.
        crop_mask: (h, w) uint8 ground-truth mask crop.
        max_ratio: Maximum allowed foreground fraction before sub-cropping.

    Returns:
        Possibly smaller (crop_img, crop_mask) pair.
    """
    h, w = crop_mask.shape
    if (crop_mask > 0).sum() / (h * w + 1e-8) < max_ratio:
        return crop_img, crop_mask
    target_area = h * w * random.uniform(0.15, 0.30)
    sub_h = max(8, min(int(np.sqrt(target_area * h / w)), h - 2))
    sub_w = max(8, min(int(target_area / sub_h), w - 2))
    ys, xs = np.where(crop_mask > 0)
    if len(ys) == 0:
        return crop_img, crop_mask
    cy, cx = random.choice(ys), random.choice(xs)
    y0 = max(0, min(cy - sub_h // 2, h - sub_h))
    x0 = max(0, min(cx - sub_w // 2, w - sub_w))
    return (
        crop_img[y0 : y0 + sub_h, x0 : x0 + sub_w].copy(),
        crop_mask[y0 : y0 + sub_h, x0 : x0 + sub_w].copy(),
    )


def cut_paste(
    good_img: np.ndarray,
    ano_img: np.ndarray,
    ano_mask: np.ndarray,
    scale_range: tuple[float, float] = (0.4, 1.2),
) -> tuple[np.ndarray, np.ndarray]:
    """Paste a scaled, colour-shifted anomaly patch onto a good image.

    The patch is randomly scaled, slightly colour-shifted, and blended with a
    Gaussian-smoothed alpha to produce a realistic defect. The paste location
    is biased toward the foreground object region.

    Args:
        good_img:    (H, W, 3) uint8 target background image.
        ano_img:     (H, W, 3) uint8 source anomaly image.
        ano_mask:    (H, W) uint8 ground-truth mask of the anomaly region.
        scale_range: (min, max) scale factor applied to the crop.

    Returns:
        synth: (H, W, 3) uint8 synthesised image.
        mask:  (H, W) float32 soft alpha mask in [0, 1].
    """
    H, W = good_img.shape[:2]
    ys, xs = np.where(ano_mask > 0)
    if len(ys) == 0:
        return good_img.copy(), np.zeros((H, W), dtype=np.float32)

    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop_img = ano_img[y0:y1, x0:x1].copy()
    crop_mask = ano_mask[y0:y1, x0:x1].copy()
    crop_img, crop_mask = maybe_subcrop_large(crop_img, crop_mask)
    ch, cw = crop_img.shape[:2]
    if ch < 4 or cw < 4:
        return good_img.copy(), np.zeros((H, W), dtype=np.float32)

    scale = random.uniform(*scale_range)
    nh = max(6, min(int(ch * scale), H // 2))
    nw = max(6, min(int(cw * scale), W // 2))
    crop_img = np.array(
        Image.fromarray(crop_img).resize((nw, nh), Image.BILINEAR)
    )
    crop_mask = np.array(
        Image.fromarray(crop_mask).resize((nw, nh), Image.NEAREST)
    )

    shift = np.random.randint(-15, 16, size=3).reshape(1, 1, 3)
    crop_img = np.clip(crop_img.astype(np.int16) + shift, 0, 255).astype(
        np.uint8
    )

    obj_mask = get_object_mask(good_img)
    oys, oxs = np.where(obj_mask)
    if len(oys) == 0:
        py = random.randint(0, H - nh)
        px = random.randint(0, W - nw)
    else:
        oy0, oy1, ox0, ox1 = oys.min(), oys.max(), oxs.min(), oxs.max()
        pyl = max(0, oy0 - nh // 4)
        pyh = max(pyl, min(H - nh, oy1 - nh // 2))
        pxl = max(0, ox0 - nw // 4)
        pxh = max(pxl, min(W - nw, ox1 - nw // 2))
        py = random.randint(pyl, pyh)
        px = random.randint(pxl, pxh)

    synth = good_img.copy()
    out_mask = np.zeros((H, W), dtype=np.float32)
    alpha = gaussian_filter((crop_mask > 0).astype(np.float32), sigma=1.0)
    region = synth[py : py + nh, px : px + nw]
    synth[py : py + nh, px : px + nw] = (
        region * (1 - alpha[:, :, None]) + crop_img * alpha[:, :, None]
    ).astype(np.uint8)
    out_mask[py : py + nh, px : px + nw] = alpha
    return synth, out_mask
