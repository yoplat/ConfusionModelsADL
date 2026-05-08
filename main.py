# ================================================================
# Spacepresso v7 - LB 0.5884
# DINOv2 memory bank + multi-layer SegHead, ensemble 0.45/0.55
# ================================================================
import gc, random, zipfile, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / "output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
PATCH_GRID = 16
FEATURE_DIM = 384
LAYERS_TO_USE = [5, 8, 11]  # 0-indexed ViT-S/14 layers 6, 9, 12
MULTILAYER_DIM = FEATURE_DIM * len(LAYERS_TO_USE)  # 1152
CORESET_RATIO = 0.01
EPOCHS = 30
SAMPLES_PER_EPOCH = 200
BATCH_SIZE = 16
LR = 1e-3
W_MB = 0.45  # memory-bank ensemble weight
W_SH = 0.55  # seg-head ensemble weight


# ── Preprocessing ─────────────────────────────────────────────────────────────
preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


# ── Datasets ──────────────────────────────────────────────────────────────────
class ImageFolderDataset(Dataset):
    """Simple dataset that loads images from a list of file paths."""

    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


class TestDataset(Dataset):
    """Dataset that returns (image_tensor, filename) pairs for inference."""

    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        return self.tf(Image.open(path).convert("RGB")), Path(path).name


class TrainSynthDataset(Dataset):
    """Dataset that synthesises anomalies on-the-fly via cut-paste augmentation.

    With 50 % probability a random anomaly patch is pasted onto a good image;
    otherwise the good image is returned unchanged with a zero mask.
    """

    def __init__(self, good_paths, sources, n, tf):
        self.good_paths, self.sources, self.n, self.tf = (
            good_paths,
            sources,
            n,
            tf,
        )

    def __len__(self):
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


# ── Feature extraction ────────────────────────────────────────────────────────
@torch.no_grad()
def extract_multilayer_patches(model, x, layers=LAYERS_TO_USE):
    """Extract and concatenate intermediate patch tokens from multiple ViT layers.

    Args:
        model: DINOv2 ViT model.
        x: (B, C, H, W) image batch on the correct device.
        layers: List of 0-indexed layer indices to sample.

    Returns:
        Tensor of shape (B, num_patches, FEATURE_DIM * len(layers)).
    """
    intermediates = model.get_intermediate_layers(
        x, n=layers, reshape=False, return_class_token=False, norm=True
    )
    return torch.cat(intermediates, dim=2)


@torch.no_grad()
def extract_patch_features(model, paths, batch_size=32):
    """Extract last-layer patch features from a list of image paths.

    Args:
        model: DINOv2 ViT model.
        paths: List of image file path strings.
        batch_size: DataLoader batch size.

    Returns:
        (N * num_patches, FEATURE_DIM) CPU float tensor.
    """
    loader = DataLoader(
        ImageFolderDataset(paths, preprocess),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )
    feats = []
    for batch in loader:
        out = model.forward_features(batch.to(device, non_blocking=True))
        feats.append(out["x_norm_patchtokens"].reshape(-1, FEATURE_DIM).cpu())
    return torch.cat(feats, dim=0)


def coreset_subsample(features, ratio=CORESET_RATIO, seed=42):
    """Randomly subsample a fraction of feature rows to form a compact coreset.

    Args:
        features: (N, D) feature matrix.
        ratio: Fraction of rows to keep.
        seed: NumPy random seed for reproducibility.

    Returns:
        (max(1, N*ratio), D) subsampled feature matrix.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(
        features.shape[0],
        size=max(1, int(features.shape[0] * ratio)),
        replace=False,
    )
    return features[idx]


# ── Memory bank construction ───────────────────────────────────────────────────
def build_memory_banks(model, data_root):
    """Build per-class L2-normalised memory banks from good training images.

    Each bank is a coreset of patch features extracted from all good-class images.

    Args:
        model: DINOv2 backbone used for feature extraction.
        data_root: Path to the dataset root (one subdirectory per class).

    Returns:
        dict mapping class_name -> (K, FEATURE_DIM) normalised coreset tensor.
    """
    print("Building memory banks...")
    banks = {}
    for class_dir in sorted(data_root.iterdir()):
        good_dir = class_dir / "train" / "good"
        if not good_dir.is_dir():
            continue
        class_name = class_dir.name
        paths = [str(p) for p in sorted(good_dir.glob("*.png"))]
        feats = extract_patch_features(model, paths)
        bank = F.normalize(coreset_subsample(feats), p=2, dim=1)
        banks[class_name] = bank
        print(f"  {class_name}: bank {tuple(bank.shape)}")
        del feats
        gc.collect()
    return banks


# ── Cut-paste synthetic anomaly ───────────────────────────────────────────────
def get_object_mask(img, threshold=30):
    """Return a boolean mask of pixels above a brightness threshold.

    Used to bias paste locations toward the foreground object rather than
    the background, so synthesised defects land on the product surface.

    Args:
        img: (H, W, 3) or (H, W) uint8 image array.
        threshold: Mean pixel value above which a pixel is considered foreground.

    Returns:
        (H, W) bool array.
    """
    return (img.mean(axis=2) if img.ndim == 3 else img) > threshold


def maybe_subcrop_large(crop_img, crop_mask, max_ratio=0.25):
    """Subcrop an anomaly patch when its foreground area fraction is too large.

    Prevents pasting defects that cover an unrealistically large portion of
    the image by randomly selecting a smaller sub-region centred on the anomaly.

    Args:
        crop_img:  (h, w, 3) uint8 cropped anomaly image.
        crop_mask: (h, w) uint8 cropped ground-truth mask.
        max_ratio: Maximum allowed foreground fraction before sub-cropping.

    Returns:
        (crop_img, crop_mask) possibly reduced to a smaller sub-region.
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
    return crop_img[y0 : y0 + sub_h, x0 : x0 + sub_w].copy(), crop_mask[
        y0 : y0 + sub_h, x0 : x0 + sub_w
    ].copy()


def cut_paste(good_img, ano_img, ano_mask, scale_range=(0.4, 1.2)):
    """Paste a scaled, colour-shifted anomaly patch onto a good image.

    The patch is randomly scaled, slightly colour-shifted, and blended with a
    Gaussian-smoothed alpha mask to produce a realistic-looking defect. The
    paste location is biased toward the foreground object region.

    Args:
        good_img:    (H, W, 3) uint8 target background image.
        ano_img:     (H, W, 3) uint8 source anomaly image.
        ano_mask:    (H, W) uint8 ground-truth mask for the anomaly region.
        scale_range: (min_scale, max_scale) for random resizing of the patch.

    Returns:
        synth: (H, W, 3) uint8 synthesised image.
        mask:  (H, W) float32 soft blending mask in [0, 1].
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


# ── Anomaly source collection ─────────────────────────────────────────────────
def collect_anomaly_sources(data_root):
    """Load all annotated anomaly image/mask pairs for cut-paste augmentation.

    Iterates over every class and anomaly subdirectory under ground_truth_train,
    loading the corresponding image from the train split. Masks with no foreground
    pixels are skipped.

    Args:
        data_root: Path to the dataset root.

    Returns:
        defaultdict mapping class_name -> list of {'image': ndarray, 'mask': ndarray}.
    """
    sources = defaultdict(list)
    for class_dir in sorted(data_root.iterdir()):
        train_dir = class_dir / "train"
        gt_dir = class_dir / "ground_truth_train"
        if not (train_dir.is_dir() and gt_dir.is_dir()):
            continue
        class_name = class_dir.name
        for ano_gt_dir in sorted(gt_dir.iterdir()):
            ano_img_dir = train_dir / ano_gt_dir.name
            for mask_path in sorted(ano_gt_dir.glob("*.png")):
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask.max() == 0:
                    continue
                img = np.array(
                    Image.open(ano_img_dir / mask_path.name).convert("RGB")
                )
                sources[class_name].append({"image": img, "mask": mask})
    return sources


# ── Model ─────────────────────────────────────────────────────────────────────
class SegHead(nn.Module):
    """Lightweight pixel-wise anomaly segmentation head operating on ViT patch tokens.

    Accepts concatenated multi-layer patch features of shape (B, num_patches, in_dim),
    reshapes them into a spatial PATCH_GRID×PATCH_GRID map, and produces a
    (B, 1, IMG_SIZE, IMG_SIZE) logit map via three convolutional layers followed by
    bilinear upsampling.
    """

    def __init__(self, in_dim=MULTILAYER_DIM, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, 1, 1)

    def forward(self, p):
        """Map patch tokens to an upsampled anomaly score map.

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


# ── Loss ──────────────────────────────────────────────────────────────────────
def bce_dice_loss(logits, target):
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


# ── Training ──────────────────────────────────────────────────────────────────
def train_seg_heads(model, data_root, anomaly_sources, output_dir):
    """Train one SegHead per class using synthetic cut-paste anomaly augmentation.

    DINOv2 features are extracted under torch.no_grad(); only SegHead parameters
    are updated. Each trained head is saved to output_dir as a state dict.

    Args:
        model:           Frozen DINOv2 backbone.
        data_root:       Path to the dataset root.
        anomaly_sources: Per-class anomaly pairs from collect_anomaly_sources().
        output_dir:      Directory where model checkpoints are written.

    Returns:
        dict mapping class_name -> trained SegHead (set to eval mode).
    """
    print("\nTraining multi-layer SegHeads...")
    seg_heads = {}
    for class_name in sorted(anomaly_sources.keys()):
        good_dir = data_root / class_name / "train" / "good"
        good_paths = [str(p) for p in sorted(good_dir.glob("*.png"))]
        sources = anomaly_sources[class_name]

        loader = DataLoader(
            TrainSynthDataset(
                good_paths, sources, SAMPLES_PER_EPOCH, preprocess
            ),
            batch_size=BATCH_SIZE,
            num_workers=2,
            pin_memory=True,
        )
        head = SegHead().to(device)
        optim = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)

        for _ in range(EPOCHS):
            head.train()
            for imgs, masks in loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).unsqueeze(1)
                with torch.no_grad():
                    patches = extract_multilayer_patches(model, imgs)
                loss = bce_dice_loss(head(patches), masks)
                optim.zero_grad()
                loss.backward()
                optim.step()

        head.eval()
        seg_heads[class_name] = head
        ckpt_path = output_dir / f"seghead_v7_{class_name}.pt"
        torch.save(head.state_dict(), ckpt_path)
        print(f"  {class_name} done  (checkpoint: {ckpt_path})")
    return seg_heads


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def memorybank_infer(model, paths, bank, batch_size=32):
    """Score images by measuring nearest-neighbour distance to the memory bank.

    Each patch is compared against all bank entries; the anomaly score for a
    patch is 1 − max_cosine_similarity, then upsampled to IMG_SIZE × IMG_SIZE.

    Args:
        model:      DINOv2 backbone.
        paths:      List of image file path strings.
        bank:       (K, FEATURE_DIM) L2-normalised memory bank tensor.
        batch_size: DataLoader batch size.

    Returns:
        scores:    (N, IMG_SIZE, IMG_SIZE) float32 ndarray of anomaly scores.
        filenames: List of N filename strings corresponding to scores.
    """
    loader = DataLoader(
        TestDataset(paths, preprocess),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )
    bank_gpu = bank.to(device)
    scores, fns = [], []
    for imgs, names in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model.forward_features(imgs)
        patches = F.normalize(out["x_norm_patchtokens"], p=2, dim=2)
        max_sim, _ = (patches @ bank_gpu.T).max(dim=2)
        dist = (1.0 - max_sim).reshape(-1, PATCH_GRID, PATCH_GRID)
        s = F.interpolate(
            dist.unsqueeze(1),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        scores.append(s.cpu().numpy())
        fns.extend(names)
    return np.concatenate(scores, axis=0), fns


@torch.no_grad()
def seghead_infer(model, paths, head, batch_size=32):
    """Score images using a trained SegHead on multi-layer DINOv2 features.

    Args:
        model:      DINOv2 backbone.
        paths:      List of image file path strings.
        head:       Trained SegHead model.
        batch_size: DataLoader batch size.

    Returns:
        scores:    (N, IMG_SIZE, IMG_SIZE) float32 ndarray of anomaly scores.
        filenames: List of N filename strings corresponding to scores.
    """
    loader = DataLoader(
        TestDataset(paths, preprocess),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )
    head.eval()
    scores, fns = [], []
    for imgs, names in loader:
        imgs = imgs.to(device, non_blocking=True)
        patches = extract_multilayer_patches(model, imgs)
        s = torch.sigmoid(head(patches)).squeeze(1).cpu().numpy()
        scores.append(s)
        fns.extend(names)
    return np.concatenate(scores, axis=0), fns


def run_inference(model, data_root, memory_banks, seg_heads):
    """Run both inference branches on the good-train and test splits for all classes.

    Training-split scores are retained to compute per-class normalisation statistics.

    Args:
        model:        DINOv2 backbone.
        data_root:    Path to the dataset root.
        memory_banks: Per-class memory banks from build_memory_banks().
        seg_heads:    Per-class SegHead models from train_seg_heads().

    Returns:
        mb_train: dict class_name -> (N_train, H, W) memory-bank scores on good images.
        mb_test:  dict class_name -> {filename: (H, W) score array} for test images.
        sh_train: dict class_name -> (N_train, H, W) seg-head scores on good images.
        sh_test:  dict class_name -> {filename: (H, W) score array} for test images.
    """
    print("\nInference...")
    mb_train, mb_test, sh_train, sh_test = {}, {}, {}, {}
    for class_name in sorted(seg_heads.keys()):
        good_dir = data_root / class_name / "train" / "good"
        good = [str(p) for p in sorted(good_dir.glob("*.png"))]
        test_dir = data_root / class_name / "test"
        test = [str(p) for p in sorted(test_dir.glob("*.png"))]

        mb_tr, _ = memorybank_infer(model, good, memory_banks[class_name])
        mb_te, mfns = memorybank_infer(model, test, memory_banks[class_name])
        sh_tr, _ = seghead_infer(model, good, seg_heads[class_name])
        sh_te, sfns = seghead_infer(model, test, seg_heads[class_name])

        mb_train[class_name] = mb_tr
        mb_test[class_name] = dict(zip(mfns, mb_te))
        sh_train[class_name] = sh_tr
        sh_test[class_name] = dict(zip(sfns, sh_te))
        print(f"  {class_name} done")
    return mb_train, mb_test, sh_train, sh_test


# ── Submission encoding ───────────────────────────────────────────────────────
def float_matrix_to_q8rle(x):
    """Encode a float [0, 1] matrix as a column-major run-length uint8 string.

    Format: 'q8rle <H> <W> <val0> <run0> <val1> <run1> ...'
    Values are quantised to [0, 255] by multiplying by 255 and rounding.
    The matrix is traversed column-major to match the competition scorer format.

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


def encode_submission(mb_test, sh_test, mb_train, sh_train, output_dir):
    """Normalise, ensemble, and encode per-class predictions into submission files.

    Per-class normalisation clips scores to the [0.5th, 99.9th] percentile range
    of training-split scores. The two branches are combined as a weighted sum
    (W_MB * memory_bank + W_SH * seg_head) and encoded via float_matrix_to_q8rle.

    Writes submission.csv and submission.zip into output_dir.

    Args:
        mb_test:    dict class_name -> {filename: score} from run_inference().
        sh_test:    dict class_name -> {filename: score} from run_inference().
        mb_train:   dict class_name -> training scores for normalisation stats.
        sh_train:   dict class_name -> training scores for normalisation stats.
        output_dir: Path where output files are written.
    """
    print("\nEncoding submission...")
    rows = []
    for class_name in sorted(mb_test.keys()):
        mb_lo, mb_hi = np.percentile(
            mb_train[class_name].flatten(), [0.5, 99.9]
        )
        sh_lo, sh_hi = np.percentile(
            sh_train[class_name].flatten(), [0.5, 99.9]
        )
        for fn in sorted(mb_test[class_name].keys()):
            n_mb = np.clip(
                (mb_test[class_name][fn] - mb_lo) / (mb_hi - mb_lo + 1e-8), 0, 1
            )
            n_sh = np.clip(
                (sh_test[class_name][fn] - sh_lo) / (sh_hi - sh_lo + 1e-8), 0, 1
            )
            ens = np.clip(W_MB * n_mb + W_SH * n_sh, 0, 1).astype(np.float32)
            rows.append({"ID": fn[:-4], "Label": float_matrix_to_q8rle(ens)})

    csv_path = output_dir / "submission.csv"
    zip_path = output_dir / "submission.zip"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with zipfile.ZipFile(
        zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        zf.write(csv_path, arcname="submission.csv")
    print(f"Done. {zip_path}  ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    """Parse arguments, orchestrate training and inference, and write the submission."""
    parser = argparse.ArgumentParser(
        description="Spacepresso v7 – ADL anomaly detection"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Dataset root directory (default: ./dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for checkpoints and submission (default: ./output)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dinov2 = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", verbose=False
    )
    dinov2 = dinov2.to(device).eval()
    for p in dinov2.parameters():
        p.requires_grad = False

    memory_banks = build_memory_banks(dinov2, args.data_root)
    anomaly_sources = collect_anomaly_sources(args.data_root)
    seg_heads = train_seg_heads(
        dinov2, args.data_root, anomaly_sources, args.output_dir
    )
    torch.cuda.empty_cache()

    mb_train, mb_test, sh_train, sh_test = run_inference(
        dinov2, args.data_root, memory_banks, seg_heads
    )
    torch.cuda.empty_cache()

    encode_submission(mb_test, sh_test, mb_train, sh_train, args.output_dir)


if __name__ == "__main__":
    main()
