# ================================================================
# v36: 3-fold SegHead ensemble + 4-fold TTA
#
# HOW TO RUN
# ----------
# 1. Put your dataset in a folder called `dataset/` next to this notebook.
#    It must contain class_01/ … class_08/, each with:
#      train/good/            clean reference images
#      train/<anomaly_type>/  anomaly images (matching ground_truth_train/)
#      ground_truth_train/    pixel-level GT masks
#      test/                  images to score for the submission
#
# 2. Open this notebook in Jupyter or Google Colab.
#    On Colab: File → Upload notebook, then Runtime → T4 GPU.
#
# 3. Run CELL 0 once to install missing packages (Colab already has most).
#
# 4. Run all cells top-to-bottom (~3-4 h on a T4 GPU).
#
# 5. submission_v36.zip appears next to the notebook when done.
#    Upload it to the competition via the Kaggle UI.
#
# ================================================================
# ARCHITECTURE
# ================================================================
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  DINOv2 ViT-S/14-reg  (frozen — never updated during training)         │
#   │  Input: 518×518 RGB image                                               │
#   │  Patch size: 14×14 px  →  37×37 = 1 369 patch tokens                  │
#   │  We tap intermediate layers 6, 9, 12 (0-indexed: 5, 8, 11)             │
#   │  and concatenate their 384-dim outputs → 1 152-dim per patch            │
#   └────────────────────────────────┬────────────────────────────────────────┘
#                                    │ (B, 1369, 1152) patch tensor
#                                    ▼
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  SegHead  (trained per class, one per fold)                             │
#   │  Reshape flat patches → (B, 1152, 37, 37) spatial grid                 │
#   │  Conv1×1 1152→128  + BN + ReLU                                         │
#   │  Conv3×3 128→128   + BN + ReLU                                         │
#   │  Conv1×1 128→1                                                          │
#   │  Bilinear upsample → (B, 1, 224, 224) logit map                        │
#   └────────────────────────────────┬────────────────────────────────────────┘
#                                    │ sigmoid → anomaly score map [0,1]
#                                    ▼
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  3-fold ensemble: average score maps from 3 independently-trained heads │
#   │  4-fold TTA: average over orig / h-flip / v-flip / h+v-flip            │
#   │  Per-class normalisation: stretch [P_LO%ile, P_HI%ile] → [0, 1]       │
#   │  Gaussian blur (σ=4): smooth out patch-boundary artefacts               │
#   └─────────────────────────────────────────────────────────────────────────┘
#
# TRAINING DATA MIX (per batch)
# ------------------------------
#   30% good images with zero mask  → teaches the head to output low scores on normals
#   40% real anomaly images with GT → direct pixel-level supervision
#   30% cut-paste synthetic         → augments the rare anomaly class
#
# EARLY STOPPING
# --------------
#   Each fold reserves its validation slice (≈1/3 of anomaly images) for
#   monitoring pixel-AP.  Training stops when AP has not improved for
#   PATIENCE=5 consecutive epochs.  The best checkpoint is restored.
#
# SCORE NORMALISATION
# -------------------
#   After inference we have raw sigmoid scores in [0,1].  Different heads
#   and TTA variants may use different score ranges, so we re-scale per class:
#     lo = P_LO-th  percentile of good-image scores  (≈ background floor)
#     hi = P_HI-th  percentile of good-image scores  (≈ hardest normal)
#     score_norm = clip((score - lo) / (hi - lo), 0, 1)
#   P_HI = 99.999 is intentionally very permissive: we don't want to clip the
#   top of the anomaly score range since real anomalies may score above any
#   normal image and AP depends on fine-grained ranking.
#
# SCORE HISTORY
# -------------
#   v33-tta4  (4-fold TTA only)       → 0.8639
#   v34       (3-fold ensemble only)  → 0.8621
#   v36       (both combined)         → TBD
# ================================================================


# %%
# ════════════════════════════════════════════════════════════════
# CELL 0 — Install dependencies  (run once per session)
# ════════════════════════════════════════════════════════════════
# Colab already ships with torch, torchvision, scipy, sklearn, pillow, pandas.
# Uncomment the line below only if something is missing.
#
# !pip install -q torch torchvision scipy scikit-learn pillow pandas


# %%
# ════════════════════════════════════════════════════════════════
# CELL 1 — Imports
# ════════════════════════════════════════════════════════════════
import os, time, copy, random, zipfile
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
from sklearn.metrics import average_precision_score
from pathlib import Path


# %%
# ════════════════════════════════════════════════════════════════
# CELL 2 — CONFIG  ← the only cell you need to edit
# ════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────────
# DATA_ROOT  : folder containing class_01/ … class_08/.
#              Default: a 'dataset' folder sitting next to this notebook.
# OUTPUT_DIR : where submission_v36.csv and submission_v36.zip will be saved.
#              Default: same folder as this notebook.
DATA_ROOT  = Path('dataset')
OUTPUT_DIR = Path('.')

# Sanity-check: fail early with a clear message if the path is wrong.
if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset not found at '{DATA_ROOT.resolve()}'.\n"
        "Create a 'dataset/' folder next to this notebook and put the "
        "competition data inside it (class_01/, class_02/, …)."
    )
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device     : {device}")
if device.type == 'cuda':
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"DATA_ROOT  : {DATA_ROOT.resolve()}")
print(f"OUTPUT_DIR : {OUTPUT_DIR.resolve()}")

# ── Vision backbone ───────────────────────────────────────────────────────────
# DINOv2 ViT-S/14-reg: 'S' = small (384-dim), '14' = 14px patches, 'reg' = registers variant.
# We use 518×518 input because it is exactly 14 × 37 → clean 37×37 patch grid.
IMG_SIZE    = 518
PATCH_GRID  = 37           # = IMG_SIZE // 14; number of patches per side
OUTPUT_SIZE = 224          # segmentation output resolution (bilinear upsample from 37×37)
FEATURE_DIM = 384          # hidden dim of ViT-S per layer

# Which intermediate ViT layers to tap (0-indexed).
# Layer 5 = early (edges/textures), 8 = mid (parts), 11 = late (semantics).
# Concatenating all three gives the head both low- and high-level cues.
LAYERS_TO_USE  = [5, 8, 11]
MULTILAYER_DIM = FEATURE_DIM * len(LAYERS_TO_USE)  # 1152

# ── Training ─────────────────────────────────────────────────────────────────
SEED              = 42
EPOCHS            = 50   # max epochs; early stopping usually kicks in much sooner
PATIENCE          = 5    # stop if val pixel-AP doesn't improve for this many epochs
SAMPLES_PER_EPOCH = 200  # synthetic samples per epoch
                         # (gradient steps per epoch = SAMPLES_PER_EPOCH / BATCH_SIZE = 25)
BATCH_SIZE = 8
LR         = 1e-3

# Training batch composition — probabilities must sum to ≤ 1.0:
P_GOOD = 0.30  # probability a sample is a clean good image (zero anomaly mask)
P_REAL = 0.40  # probability a sample is a real anomaly image with GT mask
               # remaining 0.30 → cut-paste synthetic anomaly

# ── Inference / normalisation ─────────────────────────────────────────────────
BLUR_SIGMA = 4       # Gaussian blur σ to smooth patch-boundary artefacts in score maps
P_LO       = 0.5    # lower percentile for score stretch (removes near-zero background noise)
P_HI       = 99.999 # upper percentile (very high to avoid clipping severe anomaly scores)

# ── Test-time augmentation ───────────────────────────────────────────────────
# Each entry is a list of torch dimensions to flip before inference.
# After inference the flip is inverted so all maps align before averaging.
TTA_FLIPS = [
    [],       # original
    [-1],     # horizontal flip
    [-2],     # vertical flip
    [-1, -2], # both (= 180° rotation)
]

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    # Each DataLoader worker needs its own RNG seed; otherwise they all
    # generate the same augmentation sequences.
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ── Image pre-processing ─────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Config OK.")


# %%
# ════════════════════════════════════════════════════════════════
# CELL 3 — Datasets & augmentation utilities
# ════════════════════════════════════════════════════════════════

class TestDataset(Dataset):
    """
    Minimal dataset for inference — no augmentation, just load + preprocess.
    Returns (tensor_image, filename_string) so we can keep track of which
    score map belongs to which file.
    """
    def __init__(self, paths, tf):
        self.paths = paths
        self.tf    = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.tf(img), os.path.basename(self.paths[i])


def _resize_mask(mask, size=OUTPUT_SIZE):
    """
    Binarise and resize a GT mask to (size, size) using nearest-neighbour
    interpolation — we don't want anti-aliasing to create fractional GT values.
    """
    if mask.shape == (size, size):
        return (mask > 0).astype(np.float32)
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    pil = pil.resize((size, size), Image.NEAREST)
    return (np.array(pil) > 127).astype(np.float32)


def get_object_mask(img, threshold=30):
    """
    Coarse foreground mask: pixels brighter than `threshold` across channels.
    Used by cut_paste to bias the paste location onto the object, not the
    plain background (where anomalies never actually occur).
    """
    luma = img.mean(axis=2) if img.ndim == 3 else img
    return luma > threshold


def maybe_subcrop_large(crop_img, crop_mask, max_ratio=0.25):
    """
    If the anomaly occupies more than max_ratio of its bounding box area,
    take a random sub-crop centred on an anomaly pixel.
    This prevents pasting unrealistically huge blobs.
    """
    h, w = crop_mask.shape
    if (crop_mask > 0).sum() / (h * w + 1e-8) < max_ratio:
        return crop_img, crop_mask

    # Target a sub-crop covering 15-30% of the bounding box.
    target_area = h * w * random.uniform(0.15, 0.30)
    sub_h = max(8, min(int(np.sqrt(target_area * h / w)), h - 2))
    sub_w = max(8, min(int(target_area / sub_h), w - 2))

    # Centre the sub-crop on a random anomaly pixel.
    ys, xs = np.where(crop_mask > 0)
    if len(ys) == 0:
        return crop_img, crop_mask
    cy, cx = random.choice(ys), random.choice(xs)
    y0 = max(0, min(cy - sub_h // 2, h - sub_h))
    x0 = max(0, min(cx - sub_w // 2, w - sub_w))
    return crop_img[y0:y0+sub_h, x0:x0+sub_w].copy(), crop_mask[y0:y0+sub_h, x0:x0+sub_w].copy()


def cut_paste(good_img, ano_img, ano_mask, scale_range=(0.4, 1.2)):
    """
    Create a synthetic anomaly by pasting a scaled, colour-jittered crop from
    a real anomaly image onto a clean good image.

    Steps:
      1. Crop the anomaly bounding box from (ano_img, ano_mask).
      2. Optionally sub-crop if the anomaly is very large.
      3. Randomly scale the crop.
      4. Add a small uniform colour shift to reduce texture memorisation.
      5. Paste onto the good image at a location biased toward the foreground
         object, using a soft Gaussian alpha blend at the edges.

    Returns:
      synth    — np.uint8 (H, W, 3) composite image
      out_mask — np.float32 (H, W) soft alpha mask in [0, 1]
    """
    H, W = good_img.shape[:2]
    ys, xs = np.where(ano_mask > 0)
    if len(ys) == 0:
        return good_img.copy(), np.zeros((H, W), dtype=np.float32)

    # Step 1 & 2: crop the anomaly bounding box + optional sub-crop.
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop_img  = ano_img[y0:y1, x0:x1].copy()
    crop_mask = ano_mask[y0:y1, x0:x1].copy()
    crop_img, crop_mask = maybe_subcrop_large(crop_img, crop_mask)

    ch, cw = crop_img.shape[:2]
    if ch < 4 or cw < 4:
        return good_img.copy(), np.zeros((H, W), dtype=np.float32)

    # Step 3: random scale, keeping the crop ≤ half the image on each side.
    scale     = random.uniform(*scale_range)
    nh        = max(6, min(int(ch * scale), H // 2))
    nw        = max(6, min(int(cw * scale), W // 2))
    crop_img  = np.array(Image.fromarray(crop_img).resize((nw, nh), Image.BILINEAR))
    crop_mask = np.array(Image.fromarray(crop_mask).resize((nw, nh), Image.NEAREST))

    # Step 4: uniform RGB colour jitter (±15 per channel).
    shift    = np.random.randint(-15, 16, size=3).reshape(1, 1, 3)
    crop_img = np.clip(crop_img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

    # Step 5: choose a paste location inside the object foreground.
    obj_mask = get_object_mask(good_img)
    oys, oxs = np.where(obj_mask)
    if len(oys) == 0:
        py = random.randint(0, H - nh)
        px = random.randint(0, W - nw)
    else:
        oy0, oy1 = oys.min(), oys.max()
        ox0, ox1 = oxs.min(), oxs.max()
        # Allow a small overhang outside the object bounding box.
        pyl = max(0, oy0 - nh // 4); pyh = max(pyl, min(H - nh, oy1 - nh // 2))
        pxl = max(0, ox0 - nw // 4); pxh = max(pxl, min(W - nw, ox1 - nw // 2))
        py  = random.randint(pyl, pyh)
        px  = random.randint(pxl, pxh)

    # Soft-edge alpha blend: Gaussian-smoothed binary mask avoids hard edges.
    synth    = good_img.copy()
    out_mask = np.zeros((H, W), dtype=np.float32)
    alpha    = gaussian_filter((crop_mask > 0).astype(np.float32), sigma=1.0)
    region   = synth[py:py+nh, px:px+nw]
    synth[py:py+nh, px:px+nw] = (
        region * (1 - alpha[:, :, None]) + crop_img * alpha[:, :, None]
    ).astype(np.uint8)
    out_mask[py:py+nh, px:px+nw] = alpha
    return synth, out_mask


class TrainSynthDataset(Dataset):
    """
    Generates synthetic training samples on-the-fly.

    Each __getitem__ randomly picks one of three sample types:
      1. Good image  (prob p_good) → zero mask (negative supervision)
      2. Real anomaly              → GT mask   (direct supervision)
      3. Cut-paste synthetic       → soft alpha mask (data augmentation)

    Parameters
    ----------
    good_paths : list[str]   paths to clean good-class training images
    sources    : list[dict]  pre-loaded anomaly images {"image": np.array, "mask": np.array}
    n          : int         dataset length (= samples per epoch)
    tf         : callable    torchvision transform applied to the final PIL image
    """
    def __init__(self, good_paths, sources, n, tf, p_good=P_GOOD, p_real=P_REAL):
        self.good_paths = good_paths
        self.sources    = sources
        self.n, self.tf = n, tf
        self.p_good, self.p_real = p_good, p_real

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        r = random.random()

        if r < self.p_good or not self.sources:
            # Branch 1: clean image → zero mask.
            img = np.array(Image.open(random.choice(self.good_paths)).convert("RGB"))
            return self.tf(Image.fromarray(img)), torch.zeros(OUTPUT_SIZE, OUTPUT_SIZE)

        if r < self.p_good + self.p_real:
            # Branch 2: real anomaly with GT mask.
            src = random.choice(self.sources)
            return (
                self.tf(Image.fromarray(src["image"])),
                torch.from_numpy(_resize_mask(src["mask"])).float(),
            )

        # Branch 3: cut-paste synthetic anomaly.
        good_img = np.array(Image.open(random.choice(self.good_paths)).convert("RGB"))
        src      = random.choice(self.sources)
        synth, mask = cut_paste(good_img, src["image"], src["mask"])
        return (
            self.tf(Image.fromarray(synth)),
            torch.from_numpy(mask).float(),
        )

print("Dataset classes defined.")


# %%
# ════════════════════════════════════════════════════════════════
# CELL 4 — Model definition
# ════════════════════════════════════════════════════════════════

class SegHead(nn.Module):
    """
    Lightweight anomaly segmentation head on top of frozen DINOv2 features.

    Design:
      - 1×1 conv first: mix features across the 1152 channel dimension.
      - 3×3 conv second: capture local spatial context in the 37×37 grid.
      - Bilinear upsample at the end: smooth, no checkerboard artefacts.

    Input  : (B, N_patches, MULTILAYER_DIM)   flat patch token sequence
    Output : (B, 1, OUTPUT_SIZE, OUTPUT_SIZE)  anomaly logit map (before sigmoid)
    """
    def __init__(self, in_dim=MULTILAYER_DIM, hidden=128,
                 patch_grid=PATCH_GRID, out_size=OUTPUT_SIZE):
        super().__init__()
        self.patch_grid = patch_grid
        self.out_size   = out_size
        self.conv1 = nn.Conv2d(in_dim, hidden, 1)            # 1×1: channel projection
        self.bn1   = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1) # 3×3: spatial context
        self.bn2   = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, 1, 1)                 # 1×1: collapse to score map

    def forward(self, p):
        B = p.size(0)
        # Reshape flat patch sequence → 2-D spatial grid.
        x = p.transpose(1, 2).reshape(B, -1, self.patch_grid, self.patch_grid)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return F.interpolate(x, size=(self.out_size, self.out_size),
                             mode='bilinear', align_corners=False)


def bce_dice_loss(logits, target):
    """
    BCE + soft Dice loss.

    BCE alone struggles with extreme class imbalance (anomaly pixels << good pixels).
    Dice normalises by the predicted + GT mass, making it more sensitive to
    small but correctly-localised anomalies.  Combining both gives stable
    gradients (BCE) and good recall on small defects (Dice).
    """
    bce  = F.binary_cross_entropy_with_logits(logits, target)
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    # +1.0 smoothing prevents division by zero when both pred and target are zero.
    dice = 1 - (2 * inter + 1.0) / (union + 1.0)
    return bce + dice.mean()

print("Model defined.")


# %%
# ════════════════════════════════════════════════════════════════
# CELL 5 — Data-loading helpers
# ════════════════════════════════════════════════════════════════

def collect_anomaly_kfold(data_root, n_folds=3):
    """
    Build k-fold cross-validation splits over the labelled anomaly images.

    Only images with a non-zero GT mask are included — images where the
    annotator recorded no pixel anomaly provide no pixel-AP signal.

    Fold assignment is round-robin (image i → fold i % n_folds) over the
    sorted list per class, so each fold sees roughly the same anomaly types.

    Returns
    -------
    list of n_folds tuples (train_sources, val_items):
      train_sources : dict {class_name → list[{"image": np.array, "mask": np.array}]}
      val_items     : dict {class_name → list[{"path": str, "mask": np.array}]}
    """
    all_pairs = defaultdict(list)
    for class_dir in sorted(data_root.iterdir()):
        train_dir = class_dir / "train"
        gt_dir    = class_dir / "ground_truth_train"
        if not (train_dir.is_dir() and gt_dir.is_dir()):
            continue
        class_name = class_dir.name
        for ano_gt_dir in sorted(gt_dir.iterdir()):
            ano_img_dir = train_dir / ano_gt_dir.name
            if not ano_gt_dir.is_dir():
                continue
            for mask_path in sorted(ano_gt_dir.glob("*.png")):
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask.max() == 0:
                    continue  # no pixel GT → skip
                all_pairs[class_name].append((ano_img_dir / mask_path.name, mask))

    folds = []
    for fold_idx in range(n_folds):
        train_sources = defaultdict(list)
        val_items     = defaultdict(list)
        for cn, pairs in all_pairs.items():
            for i, (img_path, mask) in enumerate(pairs):
                if i % n_folds == fold_idx:
                    # Val: keep only path (not pixels) to save RAM.
                    val_items[cn].append({"path": str(img_path), "mask": mask})
                else:
                    # Train: pre-load pixels so workers don't compete on disk.
                    train_sources[cn].append({
                        "image": np.array(Image.open(img_path).convert("RGB")),
                        "mask":  mask,
                    })
        folds.append((train_sources, val_items))
    return folds


@torch.no_grad()
def extract_multilayer_patches(model, x, layers=LAYERS_TO_USE):
    """
    Run x through DINOv2 and return concatenated intermediate patch features.

    get_intermediate_layers returns (B, N_patches, FEATURE_DIM) per layer.
    We concatenate along the channel dim → (B, N_patches, MULTILAYER_DIM).
    norm=True applies the ViT's layer-norm to each intermediate output,
    stabilising the scale across layers before concatenation.
    """
    intermediates = model.get_intermediate_layers(
        x,
        n=layers,
        reshape=False,          # keep flat (B, N, C) — SegHead reshapes itself
        return_class_token=False,
        norm=True,
    )
    return torch.cat(intermediates, dim=2)


@torch.no_grad()
def _val_pixel_ap(model, head, val_items):
    """
    Compute pixel-level average precision on a list of validation items.
    AP is the exact metric used by the competition leaderboard.
    Returns 0.0 if there are no positive pixels (avoids a sklearn crash).
    """
    head.eval()
    all_preds, all_gt = [], []
    for item in val_items:
        img     = preprocess(Image.open(item["path"]).convert("RGB")).unsqueeze(0).to(device)
        patches = extract_multilayer_patches(model, img)
        score   = torch.sigmoid(head(patches)).squeeze().cpu().numpy()  # (H, W)

        mask = (item["mask"] > 0).astype(np.uint8)
        if mask.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
            mask = (np.array(
                Image.fromarray(mask * 255).resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST)
            ) > 127).astype(np.uint8)

        all_preds.append(score.flatten())
        all_gt.append(mask.flatten())

    gt = np.concatenate(all_gt)
    if gt.max() == 0:
        return 0.0
    return float(average_precision_score(gt, np.concatenate(all_preds)))

print("Helpers defined.")


# %%
# ════════════════════════════════════════════════════════════════
# CELL 6 — Submission encoding
# ════════════════════════════════════════════════════════════════

def fn_to_id(fn):
    """Strip directory and extension to get the submission row ID."""
    return Path(fn).stem


def float_matrix_to_q8rle(x):
    """
    Encode a float [0,1] anomaly map into the competition's q8rle format.

    Format: "q8rle <H> <W> <val0> <run0> <val1> <run1> ..."
      - q8  : values quantised to uint8 (0–255)
      - rle : consecutive equal values are run-length encoded
      - Column-major order (matrix is transposed before flattening)

    Column-major order matches the competition's expected decoding.
    """
    q    = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
    h, w = q.shape
    flat = q.T.reshape(-1)  # column-major: transpose then flatten
    if flat.size == 0:
        return f"q8rle {h} {w}"

    cuts   = np.flatnonzero(flat[1:] != flat[:-1]) + 1  # positions where value changes
    starts = np.r_[0, cuts]
    ends   = np.r_[cuts, flat.size]
    parts  = ["q8rle", str(h), str(w)]
    for v, n in zip(flat[starts], ends - starts):
        parts += [str(int(v)), str(int(n))]
    return " ".join(parts)

print("Submission encoding defined.")


# %%
# ════════════════════════════════════════════════════════════════
# CELL 7 — [1/5] Load DINOv2 backbone
# ════════════════════════════════════════════════════════════════
# dinov2_vits14_reg: ViT-S/14 with register tokens.
# Register tokens improve patch-feature quality by absorbing global artefacts.
# We freeze all parameters — the backbone is a fixed feature extractor only.
#
# First run downloads ~350 MB from PyTorch Hub (cached in ~/.cache/torch/).
print("=== [1/5] Load DINOv2-S-reg ===")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', verbose=False)
dinov2 = dinov2.to(device).eval()
for p in dinov2.parameters():
    p.requires_grad = False
print(f"  Params: {sum(p.numel() for p in dinov2.parameters()):,}")


# %%
# ════════════════════════════════════════════════════════════════
# CELL 8 — [2/5] Build 3-fold splits
# ════════════════════════════════════════════════════════════════
print("\n=== [2/5] Build 3-fold splits ===")
FOLDS = collect_anomaly_kfold(DATA_ROOT, n_folds=3)
for fi, (ts, vi) in enumerate(FOLDS):
    n_train = sum(len(v) for v in ts.values())
    n_val   = sum(len(v) for v in vi.values())
    print(f"  Fold {fi}: train={n_train} anomaly images, val={n_val} anomaly images")


# %%
# ════════════════════════════════════════════════════════════════
# CELL 9 — [3/5] Train SegHeads
# ════════════════════════════════════════════════════════════════
# For each of the 3 folds, train one SegHead per class.
# Each head is trained on 2/3 of the anomaly images and validated on 1/3.
# Early stopping restores the best checkpoint by val pixel-AP.
#
# Total: 3 folds × 8 classes = 24 heads.  ~1-2 min per head on T4 → ~40-60 min.
print(f"\n=== [3/5] Train 3-fold SegHeads (patience={PATIENCE}) ===")
ALL_HEADS = []  # list of {class_name: SegHead}, one dict per fold

for fold_idx, (fold_train, fold_val) in enumerate(FOLDS):
    print(f"\n--- Fold {fold_idx} ---")
    fold_heads = {}

    for cls_idx, class_name in enumerate(sorted(fold_train.keys())):
        good_dir   = DATA_ROOT / class_name / "train" / "good"
        good_paths = [str(p) for p in sorted(good_dir.glob("*.png"))]

        # Unique seed per (fold, class) → different data shuffle per head.
        cls_seed = SEED + fold_idx * 100 + cls_idx
        g = torch.Generator()
        g.manual_seed(cls_seed)

        loader = DataLoader(
            TrainSynthDataset(good_paths, fold_train[class_name], SAMPLES_PER_EPOCH, preprocess),
            batch_size=BATCH_SIZE,
            num_workers=0,
            pin_memory=True,
            generator=g,
            worker_init_fn=worker_init_fn,
        )

        head  = SegHead().to(device)
        optim = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)

        class_val  = fold_val.get(class_name, [])
        best_ap    = -1.0
        best_state = None
        no_improve = 0
        t0 = time.time()

        for epoch in range(EPOCHS):
            head.train()
            for imgs, masks in loader:
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).unsqueeze(1)  # add channel dim
                # Backbone is frozen → no_grad saves memory and time.
                with torch.no_grad():
                    patches = extract_multilayer_patches(dinov2, imgs)
                loss = bce_dice_loss(head(patches), masks)
                optim.zero_grad()
                loss.backward()
                optim.step()

            if class_val:
                val_ap = _val_pixel_ap(dinov2, head, class_val)
                head.train()  # _val_pixel_ap calls eval(); switch back to train mode
                if val_ap > best_ap:
                    best_ap    = val_ap
                    best_state = copy.deepcopy(head.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1

            if no_improve >= PATIENCE:
                break  # early stopping: best_state already saved

        if best_state is not None:
            head.load_state_dict(best_state)  # restore best checkpoint
        head.eval()
        fold_heads[class_name] = head

        ap_msg = f" val_ap={best_ap:.4f}" if class_val else ""
        print(f"  {class_name} fold{fold_idx}  stop@epoch {epoch+1}  {time.time()-t0:.0f}s{ap_msg}")
        torch.cuda.empty_cache()  # free activations between classes

    ALL_HEADS.append(fold_heads)


# %%
# ════════════════════════════════════════════════════════════════
# CELL 10 — [4/5] Inference (3 folds × 4 TTA variants)
# ════════════════════════════════════════════════════════════════
# Two passes per head per class:
#   1. Good training images → used to compute per-class normalisation percentiles.
#   2. Test images          → the scores that go into the submission.
print("\n=== [4/5] Inference (3-fold × 4-fold TTA) ===")
SH_TRAIN_36 = {}  # {class_name: np.array (N_good, H, W)}
SH_TEST_36  = {}  # {class_name: {filename: np.array (H, W)}}

for class_name in sorted(ALL_HEADS[0].keys()):
    good_dir = DATA_ROOT / class_name / "train" / "good"
    good     = [str(p) for p in sorted(good_dir.glob("*.png"))]
    test_dir = DATA_ROOT / class_name / "test"
    test     = [str(p) for p in sorted(test_dir.glob("*.png"))]

    train_preds = []   # one (N_good, H, W) array per fold
    test_preds  = []   # one (N_test, H, W) array per fold
    test_fns    = None

    for fold_heads in ALL_HEADS:
        head = fold_heads[class_name]
        head.eval()

        # ── Pass 1: good images (for normalisation percentiles) ──
        loader = DataLoader(TestDataset(good, preprocess), batch_size=8, num_workers=0, pin_memory=True)
        tr_sc = []
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            aug  = []
            for fd in TTA_FLIPS:
                x = torch.flip(imgs, fd) if fd else imgs
                with torch.no_grad():
                    s = torch.sigmoid(head(extract_multilayer_patches(dinov2, x))).squeeze(1)
                if fd:
                    s = torch.flip(s, fd)  # undo flip so maps align before averaging
                aug.append(s)
            tr_sc.append(torch.stack(aug).mean(0).cpu().numpy())
        train_preds.append(np.concatenate(tr_sc, axis=0))

        # ── Pass 2: test images ──
        loader = DataLoader(TestDataset(test, preprocess), batch_size=8, num_workers=0, pin_memory=True)
        te_sc, fns = [], []
        for imgs, names in loader:
            imgs = imgs.to(device, non_blocking=True)
            aug  = []
            for fd in TTA_FLIPS:
                x = torch.flip(imgs, fd) if fd else imgs
                with torch.no_grad():
                    s = torch.sigmoid(head(extract_multilayer_patches(dinov2, x))).squeeze(1)
                if fd:
                    s = torch.flip(s, fd)
                aug.append(s)
            te_sc.append(torch.stack(aug).mean(0).cpu().numpy())
            fns.extend(names)
        test_preds.append(np.concatenate(te_sc, axis=0))
        if test_fns is None:
            test_fns = fns

    # Average across folds (the ensemble step).
    SH_TRAIN_36[class_name] = np.mean(train_preds, axis=0)
    SH_TEST_36[class_name]  = dict(zip(test_fns, np.mean(test_preds, axis=0)))
    print(f"  {class_name} done ({len(test)} test images)")

torch.cuda.empty_cache()


# %%
# ════════════════════════════════════════════════════════════════
# CELL 11 — [5/5] Normalise + write submission
# ════════════════════════════════════════════════════════════════
# Per-class score stretch: maps [P_LO-th, P_HI-th] percentile of good-image
# scores to [0, 1].  Then Gaussian blur smooths the patch-grid artefacts.
print("\n=== [5/5] Write submission ===")
rows = []
for class_name in sorted(SH_TEST_36.keys()):
    good_scores  = SH_TRAIN_36[class_name].flatten()
    sh_lo, sh_hi = np.percentile(good_scores, [P_LO, P_HI])

    for fn in sorted(SH_TEST_36[class_name].keys()):
        raw   = SH_TEST_36[class_name][fn]
        # Stretch scores to [0,1] relative to the good-image distribution.
        score = np.clip((raw - sh_lo) / (sh_hi - sh_lo + 1e-8), 0, 1)
        # Smooth out 14×14 patch-boundary artefacts visible in the raw score map.
        score = gaussian_filter(score, sigma=BLUR_SIGMA).astype(np.float32)
        rows.append({'ID': fn_to_id(fn), 'Label': float_matrix_to_q8rle(score)})

df       = pd.DataFrame(rows)
csv_path = OUTPUT_DIR / 'submission_v36.csv'
zip_path = OUTPUT_DIR / 'submission_v36.zip'

df.to_csv(csv_path, index=False)
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(csv_path, arcname='submission.csv')

print(f"  {len(df)} rows")
print(f"  CSV : {csv_path.resolve()}")
print(f"  ZIP : {zip_path.resolve()}  ({os.path.getsize(zip_path) / 1024 / 1024:.2f} MB)")
print("\nDone. Submit submission_v36.zip to the competition.")
