# ================================================================
# v29: High-resolution 518×518 input (was 224×224)
#
# Structural change: DINOv2 patch size = 14. At 224: 16×16=256 patches.
# At 518: 37×37=1369 patches (5.3× finer spatial resolution).
#
# Why this matters for pixel-AP:
#   - 16×16 grid: each patch covers 14×14 pixel block → anomaly
#     localization limited to 14×14 resolution
#   - 37×37 grid: same 14×14 patches but denser coverage →
#     anomaly boundaries much sharper after upsample
#   - Less need for Gaussian smoothing (fewer blocky artifacts)
#
# Changes vs v28:
#   1. IMG_SIZE: 224 → 518
#   2. PATCH_GRID: 16 → 37
#   3. CORESET_RATIO: 0.10 → 0.02 (keep bank size manageable)
#   4. Chunked feature extraction (memory control)
#   5. Output still 224×224 (competition format)
#
# Everything else from v28: DINOv2-B, multi-layer MB, top-9,
# Gaussian sigma=4, mean fusion, W=0.45/0.55, P=[0.5, 99.999]
#
# Runtime: ~1.5-2 hours (more patches per image = slower)
# ================================================================
import os, time, gc, random, re, json, zipfile
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

# ---------------- Reproducibility ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_ROOT = '/kaggle/input/datasets/cindy11102858/adl-anomaly-mirror/adl-2025-2026-anomaly-detection'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === KEY CHANGE: high-resolution input ===
IMG_SIZE = 518              # was 224
PATCH_GRID = 37             # was 16 (518 / 14 = 37)
OUTPUT_SIZE = 224           # submission format stays 224×224

FEATURE_DIM = 768           # DINOv2-B
LAYERS_TO_USE = [5, 8, 11]
MULTILAYER_DIM = FEATURE_DIM * len(LAYERS_TO_USE)  # 2304

# Adjusted for 5.3× more patches per image
CORESET_RATIO = 0.02       # was 0.10 (keeps bank ~same absolute size)
TOPK = 9

# Training
EPOCHS = 30
SAMPLES_PER_EPOCH = 200
BATCH_SIZE = 8              # smaller for 518 input
LR = 1e-3
P_GOOD = 0.30
P_REAL = 0.40

# Ensemble & normalize
W_MB = 0.45
W_SH = 0.55
P_LO = 0.5
P_HI = 99.999
SMOOTH_SIGMA = 4


preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageFolderDataset(Dataset):
    def __init__(self, paths, tf): self.paths, self.tf = paths, tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.tf(Image.open(self.paths[i]).convert('RGB'))

class TestDataset(Dataset):
    def __init__(self, paths, tf): self.paths, self.tf = paths, tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB')), os.path.basename(self.paths[i])


# ---------------- [1/8] Load DINOv2-B ----------------
print("=== [1/8] Load DINOv2-B ===")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
dinov2 = dinov2.to(device).eval()
for p in dinov2.parameters():
    p.requires_grad = False
print(f"  Params: {sum(p.numel() for p in dinov2.parameters()):,}")

# Verify 518 input works
with torch.no_grad():
    test_out = dinov2.forward_features(torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device))
    actual_grid = int(test_out['x_norm_patchtokens'].shape[1] ** 0.5)
    assert actual_grid == PATCH_GRID, f"Expected grid {PATCH_GRID}, got {actual_grid}"
    print(f"  Verified: {IMG_SIZE}×{IMG_SIZE} → {PATCH_GRID}×{PATCH_GRID} patch grid")
    del test_out
    torch.cuda.empty_cache()


@torch.no_grad()
def extract_multilayer_patches(model, x, layers=LAYERS_TO_USE):
    intermediates = model.get_intermediate_layers(
        x, n=layers, reshape=False, return_class_token=False, norm=True
    )
    return torch.cat(intermediates, dim=2)


# === Chunked feature extraction (memory-safe for 518) ===
@torch.no_grad()
def extract_and_subsample_features(model, paths, ratio=CORESET_RATIO,
                                    chunk_size=200, batch_size=4):
    """Extract multi-layer features in chunks, subsample each chunk."""
    rng = np.random.RandomState(SEED)
    all_feats = []
    
    for start in range(0, len(paths), chunk_size):
        chunk_paths = paths[start:start + chunk_size]
        loader = DataLoader(ImageFolderDataset(chunk_paths, preprocess),
                           batch_size=batch_size, num_workers=2, pin_memory=True)
        chunk_feats = []
        for batch in loader:
            ml = extract_multilayer_patches(model, batch.to(device, non_blocking=True))
            chunk_feats.append(ml.reshape(-1, MULTILAYER_DIM).cpu())
        
        chunk_feats = torch.cat(chunk_feats, dim=0)
        n_keep = max(1, int(chunk_feats.shape[0] * ratio))
        idx = rng.choice(chunk_feats.shape[0], size=n_keep, replace=False)
        all_feats.append(chunk_feats[idx])
        del chunk_feats
        gc.collect()
    
    return torch.cat(all_feats, dim=0)


# ---------------- [2/8] Build memory banks ----------------
print(f"\n=== [2/8] Build memory banks (518×518, CORESET={CORESET_RATIO}) ===")
required_classes = sorted([
    c for c in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, c, 'train', 'good'))
])
MEMORY_BANKS = {}
for class_name in required_classes:
    good_dir = os.path.join(DATA_ROOT, class_name, 'train', 'good')
    paths = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith('.png')])
    feats = extract_and_subsample_features(dinov2, paths)
    bank = F.normalize(feats, p=2, dim=1)
    MEMORY_BANKS[class_name] = bank
    print(f"  {class_name}: bank {tuple(bank.shape)} | "
          f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    del feats; gc.collect()
    torch.cuda.empty_cache()


# ---------------- Cut-paste augmentation ----------------
def get_object_mask(img, threshold=30):
    return (img.mean(axis=2) if img.ndim == 3 else img) > threshold


def maybe_subcrop_large(crop_img, crop_mask, max_ratio=0.25):
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
    return crop_img[y0:y0+sub_h, x0:x0+sub_w].copy(), crop_mask[y0:y0+sub_h, x0:x0+sub_w].copy()


def cut_paste(good_img, ano_img, ano_mask, scale_range=(0.4, 1.2)):
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
    crop_img = np.array(Image.fromarray(crop_img).resize((nw, nh), Image.BILINEAR))
    crop_mask = np.array(Image.fromarray(crop_mask).resize((nw, nh), Image.NEAREST))

    shift = np.random.randint(-15, 16, size=3).reshape(1, 1, 3)
    crop_img = np.clip(crop_img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

    obj_mask = get_object_mask(good_img)
    oys, oxs = np.where(obj_mask)
    if len(oys) == 0:
        py = random.randint(0, H - nh); px = random.randint(0, W - nw)
    else:
        oy0, oy1, ox0, ox1 = oys.min(), oys.max(), oxs.min(), oxs.max()
        pyl = max(0, oy0 - nh // 4); pyh = max(pyl, min(H - nh, oy1 - nh // 2))
        pxl = max(0, ox0 - nw // 4); pxh = max(pxl, min(W - nw, ox1 - nw // 2))
        py = random.randint(pyl, pyh); px = random.randint(pxl, pxh)

    synth = good_img.copy()
    out_mask = np.zeros((H, W), dtype=np.float32)
    alpha = gaussian_filter((crop_mask > 0).astype(np.float32), sigma=1.0)
    region = synth[py:py+nh, px:px+nw]
    synth[py:py+nh, px:px+nw] = (region * (1 - alpha[:, :, None]) +
                                 crop_img * alpha[:, :, None]).astype(np.uint8)
    out_mask[py:py+nh, px:px+nw] = alpha
    return synth, out_mask


# ---------------- [3/8] Collect anomaly sources ----------------
print("\n=== [3/8] Collect anomaly sources ===")
ANOMALY_SOURCES = defaultdict(list)
for class_name in required_classes:
    train_dir = os.path.join(DATA_ROOT, class_name, 'train')
    gt_dir = os.path.join(DATA_ROOT, class_name, 'ground_truth_train')
    if not (os.path.isdir(train_dir) and os.path.isdir(gt_dir)):
        continue
    for ano_type in sorted(os.listdir(gt_dir)):
        ano_gt_dir = os.path.join(gt_dir, ano_type)
        ano_img_dir = os.path.join(train_dir, ano_type)
        if not os.path.isdir(ano_gt_dir):
            continue
        for fn in sorted(os.listdir(ano_gt_dir)):
            mask = np.array(Image.open(os.path.join(ano_gt_dir, fn)).convert('L'))
            if mask.max() == 0:
                continue
            img = np.array(Image.open(os.path.join(ano_img_dir, fn)).convert('RGB'))
            ANOMALY_SOURCES[class_name].append({'image': img, 'mask': mask})
total = sum(len(v) for v in ANOMALY_SOURCES.values())
print(f"  Total anomaly sources: {total}")


def resize_mask(mask, size=OUTPUT_SIZE):
    """Resize mask to output size (224×224 for submission)."""
    if mask.shape == (size, size):
        return (mask > 0).astype(np.float32)
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    pil = pil.resize((size, size), Image.NEAREST)
    return (np.array(pil) > 127).astype(np.float32)


class TrainSynthDataset(Dataset):
    def __init__(self, good_paths, real_anomaly_sources, n, tf,
                 p_good=P_GOOD, p_real=P_REAL):
        self.good_paths = good_paths
        self.sources = real_anomaly_sources
        self.n = n
        self.tf = tf
        self.p_good = p_good
        self.p_real = p_real

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        r = random.random()
        if r < self.p_good or len(self.sources) == 0:
            good_img = np.array(Image.open(random.choice(self.good_paths)).convert('RGB'))
            return self.tf(Image.fromarray(good_img)), torch.zeros(OUTPUT_SIZE, OUTPUT_SIZE, dtype=torch.float32)
        elif r < self.p_good + self.p_real:
            src = random.choice(self.sources)
            mask = resize_mask(src['mask'])
            return self.tf(Image.fromarray(src['image'])), torch.from_numpy(mask).float()
        else:
            good_img = np.array(Image.open(random.choice(self.good_paths)).convert('RGB'))
            src = random.choice(self.sources)
            synth, mask = cut_paste(good_img, src['image'], src['mask'])
            return self.tf(Image.fromarray(synth)), torch.from_numpy(mask).float()


# ---------------- [4/8] SegHead (37×37 grid → 224×224 output) ----------------
class SegHead(nn.Module):
    def __init__(self, in_dim=MULTILAYER_DIM, hidden=128,
                 patch_grid=PATCH_GRID, out_size=OUTPUT_SIZE):
        super().__init__()
        self.patch_grid = patch_grid
        self.out_size = out_size
        self.conv1 = nn.Conv2d(in_dim, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, 1, 1)

    def forward(self, p):
        B = p.size(0)
        x = p.transpose(1, 2).reshape(B, -1, self.patch_grid, self.patch_grid)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return F.interpolate(x, size=(self.out_size, self.out_size),
                           mode='bilinear', align_corners=False)


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1.0) / (union + 1.0)
    return bce + dice.mean()


# ---------------- [5/8] Train SegHeads ----------------
print(f"\n=== [5/8] Train v29 SegHeads ({IMG_SIZE}×{IMG_SIZE}, grid={PATCH_GRID}) ===")
SEG_HEADS = {}
for class_name in required_classes:
    if class_name not in ANOMALY_SOURCES:
        print(f"  {class_name}: SKIP")
        continue
    good_dir = os.path.join(DATA_ROOT, class_name, 'train', 'good')
    good_paths = [os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith('.png')]
    sources = ANOMALY_SOURCES[class_name]

    loader = DataLoader(
        TrainSynthDataset(good_paths, sources, SAMPLES_PER_EPOCH, preprocess),
        batch_size=BATCH_SIZE, num_workers=2, pin_memory=True
    )
    head = SegHead().to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(EPOCHS):
        head.train()
        epoch_losses = []
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).unsqueeze(1)
            with torch.no_grad():
                patches = extract_multilayer_patches(dinov2, imgs)
            loss = bce_dice_loss(head(patches), masks)
            optim.zero_grad(); loss.backward(); optim.step()
            epoch_losses.append(loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  {class_name} epoch {epoch+1}/{EPOCHS} loss={np.mean(epoch_losses):.4f} | "
                  f"GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    SEG_HEADS[class_name] = head
    torch.save(head.state_dict(), f'/kaggle/working/seghead_v29_{class_name}.pt')
    print(f"  {class_name} done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()


# ---------------- [6/8] Inference ----------------
@torch.no_grad()
def memorybank_topk_infer(paths, bank, k=TOPK, batch_size=4):
    """MB inference at 518×518: 37×37 scores → upsample to 224×224."""
    loader = DataLoader(TestDataset(paths, preprocess),
                        batch_size=batch_size, num_workers=2, pin_memory=True)
    bank_gpu = bank.to(device)
    scores, fns = [], []
    for imgs, names in loader:
        imgs = imgs.to(device, non_blocking=True)
        patches = extract_multilayer_patches(dinov2, imgs)
        patches = F.normalize(patches, p=2, dim=2)
        sim = patches @ bank_gpu.T
        topk_sim, _ = sim.topk(k, dim=2)
        mean_topk_sim = topk_sim.mean(dim=2)
        s = F.interpolate(
            (1.0 - mean_topk_sim).reshape(-1, PATCH_GRID, PATCH_GRID).unsqueeze(1),
            size=(OUTPUT_SIZE, OUTPUT_SIZE), mode='bilinear', align_corners=False
        ).squeeze(1)
        scores.append(s.cpu().numpy()); fns.extend(names)
    return np.concatenate(scores, axis=0), fns


@torch.no_grad()
def seghead_infer(paths, head, batch_size=4):
    loader = DataLoader(TestDataset(paths, preprocess),
                        batch_size=batch_size, num_workers=2, pin_memory=True)
    head.eval()
    scores, fns = [], []
    for imgs, names in loader:
        imgs = imgs.to(device, non_blocking=True)
        patches = extract_multilayer_patches(dinov2, imgs)
        s = torch.sigmoid(head(patches)).squeeze(1).cpu().numpy()
        scores.append(s); fns.extend(names)
    return np.concatenate(scores, axis=0), fns


print("\n=== [6/8] Inference (518×518) ===")
MB_TRAIN, MB_TEST, SH_TRAIN, SH_TEST = {}, {}, {}, {}
for class_name in sorted(SEG_HEADS.keys()):
    good_dir = os.path.join(DATA_ROOT, class_name, 'train', 'good')
    good = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith('.png')])
    test_dir = os.path.join(DATA_ROOT, class_name, 'test')
    test = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')])

    mb_tr, _ = memorybank_topk_infer(good, MEMORY_BANKS[class_name])
    mb_te, mfns = memorybank_topk_infer(test, MEMORY_BANKS[class_name])
    sh_tr, _ = seghead_infer(good, SEG_HEADS[class_name])
    sh_te, sfns = seghead_infer(test, SEG_HEADS[class_name])
    MB_TRAIN[class_name] = mb_tr
    MB_TEST[class_name] = {fn: s for fn, s in zip(mfns, mb_te)}
    SH_TRAIN[class_name] = sh_tr
    SH_TEST[class_name] = {fn: s for fn, s in zip(sfns, sh_te)}
    print(f"  {class_name} done ({len(test)} test)")
torch.cuda.empty_cache()


# ---------------- [7/8] Ensemble + smooth + multi-view fusion ----------------
print("\n=== [7/8] Ensemble + smooth + multi-view fusion ===")

def parse_object_view(fn):
    m = re.match(r'^(.*)_view(\d+)\.png$', fn)
    if not m:
        raise ValueError(fn)
    return m.group(1), int(m.group(2))


PER_VIEW_ENS = {}
for class_name in sorted(MB_TEST.keys()):
    mb_lo, mb_hi = np.percentile(MB_TRAIN[class_name].flatten(), [P_LO, P_HI])
    sh_lo, sh_hi = np.percentile(SH_TRAIN[class_name].flatten(), [P_LO, P_HI])
    PER_VIEW_ENS[class_name] = {}
    for fn in MB_TEST[class_name].keys():
        n_mb = np.clip((MB_TEST[class_name][fn] - mb_lo) / (mb_hi - mb_lo + 1e-8), 0, 1)
        n_sh = np.clip((SH_TEST[class_name][fn] - sh_lo) / (sh_hi - sh_lo + 1e-8), 0, 1)
        ens = np.clip(W_MB * n_mb + W_SH * n_sh, 0, 1).astype(np.float32)
        ens = gaussian_filter(ens, sigma=SMOOTH_SIGMA).astype(np.float32)
        PER_VIEW_ENS[class_name][fn] = ens


FUSED_SCORES = {}
for class_name in sorted(PER_VIEW_ENS.keys()):
    object_views = defaultdict(list)
    for fn, score in PER_VIEW_ENS[class_name].items():
        obj_id, _ = parse_object_view(fn)
        object_views[obj_id].append((fn, score))

    print(f"  [{class_name}] {len(object_views)} objects")

    FUSED_SCORES[class_name] = {}
    for obj_id, view_list in object_views.items():
        scores_stack = np.stack([s for _, s in view_list], axis=0)
        fused = scores_stack.mean(axis=0)
        for fn, _ in view_list:
            FUSED_SCORES[class_name][fn] = fused.astype(np.float32)


# ---------------- [8/8] Encode + submit ----------------
def fn_to_id(fn):
    return Path(fn).stem


def float_matrix_to_q8rle(x):
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
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


print("\n=== [8/8] Encode v29 ===")
rows = []
for class_name in sorted(FUSED_SCORES.keys()):
    for fn in sorted(FUSED_SCORES[class_name].keys()):
        rows.append({'ID': fn_to_id(fn), 'Label': float_matrix_to_q8rle(FUSED_SCORES[class_name][fn])})

df = pd.DataFrame(rows)
csv = '/kaggle/working/submission_v29.csv'
zip_path = '/kaggle/working/submission_v29.zip'
df.to_csv(csv, index=False)
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(csv, arcname='submission.csv')
print(f"  csv {len(df)} rows, zip {os.path.getsize(zip_path)/1024/1024:.2f} MB")


kaggle_json = Path('/root/.kaggle/kaggle.json')
if not kaggle_json.exists():
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    kaggle_json.parent.mkdir(parents=True, exist_ok=True)
    with open(kaggle_json, 'w') as f:
        json.dump({"username": secrets.get_secret("KAGGLE_USERNAME"),
                   "key": secrets.get_secret("KAGGLE_KEY")}, f)
    os.chmod(kaggle_json, 0o600)

os.system(
    'kaggle competitions submit -c adl-2025-2026-anomaly-detection '
    '-f /kaggle/working/submission_v29.zip '
    '-m "v29: 518x518 high-res input (37x37 grid) + multi-layer MB + smooth + fusion"'
)
print("\n submitted")
