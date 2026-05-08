# ================================================================
# Spacepresso v7 - LB 0.5884
# DINOv2 memory bank + multi-layer SegHead, ensemble 0.45/0.55
# ================================================================
import os, time, gc, random, json, zipfile
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

# Update DATA_ROOT to your dataset path
DATA_ROOT = '/kaggle/input/datasets/cindy11102858/adl-anomaly-mirror/adl-2025-2026-anomaly-detection'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 224
PATCH_GRID = 16
FEATURE_DIM = 384
LAYERS_TO_USE = [5, 8, 11]                 # 0-indexed: layers 6, 9, 12
MULTILAYER_DIM = FEATURE_DIM * len(LAYERS_TO_USE)  # 1152
CORESET_RATIO = 0.01
EPOCHS = 30
SAMPLES_PER_EPOCH = 200
BATCH_SIZE = 16
LR = 1e-3
W_MB = 0.45
W_SH = 0.55


# ---------------- Datasets ----------------
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


# ---------------- Load DINOv2 ----------------
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
dinov2 = dinov2.to(device).eval()
for p in dinov2.parameters():
    p.requires_grad = False


# Multi-layer feature extractor (used by SegHead branch)
@torch.no_grad()
def extract_multilayer_patches(model, x, layers=LAYERS_TO_USE):
    intermediates = model.get_intermediate_layers(
        x, n=layers, reshape=False, return_class_token=False, norm=True
    )
    return torch.cat(intermediates, dim=2)  # (B, 256, 384*len(layers))


# ---------------- Memory bank (single-layer, last-layer features) ----------------
@torch.no_grad()
def extract_patch_features(model, paths, batch_size=32):
    loader = DataLoader(ImageFolderDataset(paths, preprocess),
                        batch_size=batch_size, num_workers=2, pin_memory=True)
    feats = []
    for batch in loader:
        out = model.forward_features(batch.to(device, non_blocking=True))
        feats.append(out['x_norm_patchtokens'].reshape(-1, FEATURE_DIM).cpu())
    return torch.cat(feats, dim=0)

def coreset_subsample(features, ratio=CORESET_RATIO, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(features.shape[0], size=max(1, int(features.shape[0] * ratio)), replace=False)
    return features[idx]


print("Building memory banks...")
MEMORY_BANKS = {}
for class_name in sorted(os.listdir(DATA_ROOT)):
    good_dir = os.path.join(DATA_ROOT, class_name, 'train', 'good')
    if not os.path.isdir(good_dir):
        continue
    paths = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith('.png')])
    feats = extract_patch_features(dinov2, paths)
    bank = F.normalize(coreset_subsample(feats), p=2, dim=1)
    MEMORY_BANKS[class_name] = bank
    print(f"  {class_name}: bank {tuple(bank.shape)}")
    del feats; gc.collect()


# ---------------- Cut-paste synthetic anomaly ----------------
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
    if len(ys) == 0: return crop_img, crop_mask
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
    synth[py:py+nh, px:px+nw] = (region * (1 - alpha[:,:,None]) +
                                  crop_img * alpha[:,:,None]).astype(np.uint8)
    out_mask[py:py+nh, px:px+nw] = alpha
    return synth, out_mask


# ---------------- Anomaly source collection ----------------
ANOMALY_SOURCES = defaultdict(list)
for class_name in sorted(os.listdir(DATA_ROOT)):
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


# ---------------- SegHead model (multi-layer input, in_dim=1152) ----------------
class SegHead(nn.Module):
    def __init__(self, in_dim=MULTILAYER_DIM, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, 1, 1)
    def forward(self, p):
        B = p.size(0)
        x = p.transpose(1, 2).reshape(B, -1, PATCH_GRID, PATCH_GRID)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)


class TrainSynthDataset(Dataset):
    def __init__(self, good_paths, sources, n, tf):
        self.good_paths, self.sources, self.n, self.tf = good_paths, sources, n, tf
    def __len__(self): return self.n
    def __getitem__(self, i):
        good_img = np.array(Image.open(random.choice(self.good_paths)).convert('RGB'))
        if random.random() < 0.5 and self.sources:
            src = random.choice(self.sources)
            synth, mask = cut_paste(good_img, src['image'], src['mask'])
            return self.tf(Image.fromarray(synth)), torch.from_numpy(mask).float()
        return self.tf(Image.fromarray(good_img)), torch.zeros(IMG_SIZE, IMG_SIZE)


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1.0) / (union + 1.0)
    return bce + dice.mean()


# ---------------- Train SegHead per class (multi-layer features) ----------------
print("\nTraining multi-layer SegHeads...")
SEG_HEADS = {}
for class_name in sorted(ANOMALY_SOURCES.keys()):
    good_dir = os.path.join(DATA_ROOT, class_name, 'train', 'good')
    good_paths = [os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith('.png')]
    sources = ANOMALY_SOURCES[class_name]
    
    loader = DataLoader(TrainSynthDataset(good_paths, sources, SAMPLES_PER_EPOCH, preprocess),
                        batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    head = SegHead().to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    
    for epoch in range(EPOCHS):
        head.train()
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).unsqueeze(1)
            with torch.no_grad():
                patches = extract_multilayer_patches(dinov2, imgs)  # (B, 256, 1152)
            loss = bce_dice_loss(head(patches), masks)
            optim.zero_grad(); loss.backward(); optim.step()
    SEG_HEADS[class_name] = head
    print(f"  {class_name} done")
    torch.save(head.state_dict(), f'/kaggle/working/seghead_v7_{class_name}.pt')


# ---------------- Inference ----------------
@torch.no_grad()
def memorybank_infer(paths, bank, batch_size=32):
    loader = DataLoader(TestDataset(paths, preprocess),
                        batch_size=batch_size, num_workers=2, pin_memory=True)
    bank_gpu = bank.to(device)
    scores, fns = [], []
    for imgs, names in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = dinov2.forward_features(imgs)
        patches = F.normalize(out['x_norm_patchtokens'], p=2, dim=2)
        max_sim, _ = (patches @ bank_gpu.T).max(dim=2)
        dist = (1.0 - max_sim).reshape(-1, PATCH_GRID, PATCH_GRID)
        s = F.interpolate(dist.unsqueeze(1), size=(IMG_SIZE, IMG_SIZE),
                          mode='bilinear', align_corners=False).squeeze(1)
        scores.append(s.cpu().numpy()); fns.extend(names)
    return np.concatenate(scores, axis=0), fns

@torch.no_grad()
def seghead_infer(paths, head, batch_size=32):
    loader = DataLoader(TestDataset(paths, preprocess),
                        batch_size=batch_size, num_workers=2, pin_memory=True)
    head.eval()
    scores, fns = [], []
    for imgs, names in loader:
        imgs = imgs.to(device, non_blocking=True)
        patches = extract_multilayer_patches(dinov2, imgs)  # multi-layer
        s = torch.sigmoid(head(patches)).squeeze(1).cpu().numpy()
        scores.append(s); fns.extend(names)
    return np.concatenate(scores, axis=0), fns


print("\nInference...")
MB_TRAIN, MB_TEST, SH_TRAIN, SH_TEST = {}, {}, {}, {}
for class_name in sorted(SEG_HEADS.keys()):
    good_dir = os.path.join(DATA_ROOT, class_name, 'train', 'good')
    good = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith('.png')])
    test_dir = os.path.join(DATA_ROOT, class_name, 'test')
    test = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')])
    
    mb_tr, _ = memorybank_infer(good, MEMORY_BANKS[class_name])
    mb_te, mfns = memorybank_infer(test, MEMORY_BANKS[class_name])
    sh_tr, _ = seghead_infer(good, SEG_HEADS[class_name])
    sh_te, sfns = seghead_infer(test, SEG_HEADS[class_name])
    
    MB_TRAIN[class_name] = mb_tr
    MB_TEST[class_name] = {fn: s for fn, s in zip(mfns, mb_te)}
    SH_TRAIN[class_name] = sh_tr
    SH_TEST[class_name] = {fn: s for fn, s in zip(sfns, sh_te)}
    print(f"  {class_name} done")

torch.cuda.empty_cache()


# ---------------- Train-statistics normalize + ensemble + encode ----------------
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


print("\nEncoding submission...")
rows = []
for class_name in sorted(MB_TEST.keys()):
    mb_lo, mb_hi = np.percentile(MB_TRAIN[class_name].flatten(), [0.5, 99.9])
    sh_lo, sh_hi = np.percentile(SH_TRAIN[class_name].flatten(), [0.5, 99.9])
    for fn in sorted(MB_TEST[class_name].keys()):
        n_mb = np.clip((MB_TEST[class_name][fn] - mb_lo) / (mb_hi - mb_lo + 1e-8), 0, 1)
        n_sh = np.clip((SH_TEST[class_name][fn] - sh_lo) / (sh_hi - sh_lo + 1e-8), 0, 1)
        ens = np.clip(W_MB * n_mb + W_SH * n_sh, 0, 1).astype(np.float32)
        rows.append({'ID': fn[:-4], 'Label': float_matrix_to_q8rle(ens)})

df = pd.DataFrame(rows)
df.to_csv('/kaggle/working/submission.csv', index=False)
with zipfile.ZipFile('/kaggle/working/submission.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write('/kaggle/working/submission.csv', arcname='submission.csv')
print(f"Done. submission.zip size: {os.path.getsize('/kaggle/working/submission.zip')/1024/1024:.2f} MB")


# ---------------- Submit (optional) ----------------
kaggle_json = Path('/root/.kaggle/kaggle.json')
if not kaggle_json.exists():
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    kaggle_json.parent.mkdir(parents=True, exist_ok=True)
    with open(kaggle_json, 'w') as f:
        json.dump({"username": secrets.get_secret("KAGGLE_USERNAME"),
                   "key": secrets.get_secret("KAGGLE_KEY")}, f)
    os.chmod(kaggle_json, 0o600)

!kaggle competitions submit -c adl-2025-2026-anomaly-detection \
    -f /kaggle/working/submission.zip -m "v7 reproduction"

