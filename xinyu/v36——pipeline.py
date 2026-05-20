# ================================================================
# v36: 3-fold SegHead ensemble + 4-fold TTA
#
# Combines two proven improvements:
#   v34 (3-fold ensemble): 0.8621
#   v33-tta4 (4-fold TTA): 0.8639
# Both work independently → stack them.
#
# DINOv2-S-reg @ 518, SegHead-only, early stopping patience=5,
# 4-fold TTA (orig + h-flip + v-flip + h+v-flip),
# 3-fold ensemble (average predictions from 3 SegHeads per class)
#
# Runtime: ~3-4 hours
# ================================================================
import os, time, copy, gc, random, re, json, zipfile
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

DATA_ROOT = Path('/kaggle/input/datasets/cindy11102858/adl-anomaly-mirror/adl-2025-2026-anomaly-detection')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 518
PATCH_GRID = 37
OUTPUT_SIZE = 224
FEATURE_DIM = 384
LAYERS_TO_USE = [5, 8, 11]
MULTILAYER_DIM = FEATURE_DIM * len(LAYERS_TO_USE)

EPOCHS = 50
PATIENCE = 5
SAMPLES_PER_EPOCH = 200
BATCH_SIZE = 8
LR = 1e-3
P_GOOD = 0.30
P_REAL = 0.40

BLUR_SIGMA = 4
P_LO = 0.5
P_HI = 99.999
TTA_FLIPS = [[], [-1], [-2], [-1, -2]]  # 4-fold TTA

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TestDataset(Dataset):
    def __init__(self, paths, tf): self.paths, self.tf = paths, tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB')), os.path.basename(self.paths[i])

def _resize_mask(mask, size=OUTPUT_SIZE):
    if mask.shape == (size, size):
        return (mask > 0).astype(np.float32)
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    pil = pil.resize((size, size), Image.NEAREST)
    return (np.array(pil) > 127).astype(np.float32)

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
    synth[py:py+nh, px:px+nw] = (region * (1 - alpha[:, :, None]) +
                                 crop_img * alpha[:, :, None]).astype(np.uint8)
    out_mask[py:py+nh, px:px+nw] = alpha
    return synth, out_mask

class TrainSynthDataset(Dataset):
    def __init__(self, good_paths, sources, n, tf, p_good=P_GOOD, p_real=P_REAL):
        self.good_paths, self.sources, self.n, self.tf = good_paths, sources, n, tf
        self.p_good, self.p_real = p_good, p_real
    def __len__(self): return self.n
    def __getitem__(self, i):
        r = random.random()
        if r < self.p_good or not self.sources:
            img = np.array(Image.open(random.choice(self.good_paths)).convert("RGB"))
            return self.tf(Image.fromarray(img)), torch.zeros(OUTPUT_SIZE, OUTPUT_SIZE)
        if r < self.p_good + self.p_real:
            src = random.choice(self.sources)
            return self.tf(Image.fromarray(src["image"])), torch.from_numpy(_resize_mask(src["mask"])).float()
        good_img = np.array(Image.open(random.choice(self.good_paths)).convert("RGB"))
        src = random.choice(self.sources)
        synth, mask = cut_paste(good_img, src["image"], src["mask"])
        return self.tf(Image.fromarray(synth)), torch.from_numpy(mask).float()

class SegHead(nn.Module):
    def __init__(self, in_dim=MULTILAYER_DIM, hidden=128, patch_grid=PATCH_GRID, out_size=OUTPUT_SIZE):
        super().__init__()
        self.patch_grid, self.out_size = patch_grid, out_size
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
        return F.interpolate(x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)

def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1.0) / (union + 1.0)
    return bce + dice.mean()

def collect_anomaly_kfold(data_root, n_folds=3):
    all_pairs = defaultdict(list)
    for class_dir in sorted(data_root.iterdir()):
        train_dir = class_dir / "train"
        gt_dir = class_dir / "ground_truth_train"
        if not (train_dir.is_dir() and gt_dir.is_dir()): continue
        class_name = class_dir.name
        for ano_gt_dir in sorted(gt_dir.iterdir()):
            ano_img_dir = train_dir / ano_gt_dir.name
            if not ano_gt_dir.is_dir(): continue
            for mask_path in sorted(ano_gt_dir.glob("*.png")):
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask.max() == 0: continue
                all_pairs[class_name].append((ano_img_dir / mask_path.name, mask))
    folds = []
    for fold_idx in range(n_folds):
        ts, vi = defaultdict(list), defaultdict(list)
        for cn, pairs in all_pairs.items():
            for i, (ip, m) in enumerate(pairs):
                if i % n_folds == fold_idx:
                    vi[cn].append({"path": str(ip), "mask": m})
                else:
                    ts[cn].append({"image": np.array(Image.open(ip).convert("RGB")), "mask": m})
        folds.append((ts, vi))
    return folds

@torch.no_grad()
def extract_multilayer_patches(model, x, layers=LAYERS_TO_USE):
    intermediates = model.get_intermediate_layers(x, n=layers, reshape=False, return_class_token=False, norm=True)
    return torch.cat(intermediates, dim=2)

@torch.no_grad()
def _val_pixel_ap(model, head, val_items):
    head.eval()
    all_preds, all_gt = [], []
    for item in val_items:
        img = preprocess(Image.open(item["path"]).convert("RGB")).unsqueeze(0).to(device)
        patches = extract_multilayer_patches(model, img)
        score = torch.sigmoid(head(patches)).squeeze().cpu().numpy()
        mask = (item["mask"] > 0).astype(np.uint8)
        if mask.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
            mask = (np.array(Image.fromarray(mask * 255).resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST)) > 127).astype(np.uint8)
        all_preds.append(score.flatten())
        all_gt.append(mask.flatten())
    gt = np.concatenate(all_gt)
    if gt.max() == 0: return 0.0
    return float(average_precision_score(gt, np.concatenate(all_preds)))

def fn_to_id(fn): return Path(fn).stem

def float_matrix_to_q8rle(x):
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
    h, w = q.shape
    flat = q.T.reshape(-1)
    if flat.size == 0: return f"q8rle {h} {w}"
    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, flat.size]
    parts = ["q8rle", str(h), str(w)]
    for v, n in zip(flat[starts], ends - starts):
        parts += [str(int(v)), str(int(n))]
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════

# [1/5] Load DINOv2-S-reg
print("=== [1/5] Load DINOv2-S-reg ===")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', verbose=False)
dinov2 = dinov2.to(device).eval()
for p in dinov2.parameters(): p.requires_grad = False
print(f"  Params: {sum(p.numel() for p in dinov2.parameters()):,}")

# [2/5] Build 3-fold splits
print("\n=== [2/5] Build 3-fold splits ===")
FOLDS = collect_anomaly_kfold(DATA_ROOT, n_folds=3)
for fi, (ts, vi) in enumerate(FOLDS):
    print(f"  Fold {fi}: train={sum(len(v) for v in ts.values())}, val={sum(len(v) for v in vi.values())}")

# [3/5] Train 3-fold SegHeads
print(f"\n=== [3/5] Train 3-fold SegHeads (patience={PATIENCE}) ===")
ALL_HEADS = []
for fold_idx, (fold_train, fold_val) in enumerate(FOLDS):
    print(f"\n--- Fold {fold_idx} ---")
    fold_heads = {}
    for cls_idx, class_name in enumerate(sorted(fold_train.keys())):
        good_dir = DATA_ROOT / class_name / "train" / "good"
        good_paths = [str(p) for p in sorted(good_dir.glob("*.png"))]
        cls_seed = SEED + fold_idx * 100 + cls_idx
        g = torch.Generator(); g.manual_seed(cls_seed)
        loader = DataLoader(
            TrainSynthDataset(good_paths, fold_train[class_name], SAMPLES_PER_EPOCH, preprocess),
            batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, generator=g, worker_init_fn=worker_init_fn)
        head = SegHead().to(device)
        optim = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
        class_val = fold_val.get(class_name, [])
        best_ap, best_state, no_improve = -1.0, None, 0
        t0 = time.time()
        for epoch in range(EPOCHS):
            head.train()
            for imgs, masks in loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).unsqueeze(1)
                with torch.no_grad(): patches = extract_multilayer_patches(dinov2, imgs)
                loss = bce_dice_loss(head(patches), masks)
                optim.zero_grad(); loss.backward(); optim.step()
            if class_val:
                val_ap = _val_pixel_ap(dinov2, head, class_val)
                head.train()
                if val_ap > best_ap:
                    best_ap = val_ap
                    best_state = copy.deepcopy(head.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
            if no_improve >= PATIENCE: break
        if best_state is not None: head.load_state_dict(best_state)
        head.eval()
        fold_heads[class_name] = head
        ap_msg = f" val_ap={best_ap:.4f}" if class_val else ""
        print(f"  {class_name} fold{fold_idx} stop@{epoch+1} {time.time()-t0:.0f}s{ap_msg}")
        torch.cuda.empty_cache()
    ALL_HEADS.append(fold_heads)

# [4/5] Inference: 3-fold × 4-fold TTA
print("\n=== [4/5] Inference (3-fold × 4-fold TTA) ===")
SH_TRAIN_36, SH_TEST_36 = {}, {}
for class_name in sorted(ALL_HEADS[0].keys()):
    good_dir = DATA_ROOT / class_name / "train" / "good"
    good = [str(p) for p in sorted(good_dir.glob("*.png"))]
    test_dir = DATA_ROOT / class_name / "test"
    test = [str(p) for p in sorted(test_dir.glob("*.png"))]

    train_preds, test_preds, test_fns = [], [], None
    for fold_heads in ALL_HEADS:
        head = fold_heads[class_name]; head.eval()
        # Train good (4-fold TTA)
        loader = DataLoader(TestDataset(good, preprocess), batch_size=8, num_workers=2, pin_memory=True)
        tr_sc = []
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            aug = []
            for fd in TTA_FLIPS:
                x = torch.flip(imgs, fd) if fd else imgs
                with torch.no_grad():
                    s = torch.sigmoid(head(extract_multilayer_patches(dinov2, x))).squeeze(1)
                if fd: s = torch.flip(s, fd)
                aug.append(s)
            tr_sc.append(torch.stack(aug).mean(0).cpu().numpy())
        train_preds.append(np.concatenate(tr_sc, axis=0))

        # Test (4-fold TTA)
        loader = DataLoader(TestDataset(test, preprocess), batch_size=8, num_workers=2, pin_memory=True)
        te_sc, fns = [], []
        for imgs, names in loader:
            imgs = imgs.to(device, non_blocking=True)
            aug = []
            for fd in TTA_FLIPS:
                x = torch.flip(imgs, fd) if fd else imgs
                with torch.no_grad():
                    s = torch.sigmoid(head(extract_multilayer_patches(dinov2, x))).squeeze(1)
                if fd: s = torch.flip(s, fd)
                aug.append(s)
            te_sc.append(torch.stack(aug).mean(0).cpu().numpy())
            fns.extend(names)
        test_preds.append(np.concatenate(te_sc, axis=0))
        if test_fns is None: test_fns = fns

    SH_TRAIN_36[class_name] = np.mean(train_preds, axis=0)
    SH_TEST_36[class_name] = dict(zip(test_fns, np.mean(test_preds, axis=0)))
    print(f"  {class_name} done ({len(test)} test)")
torch.cuda.empty_cache()

# [5/5] Normalize + submit
print("\n=== [5/5] Submit ===")
rows = []
for class_name in sorted(SH_TEST_36.keys()):
    sh_lo, sh_hi = np.percentile(SH_TRAIN_36[class_name].flatten(), [P_LO, P_HI])
    for fn in sorted(SH_TEST_36[class_name].keys()):
        score = np.clip((SH_TEST_36[class_name][fn] - sh_lo) / (sh_hi - sh_lo + 1e-8), 0, 1)
        score = gaussian_filter(score, sigma=BLUR_SIGMA).astype(np.float32)
        rows.append({'ID': fn_to_id(fn), 'Label': float_matrix_to_q8rle(score)})

df = pd.DataFrame(rows)
csv = '/kaggle/working/submission_v36.csv'
zip_path = '/kaggle/working/submission_v36.zip'
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
    '-f /kaggle/working/submission_v36.zip '
    '-m "v36: 3-fold ensemble + 4-fold TTA on DINOv2-S-reg@518"'
)
print("\nsubmitted")