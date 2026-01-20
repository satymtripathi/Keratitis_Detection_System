"""
SAFE + FAST Dual-Branch Training from PRECOMPUTED CACHE (Windows stable)

Key safety changes:
- NO PIL transforms (uses cv2 + torch tensors)
- NUM_WORKERS default 0 (safe on Windows). Increase carefully to 2.
- Smaller batch + grad accumulation to avoid GPU OOM / driver reset
- cv2.setNumThreads(0) to stop CPU thread explosion
- Mixed precision enabled on CUDA
- Resume from last checkpoint

Run:
python 02_train_from_precomputed_cache_SAFE.py
"""

import os, glob, json, random, time, math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import models
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ---- CRITICAL WINDOWS STABILITY ----
cv2.setNumThreads(0)          # avoids CPU thread explosion
cv2.ocl.setUseOpenCL(False)   # avoid OpenCL weirdness on some drivers


# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    CACHE_ROOT: str = r"DatasetSatyam_Precomputed"
    OUT_DIR: str = r"train_precomputed_run_SAFE_V2"

    # 3-class / 4-class
    USE_3CLASS_MERGE: bool = False

    # sizes
    GLOBAL_SIZE: int = 384
    TILE_SIZE: int = 224
    MAX_TILES: int = 24

    # training (SAFE defaults)
    EPOCHS: int = 20
    BATCH_SIZE: int = 2              # SAFE
    GRAD_ACCUM: int = 4              # effective batch = 8 but GPU sees 2
    BASE_LR: float = 2e-5
    WEIGHT_DECAY: float = 1e-4
    SEED: int = 42

    # MIL
    TOPK_POOL: int = 4
    QUALITY_BETA: float = 0.7
    TILE_DROPOUT: float = 0.10

    # loss
    FOCAL_GAMMA: float = 2.0
    LABEL_SMOOTH: float = 0.05

    # entropy warmup
    LAMBDA_ATT_ENT_START: float = 0.03
    LAMBDA_ATT_ENT_AFTER: float = 0.01
    ENT_WARM_EPOCHS: int = 5

    # weights
    W4: Tuple[float, float, float, float] = (1.0, 1.0, 2.2, 1.5)
    W3: Tuple[float, float, float] = (1.0, 1.4, 2.2)

    # DataLoader (SAFE on Windows)
    NUM_WORKERS: int = 4           # SAFE. try 2 later if stable
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = False

    # speed / stability
    CUDNN_BENCHMARK: bool = True
    LIMIT_TILES_READ: int = 24      # hard limit from disk

    # debug
    DEBUG_SAMPLES_PER_EPOCH: int = 10
    SAVE_DEBUG_IMAGES: bool = True
    DEBUG_IMAGE_EPOCH_EVERY: int = 5
    DEBUG_IMAGE_MAX_EPOCHS: int = 2
    DEBUG_IMAGE_SAMPLES_PER_EPOCH: int = 5
    DEBUG_TOPK_TILES_TO_SAVE: int = 6

    # quick sanity
    QUICK_CHECK: bool = False  # True for sanity check 
    QUICK_EPOCHS: int = 2
    QUICK_TRAIN_SAMPLES: int = 20
    QUICK_VAL_SAMPLES: int = 10

    # resume
    RESUME: bool = True

cfg = CFG()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda" and cfg.CUDNN_BENCHMARK:
    torch.backends.cudnn.benchmark = True


# =========================
# REPRO
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# LABEL MAP
# =========================
ORIG_4 = ["Edema", "Scar", "Infection", "Normal"]
if cfg.USE_3CLASS_MERGE:
    CLASSES = ["NonInfect_Other", "Normal", "Infection"]
    def map_label(lbl: str) -> str:
        return "NonInfect_Other" if lbl in ["Edema", "Scar"] else lbl
else:
    CLASSES = ORIG_4
    def map_label(lbl: str) -> str:
        return lbl

CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}


# =========================
# FAST tensor preprocess (NO PIL)
# =========================
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def _read_rgb(path: str) -> Optional[np.ndarray]:
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _read_bgr(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path)

def resize_rgb(rgb: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(rgb, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)

def to_tensor_norm(rgb_uint8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float() / 255.0
    t = (t - MEAN) / STD
    return t

def maybe_aug(rgb: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        rgb = cv2.flip(rgb, 1)
    return rgb


# =========================
# INDEX BUILD
# =========================
def find_global_files(cache_root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(cache_root, "**", "*_global.jpg"), recursive=True))

def parse_sample_from_global(global_path: str) -> Optional[dict]:
    try:
        cls = os.path.basename(os.path.dirname(global_path))
        base = global_path[:-len("_global.jpg")]
        meta_path = base + "_meta.json"

        tiles, quality = [], []
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for m in meta:
                tile_file = m.get("file", "")
                if not tile_file:
                    continue
                tile_path = os.path.join(os.path.dirname(global_path), tile_file)
                if os.path.exists(tile_path):
                    tiles.append(tile_path)
                    quality.append(float(m.get("quality", 0.0)))
        else:
            tiles = sorted(glob.glob(base + "_tile_*.jpg"))
            quality = [0.0] * len(tiles)

        return {
            "label": cls,
            "global_path": global_path,
            "tiles": tiles[:cfg.LIMIT_TILES_READ],
            "quality": np.array(quality[:cfg.LIMIT_TILES_READ], dtype=np.float32),
            "sample_id": os.path.basename(base),
        }
    except Exception:
        return None

def build_index(cache_root: str) -> List[dict]:
    globals_ = find_global_files(cache_root)
    out = []
    for gp in tqdm(globals_, desc="Indexing cache"):
        s = parse_sample_from_global(gp)
        if s is not None:
            out.append(s)
    return out


# =========================
# DATASET
# =========================
class PrecomputedFolderDataset(Dataset):
    def __init__(self, samples: List[dict], train: bool):
        self.samples = samples
        self.train = train

        # pre-make black padding tile (already resized)
        self.black_tile = np.zeros((cfg.TILE_SIZE, cfg.TILE_SIZE, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            label_str = map_label(str(s["label"]))
            y = CLASS_TO_ID[label_str]

            g_rgb = _read_rgb(s["global_path"])
            if g_rgb is None:
                return None

            if self.train:
                g_rgb = maybe_aug(g_rgb)

            g_rgb = resize_rgb(g_rgb, (cfg.GLOBAL_SIZE, cfg.GLOBAL_SIZE))
            g_t = to_tensor_norm(g_rgb)

            tile_paths = list(s["tiles"])
            q = np.array(s["quality"], dtype=np.float32)

            # normalize quality 0..1
            if q.size > 0:
                mn, mx = float(q.min()), float(q.max())
                q01 = (q - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(q)
            else:
                q01 = np.zeros((len(tile_paths),), dtype=np.float32)

            # tile dropout
            idxs = np.arange(len(tile_paths))
            if self.train and cfg.TILE_DROPOUT > 0 and len(idxs) > 6:
                keep_n = max(6, int(len(idxs) * (1 - cfg.TILE_DROPOUT)))
                idxs = np.random.choice(idxs, size=keep_n, replace=False)
                tile_paths = [tile_paths[i] for i in idxs]
                q01 = q01[idxs]

            # load tiles (up to MAX_TILES)
            tile_rgbs, loaded_paths = [], []
            for tp in tile_paths[:cfg.MAX_TILES]:
                trgb = _read_rgb(tp)
                if trgb is None:
                    continue
                trgb = resize_rgb(trgb, (cfg.TILE_SIZE, cfg.TILE_SIZE))
                tile_rgbs.append(trgb)
                loaded_paths.append(tp)

            # match q to loaded
            q01 = q01[:len(loaded_paths)] if len(loaded_paths) > 0 else np.zeros((0,), dtype=np.float32)

            # pad to MAX_TILES
            T = cfg.MAX_TILES
            if len(tile_rgbs) < T:
                pad_n = T - len(tile_rgbs)
                tile_rgbs += [self.black_tile] * pad_n
                q01 = np.concatenate([q01, np.zeros((pad_n,), dtype=np.float32)], axis=0)
                loaded_paths += [None] * pad_n
            else:
                tile_rgbs = tile_rgbs[:T]
                q01 = q01[:T]
                loaded_paths = loaded_paths[:T]

            tiles_t = torch.stack([to_tensor_norm(t) for t in tile_rgbs])  # (T,3,224,224)
            q_t = torch.from_numpy(q01.astype(np.float32))                 # (T,)


            return g_t, tiles_t, q_t, y, s["sample_id"], s["global_path"], loaded_paths

        except Exception:
            return None


def collate_precomputed(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    g, tiles, q, y, sid, gpath, tpaths = zip(*batch)
    return (
        torch.stack(g),                 # (B,3,G,G)
        torch.stack(tiles),             # (B,T,3,224,224)
        torch.stack(q),                 # (B,T)
        torch.tensor(y, dtype=torch.long),
        sid, gpath, tpaths
    )


# =========================
# MODEL
# =========================
class GatedAttention(nn.Module):
    def __init__(self, L, D, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights = nn.Linear(D, K)

    def forward(self, x):
        return self.attention_weights(self.attention_V(x) * self.attention_U(x))

class DualBranchMIL(nn.Module):
    def __init__(self, num_classes: int, topk_pool: int, quality_beta: float):
        super().__init__()
        self.topk_pool = topk_pool
        self.quality_beta = quality_beta

        tile_base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.tile_feat = tile_base.features
        self.tile_pool = nn.AdaptiveAvgPool2d(1)

        self.tile_latent = 256
        self.tile_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, self.tile_latent),
            nn.LayerNorm(self.tile_latent),
            nn.ReLU(),
        )
        self.attention = GatedAttention(self.tile_latent, 128, 1)

        global_base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.global_feat = global_base.features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.global_latent = 256
        self.global_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, self.global_latent),
            nn.LayerNorm(self.global_latent),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.tile_latent + self.global_latent, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, global_img, tiles, qnorm=None):
        """
        global_img: (B,3,G,G)
        tiles: (B,T,3,224,224)
        qnorm: (B,T)
        """
        B, T, C, H, W = tiles.shape
        tiles_ = tiles.view(B*T, C, H, W)

        t = self.tile_pool(self.tile_feat(tiles_))   # (B*T,1280,1,1)
        z = self.tile_projector(t).view(B, T, -1)    # (B,T,256)

        a_logits = self.attention(z).squeeze(-1)     # (B,T)
        if qnorm is not None:
            a_logits = a_logits + self.quality_beta * qnorm.to(a_logits.device)

        a = torch.softmax(a_logits, dim=1)           # (B,T)

        k = min(self.topk_pool, T)
        top_idx = torch.topk(a, k=k, dim=1).indices  # (B,k)

        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, z.size(-1))
        z_top = torch.gather(z, dim=1, index=idx_exp)  # (B,k,256)
        bag = z_top.mean(dim=1)                         # (B,256)

        g = self.global_pool(self.global_feat(global_img))
        g = self.global_projector(g)                    # (B,256)

        fused = torch.cat([bag, g], dim=1)
        logits = self.classifier(fused)
        return logits, a, top_idx


# =========================
# LOSS
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smooth=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smooth = label_smooth

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target,
            reduction="none",
            weight=self.alpha,
            label_smoothing=self.label_smooth
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

def attention_entropy_batch(a, eps=1e-8):
    p = a.clamp(min=eps)
    return (-(p * torch.log(p)).sum(dim=1)).mean()


# =========================
# CHECKPOINTING
# =========================
def save_ckpt(path: str, epoch: int, model, optimizer, scaler, best_acc: float):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_acc": best_acc,
        "cfg": cfg.__dict__
    }, path)

def load_ckpt(path: str, model, optimizer, scaler):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["epoch"]) + 1, float(ckpt.get("best_acc", 0.0))


# =========================
# MAIN
# =========================
def main():
    seed_everything(cfg.SEED)

    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.OUT_DIR, "checkpoints"), exist_ok=True)

    # index once
    samples = build_index(cfg.CACHE_ROOT)
    if len(samples) == 0:
        raise RuntimeError(f"No globals in {cfg.CACHE_ROOT}")

    # stratify labels
    labels = np.array([CLASS_TO_ID[map_label(str(s["label"]))] for s in samples], dtype=np.int64)

    tr_samples, va_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=cfg.SEED,
        stratify=labels
    )

    if cfg.QUICK_CHECK:
        tr_samples = tr_samples[:cfg.QUICK_TRAIN_SAMPLES]
        va_samples = va_samples[:cfg.QUICK_VAL_SAMPLES]
        epochs = cfg.QUICK_EPOCHS
        print(f"[QUICK] train={len(tr_samples)} val={len(va_samples)} epochs={epochs}")
    else:
        epochs = cfg.EPOCHS
        print(f"train={len(tr_samples)} val={len(va_samples)} epochs={epochs}")

    tr_y = np.array([CLASS_TO_ID[map_label(str(s["label"]))] for s in tr_samples], dtype=np.int64)
    counts = {i: max(1, int((tr_y == i).sum())) for i in range(len(CLASSES))}
    sample_w = np.array([1.0 / counts[int(y)] for y in tr_y], dtype=np.float32)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_ds = PrecomputedFolderDataset(tr_samples, train=True)
    val_ds = PrecomputedFolderDataset(va_samples, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0),
        collate_fn=collate_precomputed
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0),
        collate_fn=collate_precomputed
    )

    model = DualBranchMIL(len(CLASSES), cfg.TOPK_POOL, cfg.QUALITY_BETA).to(DEVICE)

    alpha = torch.tensor(cfg.W3 if cfg.USE_3CLASS_MERGE else cfg.W4, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(cfg.FOCAL_GAMMA, alpha=alpha, label_smooth=cfg.LABEL_SMOOTH)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    ckpt_dir = os.path.join(cfg.OUT_DIR, "checkpoints")
    last_ckpt = os.path.join(ckpt_dir, "last.pth")
    best_ckpt = os.path.join(ckpt_dir, "best.pth")

    start_epoch = 1
    best_acc = 0.0
    if cfg.RESUME and os.path.exists(last_ckpt):
        try:
            start_epoch, best_acc = load_ckpt(last_ckpt, model, optimizer, scaler)
            print(f"[RESUME] start_epoch={start_epoch} best_acc={best_acc:.4f}")
        except Exception as e:
            print("[RESUME] failed:", e)

    with open(os.path.join(cfg.OUT_DIR, "run_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    for epoch in range(start_epoch, epochs + 1):
        lambda_ent = cfg.LAMBDA_ATT_ENT_START if epoch <= cfg.ENT_WARM_EPOCHS else cfg.LAMBDA_ATT_ENT_AFTER

        # ---- train ----
        model.train()
        running, steps, none_batches = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for bi, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}/{epochs}")):
            if batch is None:
                none_batches += 1
                continue

            gimgs, tiles_b, q_b, y_b, *_ = batch
            gimgs = gimgs.to(DEVICE, non_blocking=True)
            tiles_b = tiles_b.to(DEVICE, non_blocking=True)
            q_b = q_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)

            with autocast(enabled=(DEVICE == "cuda")):
                logits, att, _ = model(gimgs, tiles_b, qnorm=q_b)
                cls_loss = criterion(logits, y_b)
                ent = attention_entropy_batch(att)
                loss = (cls_loss - lambda_ent * ent) / cfg.GRAD_ACCUM

            scaler.scale(loss).backward()

            if ((bi + 1) % cfg.GRAD_ACCUM == 0):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.item()) * cfg.GRAD_ACCUM
            steps += 1

        # flush last partial accumulation
        if (steps % cfg.GRAD_ACCUM) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running / max(1, steps)
        print(f"Train Loss: {train_loss:.4f} | None: {none_batches}")

        # ---- val ----
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch}/{epochs}"):
                if batch is None:
                    continue
                gimgs, tiles_b, q_b, y_b, *_ = batch
                gimgs = gimgs.to(DEVICE, non_blocking=True)
                tiles_b = tiles_b.to(DEVICE, non_blocking=True)
                q_b = q_b.to(DEVICE, non_blocking=True)

                with autocast(enabled=(DEVICE == "cuda")):
                    logits, _, _ = model(gimgs, tiles_b, qnorm=q_b)

                preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                ytrue = y_b.cpu().numpy().tolist()
                all_true.extend(ytrue)
                all_pred.extend(preds)

        all_true = np.array(all_true, dtype=np.int64)
        all_pred = np.array(all_pred, dtype=np.int64)

        val_acc = float((all_true == all_pred).mean())
        cm = confusion_matrix(all_true, all_pred)

        print(f"\nEpoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print(classification_report(all_true, all_pred, target_names=CLASSES, digits=4, zero_division=0))

        with open(os.path.join(cfg.OUT_DIR, "metrics_log.txt"), "a") as f:
            f.write(f"epoch={epoch} train_loss={train_loss:.6f} val_acc={val_acc:.6f}\n")

        save_ckpt(last_ckpt, epoch, model, optimizer, scaler, best_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt(best_ckpt, epoch, model, optimizer, scaler, best_acc)
            print(f"[BEST] saved best.pth | best_acc={best_acc:.4f}")

    print("\nDone. Best Acc:", best_acc)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
