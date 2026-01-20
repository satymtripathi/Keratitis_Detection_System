"""
infer_and_sort_from_precompute_logic.py
Matches YOUR precompute logic 1:1 (512 canonical, POLAR_MIN_PIXELS=150, PAD=2, tiles resized to 224).

Outputs:
- OUT_DIR/<PredClass>/<image>
- OUT_DIR/predictions_sorted.csv

Run:
python infer_and_sort_from_precompute_logic.py
"""

import os, glob, math, shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch import amp

from Limbus_Crop_Segmentation_System.inference_utils import load_model, predict_masks


# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    INPUT_DIR: str = r"input_images" # Folder with images for inference
    OUT_DIR: str   = r"inference_output" # Where sorted images will be saved

    SEG_CKPT: str  = r"Limbus_Crop_Segmentation_System\model_limbus_crop_unetpp_weighted.pth"
    CLS_CKPT: str  = r"training_results\checkpoints\best.pth"  # checkpoint payload with ["model"] or plain state_dict

    CLASSES: Tuple[str, ...] = ("Edema", "Scar", "Infection", "Normal")

    # MUST match precompute
    CANONICAL_SIZE: int = 512
    TILE_SAVE_SIZE: int = 224
    GLOBAL_SIZE: int = 384            # must match training global_tf size
    TILE_SIZE: int = 224             # must match training tile size

    POLAR_THETA: int = 8
    POLAR_RINGS: int = 3
    RING_EDGES_FRAC: Tuple[float, ...] = (0.0, 0.35, 0.70, 1.0)
    POLAR_MIN_PIXELS: int = 150      # EXACT from precompute
    POLAR_PAD: int = 2               # EXACT from precompute

    TOPK_POOL: int = 4
    QUALITY_BETA: float = 0.7

    SAVE_MASKED_GLOBAL: bool = True
    SAVE_MASK_OVERLAY: bool = True

cfg = CFG()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# TRANSFORMS (must match training)
# =========================
tile_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((cfg.TILE_SIZE, cfg.TILE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

global_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((cfg.GLOBAL_SIZE, cfg.GLOBAL_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# =========================
# SEGMENTER (same as precompute – mask on canonical 512)
# =========================
class LimbusSegmenter:
    def __init__(self, checkpoint: str, device: str):
        self.device = device
        self.model, self.idx_crop, self.idx_limbus, self.img_size = load_model(checkpoint, device)
        self.model.eval()
        print(f"[SegModel] idx_crop={self.idx_crop}, idx_limbus={self.idx_limbus}, img_size={self.img_size}")

    @torch.no_grad()
    def mask_on_canonical_512(self, bgr_orig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
        - rgb_512 (512x512 RGB)
        - mask01_512 (512x512 uint8 0/1)
        EXACT idea from precompute: resize image to 512 then predict masks on that tensor size.
        """
        bgr_512 = cv2.resize(bgr_orig, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE), interpolation=cv2.INTER_AREA)
        rgb_512 = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB)

        # predict_masks expects bgr input in original training size (inference_utils handles resize),
        # BUT your precompute did: model(batch_tensors) with A.Resize to 512.
        # Here we mimic that by calling predict_masks on bgr_512 then resizing (safe).
        masks = predict_masks(self.model, bgr_512, self.img_size, self.device)
        limbus = masks[self.idx_limbus]
        if limbus.dtype != np.uint8:
            mask01 = (limbus > 0.5).astype(np.uint8)
        else:
            mask01 = (limbus > 0).astype(np.uint8)

        if mask01.shape[:2] != (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE):
            mask01 = cv2.resize(mask01, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE), interpolation=cv2.INTER_NEAREST)

        return rgb_512, mask01


# =========================
# PRECOMPUTE-MATCHED TILE MAKER
# =========================
def apply_mask_rgb(img_rgb, mask01):
    out = img_rgb.copy()
    out[mask01 == 0] = 0
    return out

def tile_quality_score(tile_rgb: np.ndarray) -> float:
    if tile_rgb is None or tile_rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    valid = gray > 5
    if valid.mean() < 0.05:
        return 0.0
    g = gray[valid]
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lapv = float(lap[valid].var())
    contrast = float(g.std())
    glare_ratio = float((g > 240).mean())
    glare_pen = 1.0 - min(1.0, glare_ratio * 4.0)
    return float(max(0.0, (0.6 * math.log1p(lapv) + 0.4 * math.log1p(contrast)) * glare_pen))

def polar_tiles_precompute_matched(masked_global_512: np.ndarray, mask01_512: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    EXACT match to your process_and_save():
    - compute cx,cy,rmax on mask
    - wedge selection
    - bbox with PAD
    - crop as [y0:y1+1, x0:x1+1]
    - apply wedge mask inside crop
    - resize tile to 224 (TILE_SAVE_SIZE)
    - compute quality on resized tile
    Returns:
      tiles_224: list of (224,224,3) RGB
      q: np.array qualities (same length)
    """
    H, W = mask01_512.shape
    ys, xs = np.where(mask01_512 > 0)
    if len(xs) == 0:
        return [], np.zeros((0,), dtype=np.float32)

    cx, cy = float(xs.mean()), float(ys.mean())
    rmax = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
    if rmax < 5:
        return [], np.zeros((0,), dtype=np.float32)

    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    tt = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)

    ring_edges = [f * rmax for f in cfg.RING_EDGES_FRAC]

    tiles, qualities = [], []
    for r in range(cfg.POLAR_RINGS):
        for s in range(cfg.POLAR_THETA):
            t0 = 2 * np.pi * s / cfg.POLAR_THETA
            t1 = 2 * np.pi * (s + 1) / cfg.POLAR_THETA
            wedge = (mask01_512 > 0) & (rr >= ring_edges[r]) & (rr < ring_edges[r + 1]) & (tt >= t0) & (tt < t1)

            if wedge.sum() < cfg.POLAR_MIN_PIXELS:
                continue

            ys_w, xs_w = np.where(wedge)
            x0, y0, x1, y1 = xs_w.min(), ys_w.min(), xs_w.max(), ys_w.max()
            x0, y0 = max(0, x0 - cfg.POLAR_PAD), max(0, y0 - cfg.POLAR_PAD)
            x1, y1 = min(W - 1, x1 + cfg.POLAR_PAD), min(H - 1, y1 + cfg.POLAR_PAD)

            tile = masked_global_512[y0:y1 + 1, x0:x1 + 1].copy()
            tile_w = wedge[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
            tile[tile_w == 0] = 0

            tile_res = cv2.resize(tile, (cfg.TILE_SAVE_SIZE, cfg.TILE_SAVE_SIZE), interpolation=cv2.INTER_AREA)
            tiles.append(tile_res)
            qualities.append(tile_quality_score(tile_res))

    return tiles, np.array(qualities, dtype=np.float32)

def normalize01(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        return (arr - mn) / (mx - mn + 1e-8)
    return np.zeros_like(arr)


# =========================
# CLASSIFIER MODELS (support BOTH shapes)
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
    """
    Matches your training-from-cache: tiles are (B,T,3,224,224)
    """
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
        B, T, C, H, W = tiles.shape
        tiles_ = tiles.view(B * T, C, H, W)

        t = self.tile_pool(self.tile_feat(tiles_))
        z = self.tile_projector(t).view(B, T, -1)

        a_logits = self.attention(z).squeeze(-1)
        if qnorm is not None:
            a_logits = a_logits + self.quality_beta * qnorm.to(a_logits.device)

        a = torch.softmax(a_logits, dim=1)

        k = min(self.topk_pool, T)
        top_idx = torch.topk(a, k=k, dim=1).indices

        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, z.size(-1))
        z_top = torch.gather(z, dim=1, index=idx_exp)
        bag = z_top.mean(dim=1)

        g = self.global_pool(self.global_feat(global_img))
        g = self.global_projector(g)

        fused = torch.cat([bag, g], dim=1)
        logits = self.classifier(fused)
        return logits, a, top_idx

def load_classifier(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")  # load on CPU first
    sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt

    model = DualBranchMIL(
        num_classes=len(cfg.CLASSES),
        topk_pool=cfg.TOPK_POOL,
        quality_beta=cfg.QUALITY_BETA
    )

    model.load_state_dict(sd, strict=True)

    # ✅ Force FP32 weights
    model = model.float()

    # Move to GPU
    model = model.to(DEVICE).eval()
    return model



# =========================
# INFER ONE
# =========================
@torch.no_grad()
def infer_one(seg: LimbusSegmenter, model: nn.Module, img_path: str):
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None

    rgb_512, mask01_512 = seg.mask_on_canonical_512(bgr)
    masked_512 = apply_mask_rgb(rgb_512, mask01_512)

    tiles_224, q = polar_tiles_precompute_matched(masked_512, mask01_512)
    if len(tiles_224) == 0:
        return None

    # pad tiles to MAX_TILES with black tiles (important for batched model stability)
    T = cfg.POLAR_THETA * cfg.POLAR_RINGS  # 24
    if len(tiles_224) > T:
        tiles_224 = tiles_224[:T]
        q = q[:T]
    if len(tiles_224) < T:
        pad_n = T - len(tiles_224)
        tiles_224 += [np.zeros((cfg.TILE_SAVE_SIZE, cfg.TILE_SAVE_SIZE, 3), dtype=np.uint8)] * pad_n
        q = np.concatenate([q, np.zeros((pad_n,), dtype=np.float32)], axis=0)

    q01 = normalize01(q)

    g_t = global_tf(masked_512).unsqueeze(0).to(DEVICE)                 # (1,3,384,384)
    t_t = torch.stack([tile_tf(t) for t in tiles_224]).unsqueeze(0).to(DEVICE) # (1,24,3,224,224)
    q_t = torch.tensor(q01, dtype=torch.float32).unsqueeze(0).to(DEVICE)# (1,24)

    # ✅ Force FP32 (prevents Float vs Half mismatch)
    g_t = g_t.float()
    t_t = t_t.float()
    q_t = q_t.float()

    # ✅ No autocast
    logits, att, top_idx = model(g_t, t_t, qnorm=q_t)

    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    pred_id = int(np.argmax(probs))
    conf = float(probs[pred_id])

    # overlay for saving (optional)
    overlay = None
    if cfg.SAVE_MASK_OVERLAY:
        ov = rgb_512.copy()
        m = mask01_512.astype(bool)
        ov[m] = (0.6 * ov[m] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        overlay = ov

    return {
        "pred_id": pred_id,
        "pred_class": cfg.CLASSES[pred_id],
        "confidence": conf,
        "probs": probs,
        "masked_512": masked_512,
        "overlay_512": overlay,
        "top_idx": top_idx.detach().cpu().numpy()[0].tolist(),
        "att": att.detach().cpu().numpy()[0].tolist(),
    }


def main():
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    for c in cfg.CLASSES:
        os.makedirs(os.path.join(cfg.OUT_DIR, c), exist_ok=True)

    seg = LimbusSegmenter(cfg.SEG_CKPT, DEVICE)
    model = load_classifier(cfg.CLS_CKPT)

    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(cfg.INPUT_DIR, e))
    paths = sorted(list(set(paths)))
    if not paths:
        raise RuntimeError(f"No images found in {cfg.INPUT_DIR}")

    rows = []
    print(f"[INFO] Found {len(paths)} images. Inference on {DEVICE}...")

    for p in tqdm(paths):
        res = infer_one(seg, model, p)
        if res is None:
            rows.append({"image": os.path.basename(p), "path": p, "pred": "FAILED", "confidence": 0.0, "gt": ""})
            continue

        pred = res["pred_class"]
        conf = res["confidence"]
        probs = res["probs"]

        # copy
        dst_dir = os.path.join(cfg.OUT_DIR, pred)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(p))
        shutil.copy2(p, dst_path)

        # optional previews
        stem = os.path.splitext(os.path.basename(p))[0]
        if cfg.SAVE_MASKED_GLOBAL:
            cv2.imwrite(os.path.join(dst_dir, f"{stem}_masked512.jpg"), cv2.cvtColor(res["masked_512"], cv2.COLOR_RGB2BGR))
        if cfg.SAVE_MASK_OVERLAY and res["overlay_512"] is not None:
            cv2.imwrite(os.path.join(dst_dir, f"{stem}_maskOverlay512.jpg"), cv2.cvtColor(res["overlay_512"], cv2.COLOR_RGB2BGR))

        row = {"image": os.path.basename(p), "path": p, "copied_to": dst_path, "pred": pred, "confidence": conf, "gt": ""}
        for i, c in enumerate(cfg.CLASSES):
            row[f"prob_{c}"] = float(probs[i])
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.OUT_DIR, "predictions_sorted.csv")
    df.to_csv(csv_path, index=False)
    print("\n✅ Done")
    print("CSV:", csv_path)
    print("Output:", cfg.OUT_DIR)

if __name__ == "__main__":
    main()
