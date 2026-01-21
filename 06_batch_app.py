import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import math
from PIL import Image, ImageOps
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import streamlit as st

# Import expert utilities from the local directory
from Limbus_Crop_Segmentation_System.inference_utils import load_model, predict_masks

# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    SEG_CKPT: str  = r"Limbus_Crop_Segmentation_System/model_limbus_crop_unetpp_weighted.pth"
    CLS_CKPT: str  = r"training_results/checkpoints/best.pth"

    CLASSES: Tuple[str, ...] = ("Edema", "Scar", "Infection", "Normal")
    CLASS_COLORS = {
        "Edema": "#3498db",
        "Scar": "#9b59b6",
        "Infection": "#e74c3c",
        "Normal": "#2ecc71"
    }

    CANONICAL_SIZE: int = 512
    TILE_SAVE_SIZE: int = 224
    GLOBAL_SIZE: int = 384
    TILE_SIZE: int = 224

    POLAR_THETA: int = 8
    POLAR_RINGS: int = 3
    RING_EDGES_FRAC: Tuple[float, ...] = (0.0, 0.35, 0.70, 1.0)
    POLAR_MIN_PIXELS: int = 150
    POLAR_PAD: int = 2

    TOPK_POOL: int = 4
    QUALITY_BETA: float = 1.2

cfg = CFG()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MODEL ARCHITECTURE
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
        B, T, C, H, W = tiles.shape
        tiles_ = tiles.view(B*T, C, H, W)
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

# =========================
# HELPERS
# =========================
def tile_quality_score(tile_rgb: np.ndarray) -> float:
    if tile_rgb is None or tile_rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    valid = gray > 5
    if valid.mean() < 0.05:
        return 0.0
    g = gray[valid]
    lapv = float(cv2.Laplacian(gray, cv2.CV_32F)[valid].var())
    contrast = float(g.std())
    glare_ratio = float((g > 240).mean())
    glare_pen = 1.0 - min(1.0, glare_ratio * 4.0)
    return float(max(0.0, (0.6 * math.log1p(lapv) + 0.4 * math.log1p(contrast)) * glare_pen))

def normalize01(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(arr)

def polar_tiles_matched(masked_512, mask01_512):
    H, W = mask01_512.shape
    ys, xs = np.where(mask01_512 > 0)
    if len(xs) == 0:
        return [], np.zeros((0,)), []
    cx, cy = float(xs.mean()), float(ys.mean())
    rmax = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
    if rmax < 5:
        return [], np.zeros((0,)), []

    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    tt = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    ring_edges = [f * rmax for f in cfg.RING_EDGES_FRAC]

    tiles, q, coords = [], [], []
    for r in range(cfg.POLAR_RINGS):
        for s in range(cfg.POLAR_THETA):
            t0, t1 = 2 * np.pi * s / cfg.POLAR_THETA, 2 * np.pi * (s + 1) / cfg.POLAR_THETA
            wedge = (mask01_512 > 0) & (rr >= ring_edges[r]) & (rr < ring_edges[r + 1]) & (tt >= t0) & (tt < t1)
            if wedge.sum() < cfg.POLAR_MIN_PIXELS:
                continue
            ys_w, xs_w = np.where(wedge)
            x0, y0, x1, y1 = xs_w.min(), ys_w.min(), xs_w.max(), ys_w.max()
            x0, y0 = max(0, x0 - cfg.POLAR_PAD), max(0, y0 - cfg.POLAR_PAD)
            x1, y1 = min(W - 1, x1 + cfg.POLAR_PAD), min(H - 1, y1 + cfg.POLAR_PAD)

            tile = masked_512[y0:y1 + 1, x0:x1 + 1].copy()
            tile[wedge[y0:y1 + 1, x0:x1 + 1] == 0] = 0
            tile_res = cv2.resize(tile, (cfg.TILE_SAVE_SIZE, cfg.TILE_SAVE_SIZE), interpolation=cv2.INTER_AREA)

            tiles.append(tile_res)
            q.append(tile_quality_score(tile_res))
            coords.append((x0, y0, x1, y1))

    return tiles, np.array(q, dtype=np.float32), coords

# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="KeratitisAI - Expert System", layout="wide", page_icon="👁️")

@st.cache_resource
def load_all_models():
    model_s, idx_c, idx_l, img_s = load_model(cfg.SEG_CKPT, DEVICE)
    model_s.eval()

    ckpt = torch.load(cfg.CLS_CKPT, map_location=DEVICE)
    sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt

    model_c = DualBranchMIL(len(cfg.CLASSES), cfg.TOPK_POOL, cfg.QUALITY_BETA)
    model_c.load_state_dict(sd, strict=True)
    model_c.float().to(DEVICE).eval()

    return model_s, idx_c, idx_l, img_s, model_c

try:
    with st.spinner("Initializing Deep Learning Engine..."):
        seg_model, idx_crop, idx_limbus, img_size, clf_model = load_all_models()
except Exception as e:
    st.error(f"Initialization Failed: {e}. Please check paths.")
    st.stop()

tile_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((cfg.TILE_SIZE, cfg.TILE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

global_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((cfg.GLOBAL_SIZE, cfg.GLOBAL_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def run_one_image(pil_img: Image.Image, topk_val: int, quality_beta: float):
    pil_img = ImageOps.exif_transpose(pil_img.convert("RGB"))
    bgr_orig = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    masks = predict_masks(seg_model, bgr_orig, img_size, DEVICE)
    m_limb = masks[idx_limbus]

    bgr_512 = cv2.resize(bgr_orig, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE))
    mask_512 = cv2.resize(m_limb, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE), interpolation=cv2.INTER_NEAREST)

    rgb_512 = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB)
    masked_512 = rgb_512.copy()
    masked_512[mask_512 == 0] = 0

    tiles_224, q, tile_coords = polar_tiles_matched(masked_512, mask_512)

    if len(tiles_224) == 0:
        return None, {"pred_label": "NA", "confidence": 0.0, "probs": {c: 0.0 for c in cfg.CLASSES}, "num_tiles": 0}

    T_MAX = 24
    tiles_pad = tiles_224[:T_MAX]
    q_pad = q[:T_MAX]
    if len(tiles_pad) < T_MAX:
        pad_n = T_MAX - len(tiles_pad)
        tiles_pad += [np.zeros((cfg.TILE_SAVE_SIZE, cfg.TILE_SAVE_SIZE, 3), dtype=np.uint8)] * pad_n
        q_pad = np.concatenate([q_pad, np.zeros((pad_n,))])

    q01 = normalize01(q_pad)

    g_t = global_tf(masked_512).unsqueeze(0).to(DEVICE).float()
    t_t = torch.stack([tile_tf(t) for t in tiles_pad]).unsqueeze(0).to(DEVICE).float()
    q_t = torch.tensor(q01, dtype=torch.float32).unsqueeze(0).to(DEVICE).float()

    clf_model.topk_pool = topk_val
    clf_model.quality_beta = quality_beta

    with torch.no_grad():
        logits, att, top_idx = clf_model(g_t, t_t, qnorm=q_t)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_label = cfg.CLASSES[pred_id]
    conf = float(probs[pred_id])
    prob_map = {cfg.CLASSES[i]: float(probs[i]) for i in range(len(cfg.CLASSES))}

    return masked_512, {
        "pred_label": pred_label,
        "confidence": conf,
        "probs": prob_map,
        "num_tiles": int(len(tiles_224)),
        "tiles_224": tiles_224,
        "tile_coords": tile_coords,
        "top_idx": top_idx[0].cpu().numpy().tolist(),
        "att": att[0].detach().cpu().numpy().tolist()
    }

st.title("🛡️ KeratitisAI Diagnostic Dashboard")
st.markdown("Batch Keratitis classification using segmentation + polar tiles + MIL attention.")

st.sidebar.title("🩺 Clinical Control")
uploaded_files = st.sidebar.file_uploader(
    "Upload Slit Lamp Photography (Batch)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
st.sidebar.divider()
st.sidebar.subheader("Parameters")
topk_val = st.sidebar.slider("Attention Top-Slices", 1, 12, 4)
quality_beta = st.sidebar.slider("Quality Bias (λ)", 0.0, 3.0, 1.2, step=0.1)
show_details = st.sidebar.checkbox("Show per-image details (slower)", value=False)

if uploaded_files and len(uploaded_files) > 0:
    st.sidebar.success(f"{len(uploaded_files)} file(s) queued")

    # ---- SHOW INPUT IMAGES BY NAME (GRID) ----
    st.subheader("Uploaded Images")
    cols = st.columns(6)
    for i, uf in enumerate(uploaded_files):
        with cols[i % 6]:
            try:
                img_preview = Image.open(uf)
                st.image(img_preview, caption=uf.name, use_container_width=True)
            except Exception:
                st.write(uf.name)

    st.divider()

    # ---- RUN BATCH INFERENCE ----
    results = []
    st.subheader("Batch Results")
    pbar = st.progress(0.0)

    for i, uf in enumerate(uploaded_files):
        try:
            pil_img = Image.open(uf)
            masked_512, out = run_one_image(pil_img, topk_val=topk_val, quality_beta=quality_beta)

            if out is None:
                results.append({
                    "file": uf.name,
                    "prediction": "NA",
                    "confidence": 0.0,
                    "tiles_found": 0,
                    "note": "No valid limbus tiles",
                    **{f"p_{c}": 0.0 for c in cfg.CLASSES}
                })
            else:
                row = {
                    "file": uf.name,
                    "prediction": out["pred_label"],
                    "confidence": out["confidence"],
                    "tiles_found": out["num_tiles"],
                    "note": "",
                }
                for c in cfg.CLASSES:
                    row[f"p_{c}"] = out["probs"][c]
                results.append(row)

                if show_details:
                    with st.expander(f"Details: {uf.name}"):
                        colA, colB = st.columns([1, 1])

                        with colA:
                            st.image(pil_img, caption=f"Raw Input: {uf.name}", use_container_width=True)

                        with colB:
                            st.image(masked_512, caption="Masked Limbus (512px)", use_container_width=True)

                        # Attention boxes on masked_512
                        target_att_vis = masked_512.copy()
                        top_indices = out["top_idx"]
                        for idx in top_indices:
                            if idx < len(out["tile_coords"]):
                                x0, y0, x1, y1 = out["tile_coords"][idx]
                                cv2.rectangle(target_att_vis, (x0, y0), (x1, y1), (255, 0, 0), 4)
                        st.image(target_att_vis, caption="Top-K Attention Regions (Red Boxes)", use_container_width=True)

                        st.markdown("**Top attention tiles**")
                        cols_att = st.columns(4)
                        for j, idx in enumerate(top_indices[:4]):
                            with cols_att[j]:
                                if idx < len(out["tiles_224"]):
                                    st.image(out["tiles_224"][idx], caption=f"Rank {j+1}", use_container_width=True)

        except Exception as e:
            results.append({
                "file": uf.name,
                "prediction": "ERROR",
                "confidence": 0.0,
                "tiles_found": 0,
                "note": str(e),
                **{f"p_{c}": 0.0 for c in cfg.CLASSES}
            })

        pbar.progress((i + 1) / len(uploaded_files))

    df = pd.DataFrame(results)

    st.dataframe(
        df.style.format({
            "confidence": "{:.3f}",
            **{f"p_{c}": "{:.3f}" for c in cfg.CLASSES}
        }),
        use_container_width=True
    )

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results CSV",
        data=csv_bytes,
        file_name="keratitis_batch_results.csv",
        mime="text/csv"
    )

else:
    st.info("System Ready. Upload multiple clinical images to run batch Keratitis detection.")
