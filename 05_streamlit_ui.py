import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch import amp
import math
from PIL import Image, ImageOps
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import plotly.express as px
import streamlit as st

# Import expert utilities from the local directory
from Limbus_Crop_Segmentation_System.inference_utils import load_model, predict_masks, postprocess_mask, largest_contour

# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    SEG_CKPT: str  = r"Limbus_Crop_Segmentation_System/model_limbus_crop_unetpp_weighted.pth"
    CLS_CKPT: str  = r"training_results/checkpoints/best.pth"

    CLASSES: Tuple[str, ...] = ("Edema", "Scar", "Infection", "Normal")
    CLASS_COLORS = {
        "Edema": "#3498db",    # Blue
        "Scar": "#9b59b6",     # Purple
        "Infection": "#e74c3c", # Red
        "Normal": "#2ecc71"    # Green
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
# HELPER FUNCTIONS
# =========================

def tile_quality_score(tile_rgb: np.ndarray) -> float:
    if tile_rgb is None or tile_rgb.size == 0: return 0.0
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    valid = gray > 5
    if valid.mean() < 0.05: return 0.0
    g = gray[valid]
    lapv = float(cv2.Laplacian(gray, cv2.CV_32F)[valid].var())
    contrast = float(g.std())
    glare_ratio = float((g > 240).mean())
    glare_pen = 1.0 - min(1.0, glare_ratio * 4.0)
    return float(max(0.0, (0.6 * math.log1p(lapv) + 0.4 * math.log1p(contrast)) * glare_pen))

def normalize01(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0: return arr
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(arr)

def polar_tiles_matched(masked_512, mask01_512):
    H, W = mask01_512.shape
    ys, xs = np.where(mask01_512 > 0)
    if len(xs) == 0: return [], np.zeros((0,)), []
    cx, cy = float(xs.mean()), float(ys.mean())
    rmax = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
    if rmax < 5: return [], np.zeros((0,)), []
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    tt = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    ring_edges = [f * rmax for f in cfg.RING_EDGES_FRAC]
    tiles, q, coords = [], [], []
    for r in range(cfg.POLAR_RINGS):
        for s in range(cfg.POLAR_THETA):
            t0, t1 = 2 * np.pi * s / cfg.POLAR_THETA, 2 * np.pi * (s + 1) / cfg.POLAR_THETA
            wedge = (mask01_512 > 0) & (rr >= ring_edges[r]) & (rr < ring_edges[r + 1]) & (tt >= t0) & (tt < t1)
            if wedge.sum() < cfg.POLAR_MIN_PIXELS: continue
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
# STREAMLIT UI
# =========================

st.set_page_config(page_title="KeratitisAI - Expert System", layout="wide", page_icon="👁️")

# CSS and Styling
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    .status-card { padding: 6px 12px; border-radius: 10px; text-align: center; color: white; font-weight: 700; font-size: 18px; margin-bottom: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); width: fit-content; margin-left: auto; margin-right: auto; min-width: 140px; }
    .stButton>button { border-radius: 10px; height: 3em; width: 100%; font-weight: bold; background-color: #007bff; color: white; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    # Segmentation
    model_s, idx_c, idx_l, img_s = load_model(cfg.SEG_CKPT, DEVICE)
    model_s.eval()
    # Classification
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
    st.error(f"Initialization Failed: {e}. Please check paths: '{cfg.SEG_CKPT}' and '{cfg.CLS_CKPT}'")
    st.stop()

# Transforms
tile_tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((cfg.TILE_SIZE, cfg.TILE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
global_tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((cfg.GLOBAL_SIZE, cfg.GLOBAL_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

st.title("🛡️ KeratitisAI Diagnostic Dashboard")
st.markdown("Professional-grade Multi-Branch Multiple Instance Learning for Keratitis classification.")

# Sidebar
st.sidebar.title("🩺 Clinical Control")
uploaded_file = st.sidebar.file_uploader("Upload Slit Lamp Photography", type=["jpg", "jpeg", "png"])
st.sidebar.divider()
st.sidebar.subheader("Parameters")
topk_val = st.sidebar.slider("Attention Top-Slices", 1, 12, 4)
quality_beta = st.sidebar.slider("Quality Bias (λ)", 0.0, 3.0, 1.2, step=0.1)

if uploaded_file:
    # 1. Load and Prep
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    bgr_orig = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    H_orig, W_orig = bgr_orig.shape[:2]
    
    # 2. Segment
    with st.spinner("Analyzing anatomical structures..."):
        masks = predict_masks(seg_model, bgr_orig, img_size, DEVICE)
        m_crop = masks[idx_crop]
        m_limb = masks[idx_limbus]
        
        # Prepare 512px view for inference
        bgr_512 = cv2.resize(bgr_orig, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE))
        mask_512 = cv2.resize(m_limb, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE), interpolation=cv2.INTER_NEAREST)
        
        rgb_512 = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB)
        masked_512 = rgb_512.copy()
        masked_512[mask_512 == 0] = 0
        

    # 3. Tiling
    tiles_224, q, tile_coords = polar_tiles_matched(masked_512, mask_512)
    
    col_main, col_res = st.columns([1.2, 1])

    with col_main:
        st.subheader("Anatomical Visualization")
        # Tabs for Raw Signal and Targeted
        tab_list = ["Raw Signal", "Library of Slices", "Targeted Region"]
        tabs = st.tabs(tab_list)
        
        # We need attention results to draw on Raw Signal, so we'll fill tabs after model run
        tabs[1].image(masked_512, width=250, caption="Processed 512px MIL Global Signal")

    if len(tiles_224) > 0:
        # Prepare Tensors (MIL expects fixed batch or padded bag)
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
        conf = probs[pred_id]
        
        # --- DRAW ATTENTION ON TARGETED REGION (512px) ---
        target_att_vis = masked_512.copy()
        top_indices = top_idx[0].cpu().numpy().tolist()
        for idx in top_indices:
            x0, y0, x1, y1 = tile_coords[idx]
            cv2.rectangle(target_att_vis, (x0, y0), (x1, y1), (255, 0, 0), 4) # Red Square on Targeted Region
        
        # Fill Tabs
        tabs[0].image(image, width=250, caption="Raw Signal (Input Photography)")
        
        with tabs[1]:
            st.markdown("#### Library of Cumulative Polar Slices")
            st.markdown(f"Displaying {len(tiles_224)} extracted anatomical slices used for MIL bagging.")
            cols_grid = st.columns(6)
            for i, slice_img in enumerate(tiles_224):
                with cols_grid[i % 6]:
                    is_top = i in top_indices
                    st.image(slice_img, caption=f"S{i+1}" + (" ⭐" if is_top else ""), use_container_width=True)
        
        tabs[2].image(target_att_vis, width=250, caption="Targeted Region + AI Attention (Red)")

        with col_res:
            st.subheader("Diagnostic Status")
            bg_color = cfg.CLASS_COLORS.get(pred_label, "#34495e")
            st.markdown(f'<div class="status-card" style="background-color: {bg_color};">{pred_label.upper()}</div>', unsafe_allow_html=True)
            
            # Metrics in table format for beauty
            st.markdown("### Confidence Analysis")
            
            # Build Styled HTML Table (Simplified to ensure rendering)
            tbl_html = """<table style="width:100%; border-collapse: collapse; font-family: sans-serif; font-size: 13px;">
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 6px; border: 1px solid #ddd; text-align: left;">Condition</th>
                    <th style="padding: 6px; border: 1px solid #ddd; text-align: right;">Probability</th>
                </tr>"""
            for i, cls in enumerate(cfg.CLASSES):
                p_str = f"{(probs[i]*100):.1f}%"
                is_pred = (i == pred_id)
                bg = "#d4edda" if is_pred else "#ffffff"
                weight = "700" if is_pred else "400"
                color = cfg.CLASS_COLORS[cls] if is_pred else "#333"
                
                tbl_html += f"""
                <tr style="background-color: {bg}; font-weight: {weight};">
                    <td style="padding: 6px; border: 1px solid #ddd; color: {color};">{cls}{" (Pred)" if is_pred else ""}</td>
                    <td style="padding: 6px; border: 1px solid #ddd; text-align: right;">{p_str}</td>
                </tr>"""
            tbl_html += "</table>"
            st.write(tbl_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Top-Slices Analyzed", topk_val)

        st.divider()
        st.subheader("👁️ AI Attention Map (Clinical Hotspots)")
        st.markdown(f"The model identifies these areas as most critical for the diagnosis of **{pred_label}**.")
        
        top_indices = top_idx[0].cpu().numpy().tolist()
        cols_att = st.columns(4)
        for i, idx in enumerate(top_indices[:4]):
            with cols_att[i]:
                # Small tiles for hotspots
                st.image(tiles_pad[idx], caption=f"Rank {i+1} ({att[0, idx]:.3f})", width=180)
                
    else:
        st.warning("⚠️ Limbus segmentation yielded no valid features. Prediction may be unreliable.")

else:
    # Portfolio Style Landing
    st.image("https://images.unsplash.com/photo-1576091160550-217359f4ecf8?auto=format&fit=crop&q=80&w=2070", use_container_width=True)
    st.info("System Ready. Please upload a clinical image to proceed with automated Keratitis detection.")
    st.markdown("""
    ### About this System
    - **Dual-Branch Architecture**: Combines global ocular context with high-resolution localized Slice.
    - **Gated Attention MIL**: Learns to distinguish between noisy patches and critical diagnostic indicators.
    - **Expert Segmentation**: Hard-coded anatomical priors ensure the AI focuses on the relevant corneal surface.
    """)
