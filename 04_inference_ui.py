"""
gradio_ui_precompute_matched.py
Small UI, matches precompute logic 1:1.

Run:
python gradio_ui_precompute_matched.py
"""

import os, math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch import amp
import gradio as gr

from Limbus_Crop_Segmentation_System.inference_utils import load_model, predict_masks


@dataclass
class CFG:
    SEG_CKPT: str  = r"Limbus_Crop_Segmentation_System\model_limbus_crop_unetpp_weighted.pth"
    CLS_CKPT: str  = r"training_results\checkpoints\best.pth"

    CLASSES: Tuple[str, ...] = ("Edema", "Scar", "Infection", "Normal")

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


class LimbusSegmenter:
    def __init__(self, checkpoint: str, device: str):
        self.device = device
        self.model, self.idx_crop, self.idx_limbus, self.img_size = load_model(checkpoint, device)
        self.model.eval()
        print(f"[SegModel] idx_crop={self.idx_crop}, idx_limbus={self.idx_limbus}, img_size={self.img_size}")

    @torch.no_grad()
    def mask_on_512(self, bgr_orig: np.ndarray):
        bgr_512 = cv2.resize(bgr_orig, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE), interpolation=cv2.INTER_AREA)
        rgb_512 = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB)

        masks = predict_masks(self.model, bgr_512, self.img_size, self.device)
        limbus = masks[self.idx_limbus]
        mask01 = (limbus > 0.5).astype(np.uint8) if limbus.dtype != np.uint8 else (limbus > 0).astype(np.uint8)

        if mask01.shape[:2] != (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE):
            mask01 = cv2.resize(mask01, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE), interpolation=cv2.INTER_NEAREST)

        return rgb_512, mask01


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

def normalize01(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        return (arr - mn) / (mx - mn + 1e-8)
    return np.zeros_like(arr)

def polar_tiles_precompute_matched(masked_512: np.ndarray, mask01_512: np.ndarray):
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

    tiles, q = [], []
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

            tile = masked_512[y0:y1 + 1, x0:x1 + 1].copy()
            tile_w = wedge[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
            tile[tile_w == 0] = 0

            tile_res = cv2.resize(tile, (cfg.TILE_SAVE_SIZE, cfg.TILE_SAVE_SIZE), interpolation=cv2.INTER_AREA)
            tiles.append(tile_res)
            q.append(tile_quality_score(tile_res))

    return tiles, np.array(q, dtype=np.float32)

def mask_overlay(rgb_512, mask01):
    out = rgb_512.copy()
    m = mask01.astype(bool)
    out[m] = (0.6 * out[m] + 0.4 * np.array([0,255,0])).astype(np.uint8)
    return out

def crop_to_mask(rgb_512, mask01, pad=10):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return rgb_512
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(rgb_512.shape[1]-1, x1 + pad); y1 = min(rgb_512.shape[0]-1, y1 + pad)
    return rgb_512[y0:y1+1, x0:x1+1].copy()


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

def load_classifier(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    model = DualBranchMIL(num_classes=len(cfg.CLASSES), topk_pool=cfg.TOPK_POOL, quality_beta=cfg.QUALITY_BETA)
    model.load_state_dict(sd, strict=True)
    model = model.float()
    model.to(DEVICE).eval()
    return model


# Globals
segmenter = LimbusSegmenter(cfg.SEG_CKPT, DEVICE)
clf = load_classifier(cfg.CLS_CKPT)

@torch.no_grad()
def run_ui(img_np, topk: int):
    if img_np is None:
        return None, None, None, None, None, "No image"

    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    rgb_512, mask01_512 = segmenter.mask_on_512(bgr)
    masked_512 = apply_mask_rgb(rgb_512, mask01_512)

    tiles_224, q = polar_tiles_precompute_matched(masked_512, mask01_512)
    if len(tiles_224) == 0:
        return rgb_512, mask_overlay(rgb_512, mask01_512), masked_512, crop_to_mask(rgb_512, mask01_512), None, "No tiles"

    # pad to 24 exactly
    T = cfg.POLAR_THETA * cfg.POLAR_RINGS
    if len(tiles_224) > T:
        tiles_224 = tiles_224[:T]
        q = q[:T]
    if len(tiles_224) < T:
        pad_n = T - len(tiles_224)
        tiles_224 += [np.zeros((cfg.TILE_SAVE_SIZE, cfg.TILE_SAVE_SIZE, 3), dtype=np.uint8)] * pad_n
        q = np.concatenate([q, np.zeros((pad_n,), dtype=np.float32)], axis=0)

    q01 = normalize01(q)

    g_t = global_tf(masked_512).unsqueeze(0).to(DEVICE).float()
    t_t = torch.stack([tile_tf(t) for t in tiles_224]).unsqueeze(0).to(DEVICE).float()
    q_t = torch.tensor(q01, dtype=torch.float32).unsqueeze(0).to(DEVICE).float()

    clf.topk_pool = int(topk)

    with amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
        logits, att, top_idx = clf(g_t, t_t, qnorm=q_t)
        probs = torch.softmax(logits, dim=1).squeeze(0).float().cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred = cfg.CLASSES[pred_id]

    att_np = att.detach().cpu().numpy()[0]
    top = top_idx.detach().cpu().numpy()[0].tolist()

    # show top tiles
    top_tiles = [tiles_224[i] for i in top[:min(4, len(top))]]
    while len(top_tiles) < 4:
        top_tiles.append(np.zeros((cfg.TILE_SAVE_SIZE, cfg.TILE_SAVE_SIZE, 3), dtype=np.uint8))

    text = "Prediction: " + pred + "\n\n" + "\n".join([f"{cfg.CLASSES[i]}: {probs[i]:.4f}" for i in range(len(cfg.CLASSES))])

    return (
        rgb_512,
        mask_overlay(rgb_512, mask01_512),
        masked_512,
        crop_to_mask(rgb_512, mask01_512),
        top_tiles,
        text
    )

with gr.Blocks(title="Keratitis Inference (Precompute-matched)") as demo:
    gr.Markdown(
        "## Keratitis Inference\n"
        "Matches precompute logic: **512 mask + polar wedges + 224 tiles + global + MIL**"
    )

    with gr.Row(equal_height=False):
        # LEFT: Input + Controls
        with gr.Column(scale=1, min_width=360):
            inp = gr.Image(type="numpy", label="Input Image", height=200)
            topk = gr.Slider(2, 8, value=4, step=1, label="Top-K tiles")
            btn = gr.Button("Run", variant="primary")  # normal size

        # RIGHT: Outputs
        # RIGHT: Outputs
        with gr.Column(scale=2):
            gr.Markdown("### Outputs")

            with gr.Row():
                out1 = gr.Image(label="512 RGB (canonical)", height=200)
                out2 = gr.Image(label="Mask overlay", height=200)
            with gr.Row():
                out3 = gr.Image(label="Masked global (512)", height=200)
                out4 = gr.Image(label="Cropped around mask", height=200)

            out_txt = gr.Textbox(lines=6, label="Prediction + probabilities")

            out_tiles = gr.Gallery(label="Top-4 tiles", columns=4, height=180)

    btn.click(
        run_ui,
        inputs=[inp, topk],
        outputs=[out1, out2,out3, out4, out_tiles, out_txt]# out3, out4,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
