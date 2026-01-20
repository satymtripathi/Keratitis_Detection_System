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
    if len(xs) == 0: return [], np.zeros((0,))
    cx, cy = float(xs.mean()), float(ys.mean())
    rmax = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
    if rmax < 5: return [], np.zeros((0,))
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    tt = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    ring_edges = [f * rmax for f in cfg.RING_EDGES_FRAC]
    tiles, q = [], []
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
            tiles.append(tile_res); q.append(tile_quality_score(tile_res))
    return tiles, np.array(q, dtype=np.float32)

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="KeratitisAI - Diagnostic System", layout="wide", page_icon="👁️")

# CSS and Styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .prediction-card { padding: 20px; border-radius: 15px; text-align: center; color: white; font-weight: bold; font-size: 24px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    # Segmentation
    model_s, _, idx_l, img_s = load_model(cfg.SEG_CKPT, DEVICE)
    model_s.eval()
    # Classification
    ckpt = torch.load(cfg.CLS_CKPT, map_location=DEVICE)
    sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    model_c = DualBranchMIL(len(cfg.CLASSES), cfg.TOPK_POOL, cfg.QUALITY_BETA)
    model_c.load_state_dict(sd, strict=True)
    model_c.float().to(DEVICE).eval()
    return model_s, idx_l, img_s, model_c

try:
    with st.spinner("Loading Expert Models..."):
        seg_model, idx_limbus, img_size, clf_model = load_all_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Ensure checkpoints are in 'training_results' and 'Limbus_Crop_Segmentation_System'.")
    st.stop()

# Transforms
tile_tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((cfg.TILE_SIZE, cfg.TILE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
global_tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((cfg.GLOBAL_SIZE, cfg.GLOBAL_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

st.title("👁️ Keratitis Detection System")
st.markdown("Advanced Multi-Instance Learning for Ocular Condition Classification")

# Sidebar
st.sidebar.header("📂 Analysis Controls")
uploaded_file = st.sidebar.file_uploader("Upload Slit Lamp Image", type=["jpg", "jpeg", "png"])
topk_val = st.sidebar.slider("Top-K Attention Tiles", 1, 8, 4)
quality_beta = st.sidebar.slider("Quality Sensitivity (Beta)", 0.0, 2.0, 1.2)

if uploaded_file:
    # 1. Load and Prep
    image = Image.open(uploaded_file).convert("RGB")
    bgr_orig = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 2. Segment
    bgr_512 = cv2.resize(bgr_orig, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE))
    masks = predict_masks(seg_model, bgr_512, img_size, DEVICE)
    mask01 = (masks[idx_limbus] > 0.5).astype(np.uint8)
    if mask01.shape[:2] != (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE):
        mask01 = cv2.resize(mask01, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE), interpolation=cv2.INTER_NEAREST)
    
    rgb_512 = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB)
    masked_512 = rgb_512.copy()
    masked_512[mask01 == 0] = 0
    
    # 3. Tiling
    tiles_224, q = polar_tiles_matched(masked_512, mask01)
    
    col_main, col_res = st.columns([1.5, 1])

    with col_main:
        st.subheader("Clinical Visualizations")
        # Mask overlay
        ov = rgb_512.copy()
        ov[mask01.astype(bool)] = (0.6 * ov[mask01.astype(bool)] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        
        tabs = st.tabs(["Input", "Targeting (Limbus)", "Feature Map (Precompute)"])
        tabs[0].image(image, use_container_width=True)
        tabs[1].image(ov, use_container_width=True, caption="Green: Detected Limbus Region")
        tabs[2].image(masked_512, use_container_width=True, caption="512px Masked Canonical View")

    if len(tiles_224) > 0:
        # Prepare Tensors
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
        
        with col_res:
            st.subheader("Diagnostic Report")
            bg_color = cfg.CLASS_COLORS.get(pred_label, "#34495e")
            st.markdown(f'<div class="prediction-card" style="background-color: {bg_color};">{pred_label} ({(conf*100):.1f}%)</div>', unsafe_allow_html=True)
            
            # Metric scores
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{conf:.2%}")
            m2.metric("Tiles Selected", topk_val)
            
            # Probability Chart
            prob_df = pd.DataFrame({"Condition": cfg.CLASSES, "Probability": probs})
            fig = px.bar(prob_df, x="Probability", y="Condition", orientation='h', color="Condition",
                         color_discrete_map=cfg.CLASS_COLORS, text_auto='.2%', range_x=[0,1])
            fig.update_layout(showlegend=False, height=300, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Critical Diagnostic Features (Attention-Weighted Tiles)")
        top_indices = top_idx[0].cpu().numpy().tolist()
        cols = st.columns(4)
        for i, idx in enumerate(top_indices[:4]):
            with cols[i]:
                st.image(tiles_pad[idx], caption=f"Rank {i+1} | Score: {att[0, idx]:.3f}", use_container_width=True)
    else:
        st.warning("⚠️ Limbus not clearly detected. Please ensure the eye is centered and slit-lamp is focused.")

else:
    # Landing Page
    st.info("Please upload a slit lamp eye image from the sidebar to begin analysis.")
    st.image("https://img.freepik.com/free-vector/eye-diagnostics-medical-poster_1284-18341.jpg", width=400) # Placeholder for landing
