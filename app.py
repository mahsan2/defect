# ------------------------------------------------------------
# Streamlit app  –  Small-VLM defect classification
# weights-only checkpoint, scaler refitted at startup
# ------------------------------------------------------------
import streamlit as st
import torch, pandas as pd, numpy as np
import torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
from PIL import Image

STATE_PATH   = "small_vlm_state.pt"    # weights-only (~45 MB)
CSV_PATH     = "param_df_cleaned.csv"  # 1 814 rows
LABEL_MAP    = {0: "Powder (empty bed)", 1: "Printed region (healthy)"}

# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# ------------------------------------------------------------
# Model + data loader   (place near the top of app.py)
# ------------------------------------------------------------
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
from sklearn.preprocessing import StandardScaler

STATE_PATH = "small_vlm_state.pt"      # weights-only file
CSV_PATH   = "param_df_cleaned.csv"    # tabular data

# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading model & data…")
def load_model_and_data():
    # 1) tabular data + scaler
    df = pd.read_csv(CSV_PATH).reset_index(drop=True)
    scaler = StandardScaler().fit(df.values)

    # 2) model **with attribute name `classifier`**
    class SmallVLM(nn.Module):
        def __init__(self, n_params: int):
            super().__init__()
            backbone = models.resnet18(weights="IMAGENET1K_V1")
            backbone.fc = nn.Identity()            # 512-D
            self.cnn = backbone

            self.mlp = nn.Sequential(
                nn.Linear(n_params, 64), nn.ReLU(),
                nn.Linear(64, 32)
            )

            # ↙️ keep this exact name to match the checkpoint
            self.classifier = nn.Sequential(
                nn.Linear(512 + 32, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )

        def forward(self, img, vec):
            feats = torch.cat((self.cnn(img), self.mlp(vec)), dim=1)
            return self.classifier(feats)

    model = SmallVLM(n_params=scaler.mean_.shape[0])

    # 3) load weights (strict = True is fine now)
    state = torch.load(STATE_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, scaler, df



model, scaler, param_df = load_model_and_data()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ─────────────────────────────────────────────────────────────
st.title("Multimodal Defect Classifier")

up = st.file_uploader("Upload a slice image (file must start with layer number, e.g. **5_slice_0.png**)", type=["png", "jpg", "jpeg"])

if up:
    img_pil = Image.open(up).convert("RGB")
    st.image(img_pil, width=256)

    # --- parse layer number from filename ---
    try:
        layer_idx = int(up.name.split("_")[0])
    except Exception:
        st.error("❌ Filename must start with layer number + '_'")
        st.stop()

    if layer_idx >= len(param_df):
        st.error(f"No tabular data for layer {layer_idx}")
        st.stop()

    # --- prepare inputs ---
    x_img = transform(img_pil).unsqueeze(0)
    vec   = torch.tensor(scaler.transform(
              param_df.loc[layer_idx].values.reshape(1, -1)),
              dtype=torch.float32)

    # --- forward ---
    with torch.no_grad():
        prob = torch.softmax(model(x_img, vec), dim=1)
        conf, pred = prob.max(1)

    label = LABEL_MAP[pred.item()]
    st.markdown(f"### Prediction : **{label}** – {conf.item()*100:.1f}%")

    # --- show key parameters & rule-based tips ---
    row = param_df.loc[layer_idx]
    st.write(f"""
    **Process parameters**

    • Top-chamber T : **{row['top_chamber_temperature']:.0f} °C**  
    • Bottom flow   : **{row['bottom_flow_rate']:.1f} %**  
    • Ventilator    : **{row['ventilator_speed']:.0f} rpm**  
    • O₂            : **{row['gas_loop_oxygen']:.1f} ppm**
    """)

    tips = []
    if row['bottom_flow_rate'] < 45: tips.append("🔧 Increase bottom-flow > 45 %.")
    if row['ventilator_speed'] < 40: tips.append("🔧 Boost ventilator > 40 rpm.")
    if row['gas_loop_oxygen'] > 10:  tips.append("🔧 Purge chamber (O₂ < 10 ppm).")

    if tips:
        st.warning(" ".join(tips))
    else:
        st.success("✅ Parameters within nominal range.")


    # ⬇️ Grad-CAM (optional) -----------------------------------------
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        target_layer = model.cnn.layer4[-1]
        cam = GradCAM(model, [target_layer], device="cpu")
        heat = cam(x_img, extra_forward_args=(vec,))[0]

        rgb  = np.transpose(x_img.squeeze().numpy(), (1, 2, 0))
        cam_img = show_cam_on_image(rgb, heat, use_rgb=True)

        st.image(cam_img, caption="Grad-CAM", width=240)
    except Exception as e:
        st.info(f"Grad-CAM unavailable ({e})")
