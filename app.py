# ------------------------------------------------------------
# app.py  –  Small-VLM multimodal classifier (+ Grad-CAM & LIME)
# ------------------------------------------------------------
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
import pandas as pd
from PIL import Image
import numpy as np

# ---------- paths ----------
STATE_PATH = "small_vlm_state_fixed.pt"     # weights-only (~45 MB)
CSV_PATH   = "param_df_cleaned.csv"         # tabular data
LABEL_MAP  = {0: "Powder (empty bed)", 1: "Printed region (healthy)"}

# ---------- model ----------
class SmallVLM(nn.Module):
    def __init__(self, n_params: int):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        backbone.fc = nn.Identity()                # → 512-D global feature
        self.cnn = backbone

        self.mlp = nn.Sequential(
            nn.Linear(n_params, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )

        # keep **classifier** to match state-dict keys
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, img, vec):
        feats = torch.cat((self.cnn(img), self.mlp(vec)), dim=1)
        return self.classifier(feats)              # no self.cls

# ---------- load model + data ----------
@st.cache_resource(show_spinner="🔄 Loading model & data …")
def load_assets():
    df = pd.read_csv(CSV_PATH).reset_index(drop=True)
    scaler = StandardScaler().fit(df.values)

    model = SmallVLM(n_params=df.shape[1])
    state = torch.load(STATE_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, scaler, df

model, scaler, param_df = load_assets()

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------- Streamlit UI ----------
st.title("Multimodal Defect Classifier  (Small-VLM)")

up = st.file_uploader(
    "Upload a slice image (file name must start with layer number, e.g. **5_slice_0.png**)",
    type=["png", "jpg", "jpeg"]
)

if up is None:
    st.stop()

# ---------- show image ----------
img_pil = Image.open(up).convert("RGB")
st.image(img_pil, width=256)

# ---------- parse layer number ----------
try:
    layer_idx = int(up.name.split("_")[0])
except Exception:
    st.error("❌ Filename must begin with layer number + '_'")
    st.stop()

if layer_idx >= len(param_df):
    st.error(f"No tabular row for layer {layer_idx}")
    st.stop()

# ---------- forward pass ----------
x_img = tfm(img_pil).unsqueeze(0)
x_vec = torch.tensor(
    scaler.transform(param_df.loc[layer_idx].values.reshape(1, -1)),
    dtype=torch.float32
)

with torch.no_grad():
    prob = torch.softmax(model(x_img, x_vec), dim=1)
    conf, pred = prob.max(1)

label = LABEL_MAP[pred.item()]
st.markdown(f"### Prediction : **{label}** — {conf.item()*100:.1f}%")

# ---------- show parameters + tips ----------
row = param_df.loc[layer_idx]
st.write(f"""
**Process parameters**

• Top-chamber T = {row['top_chamber_temperature']:.0f} °C  
• Bottom flow  = {row['bottom_flow_rate']:.1f} %  
• Ventilator  = {row['ventilator_speed']:.0f} rpm  
• O₂      = {row['gas_loop_oxygen']:.1f} ppm
""")

tips = []
if row['bottom_flow_rate'] < 45:
    tips.append("🔧 Increase bottom-flow > 45 %.")
if row['ventilator_speed'] < 40:
    tips.append("🔧 Boost ventilator > 40 rpm.")
if row['gas_loop_oxygen'] > 10:
    tips.append("🔧 Purge chamber (O₂ < 10 ppm).")

st.warning(" ".join(tips)) if tips else st.success("✅ Parameters within range.")

# ---------- Grad-CAM (optional) ----------
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    cam = GradCAM(model, [model.cnn.layer4[-1]], device="cpu")
    heat = cam(x_img, extra_forward_args=(x_vec,))[0]

    rgb  = np.transpose(x_img.squeeze().numpy(), (1, 2, 0))
    cam_img = show_cam_on_image(rgb, heat, use_rgb=True)
    st.image(cam_img, caption="Grad-CAM", width=240)
except Exception as e:
    st.info(f"Grad-CAM unavailable ({e})")

# ---------- LIME tabular explanation ----------
with st.expander("🔍 Show LIME tabular explanation"):
    from lime import lime_tabular

    explainer = lime_tabular.LimeTabularExplainer(
        training_data         = param_df.values,
        feature_names         = list(param_df.columns),
        mode                  = "classification",
        class_names           = list(LABEL_MAP.values()),
        discretize_continuous = True
    )

    exp = explainer.explain_instance(
        data_row   = param_df.loc[layer_idx].values,
        predict_fn = lambda X: torch.softmax(
            model(
                torch.zeros((X.shape[0], 3, 224, 224)),          # dummy images
                torch.tensor(scaler.transform(X), dtype=torch.float32)
            ), 1
        ).detach().numpy(),
        num_features = 10
    )

    st.components.v1.html(exp.as_html(), height=400)
