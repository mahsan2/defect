# ------------------------------------------------------------
#  Small-VLM Streamlit app   –   cls = classifier.*
# ------------------------------------------------------------
import streamlit as st
import torch
import torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
import pandas as pd
from PIL import Image
import numpy as np

STATE_PATH = "small_vlm_state_fixed.pt"      # weights-only (~45 MB)
CSV_PATH   = "param_df_cleaned.csv"          # 1 814 rows
LABEL_MAP  = {0: "Powder (empty bed)", 1: "Printed region (healthy)"}

# ─────────────────────────────────────────────────────────────
class SmallVLM(nn.Module):
    def __init__(self, n_params: int):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1")
        base.fc = nn.Identity()
        self.cnn = base

        self.mlp = nn.Sequential(
            nn.Linear(n_params, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )

        #  MUST be named “classifier”  ← matches checkpoint keys
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, img, vec):
        feats = torch.cat((self.cnn(img), self.mlp(vec)), dim=1)
        return self.classifier(feats)

# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading model & data …")
def load_model_and_data():
    df      = pd.read_csv(CSV_PATH).reset_index(drop=True)
    scaler  = StandardScaler().fit(df.values)
    model   = SmallVLM(df.shape[1])
    model.load_state_dict(torch.load(STATE_PATH, map_location="cpu"),
                          strict=True)
    model.eval()
    return model, scaler, df

model, scaler, param_df = load_model_and_data()
tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# ─────────────────────────────────────────────────────────────
st.title("Multimodal Defect Classifier (Small-VLM)")

up = st.file_uploader(
    "Upload a slice image (file name must start with layer number, e.g. **5_slice_0.png**)",
    type=["png", "jpg", "jpeg"]
)

if not up:
    st.stop()

# ------------------------------------------------------------  IMAGE + TABULAR  ----
img = Image.open(up).convert("RGB")
st.image(img, width=256)

try:
    layer = int(up.name.split("_")[0])
except Exception:
    st.error("Filename must start with the layer number followed by '_'")
    st.stop()

if layer >= len(param_df):
    st.error(f"No tabular data for layer {layer}")
    st.stop()

x_img = tfm(img).unsqueeze(0)
x_vec = torch.tensor(
    scaler.transform(param_df.loc[layer].to_numpy()[None]),
    dtype=torch.float32
)

with torch.no_grad():
    prob  = F.softmax(model(x_img, x_vec), 1)
    conf, pred = prob.max(1)

st.markdown(f"### **{LABEL_MAP[pred.item()]}** — {conf.item()*100:.1f}%")

row = param_df.loc[layer]
st.write(f"""
**Process parameters**

• Top-chamber T = {row['top_chamber_temperature']:.0f} °C  
• Bottom flow   = {row['bottom_flow_rate']:.1f} %  
• Ventilator    = {row['ventilator_speed']:.0f} rpm  
• O₂            = {row['gas_loop_oxygen']:.1f} ppm
""")

tips = []
if row['bottom_flow_rate'] < 45: tips.append("🔧 Increase bottom-flow > 45 %.")
if row['ventilator_speed'] < 40: tips.append("🔧 Boost ventilator > 40 rpm.")
if row['gas_loop_oxygen'] > 10:  tips.append("🔧 Purge chamber (O₂ < 10 ppm).")

st.warning(" ".join(tips)) if tips else st.success("✅ Parameters within range.")

# ------------------------------------------------------------  GRAD-CAM  ----------
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layer = model.cnn.layer4[-1]
    cam = GradCAM(model, [target_layer], device="cpu")
    heat = cam(x_img, extra_forward_args=(x_vec,))[0]

    rgb  = np.transpose(x_img.squeeze().numpy(), (1, 2, 0))
    st.image(show_cam_on_image(rgb, heat, use_rgb=True),
             caption="Grad-CAM", width=240)
except Exception as e:
    st.info(f"Grad-CAM unavailable ({e})")

# ------------------------------------------------------------  LIME --------------
with st.expander("🔍 Show LIME tabular explanation"):
    try:
        from lime import lime_tabular

        explainer = lime_tabular.LimeTabularExplainer(
            training_data         = param_df.values,
            feature_names         = list(param_df.columns),
            class_names           = list(LABEL_MAP.values()),
            mode                  = "classification",
            discretize_continuous = True
        )

        exp = explainer.explain_instance(
            data_row    = param_df.loc[layer].values,
            predict_fn  = lambda X: torch.softmax(
                model(
                    torch.zeros((X.shape[0], 3, 224, 224)),             # dummy imgs
                    torch.tensor(scaler.transform(X), dtype=torch.float32)
                ), 1
            ).detach().numpy(),
            num_features = 10
        )

        st.components.v1.html(exp.as_html(), height=420, scrolling=True)
    except Exception as e:
        st.warning(f"LIME explanation unavailable ({e})")


