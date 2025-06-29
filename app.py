# ------------------------------------------------------------
#  Small-VLM Streamlit app  –  minimal version (no LIME / CAM)
# ------------------------------------------------------------
import streamlit as st
import torch
import torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
import pandas as pd
from PIL import Image
import numpy as np

STATE_PATH = "small_vlm_state_fixed.pt"   # your weights-only file
CSV_PATH   = "param_df_cleaned.csv"
LABEL_MAP  = {0: "Powder (empty bed)", 1: "Printed region (healthy)"}

# ─────────────────────────────────────────────────────────────
class SmallVLM(nn.Module):
    def __init__(self, n_params: int):
        super().__init__()
        res = models.resnet18(weights="IMAGENET1K_V1")
        res.fc = nn.Identity()                # 512-D global image feature
        self.cnn = res

        self.mlp = nn.Sequential(
            nn.Linear(n_params, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )

        #  name MUST be “classifier” -> matches checkpoint keys
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, img, vec):
        feats = torch.cat((self.cnn(img), self.mlp(vec)), dim=1)
        return self.classifier(feats)

@st.cache_resource(show_spinner="🔄 Loading model & data …")
def load_all():
    df      = pd.read_csv(CSV_PATH).reset_index(drop=True)
    scaler  = StandardScaler().fit(df.values)
    model   = SmallVLM(df.shape[1])
    model.load_state_dict(torch.load(STATE_PATH, map_location="cpu"),
                          strict=True)
    model.eval()
    return model, scaler, df

model, scaler, param_df = load_all()
tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# ─────────────────────────────────────────────────────────────
st.title("Multimodal Defect Classifier (Small-VLM)")

file = st.file_uploader(
    "Upload slice image (filename must start with layer number, e.g. **5_slice_0.png**)",
    type=["png", "jpg", "jpeg"]
)
if not file:
    st.stop()

# ------------------------------------------------------------
img = Image.open(file).convert("RGB")
st.image(img, width=256)

try:
    layer = int(file.name.split("_")[0])
except Exception:
    st.error("Filename must start with layer number followed by '_'")
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
    p = F.softmax(model(x_img, x_vec), 1)
    conf, pred = p.max(1)

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
if tips:
    st.warning(" ".join(tips))
else:
    st.success("✅ Parameters within range.")




