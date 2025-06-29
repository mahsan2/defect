# ------------------------------------------------------------
#  Small-VLM Streamlit app  –  image + tabular + Grad-CAM + LIME
# ------------------------------------------------------------
import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
import pandas as pd
from PIL import Image
import numpy as np

STATE_PATH = "small_vlm_state_fixed.pt"   # weights-only (~45 MB) with *classifier.* keys
CSV_PATH   = "param_df_cleaned.csv"       # tabular process data
LABEL_MAP  = {0: "Powder (empty bed)", 1: "Printed region (healthy)"}

# ─────────────────────────── model ───────────────────────────
class SmallVLM(nn.Module):
    def __init__(self, n_params: int):
        super().__init__()
        back = models.resnet18(weights="IMAGENET1K_V1")
        back.fc = nn.Identity()           # 512-D image feature
        self.cnn = back
        self.mlp = nn.Sequential(
            nn.Linear(n_params, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.classifier = nn.Sequential(  # keep name == checkpoint
            nn.Linear(512 + 32, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, img, vec):
        z = torch.cat((self.cnn(img), self.mlp(vec)), 1)
        return self.classifier(z)

@st.cache_resource(show_spinner="⏳ Loading model & data …")
def load_all():
    df = pd.read_csv(CSV_PATH).reset_index(drop=True)
    scaler = StandardScaler().fit(df.values)
    model = SmallVLM(df.shape[1])
    model.load_state_dict(torch.load(STATE_PATH, map_location="cpu"), strict=True)
    model.eval()
    return model, scaler, df

model, scaler, df = load_all()
tform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# ─────────────────────────── UI ──────────────────────────────
st.title("Multimodal Defect Classifier (Small-VLM)")

f = st.file_uploader("Upload slice image (**<layer>_something.png**)", ["png","jpg","jpeg"])
if not f:
    st.stop()

img = Image.open(f).convert("RGB")
st.image(img, width=256)

try:
    layer = int(f.name.split("_")[0])
except Exception:
    st.error("Filename must begin with layer number and underscore.")
    st.stop()

if layer >= len(df):
    st.error(f"No tabular data for layer {layer}")
    st.stop()

x_img = tform(img).unsqueeze(0)
x_vec = torch.tensor(scaler.transform(df.loc[layer].values.reshape(1,-1)), dtype=torch.float32)

with torch.no_grad():
    p = torch.softmax(model(x_img, x_vec), 1)
    conf, pred = p.max(1)

label = LABEL_MAP[pred.item()]
st.markdown(f"### **{label}** — {conf.item()*100:.1f}%")

row = df.loc[layer]
st.write(f"""
**Process parameters**

• Top-chamber T : **{row['top_chamber_temperature']:.0f} °C**  
• Bottom flow  : **{row['bottom_flow_rate']:.1f} %**  
• Ventilator  : **{row['ventilator_speed']:.0f} rpm**  
• O₂       : **{row['gas_loop_oxygen']:.1f} ppm**
""")

tips = []
if row['bottom_flow_rate'] < 45: tips.append("🔧 Increase bottom-flow > 45 %.")
if row['ventilator_speed'] < 40: tips.append("🔧 Boost ventilator > 40 rpm.")
if row['gas_loop_oxygen'] > 10: tips.append("🔧 Purge chamber (O₂ < 10 ppm).")
st.warning(" ".join(tips)) if tips else st.success("✅ Parameters within range.")

# ───────────────────── Grad-CAM (image) ─────────────────────
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    cam = GradCAM(model, [model.cnn.layer4[-1]], device="cpu")
    heat = cam(x_img, extra_forward_args=(x_vec,))[0]
    rgb  = np.transpose(x_img.squeeze().numpy(), (1,2,0))
    st.image(show_cam_on_image(rgb, heat, use_rgb=True), caption="Grad-CAM", width=240)
except Exception as e:
    st.info(f"Grad-CAM unavailable ({e})")

# ───────────────────── LIME (tabular) ───────────────────────
with st.expander("🔍 Show LIME tabular explanation"):
    from lime import lime_tabular
    explainer = lime_tabular.LimeTabularExplainer(
        training_data          = df.values,
        feature_names          = list(df.columns),
        class_names            = list(LABEL_MAP.values()),
        discretize_continuous  = True,
        mode                   = "classification"
    )
    exp = explainer.explain_instance(
        data_row   = df.loc[layer].values,
        predict_fn = lambda X: torch.softmax(
            model(
                torch.zeros((X.shape[0], 3, 224, 224)),                  # dummy images
                torch.tensor(scaler.transform(X), dtype=torch.float32)
            ), 1).detach().numpy(),
        num_features = 10
    )
    st.components.v1.html(exp.as_html(), height=420)

