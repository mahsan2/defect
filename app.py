# ------------------------------------------------------------
#  Multimodal Defect Classifier  —  small-VLM (image + tabular)
# ------------------------------------------------------------
import streamlit as st, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np
from PIL import Image

# ─────────── files you already committed ───────────
STATE_PATH = "small_vlm_state_fixed.pt"      # weighs ≈45 MB, has cnn.* keys
CSV_PATH   = "param_df_cleaned.csv"          # 1 814 rows × 20 features
LABEL_MAP  = {0: "Powder (empty bed)", 1: "Printed region (healthy)"}

# ─────────── model definition ───────────
class SmallVLM(nn.Module):
    def __init__(self, n_par: int):
        super().__init__()
        rn18 = models.resnet18(weights="IMAGENET1K_V1")
        rn18.fc = nn.Identity()                          # 512-D image feature
        self.cnn = rn18

        self.mlp = nn.Sequential(
            nn.Linear(n_par, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.classifier = nn.Sequential(                 # ←  *keep this name!*
            nn.Linear(512 + 32, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, img, vec):
        z = torch.cat((self.cnn(img), self.mlp(vec)), 1)
        return self.classifier(z)

# ─────────── load everything once & cache ───────────
@st.cache_resource(show_spinner="🔄  Loading model & data …")
def load_assets():
    df      = pd.read_csv(CSV_PATH)
    scaler  = StandardScaler().fit(df.values)
    model   = SmallVLM(df.shape[1])

    state   = torch.load(STATE_PATH, map_location="cpu")

    # In case the file still contains  “cls.*”  keys, rename on-the-fly:
    if any(k.startswith("cls.") for k in state):
        state = {k.replace("cls.", "classifier."): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, scaler, df

model, scaler, param_df = load_assets()

tfm = transforms.Compose([transforms.Resize((224, 224)),
                          transforms.ToTensor()])

# ─────────── Streamlit UI ───────────
st.title("Multimodal Defect Classifier  (Small-VLM)")

up = st.file_uploader("Upload slice image (e.g. **5_slice_0.png**)",
                      type=["png", "jpg", "jpeg"])
if not up:
    st.info("👈 Choose an image to start")
    st.stop()

img = Image.open(up).convert("RGB")
st.image(img, width=256)

# --- parse layer number from the filename ---------------------
try:
    layer = int(up.name.split("_")[0])
except ValueError:
    st.error("Filename must start with the layer number (e.g. 5_slice_0.png)")
    st.stop()

if layer >= len(param_df):
    st.error(f"Tabular data has only {len(param_df)} layers – got {layer}")
    st.stop()

# --- prepare tensors & forward pass ---------------------------
x_img = tfm(img).unsqueeze(0)                       # 1 × 3 × 224 × 224
x_vec = torch.tensor(scaler.transform(
            param_df.iloc[layer].values[None]),     # 1 × 20
        dtype=torch.float32)

with torch.no_grad():
    prob = torch.softmax(model(x_img, x_vec), 1)
conf, pred = prob.max(1)

st.markdown(f"### **{LABEL_MAP[pred.item()]}**  —  {conf.item()*100:.1f}%")

# --- show key parameters & rule-based tips --------------------
row = param_df.iloc[layer]
st.write(f"""
**Process parameters**

• Top-chamber T : **{row['top_chamber_temperature']:.0f} °C**  
• Bottom flow   : **{row['bottom_flow_rate']:.1f} %**  
• Ventilator     : **{row['ventilator_speed']:.0f} rpm**  
• O₂             : **{row['gas_loop_oxygen']:.1f} ppm**
""")

tips = []
if row['bottom_flow_rate'] < 45: tips.append("🔧 Increase bottom-flow > 45 %.")
if row['ventilator_speed'] < 40: tips.append("🔧 Boost ventilator > 40 rpm.")
if row['gas_loop_oxygen'] > 10:  tips.append("🔧 Purge chamber (O₂ < 10 ppm).")
if tips:
    st.warning(" ".join(tips))
else:
    st.success("✅ Parameters within range.")


