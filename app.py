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

#---------------------------------------------------------
# ─────────── 1. extra import ───────────
from lime import lime_tabular      #  ← NEW
# ─────────── 2. cache a single LimeTabularExplainer ───────────
@st.cache_resource(ttl=3600, show_spinner=False)
def _lime_explainer():
    return lime_tabular.LimeTabularExplainer(
        training_data         = param_df.values,          # raw, unscaled
        feature_names         = list(param_df.columns),
        class_names           = list(LABEL_MAP.values()),
        mode                  = "classification",
        discretize_continuous = True,                     # makes plots nicer
    )
# ─────────── 3. show explanation in an expander ───────────
with st.expander("🔍  Show LIME tabular explanation"):
    # --- prediction fn that LIME will call many times -------------
    def _lime_predict(X: np.ndarray) -> np.ndarray:         # X shape  (n,20)
        X_scaled = scaler.transform(X).astype("float32")
        dummy_img = torch.zeros((X.shape[0], 3, 224, 224))  # keep image branch neutral
        with torch.no_grad():
            logits = model(dummy_img, torch.tensor(X_scaled))
            return torch.softmax(logits, 1).cpu().numpy()   # (n,2)

    # --- run LIME on the *current* layer --------------------------
    exp = _lime_explainer().explain_instance(
        data_row     = row.values,      # raw, unscaled (shape 20,)
        predict_fn   = _lime_predict,
        num_features = 8,               # top-k bars to show
        num_samples  = 1000             # perturbations LIME will generate
    )

    # --- render nicely inside Streamlit ---------------------------
    st.components.v1.html(exp.as_html(), height=440, scrolling=True)

