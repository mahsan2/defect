# ------------------------------------------------------------
#  Small-VLM Streamlit app
# ------------------------------------------------------------
import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np
from PIL import Image

STATE_PATH = "small_vlm_state_fixed.pt"       # weights-only
CSV_PATH   = "param_df_cleaned.csv"           # 1 814 rows
LABEL_MAP  = {0: "Powder (empty bed)", 1: "Printed region (healthy)"}

# ─────────────────── model ───────────────────
class SmallVLM(nn.Module):
    def __init__(self, n_params: int):
        super().__init__()
        res = models.resnet18(weights=None)
        res.fc = nn.Identity()
        self.cnn = res
        self.mlp = nn.Sequential(
            nn.Linear(n_params, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, img, vec):
        feats = torch.cat((self.cnn(img), self.mlp(vec)), 1)
        return self.classifier(feats)

@st.cache_resource
def load_all():
    df     = pd.read_csv(CSV_PATH)
    scaler = StandardScaler().fit(df.values)
    model  = SmallVLM(df.shape[1])
    model.load_state_dict(torch.load(STATE_PATH, map_location="cpu"))
    model.eval()
    return model, scaler, df

model, scaler, param_df = load_all()
tfm = transforms.Compose([transforms.Resize((224,224)),
                          transforms.ToTensor()])

# ─────────────────── UI ───────────────────
st.title("Multimodal Defect Classifier")

up = st.file_uploader("Upload slice image (e.g. **5_slice_0.png**)",
                      type=["png","jpg","jpeg"])
if not up:
    st.stop()

img = Image.open(up).convert("RGB")
st.image(img, width=256)

# --- parse layer number from filename ---
try:
    layer_idx = int(up.name.split("_")[0])
except ValueError:
    st.error("Filename must start with layer number")
    st.stop()
if layer_idx >= len(param_df):
    st.error(f"No tabular data for layer {layer_idx}")
    st.stop()

# --- prediction -------------------------
x_img = tfm(img).unsqueeze(0)
row_np = param_df.iloc[layer_idx].values.reshape(1, -1)
x_vec = torch.tensor(scaler.transform(row_np), dtype=torch.float32)

with torch.no_grad():
    prob = F.softmax(model(x_img, x_vec), 1)
conf, pred = prob.max(1)

st.markdown(f"### **{LABEL_MAP[pred.item()]}** — {conf.item()*100:.1f}%")

# --- show parameters + simple rule tips --
row = param_df.iloc[layer_idx]
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

# ─────────────────── LIME (optional) ───────────────────
try:
    from lime import lime_tabular

    with st.expander("🔍 Show LIME tabular explanation"):
        @st.cache_resource
        def _get_explainer():
            return lime_tabular.LimeTabularExplainer(
                training_data         = param_df.values,
                feature_names         = list(param_df.columns),
                class_names           = list(LABEL_MAP.values()),
                mode                  = "classification",
                discretize_continuous = True
            )

        explainer = _get_explainer()

        def _predict_fn(X_raw: np.ndarray):
            X_scaled = scaler.transform(X_raw).astype("float32")
            v = torch.tensor(X_scaled)
            dummies = torch.zeros((v.shape[0], 3, 224, 224))
            with torch.no_grad():
                p = F.softmax(model(dummies, v), 1).cpu().numpy()
            return p   # shape (n,2)

        exp = explainer.explain_instance(
            data_row     = row.values,
            predict_fn   = _predict_fn,
            num_features = 10,
            num_samples  = 1000
        )

        st.components.v1.html(exp.as_html(), height=420, scrolling=True)

except Exception as e:
    st.info(f"LIME unavailable ({e})")






