import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from lime import lime_tabular

STATE_PATH = "small_vlm_state_fixed.pt"
CSV_PATH = "param_df_cleaned.csv"
LABEL_MAP = {0: "Powder (empty bed)", 1: "Printed region (healthy)"}

# ───────────────────────────────────────── Model Definition ──────────────────────────────────────
class TabularMLP(nn.Module):
    def __init__(self, n_params:int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_params,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, vec):
        return self.mlp(vec)

@st.cache_resource
def load_data():
    df = pd.read_csv(CSV_PATH)
    scaler = StandardScaler().fit(df.values)
    model = TabularMLP(df.shape[1])
    model.load_state_dict(torch.load(STATE_PATH, map_location="cpu"))
    model.eval()
    return model, scaler, df

model, scaler, param_df = load_data()

st.title("Tabular Defect Classifier + LIME")

# ───────────────────────────────────────── User Input & Prediction ─────────────────────────────
layer_idx = st.number_input("Layer index", min_value=0, max_value=len(param_df)-1, value=0, step=1)
row = param_df.loc[layer_idx].values.reshape(1, -1)
scaled = scaler.transform(row).astype("float32")
x_vec = torch.tensor(scaled)

with torch.no_grad():
    probs = torch.softmax(model(x_vec), dim=1).numpy()
pred = np.argmax(probs, axis=1)[0]
conf = probs[0, pred] * 100
st.markdown(f"**Prediction:** {LABEL_MAP[pred]} — {conf:.1f}% confidence")

st.write("**Parameter values:**")
st.json({name: float(val) for name, val in zip(param_df.columns, row.flatten())})

# ───────────────────────────────────────── LIME Explanation ──────────────────────────────────────
with st.expander("🔍 Show LIME explanation"):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=param_df.values,
        feature_names=list(param_df.columns),
        class_names=list(LABEL_MAP.values()),
        mode="classification",
        discretize_continuous=True
    )
    exp = explainer.explain_instance(
        data_row=row.flatten(),
        predict_fn=lambda X: torch.softmax(
            model(torch.tensor(scaler.transform(X).astype("float32"))), dim=1
        ).numpy(),
        num_features=8,
        num_samples=500
    )
    st.components.v1.html(exp.as_html(), height=400, scrolling=True)
