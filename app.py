import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Load model bundle ─────────────────────────────────────────
@st.cache_resource
def load_bundle():
    with open("best_model_deployment.pkl", "rb") as f:
        return pickle.load(f)

bundle    = load_bundle()
model     = bundle["model"]
features  = bundle["features"]
threshold = bundle["optimal_threshold"]
metrics   = bundle["metrics"]
label_map = bundle["label_map"]

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detector", page_icon="🔍", layout="wide")
st.title("🔍 Credit Card Fraud Detection")
st.caption(f"Model: {bundle['model_name']}  |  Optimal threshold: {threshold:.2f}")

# ── Sidebar — model metrics ───────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    for k, v in metrics.items():
        st.metric(k, f"{v:.4f}")
    st.divider()
    st.info(f"**Decision threshold:** {threshold:.2f}\n\nAdjusted via F1 optimisation (Cell 10 of notebook).")

# ── Input form ────────────────────────────────────────────────
st.subheader("Enter Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    amt         = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=1.0)
    hour        = st.slider("Hour of Day", 0, 23, 14)
    day_of_week = st.selectbox("Day of Week", list(range(7)),
                               format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    month       = st.slider("Month", 1, 12, 6)

with col2:
    age         = st.number_input("Cardholder Age", min_value=18, max_value=100, value=35)
    distance_km = st.number_input("Distance to Merchant (km)", min_value=0.0, value=10.0, step=1.0)
    city_pop    = st.number_input("City Population", min_value=100, value=50000, step=1000)

with col3:
    category_enc = st.selectbox("Merchant Category (encoded)", list(range(14)))
    state_enc    = st.selectbox("State (encoded)", list(range(50)))

# Derived features
log_amt    = np.log1p(amt)
is_night   = int(hour < 6 or hour >= 22)
is_weekend = int(day_of_week >= 5)

input_data = pd.DataFrame([[amt, log_amt, hour, day_of_week, month, age,
                             distance_km, city_pop, is_night, is_weekend,
                             category_enc, state_enc]], columns=features)

# ── Prediction ────────────────────────────────────────────────
st.divider()
if st.button("🔎 Predict", use_container_width=True, type="primary"):
    proba      = model.predict_proba(input_data)[0, 1]
    prediction = int(proba >= threshold)
    label      = label_map[prediction]

    if prediction == 1:
        st.error(f"🚨 **{label}** — Fraud probability: `{proba:.2%}`")
    else:
        st.success(f"✅ **{label}** — Fraud probability: `{proba:.2%}`")

    st.progress(float(proba), text=f"Fraud score: {proba:.2%}")

    with st.expander("Show input features"):
        st.dataframe(input_data, use_container_width=True)
