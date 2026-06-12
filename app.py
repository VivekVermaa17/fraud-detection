import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load model & scaler ──────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("This app predicts whether a credit card transaction is **fraudulent** or **legitimate** using an XGBoost model trained on real transaction data.")

st.divider()

# ── Input mode ────────────────────────────────────────────────────
mode = st.radio("Choose input method:", ["Use sample transaction", "Enter custom values"])

if mode == "Use sample transaction":
    st.info("Using a random sample row from the dataset for demonstration.")
    df = pd.read_csv("creditcard.csv")
    sample = df.sample(1)
    st.write("Sample transaction (without label):")
    st.dataframe(sample.drop(columns=["Class"]))
    input_data = sample.drop(columns=["Class"])
    actual_label = sample["Class"].values[0]

else:
    st.write("Enter transaction values:")
    col1, col2 = st.columns(2)
    
    time_val = col1.number_input("Time", value=0.0)
    amount_val = col2.number_input("Amount", value=100.0)
    
    v_values = {}
    cols = st.columns(4)
    for i in range(1, 29):
        v_values[f"V{i}"] = cols[(i-1) % 4].number_input(f"V{i}", value=0.0, format="%.4f")
    
    input_data = pd.DataFrame([{**v_values, "Time": time_val, "Amount": amount_val}])
    # reorder columns to match training order
    input_data = input_data[[f"V{i}" for i in range(1,29)] + ["Time", "Amount"]]
    actual_label = None

st.divider()

# ── Predict ───────────────────────────────────────────────────────
if st.button("🔍 Predict", type="primary"):
    scaled_input = input_data.copy()
    scaled_input[["Amount", "Time"]] = scaler.transform(scaled_input[["Amount", "Time"]])
    
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]
    
    st.subheader("Result")
    
    if prediction == 1:
        st.error(f"⚠️ FRAUD DETECTED — Confidence: {probability*100:.2f}%")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION — Fraud Probability: {probability*100:.2f}%")
    
    if actual_label is not None:
        st.write(f"**Actual label in dataset:** {'Fraud' if actual_label == 1 else 'Legit'}")

st.divider()
st.caption("Model: XGBoost trained with SMOTE on the Kaggle Credit Card Fraud dataset · AUC-ROC: 0.976")