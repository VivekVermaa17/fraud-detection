import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load model & scaler ──────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write(
    "This app uses an XGBoost model (trained with SMOTE on 284K real transactions) "
    "to predict whether a transaction is **fraudulent** or **legitimate**."
)
st.caption("Test AUC-ROC: 0.976 · Fraud Recall: 87%")

st.divider()

# ── Load sample transactions ──────────────────────────────────────
df = pd.read_csv("sample_transactions.csv")

if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0

col_a, col_b = st.columns([3, 1])
with col_a:
    st.subheader("Step 1 — Pick a transaction")
with col_b:
    if st.button("🔁 New Sample"):
        st.session_state.sample_idx = np.random.randint(0, len(df))

sample = df.iloc[[st.session_state.sample_idx]]
actual_label = sample["Class"].values[0]

base_features = sample.drop(columns=["Class"]).iloc[0]

# ── Step 2 — Adjust meaningful fields ─────────────────────────────
st.subheader("Step 2 — Adjust transaction details")
st.write("Try changing the amount or time to see how the model's prediction changes. "
         "The other 28 features (V1–V28) come from this real transaction and are "
         "anonymized via PCA — they don't have individual real-world meaning, "
         "but together they encode the transaction's underlying pattern.")

col1, col2 = st.columns(2)
amount_val = col1.number_input("Amount (€)", value=float(base_features["Amount"]), min_value=0.0, step=10.0)
time_val = col2.number_input("Time (seconds since first transaction)", value=float(base_features["Time"]), step=1000.0)

with st.expander("View underlying anonymized features (V1–V28)"):
    v_df = base_features.drop(["Time", "Amount"]).to_frame().T
    st.dataframe(v_df)

st.divider()

# ── Predict ───────────────────────────────────────────────────────
if st.button("🔍 Predict", type="primary"):
    input_row = base_features.copy()
    input_row["Amount"] = amount_val
    input_row["Time"] = time_val
    input_data = pd.DataFrame([input_row])[FEATURE_ORDER]

    scaled_input = input_data.copy()
    scaled_input[["Amount", "Time"]] = scaler.transform(scaled_input[["Amount", "Time"]])
    scaled_input = scaled_input[FEATURE_ORDER]

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("Result")

    st.write("**Fraud probability:**")
    st.progress(float(probability))
    st.write(f"{probability*100:.2f}%")

    if probability < 0.3:
        st.success("✅ Low Risk — Approve")
    elif probability < 0.7:
        st.warning("⚠️ Medium Risk — Step-up Verification (OTP)")
        st.caption("In production: an OTP would be sent to the cardholder's registered phone to confirm this transaction before approval.")
    else:
        st.error("🚫 High Risk — Hold for Manual Review")
        st.caption("In production: transaction would be paused and flagged for the fraud team to investigate.")

    st.caption(f"Actual label for this real transaction: {'Fraud' if actual_label == 1 else 'Legit'}")

st.divider()
st.caption("Model: XGBoost + SMOTE on Kaggle Credit Card Fraud dataset")