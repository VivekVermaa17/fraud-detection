import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle

# ── Load data ────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("creditcard.csv")
print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} / {len(df)} ({df['Class'].mean()*100:.3f}%)")

# ── Features & target ────────────────────────────────────────────
X = df.drop(columns=["Class"])
y = df["Class"]

# ── Scale Amount and Time ────────────────────────────────────────
scaler = StandardScaler()
X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

# ── Train/test split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── SMOTE to fix imbalance ───────────────────────────────────────
print("Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE — Fraud: {y_train_res.sum()} / Non-fraud: {(y_train_res==0).sum()}")

# ── Train XGBoost ────────────────────────────────────────────────
print("Training XGBoost...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train_res, y_train_res)

# ── Evaluate ─────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")

# ── Save model & scaler ──────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nmodel.pkl and scaler.pkl saved!")