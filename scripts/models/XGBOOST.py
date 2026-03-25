import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from xgboost import XGBClassifier

# ===============================
# PATHS
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_v1_IMPUTED_FIXED.csv"
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/xgb_untuned"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)
print("📦 Dataset loaded")
print("Shape:", df.shape)

# ===============================
# TARGET
# ===============================
y = df["delayed_15"]

print("\n🎯 Target distribution:")
print(y.value_counts(normalize=True))

# ===============================
# ⚖️ HANDLE IMBALANCE (🔥 KEY FIX)
# ===============================
neg = (y == 0).sum()
pos = (y == 1).sum()
scale_pos_weight = neg / pos

print(f"\n⚖️ scale_pos_weight: {scale_pos_weight:.3f}")

# ===============================
# DROP LEAKY / NON-PREDICTIVE FEATURES
# ===============================
DROP_COLS = [
    "delayed_15",
    "flight_date",
    "hour",
    "flight_iata",
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ===============================
# FEATURE TYPES
# ===============================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("🧱 Numeric features:", numeric_features)
print("🧱 Categorical features:", categorical_features)

# ===============================
# PREPROCESSING
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ===============================
# MODEL (UNTUNED BUT FIXED)
# ===============================
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,  # 🔥 FIX
    random_state=42,
    tree_method="hist",
    n_jobs=-1
)

# ===============================
# PIPELINE
# ===============================
pipe = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", xgb)
    ]
)

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

print("📊 Train size:", X_train.shape)
print("📊 Test size:", X_test.shape)

# ===============================
# TRAIN
# ===============================
print("\n🚀 Training XGBoost (Untuned)...")
pipe.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

# 🔥 DEBUG (VERY IMPORTANT)
print("\n🧠 Prediction distribution:")
print(pd.Series(y_pred).value_counts(normalize=True))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", round(auc, 4))

# ===============================
# SAVE METRICS
# ===============================
metrics = {
    "roc_auc": auc,
    "model": "XGBoost (untuned)",
}

with open(os.path.join(OUT_DIR, "xgb_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – XGBoost (Untuned)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close()

# ===============================
# ROC CURVE
# ===============================
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – XGBoost (Untuned)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150)
plt.close()

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(pipe, os.path.join(OUT_DIR, "xgb_untuned_model.joblib"))

print("\n✅ XGBoost (Untuned) training finished")
print("📁 Results saved to:", OUT_DIR)