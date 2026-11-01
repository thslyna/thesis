#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# ===============================
# PATHS
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_v1_IMPUTED_FIXED.csv"

OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/logreg_v2"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "logreg_model.joblib")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)
target = "delayed_15"

# ===============================
# DROP NON-PREDICTIVE
# ===============================
DROP_COLS = [
    "flight_date",
    "hour",
    "flight_iata",
    "flight_number",
    "flight_icao",
    "collected_at",
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ===============================
# FEATURES
# ===============================
numeric_features = [
    "hour_of_day",
    "day_of_week",
    "temp",
    "humidity",
    "pressure",
    "wind_speed",
    "road_avg_speed",
    "road_vehicle_count",
    "road_congestion_index",
    "prev_hour_delay_rate",
    "avg_delay_last_24h",
    "airline_7d_delay_rate",
]

categorical_features = [
    "airline_name",
    "dep_iata",
    "arr_iata",
    "weather",
]

X = df[numeric_features + categorical_features]
y = df[target]

# ===============================
# SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ===============================
# PIPELINE (IMPORTANT — KEEP THIS)
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"   # 🔥 THIS was missing before
        )),
    ]
)

# ===============================
# TRAIN
# ===============================
model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_proba)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", round(roc_auc, 4))

# ===============================
# ✅ SAVE PREDICTIONS (CRITICAL)
# ===============================
pd.DataFrame({
    "y_true": y_test.values,
    "y_prob": y_proba
}).to_csv(os.path.join(OUT_DIR, "lr_predictions.csv"), index=False)

print("📁 Predictions saved → lr_predictions.csv")

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, MODEL_PATH)

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred On Time", "Pred Delayed"],
    yticklabels=["Actual On Time", "Actual Delayed"]
)
plt.title("Confusion Matrix – Logistic Regression")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ===============================
# ROC CURVE
# ===============================
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150)
plt.close()

print("\n✅ Logistic Regression Phase 2 finished")