#!/usr/bin/env python3
# ============================================================
# Logistic Regression — V1 Baseline (WITH MODEL SAVING)
# ============================================================

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

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/logreg"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "logreg_model.joblib")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)
target = "delayed_15"

# Drop non-predictive / leaky columns
DROP_COLS = [
    "flight_date",
    "hour",
    "flight_iata",
    "flight_number",
    "flight_icao",
    "collected_at",
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ============================================================
# FEATURES
# ============================================================

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

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ============================================================
# PIPELINE
# ============================================================

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
            class_weight="balanced"
        )),
    ]
)

# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)

# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", round(roc_auc, 4))

# ============================================================
# SAVE MODEL  ⭐⭐ IMPORTANT ⭐⭐
# ============================================================

joblib.dump(model, MODEL_PATH)
print(f"\n💾 Model saved to: {MODEL_PATH}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

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

# ============================================================
# ROC CURVE
# ============================================================

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

# ============================================================
# FEATURE IMPORTANCE (COEFFICIENTS)
# ============================================================

feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefs = model.named_steps["clf"].coef_[0]

coef_df = (
    pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs)
    })
    .sort_values("abs_coef", ascending=False)
    .head(15)
)

plt.figure(figsize=(9, 6))
sns.barplot(x="coef", y="feature", data=coef_df)
plt.axvline(0, color="black")
plt.title("Top Logistic Regression Coefficients")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))
sns.histplot(y_proba[y_test == 0], bins=30, stat="density", label="On Time")
sns.histplot(y_proba[y_test == 1], bins=30, stat="density", label="Delayed")
plt.xlabel("Predicted Delay Probability")
plt.title("Predicted Probability Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "probability_distribution.png"), dpi=150)
plt.close()

print("\n✅ Logistic Regression V1 finished successfully.")
print("📁 Results saved in:", OUT_DIR)