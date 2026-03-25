import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
# PATHS (KEEP YOUR DATASET)
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/xgb_tuned"
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
# ⚖️ HANDLE CLASS IMBALANCE (🔥 KEY)
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
# XGBOOST MODEL (FIXED)
# ===============================
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,  # 🔥 FIX IMBALANCE
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
# HYPERPARAMETER SPACE
# ===============================
param_dist = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [3, 5, 7, 9],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__gamma": [0, 0.5, 1],
    "model__min_child_weight": [1, 3, 5],
}

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
# RANDOMIZED SEARCH
# ===============================
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring="roc_auc",
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\n🚀 Starting XGBoost tuning...")
search.fit(X_train, y_train)

print("\n🏆 Best parameters:")
print(search.best_params_)

# ===============================
# FINAL MODEL
# ===============================
best_model = search.best_estimator_

# ===============================
# EVALUATION
# ===============================
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

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
    "best_params": search.best_params_
}

with open(os.path.join(OUT_DIR, "xgb_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Tuned XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close()

# ===============================
# ROC CURVE
# ===============================
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve – Tuned XGBoost")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
plt.close()

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(best_model, os.path.join(OUT_DIR, "xgb_tuned_model.joblib"))

print("\n✅ Tuned XGBoost finished")
print("📁 Results saved to:", OUT_DIR)