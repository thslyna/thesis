import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
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

# ===============================
# PATHS (✅ KEEP YOUR DATASET)
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/rf_tuned"
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
# DROP NON-PREDICTIVE / LEAKY FEATURES
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
# MODEL (🔥 FIX IMBALANCE HERE)
# ===============================
rf = RandomForestClassifier(
    class_weight="balanced",   # 🔥 THIS is the key fix
    random_state=42,
    n_jobs=-1
)

# ===============================
# PIPELINE
# ===============================
pipe = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", rf)
    ]
)

# ===============================
# HYPERPARAMETER SPACE
# ===============================
param_dist = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
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

print("\n🚀 Starting Random Forest tuning...")
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

# 🔥 CRITICAL DEBUG
print("\n🧠 Prediction distribution:")
print(pd.Series(y_pred).value_counts(normalize=True))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", round(auc, 3))

# ===============================
# SAVE PREDICTIONS
# ===============================
pd.DataFrame({
    "y_true": y_test.values,
    "y_prob": y_prob
}).to_csv(os.path.join(OUT_DIR, "rf_test_predictions.csv"), index=False)

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Tuned RF")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close()

# ===============================
# ROC CURVE
# ===============================
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve – Tuned RF")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
plt.close()

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(best_model, os.path.join(OUT_DIR, "rf_model.joblib"))

print("\n✅ Done. Results saved to:", OUT_DIR)