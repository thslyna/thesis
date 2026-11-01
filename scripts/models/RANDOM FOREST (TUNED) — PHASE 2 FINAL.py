# ===============================
# RANDOM FOREST (TUNED) — PHASE 2 FINAL
# ===============================

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay

# ===============================
# PATHS
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_v1_IMPUTED_FIXED.csv"
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/rf_tuned_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)

y = df["delayed_15"]

DROP_COLS = ["delayed_15", "flight_date", "hour", "flight_iata"]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ===============================
# PREPROCESSING
# ===============================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", rf)
])

param_dist = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [None, 30],
    "model__min_samples_split": [2, 5],
    "model__max_features": ["sqrt", "log2"],
}

# ===============================
# SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ===============================
# TRAIN
# ===============================
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
model = search.best_estimator_

# ===============================
# EVAL
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
print("RF AUC:", auc)

# ✅ SAVE PREDICTIONS (CRITICAL)
pd.DataFrame({
    "y_true": y_test.values,
    "y_prob": y_prob
}).to_csv(os.path.join(OUT_DIR, "rf_tuned_predictions.csv"), index=False)

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, os.path.join(OUT_DIR, "rf_model.joblib"))