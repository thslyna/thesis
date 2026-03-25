import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)

# ===============================
# PATHS
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/v1_full_evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

# ===============================
# TARGET
# ===============================
y = df["delayed_15"]

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

print("Numeric:", numeric_features)
print("Categorical:", categorical_features)

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
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# ===============================
# DEFINE MODELS
# ===============================
models = {

    # Logistic Regression
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    ),

    # Random Forest (untuned)
    "Random Forest (untuned)": RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),

    # Random Forest (tuned - YOUR REAL PARAMS)
    "Random Forest (tuned)": RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="log2",
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    # XGBoost (untuned)
    "XGBoost (untuned)": XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    ),

    # XGBoost (tuned - YOUR REAL PARAMS)
    "XGBoost (tuned)": XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=1,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    ),
}

# ===============================
# EVALUATION LOOP
# ===============================
results = []

print("\nMODEL PERFORMANCE SUMMARY")
print("=" * 80)

for name, model in models.items():

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model)
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)

    results.append({
        "model": name,
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "brier_score": brier
    })

    # Print LaTeX-ready row
    print(f"{name:30s} & {auc:.3f} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {brier:.3f} \\\\")

# ===============================
# SAVE RESULTS
# ===============================
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, "v1_model_comparison.csv"), index=False)

with open(os.path.join(OUT_DIR, "v1_model_comparison.json"), "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to:", OUT_DIR)