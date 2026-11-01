import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
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
    brier_score_loss,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ===============================
# PATHS
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"

OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/v1_full_evaluation"
VISU_DIR = "/Users/lyna.thc/Desktop/thesis/visu"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VISU_DIR, exist_ok=True)

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

# ===============================
# MODELS
# ===============================
models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest (untuned)": RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest (tuned)": RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="log2",
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost (untuned)": XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    ),

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
# EVALUATION
# ===============================
results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Metrics
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

    print(f"AUC: {auc:.3f}")

    # ===============================
    # ROC CURVE
    # ===============================
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")

    plt.tight_layout()

    roc_path = os.path.join(VISU_DIR, f"roc_{name.replace(' ', '_')}.png")
    plt.savefig(roc_path)
    plt.close()

    print("Saved:", roc_path)

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()
    plt.title(f"Confusion Matrix - {name}")

    plt.tight_layout()

    cm_path = os.path.join(VISU_DIR, f"cm_{name.replace(' ', '_')}.png")
    plt.savefig(cm_path)
    plt.close()

    print("Saved:", cm_path)

# ===============================
# SAVE RESULTS
# ===============================
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, "v1_model_comparison.csv"), index=False)

with open(os.path.join(OUT_DIR, "v1_model_comparison.json"), "w") as f:
    json.dump(results, f, indent=4)

print("\nAll done. Results + visualizations saved.")