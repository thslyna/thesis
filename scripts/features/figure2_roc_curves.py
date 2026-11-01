import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, roc_auc_score

# ===============================
# PATHS
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"
PLOT_DIR = "/Users/lyna.thc/Desktop/thesis/results/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

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
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    ),

    "RF (untuned)": RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),

    "RF (tuned)": RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="log2",
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "XGB (untuned)": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ),

    "XGB (tuned)": XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=1,
        random_state=42,
        eval_metric="logloss"
    ),
}

# ===============================
# PLOT ROC CURVES
# ===============================
plt.figure(figsize=(8, 6))

for name, model in models.items():
    print(f"Training {name}...")

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        linewidth=1.8,
        label=f"{name} (AUC = {auc:.3f})"
    )

# ===============================
# RANDOM BASELINE
# ===============================
plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    linewidth=1.5,
    color="gray",
    alpha=0.7
)

# ===============================
# STYLE
# ===============================
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves — Model Comparison (Dataset V1)", fontsize=14)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.legend(fontsize=10, loc="lower right")

plt.tight_layout()

# ===============================
# SAVE
# ===============================
save_path = os.path.join(PLOT_DIR, "fig2_roc_overlay.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Saved Figure 2 at: {save_path}")