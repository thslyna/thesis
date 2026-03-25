import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ===============================
# PATHS
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"
MODEL_PATH = "/Users/lyna.thc/Desktop/thesis/results/rf_tuned/rf_delay_model.joblib"

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)

y = df["delayed_15"]

DROP_COLS = [
    "delayed_15",
    "flight_date",
    "hour",
    "flight_iata",
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# SAME SPLIT AS BEFORE
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load(MODEL_PATH)

# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# ===============================
# BOOTSTRAP FUNCTION
# ===============================
def bootstrap_auc(y_true, y_prob, n_bootstraps=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    scores = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))

        if len(np.unique(y_true.iloc[indices])) < 2:
            continue

        score = roc_auc_score(
            y_true.iloc[indices],
            y_prob[indices]
        )
        scores.append(score)

    scores = np.array(scores)

    mean_auc = scores.mean()
    std_auc = scores.std()
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return mean_auc, std_auc, lower, upper

# ===============================
# RUN BOOTSTRAP
# ===============================
mean_auc, std_auc, lower_ci, upper_ci = bootstrap_auc(
    y_test.reset_index(drop=True),
    y_prob
)

print("\n📊 Bootstrap Results (Tuned Random Forest)")
print("Mean AUC:", round(mean_auc, 4))
print("Std AUC:", round(std_auc, 4))
print("95% CI:", f"[{lower_ci:.4f}, {upper_ci:.4f}]")