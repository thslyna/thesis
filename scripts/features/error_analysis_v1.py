import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# ===============================
# PATHS
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"
MODEL_PATH = "/Users/lyna.thc/Desktop/thesis/results/logreg/logreg_model.joblib"
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/v1_error_analysis"

os.makedirs(OUT_DIR, exist_ok=True)

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

# ===============================
# FEATURES
# ===============================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# ===============================
# TRAIN / TEST SPLIT (SAME LOGIC AS V1)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load(MODEL_PATH)

# ===============================
# PREDICTIONS
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ===============================
# ERROR TABLE
# ===============================
results = X_test.copy()
results["y_true"] = y_test.values
results["y_pred"] = y_pred
results["y_prob"] = y_prob

def error_type(row):
    if row.y_true == 1 and row.y_pred == 0:
        return "False Negative"
    elif row.y_true == 0 and row.y_pred == 1:
        return "False Positive"
    else:
        return "Correct"

results["error_type"] = results.apply(error_type, axis=1)

results.to_csv(f"{OUT_DIR}/v1_predictions_with_errors.csv", index=False)

# ===============================
# ERROR DISTRIBUTION
# ===============================
results["error_type"].value_counts(normalize=True).to_csv(
    f"{OUT_DIR}/error_type_distribution.csv"
)

# ===============================
# NUMERICAL FEATURE ANALYSIS
# ===============================
FEATURES = [
    "hour_of_day",
    "road_congestion_index",
    "temp",
    "wind_speed",
    "prev_hour_delay_rate"
]

rows = []

for f in FEATURES:
    if f not in results.columns:
        continue
    rows.append({
        "feature": f,
        "correct_mean": results[results.error_type == "Correct"][f].mean(),
        "false_negative_mean": results[results.error_type == "False Negative"][f].mean(),
        "false_positive_mean": results[results.error_type == "False Positive"][f].mean()
    })

pd.DataFrame(rows).to_csv(
    f"{OUT_DIR}/feature_error_comparison.csv",
    index=False
)

# ===============================
# CATEGORICAL ERROR ANALYSIS
# ===============================
for f in ["weather", "airline_name"]:
    if f not in results.columns:
        continue
    pd.crosstab(
        results[f],
        results["error_type"],
        normalize="index"
    ).to_csv(f"{OUT_DIR}/{f}_error_profile.csv")

# ===============================
# HIGH CONFIDENCE ERRORS
# ===============================
hc = results[
    ((results.y_prob > 0.8) & (results.y_true == 0)) |
    ((results.y_prob < 0.2) & (results.y_true == 1))
]

hc.to_csv(f"{OUT_DIR}/high_confidence_errors.csv", index=False)

print("✅ V1 Error analysis completed")
print("📁 Results saved to:", OUT_DIR)