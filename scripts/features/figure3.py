#!/usr/bin/env python3
# ============================================================
# Figure 3 — Probability Distribution (Logistic Regression V1)
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"
PLOT_DIR = "/Users/lyna.thc/Desktop/thesis/results/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

target = "delayed_15"

# Drop same columns as your LR script
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
# FEATURES (IDENTICAL TO YOUR MODEL)
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
# PIPELINE (MATCHES YOUR LR EXACTLY)
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )),
])

# ============================================================
# TRAIN
# ============================================================

print("Training Logistic Regression...")
model.fit(X_train, y_train)

# ============================================================
# PREDICT PROBABILITIES
# ============================================================

y_proba = model.predict_proba(X_test)[:, 1]

probs_ontime = y_proba[y_test == 0]
probs_delay = y_proba[y_test == 1]

# ============================================================
# PLOT (CLEAN + THESIS STYLE)
# ============================================================

plt.figure(figsize=(8, 6))

# On-time flights
sns.histplot(
    probs_ontime,
    bins=20,
    stat="density",
    kde=True,
    label="On-time flights",
    alpha=0.5
)

# Delayed flights
sns.histplot(
    probs_delay,
    bins=20,
    stat="density",
    kde=True,
    label="Delayed flights",
    alpha=0.5
)

# Labels
plt.xlabel("Predicted Probability of Delay", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Predicted Probability Distribution — Logistic Regression", fontsize=14)

plt.legend(fontsize=11)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

save_path = os.path.join(PLOT_DIR, "fig3_probability_distribution.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"\nSaved Figure 3 at: {save_path}")
print("Done ✅")