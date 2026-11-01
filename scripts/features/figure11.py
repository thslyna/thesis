import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# PATH
# ===============================
DATA_PATH = "data/processed/flights_v1_IMPUTED_FIXED.csv"
OUT_PATH = "results/fig11_feature_shift.png"
os.makedirs("results", exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)
df["hour"] = pd.to_datetime(df["hour"])
df["month"] = df["hour"].dt.month

# ===============================
# SPLIT PERIODS
# ===============================
train = df[df["month"].isin([10, 11])]   # Oct-Nov
test = df[df["month"].isin([1, 2])]      # Jan-Feb

# ===============================
# FEATURES (FROM YOUR TABLE)
# ===============================
features = [
    "prev_hour_delay_rate",
    "avg_delay_last_24h",
    "airline_7d_delay_rate",
    "temp",
    "humidity",
    "pressure",
    "wind_speed"
]

# ===============================
# COMPUTE MEANS
# ===============================
train_vals = train[features].mean().values
test_vals = test[features].mean().values

# ===============================
# NORMALIZATION (KEY FIX)
# ===============================
max_vals = np.maximum(train_vals, test_vals)

train_norm = train_vals / max_vals
test_norm = test_vals / max_vals

# ===============================
# % CHANGE (🔥 IMPORTANT)
# ===============================
percent_change = ((test_vals - train_vals) / train_vals) * 100

# ===============================
# PLOT
# ===============================
x = np.arange(len(features))
width = 0.35

plt.figure(figsize=(12, 6))

bars1 = plt.bar(x - width/2, train_norm, width,
                label="Oct–Nov (Training)", alpha=0.8)
bars2 = plt.bar(x + width/2, test_norm, width,
                label="Jan–Feb (Test)", alpha=0.8)

# ===============================
# ADD % CHANGE LABELS
# ===============================
for i, (t, p) in enumerate(zip(test_norm, percent_change)):
    plt.text(i + width/2, t + 0.02,
             f"{p:+.1f}%",
             ha="center", fontsize=10, fontweight="bold")

# ===============================
# STYLING
# ===============================
feature_labels = [
    "Prev Hour Delay",
    "24h Avg Delay",
    "Airline 7d Delay",
    "Temperature",
    "Humidity",
    "Pressure",
    "Wind Speed"
]

plt.xticks(x, feature_labels, rotation=20)
plt.ylabel("Normalized Feature Value")
plt.xlabel("Feature")

plt.title("Feature Distribution Shift Between Training and Test Periods",
          fontsize=14, pad=15)

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
plt.close()

print(f"✅ Figure 11 saved → {OUT_PATH}")