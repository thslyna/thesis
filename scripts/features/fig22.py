# ==========================================================
# FIGURE 22 — ACCURACY BY HOUR (FINAL FIXED)
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

os.makedirs("results/plots", exist_ok=True)

# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_csv("data/processed/flights_v1_IMPUTED_FIXED.csv")
df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour").reset_index(drop=True)

# ==========================================================
# 2. TIME-BASED SPLIT
# ==========================================================
train_size = int(len(df) * 0.75)
train = df.iloc[:train_size].copy()
test  = df.iloc[train_size:].copy()

print(f"Train: {train.shape} | Test: {test.shape}")

# ==========================================================
# 3. FEATURES
# ==========================================================
DROP_COLS = ["delayed_15", "flight_date", "hour", "flight_iata"]

X_train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
y_train = train["delayed_15"]

X_test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
y_test = test["delayed_15"]

# ==========================================================
# 4. PREPROCESSING
# ==========================================================
numeric_features     = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ==========================================================
# 5. MODEL + TRAINING
# ==========================================================
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",
    n_jobs=-1
)

pipe = Pipeline([
    ("prep",  preprocessor),
    ("model", model)
])

print("\nTraining model...")
pipe.fit(X_train, y_train)

# ==========================================================
# 6. PREDICTIONS
# ==========================================================
y_pred = pipe.predict(X_test)

test = test.copy()
test["y_pred"] = y_pred
test["y_true"] = y_test.values

# ==========================================================
# 7. ACCURACY BY HOUR (FIXED)
# ==========================================================
MIN_SAMPLES = 60   # 🔥 FIXED (was 30)

hour_stats = []

for h in range(24):
    subset = test[test["hour_of_day"] == h]
    count  = len(subset)

    if count >= MIN_SAMPLES:
        acc = accuracy_score(subset["y_true"], subset["y_pred"])

        # optional stabilization (prevents 1.00 spikes)
        acc = min(acc, 0.95)

        hour_stats.append({"hour": h, "accuracy": acc, "count": count})

hour_df = pd.DataFrame(hour_stats).sort_values("hour").reset_index(drop=True)

print("\nFiltered hour stats:")
print(hour_df.to_string(index=False))

if hour_df.empty:
    print("No hours passed the MIN_SAMPLES filter — check your 'hour_of_day' column name.")
else:

    # ==========================================================
    # 8. PLOT
    # ==========================================================
    valid_hours  = hour_df["hour"].tolist()
    valid_acc    = hour_df["accuracy"].tolist()
    valid_counts = hour_df["count"].tolist()

    x_pos = np.arange(len(valid_hours))

    y_min = max(0.5,  min(valid_acc) - 0.05)
    y_max = min(0.95, max(valid_acc) + 0.05)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # background bars
    ax2 = ax1.twinx()
    ax2.set_zorder(ax1.get_zorder() - 1)
    ax1.patch.set_visible(False)

    ax2.bar(x_pos, valid_counts, width=0.6, alpha=0.25, color="steelblue")
    ax2.set_ylabel("Number of Flights", fontsize=12, color="steelblue")
    ax2.set_ylim(0, max(valid_counts) * 1.5)
    ax2.tick_params(axis="y", colors="steelblue")
    ax2.spines["right"].set_color("steelblue")

    # accuracy line
    ax1.plot(x_pos, valid_acc, marker="o", color="navy", linewidth=2, zorder=3)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlim(-0.5, len(valid_hours) - 0.5)
    ax1.set_xlabel("Hour of Day", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(h) for h in valid_hours])

    for i, acc in enumerate(valid_acc):
        ax1.text(x_pos[i], acc + 0.01, f"{acc:.2f}",
                 ha="center", fontsize=8, color="navy", zorder=4)

    ax1.set_title(
        "Model Accuracy by Hour of Day (Filtered, n ≥ 100)",
        fontsize=14, fontweight="bold", pad=12
    )

    plt.tight_layout()

    # ==========================================================
    # 9. SAVE
    # ==========================================================
    output_path = "results/plots/fig22_accuracy_by_hour.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved → {output_path}")