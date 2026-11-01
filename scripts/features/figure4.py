#!/usr/bin/env python3
# ============================================================
# Figure 4 — Ablation Study (Dataset V1) FINAL CLEAN
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix

# ===============================
# PATHS
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"

OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/v1_ablation"
PLOTS_DIR = "/Users/lyna.thc/Desktop/thesis/results/plots"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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
    "flight_number",
    "flight_icao",
    "collected_at",
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ===============================
# FEATURE GROUPS
# ===============================
WEATHER_FEATURES = ["temp", "humidity", "pressure", "wind_speed", "weather"]

ROAD_FEATURES = [
    "road_avg_speed",
    "road_vehicle_count",
    "road_congestion_index"
]

HISTORY_FEATURES = [
    "prev_hour_delay_rate",
    "avg_delay_last_24h",
    "airline_7d_delay_rate"
]

ABLATIONS = {
    "all_features": [],
    "no_weather": WEATHER_FEATURES,
    "no_road": ROAD_FEATURES,
    "no_history": HISTORY_FEATURES
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

results = []

# ===============================
# ABLATION LOOP
# ===============================
for name, drop_features in ABLATIONS.items():
    print(f"\n🚀 Running ablation: {name}")

    X_tr = X_train.drop(columns=[c for c in drop_features if c in X_train.columns])
    X_te = X_test.drop(columns=[c for c in drop_features if c in X_test.columns])

    numeric_features = X_tr.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_tr.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="log2",
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_tr, y_train)

    y_prob = pipe.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results.append({
        "ablation": name,
        "roc_auc": round(auc, 4),
        "recall_delayed": round(recall, 4),
        "false_negatives": fn
    })

# ===============================
# SAVE RESULTS
# ===============================
res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT_DIR, "ablation_results_v1.csv"), index=False)

print("\n✅ Ablation study completed")
print(res_df)

# ===============================
# OPTIONAL: FALSE NEGATIVE PLOT (fixed warning)
# ===============================
plt.figure(figsize=(7, 5))

sns.barplot(
    data=res_df,
    x="ablation",
    y="false_negatives",
    hue="ablation",
    legend=False,
    palette="Blues_r"
)

plt.xlabel("Ablation Scenario")
plt.ylabel("False Negatives")
plt.title("Impact of Feature Removal on Missed Delays")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ablation_false_negatives.png"), dpi=150)
plt.close()

# ============================================================
# ⭐ FIGURE 4 — MAIN THESIS PLOT (AUC)
# ============================================================

# Order
order = ["all_features", "no_weather", "no_road", "no_history"]
plot_df = res_df.set_index("ablation").loc[order].reset_index()

# Labels
label_map = {
    "all_features": "All features",
    "no_weather": "No weather",
    "no_road": "No road traffic",
    "no_history": "No historical"
}
plot_df["label"] = plot_df["ablation"].map(label_map)

# Baseline
baseline = plot_df.loc[plot_df["ablation"] == "all_features", "roc_auc"].values[0]

# Colors
colors = []
for val in plot_df["roc_auc"]:
    if val == baseline:
        colors.append("#2c7bb6")  # baseline
    elif val < baseline:
        colors.append("#d7191c")  # worse
    else:
        colors.append("#1a9641")  # better

# Plot
plt.figure(figsize=(8, 5))

plt.barh(
    plot_df["label"],
    plot_df["roc_auc"],
    color=colors,
    edgecolor="black",
    linewidth=0.5
)

# Baseline line (clean)
plt.axvline(baseline, linestyle="--", linewidth=1)

# Axis range (clean spacing)
plt.xlim(0.83, 0.875)

# Labels
plt.xlabel("ROC-AUC")
plt.title("Ablation Study — Feature Group Impact (Dataset V1)")

# Values
for i, v in enumerate(plot_df["roc_auc"]):
    plt.text(v + 0.0015, i, f"{v:.3f}", va="center")

# IMPORTANT: Correct order (baseline on top)
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(os.path.join(PLOTS_DIR, "fig4_ablation_v1.png"), dpi=300)
plt.close()

print("✅ Figure 4 saved (FINAL)")