import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import os

os.makedirs("results/plots", exist_ok=True)

print("="*60)
print("📊 FIGURE 18 — PHASE 2 ABLATION STUDY")
print("="*60)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data/processed/flights_v1_IMPUTED_FIXED.csv")

TARGET = "delayed_15"
DROP_COLS = ["flight_date", "hour", "flight_iata"]

X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
y = df[TARGET]

# ===============================
# FEATURE GROUPING
# ===============================
def get_category(col):
    c = col.lower()
    if "arr_iata" in c or "dep_iata" in c:
        return "Airport"
    elif "airline" in c:
        return "Airline"
    elif any(x in c for x in ["temp", "wind", "pressure", "weather"]):
        return "Weather"
    elif any(x in c for x in ["hour", "day", "month"]):
        return "Temporal"
    elif "traffic" in c or "road" in c:
        return "Traffic"
    else:
        return "Other"

feature_groups = {col: get_category(col) for col in X.columns}
groups = sorted(set(feature_groups.values()))

# ===============================
# TIME SPLIT (IMPORTANT)
# ===============================
split = int(len(X) * 0.75)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

# ===============================
# PIPELINE BUILDER
# ===============================
def build_pipeline(X_subset):

    num_cols = X_subset.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_subset.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

# ===============================
# BASELINE
# ===============================
print("\n⏳ Training baseline model...")

baseline_model = build_pipeline(X_train)
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict_proba(X_test)[:, 1]
baseline_auc = roc_auc_score(y_test, y_pred)

print(f"✅ Baseline AUC: {baseline_auc:.4f}")

# ===============================
# ABLATION
# ===============================
results = []

for group in groups:
    print(f"\n🔧 Removing: {group}")

    cols_keep = [c for c in X.columns if feature_groups[c] != group]

    X_train_sub = X_train[cols_keep]
    X_test_sub = X_test[cols_keep]

    model = build_pipeline(X_train_sub)
    model.fit(X_train_sub, y_train)

    y_pred = model.predict_proba(X_test_sub)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    drop = baseline_auc - auc

    results.append({
        "group": group,
        "auc": auc,
        "auc_drop": drop
    })

    print(f"   AUC: {auc:.4f} | Drop: {drop:.4f}")

results_df = pd.DataFrame(results).sort_values("auc_drop", ascending=True)

# ===============================
# PLOT
# ===============================
fig, ax = plt.subplots(figsize=(10, 6))

colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in results_df["auc_drop"]]

bars = ax.barh(
    results_df["group"],
    results_df["auc_drop"],
    color=colors,
    edgecolor="black"
)

ax.axvline(0, color="black", linewidth=1)

ax.set_title("Ablation Study: Impact of Feature Groups on AUC",
             fontsize=14, fontweight="bold")

ax.set_xlabel("AUC Drop (Performance Loss)", fontsize=12, fontweight="bold")
ax.set_ylabel("Removed Feature Group", fontsize=12, fontweight="bold")

# Labels
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2,
            f"{width:.3f}",
            va="center", fontsize=10)

plt.tight_layout()
plt.savefig("results/plots/fig18_ablation_auc.png", dpi=150)
plt.close()

print("\n✅ Saved → results/plots/fig18_ablation_auc.png")

print("\n" + "="*60)
print("✅ FIGURE 18 COMPLETE")
print("="*60)