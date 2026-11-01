# ==========================================================
# FIGURE 25 — COST + PERFORMANCE TRADE-OFF
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

print("="*80)
print("📊 FIGURE 25 — COST + PRECISION/RECALL TRADE-OFF")
print("="*80)

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv("data/processed/flights_v1_IMPUTED_FIXED.csv")

DROP_COLS = [
    "delayed_15", "flight_date", "hour", "flight_iata",
    "arr_iata", "airline_name"
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
y = df["delayed_15"]

# ==========================================================
# SPLIT
# ==========================================================
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ==========================================================
# MODEL
# ==========================================================
model = joblib.load("results/shap_model/xgb_shap_clean.joblib")
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================================
# COST SETUP
# ==========================================================
COST_FN = 3
COST_FP = 1

thresholds = np.linspace(0.01, 0.99, 50)

costs = []
precisions = []
recalls = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    cost = COST_FP * fp + COST_FN * fn
    costs.append(cost)

    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))

# ==========================================================
# BEST POINT
# ==========================================================
best_idx = np.argmin(costs)
best_t = thresholds[best_idx]

# ==========================================================
# PLOT
# ==========================================================
fig, ax1 = plt.subplots(figsize=(10,5))

# Precision & Recall
ax1.plot(thresholds, precisions, label="Precision")
ax1.plot(thresholds, recalls, label="Recall")

ax1.set_xlabel("Threshold")
ax1.set_ylabel("Score")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)

# Cost (secondary axis)
ax2 = ax1.twinx()
ax2.plot(thresholds, costs, linestyle="--")
ax2.set_ylabel("Cost")

# Optimal line
ax1.axvline(best_t, linestyle="--")

plt.title("Cost vs Precision–Recall Trade-off")

plt.tight_layout()
plt.savefig("results/plots/fig25_enhanced.png", dpi=150)
plt.close()

print(f"\n✅ Optimal threshold: {best_t:.2f}")
print("✅ Saved → fig25_enhanced.png")