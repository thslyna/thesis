# ==========================================================
# 📊 FIGURE 24 — CONFIDENCE CALIBRATION (FINAL CLEAN)
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

os.makedirs("results/plots", exist_ok=True)

print("="*80)
print("📊 FIGURE 24 — CONFIDENCE CALIBRATION (FINAL)")
print("="*80)

# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_csv("data/processed/flights_v1_IMPUTED_FIXED.csv")
df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour").reset_index(drop=True)

# ==========================================================
# 2. TIME-BASED SPLIT (CONSISTENT WITH THESIS)
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
numeric_features     = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ==========================================================
# 5. MODEL
# ==========================================================
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

print("\n⏳ Training model...")
pipe.fit(X_train, y_train)

# ==========================================================
# 6. PREDICT PROBABILITIES
# ==========================================================
y_prob = pipe.predict_proba(X_test)[:, 1]

# ==========================================================
# 7. CALIBRATION (FIXED — QUANTILE BINNING)
# ==========================================================
prob_true, prob_pred = calibration_curve(
    y_test,
    y_prob,
    n_bins=8,                 # fewer bins → more stable
    strategy="quantile"       # KEY FIX → equal sample bins
)

# ==========================================================
# 8. BRIER SCORE (OPTIONAL BUT STRONG)
# ==========================================================
brier = brier_score_loss(y_test, y_prob)
print(f"Brier Score: {brier:.4f}")

# ==========================================================
# 9. PLOT (CLEAN + SMOOTH)
# ==========================================================
plt.figure(figsize=(6,6))

# Perfect calibration
plt.plot([0,1], [0,1],
         linestyle="--",
         color="gray",
         label="Perfect calibration")

# Model curve
plt.plot(prob_pred, prob_true,
         marker="o",
         linewidth=2,
         color="navy",
         label="Model")

plt.xlabel("Predicted Probability", fontsize=12)
plt.ylabel("Observed Frequency", fontsize=12)

plt.title("Calibration Curve (Reliability Diagram)",
          fontweight="bold")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

# ==========================================================
# 10. SAVE
# ==========================================================
output_path = "results/plots/fig24_calibration_curve.png"
plt.savefig(output_path, dpi=150)
plt.close()

print(f"\n✅ Saved → {output_path}")

print("\n" + "="*80)
print("✅ FIGURE 24 COMPLETE (FINAL)")
print("="*80)