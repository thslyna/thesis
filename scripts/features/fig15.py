# ==========================================================
# FIGURE 15 — SHAP (CLEAN MODEL)
# ==========================================================
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

os.makedirs("results/plots", exist_ok=True)

print("="*80)
print("📊 SHAP — CLEAN MODEL")
print("="*80)

# ==========================================================
# LOAD MODEL
# ==========================================================
model = joblib.load("results/shap_model/xgb_shap_clean.joblib")

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv("data/processed/flights_v1_IMPUTED_FIXED.csv")
df["hour"] = pd.to_datetime(df["hour"])

# ==========================================================
# SAME FEATURE SET
# ==========================================================
DROP_COLS = [
    "delayed_15",
    "flight_date",
    "hour",
    "flight_iata",
    "arr_iata",
    "airline_name"
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
y = df["delayed_15"]

# ==========================================================
# SAME SPLIT
# ==========================================================
_, X_test, _, _ = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ==========================================================
# SAMPLE
# ==========================================================
X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)

# ==========================================================
# PIPELINE
# ==========================================================
prep = model.named_steps["prep"]
clf  = model.named_steps["model"]

X_trans = prep.transform(X_sample)

# ==========================================================
# SHAP
# ==========================================================
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_trans, check_additivity=False)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

feature_names = prep.get_feature_names_out()

# ==========================================================
# BAR (MAIN FIGURE)
# ==========================================================
plt.figure()
shap.summary_plot(
    shap_values,
    X_trans,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig("results/plots/fig15_shap_bar_clean.png", dpi=150)
plt.close()

# ==========================================================
# BEESWARM
# ==========================================================
plt.figure()
shap.summary_plot(
    shap_values,
    X_trans,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("results/plots/fig15_shap_summary_clean.png", dpi=150)
plt.close()

print("✅ SHAP clean plots saved")