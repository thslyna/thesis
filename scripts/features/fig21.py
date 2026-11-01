# ==========================================================
# FIGURE 21 — CONFUSION MATRIX (FINAL CONSISTENT)
# ==========================================================
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

print("="*80)
print("📊 FIGURE 21 — CONFUSION MATRIX (FINAL)")
print("="*80)

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv("data/processed/flights_v1_IMPUTED_FIXED.csv")

# ==========================================================
# SAME FEATURES AS SHAP MODEL
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
# SAME SPLIT (CRITICAL)
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ==========================================================
# LOAD MODEL (CLEAN ONE)
# ==========================================================
model = joblib.load("results/shap_model/xgb_shap_clean.joblib")

# ==========================================================
# PREDICT
# ==========================================================
y_pred = model.predict(X_test)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n📊 CONFUSION MATRIX:")
print(cm)

print(f"\nTN: {tn} | FP: {fp}")
print(f"FN: {fn} | TP: {tp}")

print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# ==========================================================
# PLOT
# ==========================================================
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["On-Time (0)", "Delayed (1)"],
    yticklabels=["On-Time (0)", "Delayed (1)"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix (Threshold = 0.5)")

plt.tight_layout()
plt.savefig("results/plots/fig21_confusion_matrix_final.png", dpi=150)
plt.close()

print("\n✅ Saved → results/plots/fig21_confusion_matrix_final.png")