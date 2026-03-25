import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

os.makedirs('results/shap_9k', exist_ok=True)

print("="*80)
print("🧠 SHAP ANALYSIS - XGBOOST (UNTUNED, 9.5K CLEAN DATA)")
print("="*80)

# ===============================
# 1. LOAD MODEL
# ===============================
model_path = 'results/xgb_untuned/xgb_untuned_model.joblib'

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    exit(1)

model = joblib.load(model_path)
print("✅ Model loaded")

# ===============================
# 2. LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])

print(f"📦 Dataset: {df.shape}")

# ===============================
# 3. RECREATE TEST SET (same split logic)
# ===============================
train_size = int(len(df) * 0.75)
df_test = df.iloc[train_size:].reset_index(drop=True)

print(f"📊 Test set: {df_test.shape}")

# ===============================
# 4. FEATURES (EXACT SAME AS TRAINING)
# ===============================
DROP_COLS = ["delayed_15", "flight_date", "hour", "flight_iata"]

X = df_test.drop(columns=[c for c in DROP_COLS if c in df_test.columns])

# ===============================
# 5. SAMPLE (speed)
# ===============================
X_sample = X.sample(n=min(500, len(X)), random_state=42)

print(f"📊 SHAP sample: {X_sample.shape}")

# ===============================
# 6. EXTRACT PIPELINE PARTS
# ===============================
preprocessor = model.named_steps['prep']
xgb_model = model.named_steps['model']

# Transform data using SAME preprocessing
X_transformed = preprocessor.transform(X_sample)

print(f"🔧 Transformed shape: {X_transformed.shape}")

# ===============================
# 7. SHAP
# ===============================
print("\nComputing SHAP values...")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_transformed)

# Binary classification fix
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print("✅ SHAP computed!")

# ===============================
# 8. FEATURE NAMES (IMPORTANT)
# ===============================
try:
    feature_names = preprocessor.get_feature_names_out()
except:
    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

# ===============================
# 9. SUMMARY PLOT
# ===============================
plt.figure()
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig('results/shap_9k/shap_summary.png', dpi=150)
plt.close()

print("📊 Saved: shap_summary.png")

# ===============================
# 10. BAR IMPORTANCE
# ===============================
plt.figure()
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig('results/shap_9k/shap_importance.png', dpi=150)
plt.close()

print("📊 Saved: shap_importance.png")

# ===============================
# 11. NUMERIC IMPORTANCE
# ===============================
mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_abs_shap
}).sort_values('importance', ascending=False)

importance_df.to_csv('results/shap_9k/feature_importance.csv', index=False)

print("\n📊 TOP 10 FEATURES:")
print(importance_df.head(10))

print("\n" + "="*80)
print("✅ SHAP ANALYSIS COMPLETE!")
print("="*80)

