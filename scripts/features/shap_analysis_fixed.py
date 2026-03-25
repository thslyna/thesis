import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os

os.chdir('/Users/lyna.thc/Desktop/thesis')
os.makedirs('results/shap', exist_ok=True)

print("🔍 SHAP Analysis: Understanding Individual Predictions\n")

# Load V1 data
df = pd.read_csv("data/processed/flights_with_weather_road_history.csv")

# Load the trained Random Forest model
print("📦 Loading trained Random Forest model...")
model_path = "results/rf_untuned/rf_untuned_model.joblib"
model = joblib.load(model_path)

print(f"✅ Model loaded")

# Prepare data
DROP_COLS = ["flight_date", "hour", "flight_iata", "delayed_15"]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
y = df["delayed_15"]

# Transform data
X_transformed = model.named_steps['prep'].transform(X)
if hasattr(X_transformed, 'toarray'):
    X_transformed = X_transformed.toarray()

feature_names_transformed = list(model.named_steps['prep'].get_feature_names_out())

print(f"\n📊 Dataset: {X_transformed.shape[0]} flights, {X_transformed.shape[1]} features")

# Sample for SHAP
np.random.seed(42)
sample_size = min(500, X_transformed.shape[0])
sample_indices = np.random.choice(X_transformed.shape[0], sample_size, replace=False)

X_sample = X_transformed[sample_indices]
y_sample = y.iloc[sample_indices]

print(f"🎲 Using sample of {sample_size} flights for SHAP analysis")

# Create SHAP explainer
print("\n⏳ Creating SHAP explainer...")
explainer = shap.TreeExplainer(model.named_steps['model'])

# Calculate SHAP values
print("⏳ Calculating SHAP values...")
shap_values = explainer.shap_values(X_sample, check_additivity=False)

if isinstance(shap_values, list):
    shap_values_plot = shap_values[1]
else:
    shap_values_plot = shap_values

print("✅ SHAP values calculated!")

# PLOT 1: Feature Importance (Bar) - CLEANER
print("\n📊 Creating SHAP feature importance plot...")
fig, ax = plt.subplots(figsize=(12, 10))
mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
top_20_idx = np.argsort(mean_abs_shap)[-20:]

# Get top 20 features - iterate properly
top_features = []
top_values = []
for idx in top_20_idx:
    top_features.append(feature_names_transformed[int(idx)])
    top_values.append(float(mean_abs_shap[int(idx)]))

# Create horizontal bar plot
y_pos = np.arange(len(top_features))
ax.barh(y_pos, top_values, color='#4ecdc4')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_features, fontsize=10)
ax.set_xlabel('Mean |SHAP value| (average impact on model output)', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance - Top 20 Features', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/shap/shap_importance_clean.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: results/shap/shap_importance_clean.png")

# PLOT 2: Summary Plot (Beeswarm) - BIGGER
print("📊 Creating SHAP summary plot...")
fig = plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values_plot, X_sample, 
                  feature_names=feature_names_transformed,
                  max_display=20, show=False)
plt.gcf().axes[0].set_xlabel('SHAP value (impact on delay prediction)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/shap/shap_summary_clean.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: results/shap/shap_summary_clean.png")

# Print top features
print("\n" + "="*70)
print("TOP 20 MOST IMPORTANT FEATURES:")
print("="*70)
for i, (feat, val) in enumerate(zip(top_features, top_values), 1):
    print(f"{i:2d}. {feat:55s} | {val:.4f}")

print("\n✅ SHAP analysis complete!")
print(f"📁 Results saved in: results/shap/")
print("\nGenerated CLEAN plots:")
print("  1. shap_summary_clean.png - Feature importance with value distribution")
print("  2. shap_importance_clean.png - Feature importance ranking")

