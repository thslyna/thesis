import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

os.chdir('/Users/lyna.thc/Desktop/thesis')
os.makedirs('results/shap', exist_ok=True)

print("🔍 SHAP Analysis - SIMPLE VERSION\n")

# Load model
model = joblib.load("results/rf_untuned/rf_untuned_model.joblib")
print("✅ Model loaded")

# Load data
df = pd.read_csv("data/processed/flights_with_weather_road_history.csv")
DROP_COLS = ["flight_date", "hour", "flight_iata", "delayed_15"]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Transform
X_transformed = model.named_steps['prep'].transform(X)
if hasattr(X_transformed, 'toarray'):
    X_transformed = X_transformed.toarray()

# Sample 500 flights
np.random.seed(42)
X_sample = X_transformed[:500]  # Just take first 500

print(f"📊 Using {X_sample.shape[0]} flights")

# SHAP
print("⏳ Calculating SHAP values...")
explainer = shap.TreeExplainer(model.named_steps['model'])
shap_values = explainer.shap_values(X_sample, check_additivity=False)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

print("✅ SHAP done!")

# Get feature names
features = list(model.named_steps['prep'].get_feature_names_out())

# PLOT 1: Bar plot
print("\n📊 Creating bar plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_sample, features, plot_type='bar', max_display=20, show=False)
plt.tight_layout()
plt.savefig('results/shap/importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: results/shap/importance.png")

# PLOT 2: Beeswarm
print("📊 Creating beeswarm plot...")
plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values, X_sample, features, max_display=20, show=False)
plt.tight_layout()
plt.savefig('results/shap/summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: results/shap/summary.png")

print("\n✅✅✅ SHAP PLOTS GENERATED! ✅✅✅")
print("📁 Check: results/shap/importance.png")
print("📁 Check: results/shap/summary.png")

