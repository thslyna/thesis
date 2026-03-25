import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('/Users/lyna.thc/Desktop/thesis')
os.makedirs('results/temporal_validation', exist_ok=True)

print("⏰ TEMPORAL VALIDATION: Train on October, Test on November\n")

# Load data
df = pd.read_csv("data/processed/flights_with_weather_road_history.csv")
df['date'] = pd.to_datetime(df['flight_date'])

# Split by month
df_oct = df[df['date'].dt.month == 10].copy()
df_nov = df[df['date'].dt.month == 11].copy()

print(f"📊 October flights: {len(df_oct)}")
print(f"📊 November flights: {len(df_nov)}")

# Prepare data
DROP_COLS = ["flight_date", "hour", "flight_iata", "delayed_15"]

X_train = df_oct.drop(columns=[c for c in DROP_COLS if c in df_oct.columns])
y_train = df_oct["delayed_15"]

X_test = df_nov.drop(columns=[c for c in DROP_COLS if c in df_nov.columns])
y_test = df_nov["delayed_15"]

print(f"\n📦 Training set: {len(X_train)} flights")
print(f"📦 Test set: {len(X_test)} flights")

# Load the pre-trained model (trained on ALL data)
model_all = joblib.load("results/rf_untuned/rf_untuned_model.joblib")

# Train NEW model on ONLY October
print("\n🔄 Training new RF model on October data only...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_features = ['hour_of_day', 'day_of_week', 'temp', 'humidity', 'pressure', 
                'wind_speed', 'road_avg_speed', 'road_vehicle_count', 
                'road_congestion_index', 'prev_hour_delay_rate', 
                'avg_delay_last_24h', 'airline_7d_delay_rate']

cat_features = ['airline_name', 'dep_iata', 'arr_iata', 'weather']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

model_oct = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

model_oct.fit(X_train, y_train)
print("✅ October model trained!")

# Test on November
print("\n🧪 Testing on November data...")
y_pred_oct = model_oct.predict(X_test)
y_pred_proba_oct = model_oct.predict_proba(X_test)[:, 1]

# Metrics
auc_oct = roc_auc_score(y_test, y_pred_proba_oct)
precision_oct = precision_score(y_test, y_pred_oct)
recall_oct = recall_score(y_test, y_pred_oct)
f1_oct = f1_score(y_test, y_pred_oct)

print("\n" + "="*70)
print("📊 TEMPORAL VALIDATION RESULTS")
print("="*70)
print(f"\n🎯 Model trained on OCTOBER, tested on NOVEMBER:")
print(f"   ROC-AUC:   {auc_oct:.3f}")
print(f"   Precision: {precision_oct:.3f}")
print(f"   Recall:    {recall_oct:.3f}")
print(f"   F1 Score:  {f1_oct:.3f}")

# Compare to original model (trained on all data)
y_pred_all = model_all.predict(X_test)
y_pred_proba_all = model_all.predict_proba(X_test)[:, 1]

auc_all = roc_auc_score(y_test, y_pred_proba_all)
precision_all = precision_score(y_test, y_pred_all)
recall_all = recall_score(y_test, y_pred_all)
f1_all = f1_score(y_test, y_pred_all)

print(f"\n🎯 Original model (trained on ALL data), tested on NOVEMBER:")
print(f"   ROC-AUC:   {auc_all:.3f}")
print(f"   Precision: {precision_all:.3f}")
print(f"   Recall:    {recall_all:.3f}")
print(f"   F1 Score:  {f1_all:.3f}")

# Comparison
print("\n" + "="*70)
print("🔍 COMPARISON:")
print("="*70)
print(f"ROC-AUC difference: {auc_oct - auc_all:+.3f} ({(auc_oct - auc_all)/auc_all * 100:+.1f}%)")
print(f"Recall difference:  {recall_oct - recall_all:+.3f} ({(recall_oct - recall_all)/recall_all * 100:+.1f}%)")

if abs(auc_oct - auc_all) < 0.02:
    print("\n✅ EXCELLENT! Model performance is STABLE across months!")
    print("   This proves the model generalizes well to new time periods.")
elif abs(auc_oct - auc_all) < 0.05:
    print("\n✅ GOOD! Model performance is reasonably stable across months.")
else:
    print("\n⚠️ WARNING! Model performance varies significantly across months.")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Metrics comparison
metrics = ['ROC-AUC', 'Precision', 'Recall', 'F1']
oct_scores = [auc_oct, precision_oct, recall_oct, f1_oct]
all_scores = [auc_all, precision_all, recall_all, f1_all]

x = np.arange(len(metrics))
width = 0.35

axes[0].bar(x - width/2, oct_scores, width, label='Oct→Nov', color='#4ecdc4')
axes[0].bar(x + width/2, all_scores, width, label='All Data→Nov', color='#ff6b6b')
axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[0].set_title('Temporal Validation: Performance Comparison', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1])

# Confusion matrices
cm_oct = confusion_matrix(y_test, y_pred_oct)
cm_all = confusion_matrix(y_test, y_pred_all)

# Just show October model CM
sns.heatmap(cm_oct, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Confusion Matrix: Oct→Nov Model', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=11)
axes[1].set_ylabel('Actual', fontsize=11)

plt.tight_layout()
plt.savefig('results/temporal_validation/temporal_validation.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n📊 Plot saved: results/temporal_validation/temporal_validation.png")
print("\n✅ Temporal validation complete!")

