import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs('results/error_9k', exist_ok=True)
os.makedirs('results/final_plots', exist_ok=True)

print("="*80)
print("🚨 ERROR ANALYSIS - 9.5K DATASET")
print("="*80)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load('results/xgb_untuned/xgb_untuned_model.joblib')

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])

train_size = int(len(df) * 0.75)
test = df.iloc[train_size:].reset_index(drop=True)

# ===============================
# FEATURES
# ===============================
DROP_COLS = ["delayed_15", "flight_date", "hour", "flight_iata"]

X = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
y_true = test['delayed_15']

# ===============================
# PREDICTIONS
# ===============================
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

test['y_prob'] = y_prob
test['y_pred'] = y_pred
test['y_true'] = y_true

# ===============================
# ERROR TYPES
# ===============================
fp = test[(test['y_true']==0)&(test['y_pred']==1)]
fn = test[(test['y_true']==1)&(test['y_pred']==0)]
correct = test[test['y_true'] == test['y_pred']]

print(f"\nFalse Positives: {len(fp)}")
print(f"False Negatives: {len(fn)}")

# ===============================
# 1. FALSE NEGATIVES
# ===============================
print("\n1️⃣ FALSE NEGATIVES ANALYSIS")

print("\nAvg weather conditions (FN):")
print(fn[['temp','wind_speed','humidity']].mean())

print("\nAvg temporal features (FN):")
print(fn[['hour_of_day','day_of_week']].mean())

print("\nDelay history (FN):")
print(fn[['prev_hour_delay_rate','avg_delay_last_24h']].mean())

# ===============================
# 2. FALSE POSITIVES
# ===============================
print("\n2️⃣ FALSE POSITIVES ANALYSIS")

print("\nAvg weather conditions (FP):")
print(fp[['temp','wind_speed','humidity']].mean())

print("\nAvg temporal features (FP):")
print(fp[['hour_of_day','day_of_week']].mean())

print("\nDelay history (FP):")
print(fp[['prev_hour_delay_rate','avg_delay_last_24h']].mean())

# ===============================
# 3. CORRECT COMPARISON
# ===============================
print("\n3️⃣ COMPARISON WITH CORRECT CASES")

print("\nCorrect cases delay history:")
print(correct[['prev_hour_delay_rate','avg_delay_last_24h']].mean())

# ===============================
# 4. VISUALIZATION (IMPROVED)
# ===============================
print("\n4️⃣ VISUALIZATION")

labels = ['Correct', 'False Positive', 'False Negative']
values = [
    len(correct),
    len(fp),
    len(fn)
]

plt.figure()
bars = plt.bar(labels, values)

plt.title("Prediction Outcome Distribution")
plt.ylabel("Number of Flights")

# add values on bars
for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, v + 10, str(v), ha='center')

plt.tight_layout()

# save both
plt.savefig('results/error_9k/error_distribution.png', dpi=150)
plt.savefig('results/final_plots/error_distribution.png', dpi=150)

print("📊 Saved: results/final_plots/error_distribution.png")

print("\n" + "="*80)
print("✅ ERROR ANALYSIS COMPLETE!")
print("="*80)