import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs('results/reliability_9k', exist_ok=True)
os.makedirs('results/final_plots', exist_ok=True)

print("="*80)
print("🔍 MODEL RELIABILITY ANALYSIS - 9.5K DATASET (FINAL)")
print("="*80)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load('results/xgb_untuned/xgb_untuned_model.joblib')
print("✅ Model loaded")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])

train_size = int(len(df) * 0.75)
test = df.iloc[train_size:].reset_index(drop=True)

print(f"\n📦 Test set: {len(test):,} flights")

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
test['correct'] = (y_pred == y_true)

overall_acc = test['correct'].mean() * 100
print(f"✅ Overall accuracy: {overall_acc:.1f}%")

# ===============================
# 1. ACCURACY BY HOUR
# ===============================
print("\n1️⃣ Accuracy by hour")

hour_acc = test.groupby('hour_of_day')['correct'].mean() * 100
hour_acc = hour_acc.sort_index()

print("\nWorst hours:")
print(hour_acc.nsmallest(5))

print("\nBest hours:")
print(hour_acc.nlargest(5))

plt.figure()
hour_acc.plot(marker='o')
plt.axhline(overall_acc, linestyle='--')
plt.title('Model Accuracy by Hour of Day')
plt.ylabel('Accuracy (%)')
plt.xlabel('Hour')
plt.tight_layout()

plt.savefig('results/reliability_9k/hour.png', dpi=150)
plt.savefig('results/final_plots/reliability_hour.png', dpi=150)

# ===============================
# 2. ACCURACY BY WEATHER (CLEANED)
# ===============================
print("\n2️⃣ Accuracy by weather (cleaned)")

test_clean = test[test['weather'] != 'Unknown'].copy()

weather_acc = test_clean.groupby('weather')['correct'].mean() * 100
weather_acc = weather_acc.sort_values()

print(weather_acc)

plt.figure()
weather_acc.plot(kind='barh')
plt.axvline(test_clean['correct'].mean()*100, linestyle='--')
plt.title('Model Accuracy by Weather Condition')
plt.xlabel('Accuracy (%)')
plt.tight_layout()

plt.savefig('results/reliability_9k/weather.png', dpi=150)
plt.savefig('results/final_plots/reliability_weather.png', dpi=150)

# ===============================
# 3. ACCURACY BY AIRLINE (CLEANED)
# ===============================
print("\n3️⃣ Accuracy by airline (cleaned)")

test_air = test[~test['airline_name'].str.contains('202', na=False)]

air_acc = test_air.groupby('airline_name')['correct'].mean() * 100

print("\nWorst airlines:")
print(air_acc.nsmallest(5))

print("\nBest airlines:")
print(air_acc.nlargest(5))

# ===============================
# 4. ERROR COUNTS
# ===============================
print("\n4️⃣ Error analysis")

fp = test[(test['y_true']==0)&(test['y_pred']==1)]
fn = test[(test['y_true']==1)&(test['y_pred']==0)]

print(f"\nFalse Positives: {len(fp)}")
print(f"False Negatives: {len(fn)}")

# ===============================
# 5. CONFIDENCE CALIBRATION
# ===============================
print("\n5️⃣ Confidence calibration")

test['confidence_bin'] = pd.cut(
    test['y_prob'],
    bins=[0,0.2,0.4,0.6,0.8,1],
    labels=['Very Low','Low','Medium','High','Very High']
)

conf = test.groupby('confidence_bin', observed=False)['correct'].mean() * 100
print(conf)

plt.figure()
conf.plot(kind='bar')
plt.axhline(overall_acc, linestyle='--')
plt.title('Confidence vs Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Confidence Level')
plt.tight_layout()

plt.savefig('results/reliability_9k/confidence.png', dpi=150)
plt.savefig('results/final_plots/reliability_confidence.png', dpi=150)

print("\n" + "="*80)
print("✅ RELIABILITY ANALYSIS COMPLETE!")
print("="*80)
