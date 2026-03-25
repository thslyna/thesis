import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*80)
print("🔍 MODEL RELIABILITY ANALYSIS")
print("="*80)

# Create output folder
os.makedirs('results/reliability', exist_ok=True)

# Load data
preds = pd.read_csv('results/rf_tuned/rf_v1_test_predictions.csv')
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')

# Convert date
df['flight_date'] = pd.to_datetime(df['flight_date'])

# Select test period (Jan-Feb)
test = df[df['flight_date'] >= '2026-01-01'].reset_index(drop=True)

# Align sizes (IMPORTANT FIX)
min_len = min(len(test), len(preds))
test = test.iloc[:min_len].copy()
preds = preds.iloc[:min_len].copy()

# Add predictions
test['y_prob'] = preds['y_prob']
test['y_pred'] = (test['y_prob'] >= 0.5).astype(int)
test['y_true'] = preds['y_true']
test['correct'] = (test['y_pred'] == test['y_true']).astype(int)

print(f"\n📦 Test set: {len(test):,} flights")
print(f"✅ Overall accuracy: {test['correct'].mean()*100:.1f}%")

# =========================================================
# 1. ACCURACY BY HOUR
# =========================================================
print("\n1️⃣ Accuracy by hour")

hourly = test.groupby('hour_of_day')['correct'].mean() * 100

print("\nWorst hours:")
print(hourly.sort_values().head())

print("\nBest hours:")
print(hourly.sort_values(ascending=False).head())

plt.figure()
plt.bar(hourly.index, hourly.values)
plt.axhline(test['correct'].mean()*100, linestyle='--')
plt.xlabel("Hour")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy by Hour")
plt.savefig('results/reliability/accuracy_by_hour.png')
plt.close()

# =========================================================
# 2. ACCURACY BY WEATHER
# =========================================================
print("\n2️⃣ Accuracy by weather")

weather = test.groupby('weather')['correct'].mean() * 100
weather = weather.sort_values()

print(weather)

plt.figure()
plt.barh(weather.index, weather.values)
plt.axvline(test['correct'].mean()*100, linestyle='--')
plt.xlabel("Accuracy (%)")
plt.title("Accuracy by Weather")
plt.savefig('results/reliability/accuracy_by_weather.png')
plt.close()

# =========================================================
# 3. ACCURACY BY AIRLINE
# =========================================================
print("\n3️⃣ Accuracy by airline")

airline = test.groupby('airline_name')['correct'].mean() * 100
airline = airline.sort_values()

print("\nWorst airlines:")
print(airline.head())

print("\nBest airlines:")
print(airline.tail())

plt.figure(figsize=(8,6))
plt.barh(airline.index, airline.values)
plt.axvline(test['correct'].mean()*100, linestyle='--')
plt.xlabel("Accuracy (%)")
plt.title("Accuracy by Airline")
plt.savefig('results/reliability/accuracy_by_airline.png')
plt.close()

# =========================================================
# 4. ERROR TYPES
# =========================================================
print("\n4️⃣ Error analysis")

fp = test[(test['y_true']==0) & (test['y_pred']==1)]
fn = test[(test['y_true']==1) & (test['y_pred']==0)]

print(f"False Positives: {len(fp)}")
print(f"False Negatives: {len(fn)}")

# =========================================================
# 5. CONFIDENCE ANALYSIS
# =========================================================
print("\n5️⃣ Confidence analysis")

test['confidence_bin'] = pd.cut(test['y_prob'],
                               bins=[0,0.3,0.5,0.7,0.9,1.0])

conf = test.groupby('confidence_bin')['correct'].mean() * 100
print(conf)

plt.figure()
plt.bar(range(len(conf)), conf.values)
plt.xticks(range(len(conf)), [str(x) for x in conf.index], rotation=30)
plt.axhline(test['correct'].mean()*100, linestyle='--')
plt.ylabel("Accuracy (%)")
plt.title("Confidence Calibration")
plt.savefig('results/reliability/confidence_calibration.png')
plt.close()

print("\n" + "="*80)
print("🎯 DONE - Reliability analysis complete")
print("="*80)

