import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json

os.makedirs('results/drift_9k', exist_ok=True)
os.makedirs('results/final_plots', exist_ok=True)

print("="*80)
print("🔍 TEMPORAL DRIFT ANALYSIS (OCT-NOV vs JAN-FEB)")
print("="*80)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])
df['month'] = df['hour'].dt.month

train = df[df['month'].isin([10,11])]
future = df[df['month'].isin([1,2])]

print(f"\n📦 Train (Oct-Nov): {len(train)}")
print(f"📦 Future (Jan-Feb): {len(future)}")

# ===============================
# 1. TARGET DRIFT
# ===============================
print("\n" + "="*80)
print("1️⃣ DELAY RATE DRIFT")
print("="*80)

train_delay = train['delayed_15'].mean()
future_delay = future['delayed_15'].mean()

print(f"Oct-Nov delay rate: {train_delay*100:.1f}%")
print(f"Jan-Feb delay rate: {future_delay*100:.1f}%")
print(f"Change: {(future_delay - train_delay)*100:+.1f} percentage points")

# ===============================
# 2. NUMERIC FEATURE DRIFT
# ===============================
print("\n" + "="*80)
print("2️⃣ NUMERIC FEATURE DRIFT")
print("="*80)

numeric_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['delayed_15']]

drift_results = []

for col in numeric_cols:
    train_mean = train[col].mean()
    future_mean = future[col].mean()
    
    if train_mean != 0:
        change_pct = (future_mean - train_mean) / abs(train_mean)
    else:
        change_pct = 0

    drift_results.append({
        'feature': col,
        'train_mean': float(train_mean),
        'future_mean': float(future_mean),
        'pct_change': float(change_pct * 100)
    })

drift_df = pd.DataFrame(drift_results)
drift_df = drift_df.sort_values('pct_change', key=abs, ascending=False)

print("\n📊 TOP 10 MOST CHANGED FEATURES:\n")
print(drift_df.head(10).to_string(index=False))

# ===============================
# 3. WEATHER DISTRIBUTION
# ===============================
print("\n" + "="*80)
print("3️⃣ WEATHER DISTRIBUTION CHANGE")
print("="*80)

train_weather = train['weather'].value_counts(normalize=True)
future_weather = future['weather'].value_counts(normalize=True)

weather_df = pd.concat([train_weather, future_weather], axis=1)
weather_df.columns = ['Oct-Nov', 'Jan-Feb']
weather_df = weather_df.fillna(0)

print("\n📊 Weather distribution:\n")
print((weather_df * 100).round(1))

# ===============================
# 4. TRAFFIC COMPARISON
# ===============================
print("\n" + "="*80)
print("4️⃣ TRAFFIC SHIFT")
print("="*80)

traffic_cols = [
    'road_avg_speed',
    'road_vehicle_count',
    'road_congestion_index'
]

for col in traffic_cols:
    if col in df.columns:
        t_mean = train[col].mean()
        f_mean = future[col].mean()
        
        print(f"\n{col}:")
        print(f"  Oct-Nov: {t_mean:.2f}")
        print(f"  Jan-Feb: {f_mean:.2f}")
        print(f"  Change: {f_mean - t_mean:+.2f}")

# ===============================
# 📊 FINAL PLOT: FEATURE DRIFT
# ===============================
top_features = drift_df.head(5)

features = top_features['feature']
train_vals = top_features['train_mean']
future_vals = top_features['future_mean']

x = np.arange(len(features))
width = 0.35

plt.figure()

plt.bar(x - width/2, train_vals, width, label='Train (Oct–Nov)')
plt.bar(x + width/2, future_vals, width, label='Future (Jan–Feb)')

plt.xticks(x, features, rotation=30)
plt.ylabel("Feature Value")
plt.title("Top Feature Drift (Train vs Future)")
plt.legend()

plt.tight_layout()
plt.savefig('results/final_plots/feature_drift.png', dpi=150)

print("📊 Saved: results/final_plots/feature_drift.png")

# ===============================
# 5. SAVE SUMMARY
# ===============================
summary = {
    'train_size': int(len(train)),
    'future_size': int(len(future)),
    'delay_rate_train': float(train_delay),
    'delay_rate_future': float(future_delay),
    'top_changed_features': drift_df.head(10).to_dict('records')
}

with open('results/drift_9k/summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("\n💾 Saved: results/drift_9k/summary.json")

print("\n" + "="*80)
print("✅ DRIFT ANALYSIS COMPLETE!")
print("="*80)
