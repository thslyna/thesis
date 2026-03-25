import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
import os

os.makedirs('results/traffic_9k', exist_ok=True)

print("="*80)
print("🚗 TRAFFIC CORRELATION ANALYSIS - 9.5K DATASET (FINAL)")
print("="*80)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])

print(f"\n📦 Dataset: {len(df):,} flights")
print(f"📊 Delay rate: {df['delayed_15'].mean()*100:.1f}%")

traffic_features = [
    'road_avg_speed',
    'road_vehicle_count',
    'road_congestion_index'
]

# ===============================
# 1. CORRELATION (CORRECT)
# ===============================
print("\n" + "="*80)
print("1️⃣  POINT-BISERIAL CORRELATION")
print("="*80)

correlations = {}

for feature in traffic_features:
    if feature in df.columns:
        temp_df = df[[feature, 'delayed_15']].dropna()

        r, p_value = stats.pointbiserialr(
            temp_df['delayed_15'],
            temp_df[feature]
        )

        correlations[feature] = {
            'correlation': float(r),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }

        print(f"\n{feature}:")
        print(f"  Correlation (r): {r:+.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'}")

# ===============================
# SUMMARY METRICS
# ===============================
avg_r = float(np.mean([abs(c['correlation']) for c in correlations.values()]))
max_r = float(max([abs(c['correlation']) for c in correlations.values()]))

print("\n📊 SUMMARY:")
print(f"  Average |r|: {avg_r:.4f}")
print(f"  Max |r|: {max_r:.4f}")

if avg_r < 0.05:
    verdict = "NEGLIGIBLE"
elif avg_r < 0.15:
    verdict = "WEAK"
else:
    verdict = "MODERATE"

print(f"\n🎯 VERDICT: Traffic impact = {verdict}")

# ===============================
# 2. TIME OF DAY
# ===============================
print("\n" + "="*80)
print("2️⃣  BY TIME OF DAY")
print("="*80)

df['hour_num'] = df['hour'].dt.hour

periods = {
    'Morning Rush': (6, 9),
    'Midday': (10, 15),
    'Evening Rush': (17, 20),
    'Off-Peak': [(21, 23), (0, 5)]
}

period_results = {}

for name, hours in periods.items():
    if isinstance(hours, list):
        mask = ((df['hour_num'] >= hours[0][0]) & (df['hour_num'] <= hours[0][1])) | \
               ((df['hour_num'] >= hours[1][0]) & (df['hour_num'] <= hours[1][1]))
    else:
        mask = (df['hour_num'] >= hours[0]) & (df['hour_num'] <= hours[1])

    subset = df[mask]

    if len(subset) > 50:
        vals = []
        for feature in traffic_features:
            temp = subset[[feature, 'delayed_15']].dropna()
            if len(temp) > 30:
                r, _ = stats.pointbiserialr(temp['delayed_15'], temp[feature])
                vals.append(abs(r))

        avg_period = float(np.mean(vals)) if vals else 0.0
        period_results[name] = avg_period

        print(f"{name:15s}: |r| = {avg_period:.4f} (n={len(subset)})")

# ===============================
# 3. VISUALIZATION
# ===============================
print("\n" + "="*80)
print("3️⃣  VISUALIZATION")
print("="*80)

features_clean = [f.replace('road_', '').replace('_', ' ') for f in traffic_features]
values = [correlations[f]['correlation'] for f in traffic_features]

plt.figure()
plt.bar(features_clean, values)
plt.axhline(0)
plt.title("Traffic vs Delay Correlation")
plt.ylabel("Correlation (r)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('results/traffic_9k/correlation.png', dpi=150)
print("📊 Saved: correlation.png")

# ===============================
# 4. FINAL CONCLUSION
# ===============================
print("\n" + "="*80)
print("4️⃣  FINAL CONCLUSION")
print("="*80)

if avg_r < 0.05:
    conclusion = "Traffic has negligible predictive value"
elif avg_r < 0.15:
    conclusion = "Traffic has weak but insignificant effect"
else:
    conclusion = "Traffic shows meaningful relationship"

print(f"\n🧠 Conclusion: {conclusion}")

# ===============================
# SAVE SUMMARY (SAFE JSON)
# ===============================
summary = {
    'dataset_size': int(len(df)),
    'avg_abs_correlation': avg_r,
    'max_abs_correlation': max_r,
    'verdict': str(verdict),
    'conclusion': str(conclusion),
    'feature_results': correlations,
    'time_periods': period_results
}

with open('results/traffic_9k/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n💾 Saved: results/traffic_9k/summary.json")

print("\n" + "="*80)
print("✅ TRAFFIC ANALYSIS COMPLETE!")
print("="*80)

