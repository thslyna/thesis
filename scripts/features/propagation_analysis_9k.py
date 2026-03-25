import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

os.makedirs('results/propagation_9k', exist_ok=True)

print("="*80)
print("🦠 DELAY PROPAGATION ANALYSIS - 9.5K DATASET (CLEAN VERSION)")
print("="*80)

# Load dataset
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])

# Sort properly
df = df.sort_values('hour').reset_index(drop=True)

print(f"\n📦 Dataset: {len(df):,} flights")

# ==========================================================
# 1. HOURLY AGGREGATION (KEY FIX)
# ==========================================================

df['hour_group'] = df['hour'].dt.floor('h')

hourly = df.groupby('hour_group')['delayed_15'].mean().reset_index()

print(f"\n📊 Hourly observations: {len(hourly)}")

# ==========================================================
# 2. CONTAGION EFFECT (CLEAN VERSION)
# ==========================================================

print("\n1️⃣  CONTAGION EFFECT (HOURLY)")
print("-"*80)

# Next hour delay
hourly['next_hour_delay'] = hourly['delayed_15'].shift(-1)

# Remove last row
hourly = hourly.dropna()

# Define "delayed hour" (system-level, not flight-level)
hourly['current_delayed'] = (hourly['delayed_15'] > 0.5).astype(int)

delayed_next = hourly[hourly['current_delayed'] == 1]['next_hour_delay'].mean()
ontime_next = hourly[hourly['current_delayed'] == 0]['next_hour_delay'].mean()

contagion = delayed_next - ontime_next

print(f"\nWhen CURRENT HOUR is DELAYED (>50% flights delayed):")
print(f"  → Next hour delay rate: {delayed_next*100:.1f}%")

print(f"\nWhen CURRENT HOUR is ON-TIME:")
print(f"  → Next hour delay rate: {ontime_next*100:.1f}%")

print(f"\n🦠 CONTAGION EFFECT: {contagion*100:+.1f} percentage points")

# ==========================================================
# 3. VISUALIZATION
# ==========================================================

plt.figure(figsize=(10, 6))

categories = ['After\nDelayed Hour', 'After\nOn-Time Hour']
values = [delayed_next * 100, ontime_next * 100]

plt.bar(categories, values)

plt.ylabel('Next Hour Delay Rate (%)')
plt.title('Delay Propagation Effect (Hourly Aggregation)')
plt.ylim(0, max(values) * 1.2)

for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')

plt.tight_layout()
plt.savefig('results/propagation_9k/contagion_effect.png', dpi=150)

print(f"\n📊 Saved: results/propagation_9k/contagion_effect.png")

# ==========================================================
# 4. TEMPORAL DECAY
# ==========================================================

print("\n2️⃣  TEMPORAL DECAY")
print("-"*80)

decay_results = []

for h in [1, 2, 3]:
    hourly[f'delay_plus_{h}h'] = hourly['delayed_15'].shift(-h)
    
    temp = hourly.dropna()

    delayed_future = temp[temp['current_delayed'] == 1][f'delay_plus_{h}h'].mean()
    ontime_future = temp[temp['current_delayed'] == 0][f'delay_plus_{h}h'].mean()

    effect = delayed_future - ontime_future

    decay_results.append({
        'hours_ahead': h,
        'effect_pp': effect * 100
    })

    print(f"{h} hour(s) later: {effect*100:+.1f}pp")

# ==========================================================
# 5. SAVE RESULTS
# ==========================================================

summary = {
    'dataset_size': int(len(df)),
    'hourly_points': int(len(hourly)),
    'contagion_effect_pp': float(contagion * 100),
    'delayed_next_rate': float(delayed_next * 100),
    'ontime_next_rate': float(ontime_next * 100),
    'temporal_decay': decay_results
}

with open('results/propagation_9k/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n�� Summary saved: results/propagation_9k/summary.json")

print("\n" + "="*80)
print("✅ CLEAN PROPAGATION ANALYSIS COMPLETE!")
print("="*80)

