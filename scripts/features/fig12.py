import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# ===============================
# GLOBAL STYLE (CONSISTENT + CLEAN)
# ===============================
sns.set_theme(style="whitegrid", context="talk")

plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 11
})

os.makedirs('results/plots/propagation_9k', exist_ok=True)

print("="*80)
print("✈️  FLIGHT-LEVEL PROPAGATION ANALYSIS - FINAL")
print("="*80)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])
df = df.sort_values('hour').reset_index(drop=True)

print(f"\n📦 Dataset: {len(df):,} flights")
print(f"📊 Delay rate: {df['delayed_15'].mean()*100:.1f}%")

# =============================================================================
# 1. IMMEDIATE PROPAGATION (CORE RESULT)
# =============================================================================
propagation_results = []

for idx in range(len(df) - 10):
    current = df.iloc[idx]
    current_time = current['hour']
    current_delayed = current['delayed_15']
    
    next_window_end = current_time + pd.Timedelta(hours=2)
    next_flights = df[
        (df['hour'] > current_time) & 
        (df['hour'] <= next_window_end)
    ].head(5)
    
    if len(next_flights) >= 3:
        propagation_results.append({
            'current_delayed': current_delayed,
            'next_5_delay_rate': next_flights['delayed_15'].mean()
        })

prop_df = pd.DataFrame(propagation_results)

# FINAL VALUES (LOCKED)
after_delayed_prop = prop_df[prop_df['current_delayed'] == 1]['next_5_delay_rate'].mean()
after_ontime_prop = prop_df[prop_df['current_delayed'] == 0]['next_5_delay_rate'].mean()
contagion_next5 = after_delayed_prop - after_ontime_prop

print(f"\n🦠 Contagion Effect: {contagion_next5*100:+.1f} pp")

# =============================================================================
# 2. PROPAGATION DECAY
# =============================================================================
position_results = []

for position in [1, 2, 3, 5, 10]:
    pos_data = []
    
    for idx in range(len(df) - position - 5):
        current = df.iloc[idx]
        current_time = current['hour']
        current_delayed = current['delayed_15']
        
        next_window = df[df['hour'] > current_time].head(position + 5)
        
        if len(next_window) >= position:
            target = next_window.iloc[position - 1]
            time_gap = (target['hour'] - current_time).total_seconds() / 60
            
            if time_gap <= 240:
                pos_data.append({
                    'current_delayed': current_delayed,
                    'target_delayed': target['delayed_15']
                })
    
    if len(pos_data) > 100:
        pos_df = pd.DataFrame(pos_data)
        
        after_delayed_pos = pos_df[pos_df['current_delayed'] == 1]['target_delayed'].mean()
        after_ontime_pos = pos_df[pos_df['current_delayed'] == 0]['target_delayed'].mean()
        
        position_results.append({
            'position': position,
            'effect_pp': (after_delayed_pos - after_ontime_pos) * 100
        })

# =============================================================================
# 3. AIRLINE PROPAGATION
# =============================================================================
top_airlines = df['airline_name'].value_counts().head(10).index

airline_prop = []

for airline in top_airlines:
    airline_df = df[df['airline_name'] == airline].reset_index(drop=True)
    
    if len(airline_df) < 100:
        continue
    
    results = []
    
    for idx in range(len(airline_df) - 5):
        current = airline_df.iloc[idx]
        current_delayed = current['delayed_15']
        current_time = current['hour']
        
        next_flights = airline_df[
            airline_df['hour'] > current_time
        ].head(3)
        
        if len(next_flights) >= 2:
            results.append({
                'current_delayed': current_delayed,
                'next_delay_rate': next_flights['delayed_15'].mean()
            })
    
    if len(results) > 50:
        res_df = pd.DataFrame(results)
        
        after_delayed_air = res_df[res_df['current_delayed'] == 1]['next_delay_rate'].mean()
        after_ontime_air = res_df[res_df['current_delayed'] == 0]['next_delay_rate'].mean()
        
        airline_prop.append({
            'airline': airline,
            'effect_pp': (after_delayed_air - after_ontime_air) * 100
        })

airline_prop_df = pd.DataFrame(airline_prop).sort_values('effect_pp')

# =============================================================================
# 🔥 FINAL PLOT 1 — OVERALL PROPAGATION
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

labels = ['After Delayed', 'After On-Time']
values = [after_delayed_prop * 100, after_ontime_prop * 100]

bars = ax.bar(labels, values,
              color=['#d62828', '#2a9d8f'],
              edgecolor='black', linewidth=1.5)

# Value labels
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.8,
            f"{val:.1f}%",
            ha='center', fontweight='bold')

# Delta annotation
ax.annotate(f"Δ = {contagion_next5*100:+.1f} pp",
            xy=(0.5, max(values)+3),
            ha='center',
            fontsize=13,
            fontweight='bold',
            bbox=dict(boxstyle='round', fc='#fff3b0', ec='black'))

ax.set_title("Overall Delay Propagation Effect")
ax.set_ylabel("Delay Rate (%)")

# ✅ Balanced axis (not misleading, not flat)
ax.set_ylim(min(values)-5, max(values)+6)

sns.despine()
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('results/plots/propagation_9k/flight_level_contagion.png', dpi=300)

# =============================================================================
# 🔥 FINAL PLOT 2 — DECAY
# =============================================================================
if position_results:
    fig, ax = plt.subplots(figsize=(10, 6))
    pos_df = pd.DataFrame(position_results)

    ax.plot(pos_df['position'], pos_df['effect_pp'],
            marker='o', linewidth=2)

    ax.axhline(0, linestyle='--', alpha=0.6)

    # highlight peak
    peak = pos_df.loc[pos_df['effect_pp'].idxmax()]
    ax.scatter(peak['position'], peak['effect_pp'], s=120, zorder=3)

    ax.set_title("Delay Propagation Decay Pattern")
    ax.set_xlabel("Flight Position")
    ax.set_ylabel("Effect (percentage points)")

    sns.despine()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/propagation_9k/propagation_decay.png', dpi=300)

# =============================================================================
# 🔥 FINAL PLOT 3 — AIRLINE
# =============================================================================
if len(airline_prop_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['#d62828' if x > 0 else '#2a9d8f'
              for x in airline_prop_df['effect_pp']]

    ax.barh(airline_prop_df['airline'],
            airline_prop_df['effect_pp'],
            color=colors,
            edgecolor='black')

    ax.axvline(0, color='black', linewidth=1)

    ax.set_title("Airline-Specific Delay Propagation")
    ax.set_xlabel("Propagation Effect (percentage points)")

    sns.despine()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/propagation_9k/airline_propagation.png', dpi=300)

print("\n✅ FINAL VERSION — READY FOR THESIS")