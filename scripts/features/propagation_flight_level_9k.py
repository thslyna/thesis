import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

os.makedirs('results/propagation_9k', exist_ok=True)

print("="*80)
print("✈️  FLIGHT-LEVEL PROPAGATION ANALYSIS - 9.5K DATASET")
print("="*80)

# Load dataset
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])
df = df.sort_values('hour').reset_index(drop=True)

print(f"\n📦 Dataset: {len(df):,} flights")
print(f"📊 Delay rate: {df['delayed_15'].mean()*100:.1f}%")

# =============================================================================
# 1. IMMEDIATE NEXT FLIGHT PROPAGATION (within same hour or next hour)
# =============================================================================
print("\n" + "="*80)
print("1️⃣  IMMEDIATE PROPAGATION: Next 5 Flights After Each Flight")
print("="*80)

propagation_results = []

for idx in range(len(df) - 10):  # Leave buffer for next flights
    current_flight = df.iloc[idx]
    current_time = current_flight['hour']
    current_delayed = current_flight['delayed_15']
    
    # Look at next 5 flights within 2-hour window
    next_window_end = current_time + pd.Timedelta(hours=2)
    next_flights = df[
        (df['hour'] > current_time) & 
        (df['hour'] <= next_window_end)
    ].head(5)
    
    if len(next_flights) >= 3:  # Need at least 3 flights for meaningful analysis
        propagation_results.append({
            'current_delayed': current_delayed,
            'next_5_delay_rate': next_flights['delayed_15'].mean(),
            'next_5_count': len(next_flights),
            'time_gap_minutes': (next_flights.iloc[0]['hour'] - current_time).total_seconds() / 60
        })

prop_df = pd.DataFrame(propagation_results)

# Calculate contagion
after_delayed = prop_df[prop_df['current_delayed'] == 1]['next_5_delay_rate'].mean()
after_ontime = prop_df[prop_df['current_delayed'] == 0]['next_5_delay_rate'].mean()
contagion_next5 = after_delayed - after_ontime

print(f"\nAnalyzed {len(prop_df):,} flight sequences")
print(f"\nWhen CURRENT flight is DELAYED:")
print(f"  → Next 5 flights delay rate: {after_delayed*100:.1f}%")
print(f"  → Sample size: {len(prop_df[prop_df['current_delayed'] == 1]):,}")

print(f"\nWhen CURRENT flight is ON-TIME:")
print(f"  → Next 5 flights delay rate: {after_ontime*100:.1f}%")
print(f"  → Sample size: {len(prop_df[prop_df['current_delayed'] == 0]):,}")

print(f"\n🦠 CONTAGION EFFECT (next 5 flights): {contagion_next5*100:+.1f} percentage points")

# Statistical significance test
from scipy import stats
delayed_group = prop_df[prop_df['current_delayed'] == 1]['next_5_delay_rate']
ontime_group = prop_df[prop_df['current_delayed'] == 0]['next_5_delay_rate']
t_stat, p_value = stats.ttest_ind(delayed_group, ontime_group)
print(f"📊 Statistical test: t={t_stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print(f"   ✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print(f"   ⚠️  Not significant (p > 0.05)")

# =============================================================================
# 2. DISTANCE-BASED PROPAGATION (1st, 2nd, 3rd flight after)
# =============================================================================
print("\n" + "="*80)
print("2️⃣  PROPAGATION BY FLIGHT POSITION")
print("="*80)

position_results = []

for position in [1, 2, 3, 5, 10]:
    pos_data = []
    
    for idx in range(len(df) - position - 5):
        current = df.iloc[idx]
        current_delayed = current['delayed_15']
        current_time = current['hour']
        
        # Get the Nth flight after current
        next_window = df[df['hour'] > current_time].head(position + 5)
        
        if len(next_window) >= position:
            target_flight = next_window.iloc[position - 1]
            time_gap = (target_flight['hour'] - current_time).total_seconds() / 60
            
            # Only consider if within reasonable window (4 hours)
            if time_gap <= 240:
                pos_data.append({
                    'current_delayed': current_delayed,
                    'target_delayed': target_flight['delayed_15'],
                    'time_gap_min': time_gap
                })
    
    if len(pos_data) > 100:
        pos_df = pd.DataFrame(pos_data)
        
        after_delayed = pos_df[pos_df['current_delayed'] == 1]['target_delayed'].mean()
        after_ontime = pos_df[pos_df['current_delayed'] == 0]['target_delayed'].mean()
        effect = after_delayed - after_ontime
        
        avg_gap = pos_df['time_gap_min'].mean()
        
        position_results.append({
            'position': position,
            'effect_pp': effect * 100,
            'avg_time_gap_min': avg_gap,
            'sample_size': len(pos_df)
        })
        
        print(f"\n{position}{'st' if position==1 else 'nd' if position==2 else 'rd' if position==3 else 'th'} flight after: {effect*100:+.1f}pp (avg gap: {avg_gap:.0f} min, n={len(pos_df):,})")

# =============================================================================
# 3. AIRLINE-SPECIFIC PROPAGATION
# =============================================================================
print("\n" + "="*80)
print("3️⃣  AIRLINE-SPECIFIC PROPAGATION")
print("="*80)

# Get top airlines
top_airlines = df['airline_name'].value_counts().head(10).index

airline_prop = []

for airline in top_airlines:
    airline_flights = df[df['airline_name'] == airline].reset_index(drop=True)
    
    if len(airline_flights) < 100:
        continue
    
    airline_results = []
    
    for idx in range(len(airline_flights) - 5):
        current = airline_flights.iloc[idx]
        current_delayed = current['delayed_15']
        current_time = current['hour']
        
        # Next 3 flights from SAME airline
        next_same_airline = airline_flights[
            airline_flights['hour'] > current_time
        ].head(3)
        
        if len(next_same_airline) >= 2:
            airline_results.append({
                'current_delayed': current_delayed,
                'next_delay_rate': next_same_airline['delayed_15'].mean()
            })
    
    if len(airline_results) > 50:
        airline_df = pd.DataFrame(airline_results)
        
        after_delayed = airline_df[airline_df['current_delayed'] == 1]['next_delay_rate'].mean()
        after_ontime = airline_df[airline_df['current_delayed'] == 0]['next_delay_rate'].mean()
        effect = after_delayed - after_ontime
        
        airline_prop.append({
            'airline': airline,
            'effect_pp': effect * 100,
            'sample_size': len(airline_df)
        })

# Sort by effect size
airline_prop_df = pd.DataFrame(airline_prop).sort_values('effect_pp', ascending=False)

print(f"\nTop 5 airlines with STRONGEST propagation:")
for idx, row in airline_prop_df.head(5).iterrows():
    print(f"  {row['airline']:30s}: {row['effect_pp']:+6.1f}pp (n={row['sample_size']:,})")

print(f"\nTop 5 airlines with WEAKEST/NEGATIVE propagation:")
for idx, row in airline_prop_df.tail(5).iterrows():
    print(f"  {row['airline']:30s}: {row['effect_pp']:+6.1f}pp (n={row['sample_size']:,})")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "="*80)
print("📊 CREATING VISUALIZATIONS")
print("="*80)

# Plot 1: Contagion effect (next 5 flights)
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['After\nDelayed Flight', 'After\nOn-Time Flight']
values = [after_delayed * 100, after_ontime * 100]
colors = ['#e74c3c', '#2ecc71']

bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Next 5 Flights Delay Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Flight-Level Delay Contagion (9.5k Dataset)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(values) * 1.2)

for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, value + 2, 
            f'{value:.1f}%', ha='center', fontsize=13, fontweight='bold')

# Add difference annotation
ax.text(0.5, max(values) * 1.1, f'Δ = {contagion_next5*100:+.1f}pp', 
        ha='center', fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('results/propagation_9k/flight_level_contagion.png', dpi=150, bbox_inches='tight')
print("✅ Saved: results/propagation_9k/flight_level_contagion.png")

# Plot 2: Propagation by position
if position_results:
    fig, ax = plt.subplots(figsize=(10, 6))
    pos_df = pd.DataFrame(position_results)
    
    ax.plot(pos_df['position'], pos_df['effect_pp'], marker='o', linewidth=2, 
            markersize=10, color='#3498db')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Flight Position After Current Flight', fontsize=12, fontweight='bold')
    ax.set_ylabel('Propagation Effect (percentage points)', fontsize=12, fontweight='bold')
    ax.set_title('Delay Propagation Decay Pattern', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/propagation_9k/propagation_decay.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/propagation_9k/propagation_decay.png")

# Plot 3: Airline-specific
if len(airline_prop_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    airlines_sorted = airline_prop_df.sort_values('effect_pp')
    colors_airline = ['#e74c3c' if x > 0 else '#2ecc71' for x in airlines_sorted['effect_pp']]
    
    ax.barh(range(len(airlines_sorted)), airlines_sorted['effect_pp'], 
            color=colors_airline, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(airlines_sorted)))
    ax.set_yticklabels(airlines_sorted['airline'], fontsize=10)
    ax.set_xlabel('Propagation Effect (percentage points)', fontsize=12, fontweight='bold')
    ax.set_title('Airline-Specific Delay Propagation', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('results/propagation_9k/airline_propagation.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/propagation_9k/airline_propagation.png")

# =============================================================================
# SAVE SUMMARY
# =============================================================================
summary = {
    'dataset_size': len(df),
    'overall_delay_rate_pct': df['delayed_15'].mean() * 100,
    'flight_level_analysis': {
        'next_5_flights_contagion_pp': contagion_next5 * 100,
        'after_delayed_rate_pct': after_delayed * 100,
        'after_ontime_rate_pct': after_ontime * 100,
        'statistical_significance_p': p_value,
        'sample_size': len(prop_df)
    },
    'position_analysis': position_results,
    'airline_analysis': airline_prop_df.to_dict('records')
}

with open('results/propagation_9k/flight_level_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✅ Summary saved: results/propagation_9k/flight_level_summary.json")

print("\n" + "="*80)
print("✅ FLIGHT-LEVEL PROPAGATION ANALYSIS COMPLETE!")
print("="*80)

