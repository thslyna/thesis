import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('/Users/lyna.thc/Desktop/thesis')
os.makedirs('results', exist_ok=True)

print("🔍 Weather × Road Traffic Interaction Analysis\n")

# Load data
df = pd.read_csv("data/processed/flights_with_weather_road_history.csv")

# Classify weather
def classify_weather(w):
    if pd.isna(w):
        return 'Unknown'
    w = w.lower()
    if any(x in w for x in ['rain', 'drizzle', 'shower']):
        return 'Rainy'
    elif any(x in w for x in ['snow', 'sleet']):
        return 'Snow'
    elif any(x in w for x in ['fog', 'mist', 'haze']):
        return 'Fog'
    elif any(x in w for x in ['clear', 'sun']):
        return 'Clear'
    elif any(x in w for x in ['cloud', 'overcast']):
        return 'Cloudy'
    else:
        return 'Other'

df['weather_category'] = df['weather'].apply(classify_weather)

# Analyze by weather condition
results = []
for weather in df['weather_category'].unique():
    subset = df[df['weather_category'] == weather]
    if len(subset) > 100:  # Minimum sample size
        corr = subset['delayed_15'].corr(subset['road_congestion_index'])
        results.append({
            'weather': weather,
            'correlation': corr,
            'n_flights': len(subset),
            'delay_rate': subset['delayed_15'].mean(),
            'avg_road_congestion': subset['road_congestion_index'].mean()
        })
        print(f"{weather:15} | Flights: {len(subset):4} | Correlation: {corr:6.3f} | Delay Rate: {subset['delayed_15'].mean()*100:.1f}% | Avg Road Congestion: {subset['road_congestion_index'].mean():.1f}")

# Sort by correlation
results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Correlation by weather
ax1 = axes[0, 0]
bars = ax1.barh(results_df['weather'], results_df['correlation'])
for i, bar in enumerate(bars):
    if results_df.iloc[i]['correlation'] > 0:
        bar.set_color('#4ecdc4')
    else:
        bar.set_color('#ff6b6b')
ax1.set_xlabel('Correlation (Road → Delay)', fontsize=11)
ax1.set_title('Road Traffic Impact by Weather Condition', fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(axis='x', alpha=0.3)

# 2. Delay rate by weather
ax2 = axes[0, 1]
ax2.barh(results_df['weather'], results_df['delay_rate']*100, color='#95e1d3')
ax2.set_xlabel('Delay Rate (%)', fontsize=11)
ax2.set_title('Delay Frequency by Weather', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. Sample size
ax3 = axes[1, 0]
ax3.barh(results_df['weather'], results_df['n_flights'], color='#ffeaa7')
ax3.set_xlabel('Number of Flights', fontsize=11)
ax3.set_title('Sample Size by Weather Condition', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Average road congestion
ax4 = axes[1, 1]
ax4.barh(results_df['weather'], results_df['avg_road_congestion'], color='#fab1a0')
ax4.set_xlabel('Average Road Congestion Index', fontsize=11)
ax4.set_title('Road Congestion by Weather', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/weather_road_interaction.png', dpi=150, bbox_inches='tight')
print("\n📊 Plot saved: results/weather_road_interaction.png")

# Key findings
print(f"\n🎯 KEY FINDINGS:")
top = results_df.iloc[0]
bottom = results_df.iloc[-1]

print(f"\n   STRONGEST correlation: {top['weather']} (r={top['correlation']:.3f})")
print(f"   WEAKEST correlation: {bottom['weather']} (r={bottom['correlation']:.3f})")

if results_df[results_df['weather'] == 'Rainy'].shape[0] > 0:
    rainy = results_df[results_df['weather'] == 'Rainy'].iloc[0]
    print(f"\n   Rainy conditions: correlation = {rainy['correlation']:.3f}, delay rate = {rainy['delay_rate']*100:.1f}%")
    
if results_df[results_df['weather'] == 'Clear'].shape[0] > 0:
    clear = results_df[results_df['weather'] == 'Clear'].iloc[0]
    print(f"   Clear conditions: correlation = {clear['correlation']:.3f}, delay rate = {clear['delay_rate']*100:.1f}%")

