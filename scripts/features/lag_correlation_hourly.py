import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('/Users/lyna.thc/Desktop/thesis')
os.makedirs('results', exist_ok=True)

print("🔍 Time-of-Day Analysis: Does road traffic matter more during rush hours?\n")

# Load data
df = pd.read_csv("data/processed/flights_with_weather_road_history.csv")
df['datetime'] = pd.to_datetime(df['hour'], utc=True)
df = df.sort_values('datetime').reset_index(drop=True)

# Define time periods
def classify_period(hour):
    if 6 <= hour < 9:
        return 'Morning Rush (6-9am)'
    elif 17 <= hour < 20:
        return 'Evening Rush (5-8pm)'
    else:
        return 'Off-Peak'

df['period'] = df['hour_of_day'].apply(classify_period)

# Calculate correlation for each period
results = []
for period in ['Morning Rush (6-9am)', 'Evening Rush (5-8pm)', 'Off-Peak']:
    subset = df[df['period'] == period]
    if len(subset) > 50:
        corr = subset['delayed_15'].corr(subset['road_congestion_index'])
        results.append({
            'period': period,
            'correlation': corr,
            'n_flights': len(subset),
            'delay_rate': subset['delayed_15'].mean()
        })
        print(f"{period:25} | Flights: {len(subset):4} | Correlation: {corr:.3f} | Delay Rate: {subset['delayed_15'].mean()*100:.1f}%")
    else:
        print(f"{period:25} | Flights: {len(subset):4} | SKIPPED (too few flights)")

# Plot only if we have results
if len(results) > 0:
    results_df = pd.DataFrame(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Correlation by period
    colors = ['#ff6b6b', '#4ecdc4', '#95e1d3'][:len(results_df)]
    ax1.bar(results_df['period'], results_df['correlation'], color=colors)
    ax1.set_ylabel('Correlation (Road Traffic → Delay)', fontsize=11)
    ax1.set_title('Road Traffic Impact by Time of Day', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(results_df['period'], rotation=15, ha='right')
    for i, row in results_df.iterrows():
        ax1.text(i, row['correlation'] + 0.01 if row['correlation'] > 0 else row['correlation'] - 0.02, 
                 f"{row['correlation']:.3f}", ha='center', fontsize=10, fontweight='bold')

    # Delay rate by period
    ax2.bar(results_df['period'], results_df['delay_rate']*100, color=colors)
    ax2.set_ylabel('Delay Rate (%)', fontsize=11)
    ax2.set_title('Delays by Time of Day', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels(results_df['period'], rotation=15, ha='right')
    for i, row in results_df.iterrows():
        ax2.text(i, row['delay_rate']*100 + 2, f"{row['delay_rate']*100:.1f}%", 
                 ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/rush_hour_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 Plot saved: results/rush_hour_analysis.png")

    # Key finding
    print(f"\n🎯 KEY FINDINGS:")
    
    evening_data = results_df[results_df['period'] == 'Evening Rush (5-8pm)']
    offpeak_data = results_df[results_df['period'] == 'Off-Peak']
    
    if len(evening_data) > 0:
        evening_corr = evening_data['correlation'].values[0]
        evening_delay = evening_data['delay_rate'].values[0]
        print(f"   Evening Rush: correlation = {evening_corr:.3f}, delay rate = {evening_delay*100:.1f}%")
    
    if len(offpeak_data) > 0:
        offpeak_corr = offpeak_data['correlation'].values[0]
        offpeak_delay = offpeak_data['delay_rate'].values[0]
        print(f"   Off-Peak: correlation = {offpeak_corr:.3f}, delay rate = {offpeak_delay*100:.1f}%")
    
    if len(evening_data) > 0 and len(offpeak_data) > 0:
        if abs(evening_corr) > abs(offpeak_corr):
            print(f"\n   ⚠️ SURPRISING: Evening rush shows NEGATIVE correlation!")
            print(f"   This suggests road traffic during evening rush is INVERSELY related to delays")
            print(f"   Possible explanation: Heavy traffic = more careful scheduling/padding")
        else:
            print(f"\n   ✅ Off-peak shows STRONGER positive correlation")
            print(f"   Road traffic matters more when airport is less busy")

else:
    print("\n❌ Not enough data for time-of-day analysis")

