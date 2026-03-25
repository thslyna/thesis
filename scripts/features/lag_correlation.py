import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set working directory
os.chdir('/Users/lyna.thc/Desktop/thesis')

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

print("🔍 Loading data...")

# Load V1 data
df = pd.read_csv("data/processed/flights_with_weather_road_history.csv")

print(f"Loaded {len(df)} flights")

# Parse datetime
df['datetime'] = pd.to_datetime(df['hour'], utc=True)

# Sort by datetime
df = df.sort_values('datetime').reset_index(drop=True)

# Use existing delayed_15 column
print(f"\nDelayed flights: {df['delayed_15'].sum()} / {len(df)} ({df['delayed_15'].mean()*100:.1f}%)")
print("\n🔍 Testing if road traffic PREDICTS future delays...\n")

# Test different time lags (in number of flights)
lags = [0, 5, 10, 15, 20, 30]
correlations = []

for lag in lags:
    # Shift road traffic back by lag flights
    df[f'road_lag_{lag}'] = df['road_congestion_index'].shift(lag)
    
    # Correlation with current delays
    corr = df['delayed_15'].corr(df[f'road_lag_{lag}'])
    correlations.append(corr)
    
    print(f"Lag {lag} flights: correlation = {corr:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(lags, correlations, marker='o', linewidth=2, markersize=8)
plt.xlabel('Lag (number of flights)', fontsize=12)
plt.ylabel('Correlation with Delay', fontsize=12)
plt.title('Road Traffic → Delay: Which lag predicts best?', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No correlation')
plt.legend()
plt.tight_layout()
plt.savefig('results/lag_correlation.png', dpi=150, bbox_inches='tight')
print("\n📊 Plot saved: results/lag_correlation.png")

# Find best lag
best_lag_idx = np.argmax(np.abs(correlations))
best_lag = lags[best_lag_idx]
best_corr = correlations[best_lag_idx]

print(f"\n🎯 BEST LAG: {best_lag} flights")
print(f"   Correlation: {best_corr:.3f}")

if abs(best_corr) > 0.1:
    print(f"\n✅ Correlation found! Road traffic appears to have predictive power.")
else:
    print(f"\n⚠️ Weak correlation. Road traffic may not have strong temporal predictive power.")

