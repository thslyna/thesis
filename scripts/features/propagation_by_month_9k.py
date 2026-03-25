import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('results/final_plots', exist_ok=True)

print("="*80)
print("📆 PROPAGATION ANALYSIS BY MONTH")
print("="*80)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])
df = df.sort_values('hour').reset_index(drop=True)
df['month'] = df['hour'].dt.month

# ===============================
# FUNCTION: CONTAGION EFFECT
# ===============================
def compute_contagion(data):
    results = []

    for idx in range(len(data) - 50):
        current = data.iloc[idx]
        t = current['hour']
        delayed = current['delayed_15']

        next_window = data[
            (data['hour'] > t) &
            (data['hour'] <= t + pd.Timedelta(hours=2))
        ].head(5)

        if len(next_window) >= 3:
            results.append({
                'current_delayed': delayed,
                'next_rate': next_window['delayed_15'].mean()
            })

    df_res = pd.DataFrame(results)

    if len(df_res) == 0:
        return np.nan

    after_delayed = df_res[df_res['current_delayed']==1]['next_rate'].mean()
    after_ontime = df_res[df_res['current_delayed']==0]['next_rate'].mean()

    return (after_delayed - after_ontime) * 100

# ===============================
# RUN BY MONTH GROUPS
# ===============================
groups = {
    "Oct-Nov (Train)": [10,11],
    "December": [12],
    "Jan-Feb (Future)": [1,2]
}

results = {}

for name, months in groups.items():
    subset = df[df['month'].isin(months)].copy()

    print("\n" + "-"*60)
    print(name)
    print("-"*60)

    print(f"Flights: {len(subset)}")
    print(f"Delay rate: {subset['delayed_15'].mean()*100:.1f}%")

    contagion = compute_contagion(subset)

    print(f"🦠 Contagion effect: {contagion:+.2f} pp")

    results[name] = contagion

# ===============================
# SUMMARY
# ===============================
print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)

for k, v in results.items():
    print(f"{k:20s}: {v:+.2f} pp")

if not np.isnan(results["Oct-Nov (Train)"]) and not np.isnan(results["Jan-Feb (Future)"]):
    drop = results["Oct-Nov (Train)"] - results["Jan-Feb (Future)"]
    print(f"\n🔥 PROPAGATION CHANGE: {drop:+.2f} pp")

# ===============================
# 📊 FINAL PLOT
# ===============================
labels = list(results.keys())
values = list(results.values())

plt.figure()
bars = plt.bar(labels, values)

plt.axhline(0)
plt.ylabel("Contagion Effect (pp)")
plt.title("Delay Propagation Over Time")

# annotate values
for bar, v in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        v + (1 if v > 0 else -2),
        f"{v:+.1f}",
        ha='center'
    )

plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('results/final_plots/propagation_by_month.png', dpi=150)

print("📊 Saved: results/final_plots/propagation_by_month.png")

print("\n" + "="*80)
print("✅ DONE")
print("="*80)