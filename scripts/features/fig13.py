# ===============================
# FIGURE 4.8 TOP — AIRLINE PROPAGATION (UPDATED)
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="talk")

os.makedirs('results/plots', exist_ok=True)

print("📊 FIGURE 4.8 TOP — AIRLINE-SPECIFIC PROPAGATION")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])
df = df.sort_values('hour').reset_index(drop=True)

# ===============================
# COMPUTE PROPAGATION PER AIRLINE
# (same method as Table 4.14)
# ===============================
MIN_FLIGHTS = 50

airline_results = []

for airline, group in df.groupby('airline_name'):
    group = group.sort_values('hour').reset_index(drop=True)

    if len(group) < MIN_FLIGHTS:
        continue

    # skip corrupted datetime entries
    if any(c.isdigit() for c in airline[:4]):
        continue

    after_delayed_list = []
    after_ontime_list  = []

    for i in range(len(group) - 1):
        current_delayed = group.loc[i, 'delayed_15']
        next_delayed    = group.loc[i + 1, 'delayed_15']

        if current_delayed == 1:
            after_delayed_list.append(next_delayed)
        else:
            after_ontime_list.append(next_delayed)

    if not after_delayed_list or not after_ontime_list:
        continue

    after_delayed = pd.Series(after_delayed_list).mean() * 100
    after_ontime  = pd.Series(after_ontime_list).mean()  * 100
    effect        = after_delayed - after_ontime

    airline_results.append({
        'airline':    airline,
        'effect_pp':  round(effect, 1)
    })

# ===============================
# TOP 10 BY CONTAGION EFFECT
# ===============================
airline_prop_df = (pd.DataFrame(airline_results)
                     .sort_values('effect_pp', ascending=False)
                     .head(10)
                     .sort_values('effect_pp'))   # ascending for barh

print("\nTop 10 airlines by contagion:")
print(airline_prop_df.to_string(index=False))

# ===============================
# PLOT
# ===============================
fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#d62828' if x > 0 else '#2a9d8f'
          for x in airline_prop_df['effect_pp']]

bars = ax.barh(
    airline_prop_df['airline'],
    airline_prop_df['effect_pp'],
    color=colors,
    edgecolor='black',
    linewidth=1.2
)

ax.axvline(0, color='black', linewidth=1)

for bar, val in zip(bars, airline_prop_df['effect_pp']):
    ax.text(
        val + (0.5 if val >= 0 else -0.5),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.1f}",
        va='center',
        ha='left' if val >= 0 else 'right',
        fontsize=11,
        fontweight='bold'
    )

ax.set_title("Airline-Specific Delay Propagation", fontsize=14, fontweight='bold')
ax.set_xlabel("Propagation Effect (percentage points)")
ax.set_ylabel("Airline")

sns.despine()
ax.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()

# ===============================
# SAVE
# ===============================
output_path = "results/plots/fig_airline_propagation_updated.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Saved → {output_path}")