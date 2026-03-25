import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os

os.chdir('/Users/lyna.thc/Desktop/thesis')
os.makedirs('results/cost_benefit', exist_ok=True)

print("💰 COST-BENEFIT ANALYSIS: Business Value of Delay Prediction\n")

# Load data and model (FIXED)
df = pd.read_csv("data/processed/flights_v1_IMPUTED_FIXED.csv")
df['hour'] = pd.to_datetime(df['hour'])

model = joblib.load("results/xgb_untuned/xgb_untuned_model.joblib")

# 🔥 TEST SPLIT (ADDED)
train_size = int(len(df) * 0.75)
test = df.iloc[train_size:].reset_index(drop=True)

# Prepare data (FIXED → use test instead of df)
DROP_COLS = ["flight_date", "hour", "flight_iata", "delayed_15"]
X = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
y = test["delayed_15"]

# Get predictions
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

# Confusion matrix
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

print("="*70)
print("📊 PREDICTION RESULTS:")
print("="*70)
print(f"True Negatives (TN):  {tn:4d} - Correctly predicted on-time")
print(f"False Positives (FP): {fp:4d} - False alarms (predicted delay, was on-time)")
print(f"False Negatives (FN): {fn:4d} - Missed delays (predicted on-time, was delayed)")
print(f"True Positives (TP):  {tp:4d} - Correctly predicted delays")

# COST PARAMETERS (unchanged)
print("\n" + "="*70)
print("💶 COST PARAMETERS (Industry Estimates):")
print("="*70)

COST_PER_DELAY = 5000
print(f"Cost per delay (baseline):           €{COST_PER_DELAY:,}")

PREVENTION_RATE = 0.30
PREVENTION_SAVINGS = COST_PER_DELAY * 0.80

print(f"Delay prevention rate:               {PREVENTION_RATE*100:.0f}%")
print(f"Savings per prevented delay:         €{PREVENTION_SAVINGS:,}")

COST_FALSE_ALARM = 200
print(f"Cost per false alarm:                €{COST_FALSE_ALARM:,}")

# BASELINE
print("\n" + "="*70)
print("📊 BASELINE (No Prediction System):")
print("="*70)

total_delays = tp + fn
baseline_cost = total_delays * COST_PER_DELAY

print(f"Total delays: {total_delays:,}")
print(f"Total cost: €{baseline_cost:,}")

# WITH SYSTEM
print("\n" + "="*70)
print("🤖 WITH PREDICTION SYSTEM:")
print("="*70)

prevented_delays = int(tp * PREVENTION_RATE)
unprevented_delays = tp - prevented_delays

savings_from_prevention = prevented_delays * PREVENTION_SAVINGS
cost_of_unprevented = unprevented_delays * COST_PER_DELAY

print(f"\nTrue Positives (TP = {tp}):")
print(f"  Delays prevented:   {prevented_delays:4d} × €{PREVENTION_SAVINGS:,} = €{savings_from_prevention:,}")
print(f"  Delays unprevented: {unprevented_delays:4d} × €{COST_PER_DELAY:,} = €{cost_of_unprevented:,}")

cost_of_missed = fn * COST_PER_DELAY
print(f"\nFalse Negatives (FN = {fn}):")
print(f"  Missed delays:      {fn:4d} × €{COST_PER_DELAY:,} = €{cost_of_missed:,}")

cost_of_false_alarms = fp * COST_FALSE_ALARM
print(f"\nFalse Positives (FP = {fp}):")
print(f"  False alarms:       {fp:4d} × €{COST_FALSE_ALARM:,} = €{cost_of_false_alarms:,}")

total_cost_with_system = cost_of_unprevented + cost_of_missed + cost_of_false_alarms

print("\n" + "="*70)
print(f"Total cost with system: €{total_cost_with_system:,}")

# NET BENEFIT
net_savings = baseline_cost - total_cost_with_system
roi_percent = (net_savings / baseline_cost) * 100

print("\n" + "="*70)
print("💰 NET BENEFIT:")
print("="*70)
print(f"Baseline cost (no system):    €{baseline_cost:,}")
print(f"Cost with system:             €{total_cost_with_system:,}")
print(f"NET SAVINGS:                  €{net_savings:,}")
print(f"ROI:                          {roi_percent:.1f}%")

# Per-flight savings (FIXED → use test)
flights_per_month = len(test) / 2
savings_per_flight = net_savings / len(test)
monthly_savings = (net_savings / 2)
yearly_savings = monthly_savings * 12

print(f"\nSavings per flight:           €{savings_per_flight:.2f}")
print(f"Monthly savings:              €{monthly_savings:,.0f}")
print(f"Projected yearly savings:     €{yearly_savings:,.0f}")

# ===== EVERYTHING BELOW UNCHANGED =====
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
categories = ['Baseline\n(No System)', 'With\nPrediction\nSystem']
costs = [baseline_cost, total_cost_with_system]
colors = ['#ff6b6b', '#4ecdc4']
bars = ax1.bar(categories, costs, color=colors, width=0.6)
ax1.set_ylabel('Total Cost (€)', fontsize=12, fontweight='bold')
ax1.set_title('Total Cost Comparison', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for bar, cost in zip(bars, costs):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'€{cost:,.0f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.annotate('', xy=(1, total_cost_with_system), xytext=(1, baseline_cost),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax1.text(1.15, (baseline_cost + total_cost_with_system)/2, 
         f'SAVINGS\n€{net_savings:,.0f}\n({roi_percent:.1f}% ROI)',
         fontsize=10, fontweight='bold', color='green',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax2 = axes[0, 1]
components = ['Prevented\nDelays\n(Savings)', 'Unprevented\nDelays', 'Missed\nDelays', 'False\nAlarms']
values = [savings_from_prevention, -cost_of_unprevented, -cost_of_missed, -cost_of_false_alarms]
colors_comp = ['green', 'orange', 'red', 'yellow']

bars2 = ax2.bar(components, values, color=colors_comp)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_ylabel('Impact (€)', fontsize=12, fontweight='bold')
ax2.set_title('Cost-Benefit Components', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars2, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'€{abs(val):,.0f}',
             ha='center', va='bottom' if val > 0 else 'top', 
             fontsize=9, fontweight='bold')

ax3 = axes[1, 0]
outcomes = [f'TP\n{tp}', f'FP\n{fp}', f'FN\n{fn}', f'TN\n{tn}']
counts = [tp, fp, fn, tn]
colors_out = ['#4ecdc4', '#ffeaa7', '#ff6b6b', '#95e1d3']
bars3 = ax3.bar(outcomes, counts, color=colors_out)
ax3.set_ylabel('Number of Flights', fontsize=12, fontweight='bold')
ax3.set_title('Prediction Outcomes Distribution', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for bar, count in zip(bars3, counts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:,}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4 = axes[1, 1]
periods = ['Per\nFlight', 'Monthly', 'Yearly']
period_savings = [savings_per_flight, monthly_savings, yearly_savings]
bars4 = ax4.bar(periods, period_savings, color='#4ecdc4')
ax4.set_ylabel('Savings (€)', fontsize=12, fontweight='bold')
ax4.set_title('Projected Savings', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

for bar, saving in zip(bars4, period_savings):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'€{saving:,.0f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/cost_benefit/cost_benefit_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n📊 Visualization saved: results/cost_benefit/cost_benefit_analysis.png")

summary_df.to_csv('results/cost_benefit/cost_benefit_summary.csv', index=False)
print("📄 Summary saved: results/cost_benefit/cost_benefit_summary.csv")

print("\n✅ Cost-benefit analysis complete!")
