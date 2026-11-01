# ==========================================================
# TABLE 4.14 — DELAY CONTAGION BY AIRLINE (TOP 10)
# ==========================================================
import pandas as pd
import numpy as np
import os

os.makedirs("results/tables", exist_ok=True)

# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_csv("data/processed/flights_v1_IMPUTED_FIXED.csv")
df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour").reset_index(drop=True)

print(f"Dataset loaded: {df.shape}")

# ==========================================================
# 2. CHECK AIRLINE COLUMN NAME
# ==========================================================
# 🔴 Check which column holds the airline name/code
print("\nAvailable columns:", df.columns.tolist())
# Change "airline_iata" below to whatever your column is called
AIRLINE_COL = "airline_name"   # ← was "airline_iata"

# ==========================================================
# 3. COMPUTE CONTAGION PER AIRLINE
# ==========================================================
MIN_FLIGHTS = 50   # minimum flights for airline to be included

results = []

for airline, group in df.groupby(AIRLINE_COL):
    group = group.sort_values("hour").reset_index(drop=True)
    
    if len(group) < MIN_FLIGHTS:
        continue
    
    after_delayed_rates = []
    after_ontime_rates  = []
    
    for i in range(len(group) - 1):
        current_delayed = group.loc[i, "delayed_15"]
        next_delayed    = group.loc[i + 1, "delayed_15"]
        
        if current_delayed == 1:
            after_delayed_rates.append(next_delayed)
        else:
            after_ontime_rates.append(next_delayed)
    
    if not after_delayed_rates or not after_ontime_rates:
        continue
    
    after_delayed = np.mean(after_delayed_rates) * 100
    after_ontime  = np.mean(after_ontime_rates)  * 100
    contagion     = after_delayed - after_ontime
    
    results.append({
        "Airline":        airline,
        "Flights":        len(group),
        "After Delayed":  round(after_delayed, 1),
        "After On-time":  round(after_ontime,  1),
        "Contagion (pp)": round(contagion,      1),
    })

# ==========================================================
# 4. SORT BY CONTAGION (highest to lowest, like your table)
# ==========================================================
results_df = (pd.DataFrame(results)
                .sort_values("Contagion (pp)", ascending=False)
                .reset_index(drop=True))

print("\n=== TABLE 4.14 VALUES ===")
print(results_df.to_string(index=False))

# ==========================================================
# 5. SAVE TO CSV
# ==========================================================
output_path = "results/tables/table4_14_contagion_by_airline.csv"
results_df.to_csv(output_path, index=False)
print(f"\nSaved → {output_path}")