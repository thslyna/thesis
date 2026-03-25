#!/usr/bin/env python3
"""
features_engineering.py

Reads the merged training_data.csv, creates model-friendly features and the target
label (delayed_15). Saves results to ~/thesis/data/processed/training_features.csv.

Fixes included:
 - Safely convert delay_min to numeric (coerce errors -> NaN)
 - Build delayed_15 using fillna(0)
 - Robust datetime parsing and timezone handling
 - Helpful prints for progress
"""

import os
import pandas as pd
from datetime import timezone
import pytz

ROOT = os.path.expanduser("~/thesis")
IN_PATH = os.path.join(ROOT, "data", "processed", "training_data.csv")
OUT_DIR = os.path.join(ROOT, "data", "processed")
OUT_PATH = os.path.join(OUT_DIR, "training_features.csv")

print("📦 Loading data...")
if not os.path.exists(IN_PATH):
    raise SystemExit(f"❌ Input not found: {IN_PATH}")

df = pd.read_csv(IN_PATH, low_memory=False)

print("🔎 Inspecting columns:", df.columns.tolist())

# --- Ensure delay_min is numeric (fix for your error) ---
print("🔧 Converting delay_min to numeric (coerce invalid -> NaN)...")
df['delay_min'] = pd.to_numeric(df.get('dep_delay')      # try common names first
                                if 'dep_delay' in df.columns else df.get('delay_min'),
                                errors='coerce')

# If above produced NaNs for many, also check other possible columns:
if df['delay_min'].isna().all():
    # Try 'arr_delay' or 'delay' fallbacks
    for alt in ['arr_delay', 'delay']:
        if alt in df.columns:
            df['delay_min'] = pd.to_numeric(df[alt], errors='coerce')
            if not df['delay_min'].isna().all():
                print(f"  → used fallback column: {alt}")
                break

n_invalid = df['delay_min'].isna().sum()
print(f"  delay_min NaNs after conversion: {n_invalid} / {len(df)}")

# --- Target label: delayed >= 15 min
# Treat missing delay as 0 (not delayed) for label creation
df['delayed_15'] = (df['delay_min'].fillna(0) >= 15).astype(int)
print("✅ Created target column 'delayed_15'")

# --- Parse datetimes and create temporal features ---
# We expect 'dep_scheduled' or 'dep_scheduled' style columns in input.
date_col = None
for c in ['dep_scheduled', 'departure.scheduled', 'dep_scheduled_utc', 'dep_scheduled']:
    if c in df.columns:
        date_col = c
        break

if date_col is None:
    # fallback to flight_date + scheduled time if available (less ideal)
    print("⚠️ No dep_scheduled column found. Attempting to use 'flight_date' if present.")
    date_col = None

if date_col:
    print(f"🔧 Parsing datetimes from column: {date_col}")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
    # convert to Istanbul timezone for 'hour'
    try:
        ist = pytz.timezone("Europe/Istanbul")
    except Exception:
        ist = pytz.FixedOffset(180)  # fallback +03:00
    df['hour'] = df[date_col].dt.tz_convert(ist).dt.floor('H')
else:
    # Create a synthetic 'hour' from flight_date if available (no timezone info)
    if 'flight_date' in df.columns:
        df['hour'] = pd.to_datetime(df['flight_date'], errors='coerce').dt.floor('D')
        print("  → Created 'hour' from 'flight_date' (day precision).")
    else:
        # last resort: current UTC hour for all (not ideal)
        print("❗ No datetime available. Filling 'hour' with current UTC hour (placeholder).")
        now = pd.to_datetime("now", utc=True).floor('H')
        df['hour'] = now

# Create simple temporal features
df['hour_of_day'] = df['hour'].dt.hour
df['day_of_week'] = df['hour'].dt.weekday
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

# --- Keep only the useful columns for modeling ---
# Identify commonly useful columns if present
keep = [
    'hour',
    'dep_scheduled', 'dep_actual', 'delay_min',
    'airline_name', 'flight_number',
    'temp', 'humidity', 'pressure', 'wind_speed', 'weather',
    'avg_speed', 'total_vehicles',
    'hour_of_day', 'day_of_week', 'is_weekend',
    'delayed_15'
]

# Only keep those that exist in df
keep_existing = [c for c in keep if c in df.columns]
# But ensure we always include our derived temporal features and delayed_15
for must in ['hour', 'hour_of_day', 'day_of_week', 'is_weekend', 'delayed_15']:
    if must not in keep_existing:
        keep_existing.append(must)

print("📋 Keeping columns:", keep_existing)

out_df = df[keep_existing].copy()

# Save
os.makedirs(OUT_DIR, exist_ok=True)
out_df.to_csv(OUT_PATH, index=False)
print(f"✅ Features saved to {OUT_PATH} with shape {out_df.shape}")
print("Columns:", out_df.columns.tolist())