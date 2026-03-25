#!/bin/bash
set -e

# Explicit HOME (launchd does NOT set it)
export HOME="/Users/lyna.thc"

# Load keys
if [ -f "$HOME/.thesis_env" ]; then
  source "$HOME/.thesis_env"
else
  echo "❌ .thesis_env not found" >> "$HOME/Desktop/thesis/data/logs/logs_weather.txt"
  exit 78
fi

# Sanity check
if [ -z "$OPENWEATHER_KEY" ]; then
  echo "❌ OPENWEATHER_KEY missing" >> "$HOME/Desktop/thesis/data/logs/logs_weather.txt"
  exit 78
fi

# Run collector (use the SAME python that worked manually)
"$HOME/Desktop/thesis/venv/bin/python3" \
"$HOME/Desktop/thesis/scripts/data_collection/collect_weather.py" \
>> "$HOME/Desktop/thesis/data/logs/logs_weather.txt" 2>&1