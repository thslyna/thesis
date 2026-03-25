#!/bin/bash
set -e

# -----------------------
# Explicit environment
# -----------------------
export HOME="/Users/lyna.thc"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"

# -----------------------
# Load API keys
# -----------------------
if [ -f "${HOME}/.thesis_env" ]; then
  . "${HOME}/.thesis_env"
else
  echo "❌ .thesis_env not found" >> /tmp/flights_wrapper_error.log
  exit 1
fi

# -----------------------
# Absolute paths
# -----------------------
PYTHON="/opt/homebrew/bin/python3"
SCRIPT="/Users/lyna.thc/Desktop/thesis/scripts/data_collection/collect_flights.py"
LOG="/Users/lyna.thc/Desktop/thesis/data/logs/logs_flights.txt"

# -----------------------
# Log rotation
# -----------------------
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG")" -gt 500000 ]; then
  mv "$LOG" "${LOG}.$(date +%Y%m%d%H%M%S)"
fi

# -----------------------
# Run
# -----------------------
echo "=== run_flights_wrapper START $(date -u +'%Y-%m-%dT%H:%M:%SZ') ===" >> "$LOG"
"$PYTHON" "$SCRIPT" >> "$LOG" 2>&1
EXIT=$?
echo "=== run_flights_wrapper END $(date -u +'%Y-%m-%dT%H:%M:%SZ') exit=$EXIT ===" >> "$LOG"

exit $EXIT