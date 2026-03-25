#!/usr/bin/env python3
"""
opensky_get_token.py
Reads ~/opensky_credentials.json (accepts keys clientId/client_id and clientSecret/client_secret),
attempts to request an OAuth2 client_credentials token from a small list of possible endpoints,
and prints the full response so you can see what's happening (status, body).

Save this file and run:
  python3 ~/thesis/scripts/data_collection/opensky_get_token.py
"""

import json
import os
import sys
import requests
from requests.auth import HTTPBasicAuth

CRED_PATH = os.path.expanduser("~/opensky_credentials.json")

if not os.path.exists(CRED_PATH):
    print("❌ Credentials file not found:", CRED_PATH)
    print("Create it with JSON like: {\"clientId\":\"...\",\"clientSecret\":\"...\"}")
    sys.exit(1)

with open(CRED_PATH, "r", encoding="utf-8") as f:
    try:
        creds = json.load(f)
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        sys.exit(1)

# Accept multiple key namings
client_id = creds.get("client_id") or creds.get("clientId") or creds.get("client")
client_secret = creds.get("client_secret") or creds.get("clientSecret") or creds.get("clientSecret")

if not client_id or not client_secret:
    print("❌ Credentials file missing client id / secret.")
    print("Current JSON keys found:", list(creds.keys()))
    print('Make sure it contains "clientId" and "clientSecret" (or "client_id" and "client_secret").')
    sys.exit(1)

# Candidate token endpoints to try (we'll try them and show results)
ENDPOINTS = [
    "https://opensky-network.org/api/token",
    "https://opensky-network.org/oauth/token",
    "https://opensky-network.org/api/auth/token",
    "https://opensky-network.org/o/token/",
    "https://opensky-network.org/oauth2/token",
]

print("🔄 Requesting access token...")
headers = {"User-Agent": "thesis-opensky-token-check/1.0"}
data = {"grant_type": "client_credentials"}

# Try POST with HTTP Basic auth (standard client_credentials) and also try form fields if Basic fails
for url in ENDPOINTS:
    try:
        r = requests.post(url, auth=HTTPBasicAuth(client_id, client_secret), data=data, headers=headers, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"→ {url}  --- network error:", str(e))
        continue

    print(f"\n→ Tried: {url}")
    print("  status:", r.status_code)
    # Print small body safely
    txt = r.text.strip()
    if len(txt) > 2000:
        txt_short = txt[:1800] + "\n...[truncated]..."
    else:
        txt_short = txt
    print("  body (first 2000 chars):\n", txt_short)

    # If success and json contains access_token, print it and exit
    try:
        j = r.json()
        if "access_token" in j:
            print("\n✅ Got access token!")
            print("access_token:", j["access_token"])
            if "expires_in" in j:
                print("expires_in (s):", j["expires_in"])
            # Save to file for later usage
            out = os.path.expanduser("~/opensky_token.json")
            with open(out, "w", encoding="utf-8") as fw:
                json.dump(j, fw, indent=2)
            print("Saved full token response to:", out)
            sys.exit(0)
    except Exception:
        pass

    # If 404 or 401 or 400, keep trying other endpoints
    if r.status_code in (200, 201):
        print("Note: 200/201 but no access_token in response. Check response body above.")
    else:
        print("Note: non-200 status, trying next possible endpoint...")

# If we got here, nothing worked
print("\n❌ All endpoints tried and no access token returned.")
print("What to do next (one-by-one):")
print("1) Open your OpenSky 'Account' page -> API Client. Click `Reset Credential` (if available) to get a fresh clientId/clientSecret.")
print('   Save them to ~/opensky_credentials.json with JSON keys: {"clientId":"...","clientSecret":"..."}')
print("2) If you created a credential, inspect the OpenSky docs page linked from your Account page. Check the token URL they specify.")
print("3) If OpenSky's API uses HTTP Basic + client_credentials (standard), this script should succeed once you have the correct endpoint.")
print("4) If the site gives you a single-line 'client id:secret' then split it into JSON as above.")
print("\nIf you want, paste here the exact response body (or the saved ~/opensky_token.json) and I will explain the meaning.")
sys.exit(1)