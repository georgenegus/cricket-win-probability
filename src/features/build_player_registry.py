import os
import urllib.request
import pandas as pd
from pathlib import Path


def setup_player_registry():
    dest_path = Path("data/processed/player_meta.csv")
    if dest_path.exists():
        print(f"Player registry already exists at {dest_path}")
        return

    os.makedirs(dest_path.parent, exist_ok=True)

    # Download from URL
    url = "https://raw.githubusercontent.com/mavaali/cricket-mcp/main/data/player_meta.csv"
    print(f"Downloading player registry from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode('utf-8')
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Downloaded and saved player registry to {dest_path}")
    except Exception as e:
        print(f"Error downloading player registry: {e}")


def normalize_batting_style(style):
    if not isinstance(style, str) or pd.isna(style) or style == "NA":
        return "RHB"  # default to RHB
    s = style.lower()
    if "left" in s:
        return "LHB"
    return "RHB"


def normalize_bowling_style(style):
    if not isinstance(style, str) or pd.isna(style) or style == "NA":
        return "RAP"  # default to right-arm pace

    s = style.lower()
    if "left-arm" in s or "left arm" in s:
        if "orthodox" in s or "sla" in s:
            return "SLA"
        elif "unorthodox" in s or "chinaman" in s or "wrist spin" in s or "wristspin" in s or "spin" in s:
            return "LAA"
        else:
            return "LAP"
    else:
        # right arm or others
        if "offbreak" in s or "off break" in s or "off-break" in s or "offspin" in s or "off spin" in s:
            return "OB"
        elif "legbreak" in s or "leg break" in s or "leg-break" in s or "googly" in s or "legspin" in s or "leg spin" in s:
            return "LB"
        elif "unorthodox" in s or "wrist spin" in s or "wristspin" in s:
            return "LAA"
        elif "spin" in s or "slow" in s:
            return "OB"
        else:
            return "RAP"


def load_player_styles():
    setup_player_registry()
    csv_path = Path("data/processed/player_meta.csv")
    if not csv_path.exists():
        print("Player registry file not found. Returning empty mapping.")
        return {}

    try:
        df = pd.read_csv(csv_path)
        # Drop rows with no cricsheet_id
        df = df[df["cricsheet_id"].notna()]

        mapping = {}
        for _, row in df.iterrows():
            cid = str(row["cricsheet_id"]).strip()
            bat = normalize_batting_style(row.get("batting_style"))
            bowl = normalize_bowling_style(row.get("bowling_style"))
            mapping[cid] = {"bat": bat, "bowl": bowl}

        return mapping
    except Exception as e:
        print(f"Error loading player registry mapping: {e}")
        return {}
