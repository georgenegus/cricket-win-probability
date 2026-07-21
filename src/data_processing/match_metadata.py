from pathlib import Path
import yaml
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader


def _extract_single_match_metadata(file_path: Path) -> dict:
    """
    Parses a single YAML match file and extracts match-level metadata.
    Designed to run in parallel processes.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            match_data = yaml.load(f, Loader=SafeLoader)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    if not match_data:
        return None

    info = match_data.get("info", {})
    match_id = file_path.stem

    # Dates
    dates = info.get("dates", [])
    date_str = str(dates[0]) if dates else ""

    # Teams
    teams = info.get("teams", [])
    if len(teams) != 2:
        return None

    # Innings to find who batted first
    innings_data = match_data.get("innings", [])
    if not innings_data:
        return None

    first_innings_name = list(innings_data[0].keys())[0]
    team_1 = innings_data[0][first_innings_name].get("team")

    if team_1 not in teams:
        return None

    team_2 = teams[1] if teams[0] == team_1 else teams[0]

    # Outcome
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    method = outcome.get("method", "")

    win_margin = 0
    win_type = None

    by_info = outcome.get("by", {})
    if "runs" in by_info:
        win_margin = int(by_info["runs"])
        win_type = "runs"
    elif "wickets" in by_info:
        win_margin = int(by_info["wickets"])
        win_type = "wickets"
    elif outcome.get("result") == "tie":
        win_type = "tie"
    elif outcome.get("result") == "no result":
        win_type = "no_result"
    elif "eliminator" in outcome:
        win_type = "tie"

    toss = info.get("toss", {})
    toss_winner = toss.get("winner", "")
    toss_decision = toss.get("decision", "")

    players = info.get("players", {})
    team_1_players = players.get(team_1, [])
    team_2_players = players.get(team_2, [])

    return {
        "match_id": match_id,
        "date": date_str,
        "competition": file_path.parent.name,
        "team_1": team_1,  # Batted first
        "team_2": team_2,  # Batted second (chasing)
        "team_1_players": ",".join(team_1_players),
        "team_2_players": ",".join(team_2_players),
        "winner": winner,
        "win_margin": win_margin,
        "win_type": win_type,
        "method": method,
        "team_type": info.get("team_type", ""),
        "gender": info.get("gender", "male"),
        "venue": info.get("venue", ""),
        "city": info.get("city", ""),
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
    }


from src.utils.cache_manager import CacheManager


class MatchMetadataExtractor:
    """
    Extracts match-level metadata (teams, winner, toss, date, venue, outcome) from Cricsheet YAMLs.
    Used to build historical match context for Elo calculations.
    """

    def __init__(self, raw_data_path: str, processed_path: str = "data/processed", format_type: str = "t20"):
        self.format_type = format_type
        base_path = Path(raw_data_path)
        if base_path.name in ("t20", "od"):
            self.raw_data_path = base_path
        else:
            self.raw_data_path = base_path / format_type
            
        self.processed_path = Path(processed_path)
        self.cache_file = self.processed_path / f"{format_type}_match_metadata.parquet"
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def extract_all(self, force=False) -> pd.DataFrame:
        if not force and CacheManager.is_cache_valid(self.cache_file, [Path(__file__)]):
            print(f"Loading match metadata from cache: {self.cache_file}")
            return pd.read_parquet(self.cache_file)

        print("Extracting match metadata from YAML files...")
        file_paths = list(self.raw_data_path.glob("**/*.yaml"))
        
        # Deduplicate paths by file stem (match_id)
        seen_stems = set()
        unique_paths = []
        for p in file_paths:
            if p.stem not in seen_stems:
                seen_stems.add(p.stem)
                unique_paths.append(p)
        file_paths = unique_paths

        all_metadata = []
        try:
            max_workers = min(4, os.cpu_count() or 2)
            print(f"Using {max_workers} parallel workers to parse match metadata...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(
                    _extract_single_match_metadata, file_paths, chunksize=50
                )
                for res in results:
                    if res is not None:
                        all_metadata.append(res)
        except Exception as e:
            print(f"Parallel metadata extraction failed: {e}. Falling back to sequential.")
            all_metadata = []
            for file_path in file_paths:
                res = _extract_single_match_metadata(file_path)
                if res is not None:
                    all_metadata.append(res)

        df = pd.DataFrame(all_metadata)

        # Sort chronologically by date and match_id
        df = df.sort_values(by=["date", "match_id"]).reset_index(drop=True)

        # Save to cache
        print(f"Saving match metadata to cache: {self.cache_file}")
        df.to_parquet(self.cache_file, index=False, compression="snappy")
        CacheManager.register_cache(self.cache_file, [Path(__file__)])
        return df

    def clear_cache(self):
        CacheManager.clear_cache(self.cache_file)
