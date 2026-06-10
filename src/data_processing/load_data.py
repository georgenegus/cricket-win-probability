from pathlib import Path
import yaml
import pandas as pd


class CricketDataLoader:
    """
    Loads Cricsheet T20 YAML files and converts them into a structured DataFrame.
    Data is persisted to parquet for fast reloading on subsequent runs.
    """

    def __init__(self, data_path: str, processed_path: str = "data/processed"):
        self.data_path = Path(data_path)
        self.processed_path = Path(processed_path)
        self.cache_file = self.processed_path / "cricket_data.parquet"
        
        # Create processed directory if it doesn't exist
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_match(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single match YAML file and convert to ball-by-ball DataFrame.
        """

        with open(file_path, "r") as f:
            match = yaml.safe_load(f)

        match_id = file_path.stem
        innings_data = match.get("innings", [])

        rows = []

        for innings_index, innings in enumerate(innings_data):
            innings_name = list(innings.keys())[0]
            innings_info = innings[innings_name]

            batting_team = innings_info.get("team")
            deliveries = innings_info.get("deliveries", [])

            over_num = 0

            for delivery in deliveries:
                for ball_str, ball_data in delivery.items():

                    over, ball = map(int, str(ball_str).split("."))

                    runs = ball_data.get("runs", {})
                    wickets = ball_data.get("wickets", [])

                    rows.append({
                        "match_id": match_id,
                        "innings": innings_index + 1,
                        "batting_team": batting_team,
                        "over": over,
                        "ball": ball,
                        "runs_off_bat": runs.get("batsman", 0),
                        "extras": runs.get("extras", 0),
                        "total_runs": runs.get("total", 0),
                        "wicket": 1 if len(wickets) > 0 else 0,
                        "batter": ball_data.get("batsman"),
                        "bowler": ball_data.get("bowler"),
                    })

        return pd.DataFrame(rows)

    def load_all(self) -> pd.DataFrame:
        """
        Load all matches in dataset directory.
        Uses cached parquet file if available, otherwise loads from YAML and caches.
        """
        
        # Check if parquet cache exists
        if self.cache_file.exists():
            print(f"Loading from cache: {self.cache_file}")
            df = pd.read_parquet(self.cache_file)
            print(f"Loaded {len(df)} rows from cache")
            return df
        
        print(f"Loading from YAML files...")
        all_dfs = []

        for file_path in self.data_path.glob("**/*.yaml"):
            df = self.load_match(file_path)
            all_dfs.append(df)

        df = pd.concat(all_dfs, ignore_index=True)
        
        # Optimize dtypes for memory efficiency
        df["innings"] = df["innings"].astype("int8")
        df["over"] = df["over"].astype("int8")
        df["ball"] = df["ball"].astype("int8")
        df["runs_off_bat"] = df["runs_off_bat"].astype("int8")
        df["extras"] = df["extras"].astype("int8")
        df["total_runs"] = df["total_runs"].astype("int8")
        df["wicket"] = df["wicket"].astype("int8")
        df["batting_team"] = df["batting_team"].astype("category")
        df["batter"] = df["batter"].astype("category")
        df["bowler"] = df["bowler"].astype("category")
        
        # Save to parquet cache
        print(f"Saving to cache: {self.cache_file}")
        df.to_parquet(self.cache_file, index=False, compression="snappy")
        
        print(f"Loaded {len(df)} rows and saved to cache")
        return df
    
    def clear_cache(self):
        """Remove the cache file to force reloading from YAML."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"Cleared cache: {self.cache_file}")
        else:
            print(f"No cache file found: {self.cache_file}")