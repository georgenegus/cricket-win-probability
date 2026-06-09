from pathlib import Path
import yaml
import pandas as pd


class CricketDataLoader:
    """
    Loads Cricsheet T20 YAML files and converts them into a structured DataFrame.
    Designed to be reusable across formats and datasets.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

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

                    over, ball = map(int, ball_str.split("."))

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
        """

        all_dfs = []

        for file_path in self.data_path.glob("*.yaml"):
            df = self.load_match(file_path)
            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)