import os
import pandas as pd
from pathlib import Path
import math


from src.utils.cache_manager import CacheManager


class EloEngine:
    """
    Computes chronological rolling Elo ratings per competition pool.
    Extensible to support dynamic K-factor, margin of victory scaling,
    off-season decay, and home/away adjustments.
    """


    def __init__(
        self,
        k_factor=None,
        use_dynamic_k=False,
        use_margin=False,
        use_decay=False,
        use_home_away=False,
        decay_days=None,
        decay_rate=None,
        margin_runs_coef=None,
        margin_runs_div=None,
        margin_wickets_coef=None,
        k_max=None,
        k_min=None,
        k_scale=None,
        processed_path="data/processed",
        format_type="t20",
    ):
        self.use_dynamic_k = use_dynamic_k
        self.use_margin = use_margin
        self.use_decay = use_decay
        self.use_home_away = use_home_away
        self.processed_path = Path(processed_path)
        self.format_type = format_type
        self.cache_file = self.processed_path / f"{format_type}_elo_ratings.parquet"
        os.makedirs(self.processed_path, exist_ok=True)

        # Set format-specific defaults. odw is also a 50-over format -- cadence/margin-scale
        # constants are tied to per-innings ball volume and score scale (odw par 220.88 is
        # within 15% of od's 260), not a men's-population fit, so widening is safe.
        if format_type in ("od", "odw"):
            self.k_factor = float(k_factor if k_factor is not None else 25.0)
            self.decay_days = float(decay_days if decay_days is not None else 120.0)
            self.decay_rate = float(decay_rate if decay_rate is not None else 0.97)
            self.margin_runs_coef = float(margin_runs_coef if margin_runs_coef is not None else 0.3)
            self.margin_runs_div = float(margin_runs_div if margin_runs_div is not None else 25.0)
            self.margin_wickets_coef = float(margin_wickets_coef if margin_wickets_coef is not None else 0.08)
            
            # Dynamic K-factor defaults
            self.k_max = float(k_max if k_max is not None else 35.0)
            self.k_min = float(k_min if k_min is not None else 25.0)
            self.k_scale = float(k_scale if k_scale is not None else 30.0)
        else:
            self.k_factor = float(k_factor if k_factor is not None else 35.0)
            self.decay_days = float(decay_days if decay_days is not None else 90.0)
            self.decay_rate = float(decay_rate if decay_rate is not None else 0.95)
            self.margin_runs_coef = float(margin_runs_coef if margin_runs_coef is not None else 0.5)
            self.margin_runs_div = float(margin_runs_div if margin_runs_div is not None else 10.0)
            self.margin_wickets_coef = float(margin_wickets_coef if margin_wickets_coef is not None else 0.1)

            # Dynamic K-factor defaults
            self.k_max = float(k_max if k_max is not None else 50.0)
            self.k_min = float(k_min if k_min is not None else 20.0)
            self.k_scale = float(k_scale if k_scale is not None else 30.0)

        self.pools = {}  # comp -> {team: elo}
        self.match_counts = {}  # comp -> {team: count}
        self.last_match_dates = {}  # comp -> {team: date_str}

    city_to_country = {
        # Australia
        "Melbourne": "Australia", "Sydney": "Australia", "Brisbane": "Australia", "Adelaide": "Australia", 
        "Hobart": "Australia", "Perth": "Australia", "Geelong": "Australia", "Canberra": "Australia", 
        "Gold Coast": "Australia", "Cairns": "Australia", "Darwin": "Australia", "Victoria": "Australia",
        # England & Wales
        "London": "England", "Birmingham": "England", "Leeds": "England", "Manchester": "England", 
        "Southampton": "England", "Bristol": "England", "Nottingham": "England", "Cardiff": "England", 
        "Chester-le-Street": "England", "Taunton": "England", "Hove": "England", "Chelmsford": "England", 
        "Derby": "England", "Worcester": "England", "Northampton": "England", "Leicester": "England",
        # India
        "Mumbai": "India", "Delhi": "India", "Kolkata": "India", "Bangalore": "India", "Bengaluru": "India", "Chennai": "India", 
        "Hyderabad": "India", "Ahmedabad": "India", "Pune": "India", "Cuttack": "India", "Guwahati": "India", 
        "Lucknow": "India", "Dharamshala": "India", "Dharamsala": "India", "Ranchi": "India", "Indore": "India", "Rajkot": "India", 
        "Visakhapatnam": "India", "Nagpur": "India", "Kanpur": "India", "Raipur": "India", "Mohali": "India", "Chandigarh": "India",
        # New Zealand
        "Auckland": "New Zealand", "Wellington": "New Zealand", "Christchurch": "New Zealand", 
        "Hamilton": "New Zealand", "Napier": "New Zealand", "Mount Maunganui": "New Zealand", 
        "Dunedin": "New Zealand", "Nelson": "New Zealand", "Queenstown": "New Zealand",
        # South Africa
        "Johannesburg": "South Africa", "Centurion": "South Africa", "Cape Town": "South Africa", 
        "Durban": "South Africa", "Port Elizabeth": "South Africa", "Gqeberha": "South Africa", "Bloemfontein": "South Africa", 
        "East London": "South Africa", "Paarl": "South Africa", "Kimberley": "South Africa", "Potchefstroom": "South Africa", "Benoni": "South Africa",
        # West Indies
        "Trinidad": "West Indies", "St Lucia": "West Indies", "Antigua": "West Indies", 
        "Barbados": "West Indies", "Grenada": "West Indies", "Dominica": "West Indies", 
        "Guyana": "West Indies", "St Kitts": "West Indies", "Jamaica": "West Indies",
        "Gros Islet": "West Indies", "Kingstown": "West Indies", "Tarouba": "West Indies",
        "Bridgetown": "West Indies", "St George's": "West Indies", "Providence": "West Indies",
        "Basseterre": "West Indies", "Coolidge": "West Indies", "Cave Hill": "West Indies", "Kingston": "West Indies",
        # Bangladesh
        "Dhaka": "Bangladesh", "Chattogram": "Bangladesh", "Chittagong": "Bangladesh", "Sylhet": "Bangladesh", "Fatullah": "Bangladesh", "Khulna": "Bangladesh",
        # Pakistan
        "Karachi": "Pakistan", "Lahore": "Pakistan", "Rawalpindi": "Pakistan", "Multan": "Pakistan", "Peshawar": "Pakistan", "Faisalabad": "Pakistan",
        # Zimbabwe
        "Harare": "Zimbabwe", "Bulawayo": "Zimbabwe",
        # Ireland
        "Dublin": "Ireland", "Belfast": "Ireland", "Bready": "Ireland", "Malahide": "Ireland", "Londonderry": "Ireland", "Derry": "Ireland",
        # Scotland
        "Edinburgh": "Scotland", "Glasgow": "Scotland", "Aberdeen": "Scotland", "Arbroath": "Scotland", "Dundee": "Scotland",
        # Netherlands
        "Amstelveen": "Netherlands", "Rotterdam": "Netherlands", "Utrecht": "Netherlands", "The Hague": "Netherlands", "Deventer": "Netherlands",
        # UAE
        "Dubai": "United Arab Emirates", "Abu Dhabi": "United Arab Emirates", "Sharjah": "United Arab Emirates", "Ajman": "United Arab Emirates",
        # Oman
        "Al Amerat": "Oman", "Al Amarat": "Oman", "Muscat": "Oman",
        # Namibia
        "Windhoek": "Namibia",
        # Nepal
        "Kirtipur": "Nepal", "Kathmandu": "Nepal",
        # PNG
        "Port Moresby": "PNG"
    }

    venue_keyword_to_country = {
        "Melbourne": "Australia", "Adelaide": "Australia", "Sydney": "Australia", "Brisbane": "Australia",
        "Hobart": "Australia", "Perth": "Australia", "Manuka": "Australia", "Junction Oval": "Australia",
        "Lord's": "England", "Oval": "England", "Trent Bridge": "England", "Edgbaston": "England", 
        "Headingley": "England", "Old Trafford": "England", "Sophia Gardens": "England", "Cardiff": "England", "Rose Bowl": "England",
        "Wankhede": "India", "Eden Gardens": "India", "Chinnaswamy": "India", "Chepauk": "India",
        "Rajiv Gandhi": "India", "Narendra Modi": "India", "Feroz Shah": "India", "Arun Jaitley": "India",
        "Ekana": "India", "Barsapara": "India", "Barabati": "India", "DY Patil": "India",
        "Basin Reserve": "New Zealand", "Seddon Park": "New Zealand", "Hagley": "New Zealand", 
        "Bay Oval": "New Zealand", "McLean Park": "New Zealand", "Eden Park": "New Zealand", "Westpac": "New Zealand",
        "Wanderers": "South Africa", "Newlands": "South Africa", "Kingsmead": "South Africa",
        "St George's Park": "South Africa", "Boland Park": "South Africa", "Buffalo Park": "South Africa",
        "Kensington Oval": "West Indies", "Providence Stadium": "West Indies", "Warner Park": "West Indies",
        "Sabina Park": "West Indies", "Brian Lara": "West Indies", "Daren Sammy": "West Indies",
        "Sher-e-Bangla": "Bangladesh", "Zahur Ahmed": "Bangladesh",
        "Gaddafi": "Pakistan", "National Stadium, Karachi": "Pakistan",
        "Harare Sports Club": "Zimbabwe", "Queens Sports Club": "Zimbabwe",
        "Malahide": "Ireland", "Stormont": "Ireland", "Bready": "Ireland",
        "Grange Cricket Club": "Scotland",
        "VRA Ground": "Netherlands", "Hazelaarweg": "Netherlands",
        "Amini Park": "Papua New Guinea",
        "Al Amerat": "Oman",
        "Tribhuvan": "Nepal",
        "Gymkhana": "Kenya"
    }

    def get_country_from_location(self, venue: str, city: str) -> str | None:
        if isinstance(city, str) and city.strip():
            city_name = city.strip()
            if city_name in self.city_to_country:
                return self.city_to_country[city_name]

        if not isinstance(venue, str):
            return None

        # Check venue keywords
        for kw, country in self.venue_keyword_to_country.items():
            if kw.lower() in venue.lower():
                return country

        # Check city mapping keys in venue string
        for city_name, country in self.city_to_country.items():
            if city_name.lower() in venue.lower():
                return country

        return None

    def detect_home_team(self, venue: str, city: str, team_1: str, team_2: str, competition: str) -> str | None:
        if competition != "t20s":
            return None

        country = self.get_country_from_location(venue, city)
        if not country or not isinstance(team_1, str) or not isinstance(team_2, str):
            return None

        c_lower = country.lower()
        t1_lower = team_1.lower()
        t2_lower = team_2.lower()

        if c_lower in t1_lower or t1_lower in c_lower:
            return team_1
        elif c_lower in t2_lower or t2_lower in c_lower:
            return team_2

        return None

    def get_elo(self, competition: str, team: str) -> float:
        comp_pool = self.pools.setdefault(competition, {})
        return comp_pool.setdefault(team, 1500.0)

    def get_match_count(self, competition: str, team: str) -> int:
        comp_counts = self.match_counts.setdefault(competition, {})
        return comp_counts.setdefault(team, 0)

    def get_last_date(self, competition: str, team: str) -> str:
        comp_dates = self.last_match_dates.setdefault(competition, {})
        return comp_dates.get(team, None)

    def compute_expected(self, elo_1: float, elo_2: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((elo_2 - elo_1) / 400.0))

    def calculate_decayed_elo(
        self, elo: float, last_date_str: str, current_date_str: str
    ) -> float:
        if not last_date_str or not current_date_str:
            return elo
        try:
            d1 = last_date_str if isinstance(last_date_str, pd.Timestamp) else pd.to_datetime(last_date_str)
            d2 = current_date_str if isinstance(current_date_str, pd.Timestamp) else pd.to_datetime(current_date_str)
            days = (d2 - d1).days
            if days > self.decay_days:
                # decay towards 1500 for every decay_days of inactivity
                periods = (days - self.decay_days) / self.decay_days
                decay_factor = self.decay_rate**periods
                return 1500.0 + (elo - 1500.0) * decay_factor
        except Exception as e:
            print(f"Decay datetime parse error: {e}")
        return elo

    def compute_margin_multiplier(self, margin: int, win_type: str) -> float:
        if win_type == "wickets":
            return 1.0 + self.margin_wickets_coef * (margin - 1)
        elif win_type == "runs" and margin > 0:
            return 1.0 + self.margin_runs_coef * math.log(1.0 + margin / self.margin_runs_div)
        return 1.0

    def compute_dynamic_k(self, matches_played: int) -> float:
        return self.k_min + (self.k_max - self.k_min) / (1.0 + matches_played / self.k_scale)

    def run(self, metadata_df: pd.DataFrame, force=False) -> pd.DataFrame:
        if not force and CacheManager.is_cache_valid(self.cache_file, [Path(__file__)]):
            print(f"Loading Elo ratings from cache: {self.cache_file}")
            return pd.read_parquet(self.cache_file)


        print("Computing rolling Elo ratings...")
        # Reset internal states to ensure clean run
        self.pools = {}
        self.match_counts = {}
        self.last_match_dates = {}

        elo_records = []

        # Sort chronologically by date and match_id
        metadata_df = metadata_df.sort_values(by=["date", "match_id"]).reset_index(
            drop=True
        )

        for _, row in metadata_df.iterrows():
            match_id = row["match_id"]
            comp = row["competition"]
            t1 = row["team_1"]
            t2 = row["team_2"]
            winner = row["winner"]
            date_str = row["date"]
            margin = row["win_margin"]
            win_type = row["win_type"]
            venue = row.get("venue", "")
            city = row.get("city", "")

            # Detect home team
            home_team = self.detect_home_team(venue, city, t1, t2, comp)

            # Get current Elo ratings
            e1 = self.get_elo(comp, t1)
            e2 = self.get_elo(comp, t2)

            # Apply decay if enabled
            if self.use_decay:
                e1 = self.calculate_decayed_elo(
                    e1, self.get_last_date(comp, t1), date_str
                )
                e2 = self.calculate_decayed_elo(
                    e2, self.get_last_date(comp, t2), date_str
                )
                self.pools[comp][t1] = e1
                self.pools[comp][t2] = e2

            # Record pre-match Elo ratings
            elo_records.append(
                {
                    "match_id": match_id,
                    "team_1": t1,
                    "team_2": t2,
                    "team_1_elo": e1,
                    "team_2_elo": e2,
                    "home_team": home_team,
                    "toss_winner": row.get("toss_winner", ""),
                    "toss_decision": row.get("toss_decision", ""),
                    "match_date": row.get("date", ""),
                    "team_1_players": row.get("team_1_players", ""),
                    "team_2_players": row.get("team_2_players", ""),
                }
            )

            # Update match counts and last played dates
            self.match_counts.setdefault(comp, {})[t1] = self.get_match_count(comp, t1) + 1
            self.match_counts.setdefault(comp, {})[t2] = self.get_match_count(comp, t2) + 1
            self.last_match_dates.setdefault(comp, {})[t1] = date_str
            self.last_match_dates.setdefault(comp, {})[t2] = date_str

            # Update Elo ratings if there is a clear winner
            if winner and (winner in [t1, t2]):
                # Dynamic K or Static K
                k1 = (
                    self.compute_dynamic_k(self.get_match_count(comp, t1) - 1)
                    if self.use_dynamic_k
                    else self.k_factor
                )
                k2 = (
                    self.compute_dynamic_k(self.get_match_count(comp, t2) - 1)
                    if self.use_dynamic_k
                    else self.k_factor
                )

                # Expected outcome with home advantage adjustment
                h1 = 0.0
                h2 = 0.0
                if self.use_home_away and home_team:
                    if home_team == t1:
                        h1 = 50.0
                    elif home_team == t2:
                        h2 = 50.0

                exp1 = self.compute_expected(e1 + h1, e2 + h2)
                exp2 = 1.0 - exp1

                # Actual outcome
                s1 = 1.0 if winner == t1 else 0.0
                s2 = 1.0 - s1

                # Margin of victory multiplier
                mult = (
                    self.compute_margin_multiplier(margin, win_type)
                    if self.use_margin
                    else 1.0
                )

                # Adjust Elos
                new_e1 = e1 + k1 * (s1 - exp1) * mult
                new_e2 = e2 + k2 * (s2 - exp2) * mult

                # Save updated ratings to pool
                self.pools[comp][t1] = new_e1
                self.pools[comp][t2] = new_e2

        df = pd.DataFrame(elo_records)
        print(f"Saving Elo ratings to cache: {self.cache_file}")
        df.to_parquet(self.cache_file, index=False, compression="snappy")
        CacheManager.register_cache(self.cache_file, [Path(__file__)])
        return df

    def clear_cache(self):
        CacheManager.clear_cache(self.cache_file)

