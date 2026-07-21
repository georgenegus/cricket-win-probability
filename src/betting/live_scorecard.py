"""
Live Scorecard Feed Module

Provides data structures and connectors for real-time live cricket scorecards.
Supports web scraping (ESPNCricinfo) and interactive manual entry for paper-trading and testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
import time
from typing import Any, Dict, Generator, List, Optional
import requests
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


@dataclass
class DeliveryUpdate:
    """Represents the state after a single delivery in a live match."""
    match_id: str
    innings: int                    # 1 or 2
    over: int                       # 0-indexed over number (0-19 for T20, 0-49 for ODI)
    ball: int                       # 1-6 (legal delivery number within the over)
    runs_this_ball: int             # Total runs conceded on this delivery (including extras)
    is_wicket: bool                 # Was a wicket taken?
    is_wide: bool                   # Wide delivery (extra ball)
    is_noball: bool                 # No-ball (extra ball)
    batting_team: str               # Name of batting team
    bowling_team: str               # Name of bowling team
    striker: str                    # Current batter on strike
    non_striker: str                # Current non-striker
    bowler: str                     # Current bowler
    total_runs: int                 # Cumulative innings total after this delivery
    total_wickets: int              # Cumulative wickets after this delivery
    target: Optional[int] = None    # Chase target (only for innings 2)
    player_dismissed: Optional[str] = None  # Name of dismissed player, if any


class LiveScorecardFeed(ABC):
    """Abstract base class for live scorecard feeds."""

    @abstractmethod
    def connect(self, match_url_or_id: str) -> None:
        """Connect to a live match scorecard feed."""
        pass

    @abstractmethod
    def get_current_state(self) -> Dict[str, Any]:
        """
        Returns the full current match state as a dictionary.

        Returns:
            Dict containing match_id, innings, overs_completed, runs_scored,
            wickets_lost, target, batting_team, bowling_team, striker,
            non_striker, bowler, recent_deliveries, is_innings_break, is_match_complete.
        """
        pass

    @abstractmethod
    def poll(self, interval_seconds: float = 10.0) -> Generator[DeliveryUpdate, None, None]:
        """
        Generator yielding DeliveryUpdate objects as new deliveries occur.
        Polls the feed at interval_seconds and yields new delivery updates.
        """
        pass


class ESPNCricInfoScraper(LiveScorecardFeed):
    """
    Scrapes ESPNcricinfo live scorecards and commentary feeds for ball-by-ball updates.
    Uses ESPNcricinfo's public web summary/play-by-play API endpoints and page parsing.
    """

    SUMMARY_API = "https://site.web.api.espn.com/apis/site/v2/sports/cricket/summary?event={match_id}"
    PLAY_BY_PLAY_API = "https://site.web.api.espn.com/apis/site/v2/sports/cricket/playbyplay?event={match_id}"

    def __init__(self, timeout: float = 10.0):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })
        self.timeout = timeout
        self.match_id: Optional[str] = None
        self.last_seen_delivery_key: Optional[tuple] = None  # (innings, over, ball)

    @staticmethod
    def extract_match_id(url_or_id: str) -> str:
        """
        Extracts numeric match ID from full URL or returns raw ID string.
        Examples:
            'https://www.espncricinfo.com/series/ipl-2024-1410320/csk-vs-rcb-1st-match-1411242/full-scorecard' -> '1411242'
            '1411242' -> '1411242'
        """
        url_or_id = url_or_id.strip()
        if url_or_id.isdigit():
            return url_or_id

        matches = re.findall(r'-(\d+)(?:/|$|\?)', url_or_id)
        if matches:
            return matches[-1]

        digits = re.findall(r'\b\d{5,8}\b', url_or_id)
        if digits:
            return digits[-1]

        raise ValueError(f"Could not extract ESPNcricinfo match ID from: {url_or_id}")

    def connect(self, match_url_or_id: str) -> None:
        """Connects to a specific Cricinfo match by URL or ID."""
        self.match_id = self.extract_match_id(match_url_or_id)
        self.last_seen_delivery_key = None

    def get_current_state(self) -> Dict[str, Any]:
        """Fetches and parses the current match state from Cricinfo API."""
        if not self.match_id:
            raise RuntimeError("Scraper not connected to a match. Call connect() first.")

        url = self.SUMMARY_API.format(match_id=self.match_id)
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        header = data.get("header", {})
        competitions = header.get("competitions", [{}])[0]
        competitors = competitions.get("competitors", [])

        batting_team = "Unknown"
        bowling_team = "Unknown"
        runs_scored = 0
        wickets_lost = 0
        overs_completed = 0.0
        target = None
        innings = 1
        is_complete = header.get("completed", False)
        is_break = False

        for comp in competitors:
            team_name = comp.get("team", {}).get("displayName", "")
            is_batting = comp.get("isBatting", False)
            linescores = comp.get("linescores", [])

            if is_batting:
                batting_team = team_name
                innings = len(linescores) if linescores else 1
                if linescores:
                    current_line = linescores[-1]
                    runs_scored = int(current_line.get("value", 0))
                    wickets_lost = int(current_line.get("wickets", 0))
                    overs_completed = float(current_line.get("overs", 0.0))
            else:
                bowling_team = team_name
                if linescores and len(linescores) >= 1:
                    target = int(linescores[0].get("value", 0)) + 1

        striker = "Batter 1"
        non_striker = "Batter 2"
        bowler = "Bowler 1"

        return {
            "match_id": self.match_id,
            "innings": innings,
            "overs_completed": overs_completed,
            "runs_scored": runs_scored,
            "wickets_lost": wickets_lost,
            "target": target if innings == 2 else None,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "striker": striker,
            "non_striker": non_striker,
            "bowler": bowler,
            "recent_deliveries": [],
            "is_innings_break": is_break,
            "is_match_complete": is_complete,
        }

    def poll(self, interval_seconds: float = 10.0) -> Generator[DeliveryUpdate, None, None]:
        """
        Polls ESPNcricinfo for new delivery updates.
        """
        if not self.match_id:
            raise RuntimeError("Scraper not connected to a match. Call connect() first.")

        while True:
            try:
                state = self.get_current_state()
                if state["is_match_complete"]:
                    return

                key = (state["innings"], int(state["overs_completed"]), int(round((state["overs_completed"] % 1) * 10)))
                if key != self.last_seen_delivery_key and key != (1, 0, 0):
                    self.last_seen_delivery_key = key
                    over_num = key[1]
                    ball_num = key[2]
                    yield DeliveryUpdate(
                        match_id=self.match_id,
                        innings=state["innings"],
                        over=over_num,
                        ball=ball_num,
                        runs_this_ball=0,
                        is_wicket=False,
                        is_wide=False,
                        is_noball=False,
                        batting_team=state["batting_team"],
                        bowling_team=state["bowling_team"],
                        striker=state["striker"],
                        non_striker=state["non_striker"],
                        bowler=state["bowler"],
                        total_runs=state["runs_scored"],
                        total_wickets=state["wickets_lost"],
                        target=state["target"],
                    )
            except Exception:
                pass
            time.sleep(interval_seconds)


class ManualEntryFeed(LiveScorecardFeed):
    """
    Interactive CLI feed allowing manual entry of live match deliveries.
    Useful for testing and paper trading without active scraper network dependencies.
    """

    def __init__(self):
        self.match_id = "manual_match"
        self.batting_team = "Team A"
        self.bowling_team = "Team B"
        self.format_type = "t20"
        self.max_overs = 20
        self.innings = 2
        self.target: Optional[int] = 160

        self.runs_scored = 0
        self.wickets_lost = 0
        self.legal_deliveries = 0
        self.striker = "Striker"
        self.non_striker = "Non-Striker"
        self.bowler = "Bowler"
        self.history: List[DeliveryUpdate] = []

    def connect(self, match_url_or_id: str) -> None:
        """Sets up match metadata for manual entry."""
        self.match_id = match_url_or_id or "manual_match"
        self.runs_scored = 0
        self.wickets_lost = 0
        self.legal_deliveries = 0
        self.history.clear()

    def set_match_info(
        self,
        batting_team: str,
        bowling_team: str,
        target: Optional[int] = None,
        innings: int = 2,
        format_type: str = "t20",
    ) -> None:
        """Configures match teams, target, and format."""
        self.batting_team = batting_team
        self.bowling_team = bowling_team
        self.target = target
        self.innings = innings
        self.format_type = format_type
        self.max_overs = 50 if format_type == "od" else 20

    def get_current_state(self) -> Dict[str, Any]:
        """Returns the current accumulated manual match state."""
        overs_completed = (self.legal_deliveries // 6) + (self.legal_deliveries % 6) / 10.0
        is_complete = (
            self.wickets_lost >= 10
            or self.legal_deliveries >= self.max_overs * 6
            or (self.target is not None and self.runs_scored >= self.target)
        )
        return {
            "match_id": self.match_id,
            "innings": self.innings,
            "overs_completed": overs_completed,
            "runs_scored": self.runs_scored,
            "wickets_lost": self.wickets_lost,
            "target": self.target,
            "batting_team": self.batting_team,
            "bowling_team": self.bowling_team,
            "striker": self.striker,
            "non_striker": self.non_striker,
            "bowler": self.bowler,
            "recent_deliveries": self.history[-6:],
            "is_innings_break": False,
            "is_match_complete": is_complete,
        }

    @staticmethod
    def parse_delivery_input(input_str: str) -> Dict[str, Any]:
        """
        Parses manual input string into delivery components.
        Examples:
            '0'   -> 0 runs
            '1'   -> 1 run
            '4'   -> 4 runs
            '6'   -> 6 runs
            'w'   -> 0 runs, wicket
            '0w'  -> 0 runs, wicket
            '1wd' -> 1 run wide (0 runs off bat, 1 extra)
            '2nb' -> 2 runs no-ball
        """
        s = input_str.strip().lower()
        if not s:
            return {"runs": 0, "wicket": False, "wide": False, "noball": False}

        is_wicket = "w" in s and "wd" not in s
        is_wide = "wd" in s
        is_noball = "nb" in s

        digits = re.findall(r'\d+', s)
        if digits:
            runs = int(digits[0])
        elif is_wide or is_noball:
            runs = 1
        elif is_wicket:
            runs = 0
        else:
            runs = 0

        return {
            "runs": runs,
            "wicket": is_wicket,
            "wide": is_wide,
            "noball": is_noball,
        }

    def process_delivery_string(self, raw_input: str) -> DeliveryUpdate:
        """Processes a delivery string and updates internal state counters."""
        parsed = self.parse_delivery_input(raw_input)

        runs = parsed["runs"]
        is_wicket = parsed["wicket"]
        is_wide = parsed["wide"]
        is_noball = parsed["noball"]

        self.runs_scored += runs
        if is_wicket:
            self.wickets_lost += 1

        is_legal = not is_wide and not is_noball
        if is_legal:
            self.legal_deliveries += 1

        over_num = (self.legal_deliveries - (1 if is_legal else 0)) // 6
        ball_num = ((self.legal_deliveries - 1) % 6) + 1 if is_legal else (self.legal_deliveries % 6)

        update = DeliveryUpdate(
            match_id=self.match_id,
            innings=self.innings,
            over=over_num,
            ball=ball_num,
            runs_this_ball=runs,
            is_wicket=is_wicket,
            is_wide=is_wide,
            is_noball=is_noball,
            batting_team=self.batting_team,
            bowling_team=self.bowling_team,
            striker=self.striker,
            non_striker=self.non_striker,
            bowler=self.bowler,
            total_runs=self.runs_scored,
            total_wickets=self.wickets_lost,
            target=self.target,
        )
        self.history.append(update)
        return update

    def poll(self, interval_seconds: float = 0.0) -> Generator[DeliveryUpdate, None, None]:
        """
        Interactive loop yielding DeliveryUpdates as user types delivery inputs.
        """
        print(f"\n--- Manual Entry Feed Started ({self.batting_team} vs {self.bowling_team}) ---")
        if self.target:
            print(f"Target: {self.target} runs")

        while True:
            state = self.get_current_state()
            if state["is_match_complete"]:
                print("Match completed!")
                break

            prompt = f"[{state['runs_scored']}/{state['wickets_lost']} ({state['overs_completed']:.1f} ov)] Delivery> "
            try:
                user_in = input(prompt)
            except EOFError:
                break

            if user_in.strip().lower() in ("quit", "exit"):
                break

            update = self.process_delivery_string(user_in)
            yield update
