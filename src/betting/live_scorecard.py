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
    # ESPN's summary feed is league-scoped: the league id must sit in the path. The bare
    # SUMMARY_API above now 404s, so LEAGUE_SUMMARY_API is preferred whenever a league id
    # is known (it always is for games discovered via list_live_games).
    LEAGUE_SUMMARY_API = "https://site.web.api.espn.com/apis/site/v2/sports/cricket/{league_id}/summary?event={match_id}"
    PLAY_BY_PLAY_API = "https://site.web.api.espn.com/apis/site/v2/sports/cricket/playbyplay?event={match_id}"
    # Single cross-league scoreboard: lists every cricket event with a pre/in/post state.
    # NOTE: was "site.api.espn.com" -- that host started returning a hard 403 (Akamai
    # "Access Denied") for this endpoint specifically, while the identical path on
    # "site.web.api.espn.com" (the same host SUMMARY_API/PLAY_BY_PLAY_API already use)
    # returns 200 with the same sports/leagues/events payload shape. Swapped host, not path.
    HEADER_API = "https://site.web.api.espn.com/apis/v2/scoreboard/header?sport=cricket"

    # Module-shared short-TTL cache for the live-games list so repeated UI refreshes
    # don't hammer ESPN. Keyed as (timestamp, payload).
    _live_cache: tuple = (0.0, None)
    _LIVE_CACHE_TTL = 20.0  # seconds

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
        self.league_id: Optional[str] = None  # ESPN league id needed for the summary path
        self.last_seen_delivery_key: Optional[tuple] = None  # (innings, over, ball)
        self._last_summary: Optional[Dict[str, Any]] = None  # cached summary JSON for get_lineups()

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

    @classmethod
    def list_live_games(cls, include_upcoming: bool = True, timeout: float = 10.0) -> List[Dict[str, Any]]:
        """
        Returns every current cricket match from ESPN's cross-league scoreboard header.

        Each item: {event_id, name, team_a, team_b, team_a_score, team_b_score,
                    league_name, state ('pre'|'in'|'post'), status_detail, event_type,
                    class_card}.
        'in' == live. When include_upcoming is True, 'pre' fixtures are kept too so the
        UI can show what's coming up when nothing is currently live. Results are cached
        for a few seconds (see _LIVE_CACHE_TTL) to avoid hammering ESPN on UI refreshes.

        event_type / class_card are ESPN's own match-type classification (e.g. "Test",
        "ODI", "T20", "List A") straight off this same payload -- callers use them (see
        format_spec.classify_espn_event) to identify match types we don't model (Test,
        First-Class) rather than guessing format from the name/league text alone.
        """
        now = time.time()
        ts, cached = cls._live_cache
        if cached is not None and (now - ts) < cls._LIVE_CACHE_TTL:
            data = cached
        else:
            session = requests.Session()
            session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            })
            resp = session.get(cls.HEADER_API, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            cls._live_cache = (now, data)

        keep_states = {"in", "pre"} if include_upcoming else {"in"}
        games: List[Dict[str, Any]] = []
        for sport in data.get("sports", []):
            for league in sport.get("leagues", []):
                league_name = league.get("name", "")
                league_id = str(league.get("id", ""))
                for ev in league.get("events", []):
                    full_status = ev.get("fullStatus", {}) or {}
                    state = (full_status.get("type", {}) or {}).get("state", "")
                    if state not in keep_states:
                        continue
                    competitors = ev.get("competitors", []) or []
                    if len(competitors) < 2:
                        continue
                    a, b = competitors[0], competitors[1]
                    class_info = ev.get("class", {}) or {}
                    games.append({
                        "event_id": str(ev.get("id", "")),
                        "league_id": league_id,
                        "name": ev.get("name", ""),
                        "team_a": a.get("displayName", ""),
                        "team_b": b.get("displayName", ""),
                        "team_a_score": a.get("score", ""),
                        "team_b_score": b.get("score", ""),
                        "league_name": league_name,
                        "state": state,
                        "status_detail": full_status.get("summary", "") or full_status.get("longSummary", ""),
                        "event_type": ev.get("eventType", ""),
                        "class_card": class_info.get("generalClassCard", ""),
                    })
        # Live first, then upcoming.
        games.sort(key=lambda g: 0 if g["state"] == "in" else 1)
        return games

    def connect(self, match_url_or_id: str, league_id: Optional[str] = None) -> None:
        """Connects to a specific Cricinfo match by URL or ID.

        league_id is required to build the (league-scoped) summary URL. When not passed,
        it is parsed from a full ESPN scorecard URL of the form
        '.../cricket/series/{league_id}/scorecard/{event_id}/...'.
        """
        self.match_id = self.extract_match_id(match_url_or_id)
        self.league_id = str(league_id) if league_id else None
        if not self.league_id:
            m = re.search(r'/series/(\d+)/', str(match_url_or_id))
            if m:
                self.league_id = m.group(1)
        self.last_seen_delivery_key = None
        self._last_summary = None

    def _summary_url(self) -> str:
        """Preferred league-scoped summary URL, falling back to the bare (legacy) one."""
        if self.league_id:
            return self.LEAGUE_SUMMARY_API.format(league_id=self.league_id, match_id=self.match_id)
        return self.SUMMARY_API.format(match_id=self.match_id)

    def get_current_state(self) -> Dict[str, Any]:
        """Fetches and parses the current match state from Cricinfo API."""
        if not self.match_id:
            raise RuntimeError("Scraper not connected to a match. Call connect() first.")

        url = self._summary_url()
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        self._last_summary = data  # cached so get_lineups() can reuse it without re-fetching

        header = data.get("header", {})
        competitions = header.get("competitions", [{}])[0]
        competitors = competitions.get("competitors", [])
        status = competitions.get("status", {}) or {}
        status_type = status.get("type", {}) or {}

        batting_team = "Unknown"
        bowling_team = "Unknown"
        runs_scored = 0
        wickets_lost = 0
        overs_completed = 0.0
        target = None

        # ESPN marks the batting side inside the *current* linescore (competitor-level
        # isBatting is null) and also exposes status.battingTeamId; use both defensively.
        batting_team_id = status.get("battingTeamId")
        period = status.get("period")
        state = status_type.get("state", "")
        is_complete = state == "post" or header.get("completed", False)
        is_break = "break" in str(status_type.get("description", "")).lower()

        def _current_linescore(ls_list):
            if not ls_list:
                return {}
            for ls in ls_list:
                if ls.get("isCurrent"):
                    return ls
            return ls_list[-1]

        first_innings_runs = None
        for comp in competitors:
            team = comp.get("team", {}) or {}
            team_name = team.get("displayName", "")
            team_id = team.get("id")
            cl = _current_linescore(comp.get("linescores", []) or [])
            is_this_batting = bool(cl.get("isBatting")) or (
                batting_team_id is not None and str(team_id) == str(batting_team_id)
            )

            if is_this_batting:
                batting_team = team_name
                runs_scored = int(cl.get("runs", 0) or 0)
                wickets_lost = int(cl.get("wickets", 0) or 0)
                overs_completed = float(cl.get("overs", 0.0) or 0.0)
                if cl.get("target"):
                    target = int(cl["target"])
            else:
                bowling_team = team_name
                # The bowling side's total is their (completed) 1st innings when chasing.
                first_innings_runs = int(cl.get("runs", 0) or 0)

        # Innings number: prefer status.period, else infer from whether a 1st innings total exists.
        if period in (1, 2):
            innings = int(period)
        else:
            innings = 2 if (first_innings_runs and first_innings_runs > 0) else 1

        # Chase target fallback when not carried on the batting linescore.
        if innings == 2 and not target and first_innings_runs:
            target = first_innings_runs + 1

        striker = "Batter 1"
        non_striker = "Batter 2"
        bowler = "Bowler 1"

        # Venue: ESPN summary exposes it under gameInfo.venue.fullName (fallbacks kept
        # defensive because the shape varies across competitions).
        venue = "Unknown"
        game_info = data.get("gameInfo", {}) or {}
        venue_obj = game_info.get("venue", {}) or {}
        if venue_obj.get("fullName"):
            venue = venue_obj["fullName"]
        elif competitions.get("venue", {}).get("fullName"):
            venue = competitions["venue"]["fullName"]

        return {
            "match_id": self.match_id,
            "innings": innings,
            "overs_completed": overs_completed,
            "runs_scored": runs_scored,
            "wickets_lost": wickets_lost,
            "target": target if innings == 2 else None,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "striker": striker,
            "non_striker": non_striker,
            "bowler": bowler,
            "recent_deliveries": [],
            "is_innings_break": is_break,
            "is_match_complete": is_complete,
        }

    def get_lineups(self) -> Dict[str, List[Dict[str, Optional[str]]]]:
        """
        Returns {team_display_name: [{"id": athlete_id_or_None, "name": player_name}, ...]}
        scraped from the summary 'rosters' section. Returns {} when lineups aren't published
        yet (pre-toss, minor fixtures), which signals callers to fall back to a historical
        squad.

        `id` is ESPN's athlete id, which is the same id space as Cricinfo's numeric player id
        (confirmed: e.g. West Indies' Brandon King is athlete id "670035" here, and
        player_meta.csv lists cricinfo_id 670035 for the same player under unique_name
        "BA King"). Callers can use it to resolve a live player to the Cricsheet-style name
        the historical Elo ratings are keyed by, since ESPN's displayName often spells out a
        full first name ("Brandon King") that a bare string match against "BA King" would
        never hit.
        """
        if not self.match_id:
            raise RuntimeError("Scraper not connected to a match. Call connect() first.")

        data = getattr(self, "_last_summary", None)
        if data is None:
            resp = self.session.get(self._summary_url(), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            self._last_summary = data

        lineups: Dict[str, List[Dict[str, Optional[str]]]] = {}
        for roster in data.get("rosters", []) or []:
            team_name = (roster.get("team", {}) or {}).get("displayName", "")
            players: List[Dict[str, Optional[str]]] = []
            for entry in roster.get("roster", []) or []:
                athlete = entry.get("athlete", {}) or {}
                name = athlete.get("displayName") or athlete.get("fullName") or entry.get("displayName")
                if name:
                    players.append({"id": athlete.get("id"), "name": str(name)})
            if team_name and players:
                lineups[team_name] = players
        return lineups

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
        self.max_overs = 50 if format_type in ("od", "odw") else 20

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
