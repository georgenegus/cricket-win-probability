import os
import time
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
import betfairlightweight
from betfairlightweight.filters import market_filter
from dotenv import load_dotenv


class BetfairClient:
    """
    Thin wrapper around betfairlightweight for cricket market discovery,
    odds retrieval, and team name matching.
    """

    # Common team name overrides for cricket franchise/international variations
    TEAM_NAME_OVERRIDES = {
        "mumbai indians": "Mumbai Indians",
        "chennai super kings": "Chennai Super Kings",
        "royal challengers bengaluru": "Royal Challengers Bangalore",
        "royal challengers bangalore": "Royal Challengers Bangalore",
        "kolkata knight riders": "Kolkata Knight Riders",
        "delhi capitals": "Delhi Capitals",
        "sunrisers hyderabad": "Sunrisers Hyderabad",
        "rajasthan royals": "Rajasthan Royals",
        "punjab kings": "Punjab Kings",
        "gujarat titans": "Gujarat Titans",
        "lucknow super giants": "Lucknow Super Giants",
    }

    def __init__(self, env_path="credentials.env"):
        self.env_path = env_path
        load_dotenv(env_path, override=True)
        
        self.username = os.getenv("BETFAIR_USERNAME")
        self.password = os.getenv("BETFAIR_PASSWORD")
        self.app_key = os.getenv("BETFAIR_APP_KEY")

        if not all([self.username, self.password, self.app_key]):
            raise ValueError(
                f"Missing Betfair credentials in {env_path}. "
                "Ensure BETFAIR_USERNAME, BETFAIR_PASSWORD, and BETFAIR_APP_KEY are set."
            )

        self.trading = betfairlightweight.APIClient(
            username=self.username,
            password=self.password,
            app_key=self.app_key,
        )
        self.is_logged_in = False
        self.last_login_time = None

    def login(self):
        """Authenticates with Betfair API. Automatically called before queries if needed."""
        print(f"Logging into Betfair API for user: {self.username}...")
        self.trading.login_interactive()
        self.is_logged_in = True
        self.last_login_time = datetime.now(timezone.utc)
        print("Betfair API login successful!")
        return self

    def _ensure_login(self):
        """Re-authenticates if session token is missing or older than 3 hours."""
        if not self.is_logged_in or self.last_login_time is None:
            self.login()
        elif datetime.now(timezone.utc) - self.last_login_time > timedelta(hours=3):
            print("Session token expiring, re-authenticating...")
            self.trading.keep_alive()
            self.last_login_time = datetime.now(timezone.utc)

    def get_upcoming_cricket_events(self, hours_ahead=48):
        """
        Returns upcoming cricket events starting within N hours.

        Returns: list of dicts:
            [{
                "event_id": str,
                "event_name": str,
                "open_date": str,
                "country_code": str
            }, ...]
        """
        self._ensure_login()

        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=hours_ahead)

        # Event Type ID 4 = Cricket
        cricket_filter = market_filter(
            event_type_ids=["4"],
            market_start_time={
                "from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": future.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )

        events_result = self.trading.betting.list_events(filter=cricket_filter)

        results = []
        for e in events_result:
            event = e.event
            venue_name = getattr(event, "venue", None) or "N/A"
            results.append(
                {
                    "event_id": str(event.id),
                    "event_name": str(event.name),
                    "open_date": str(event.open_date),
                    "country_code": str(getattr(event, "country_code", "N/A")),
                    "venue": str(venue_name),
                }
            )

        print(f"Retrieved {len(results)} upcoming cricket events from Betfair.")
        return results

    def get_match_odds(self, event_id):
        """
        Returns Match Winner market odds for a specific cricket event.

        Args:
            event_id (str): Betfair event ID

        Returns: dict or None
        """
        self._ensure_login()

        # Step 1: Find Match Winner market catalogue for event
        m_filter = market_filter(
            event_ids=[str(event_id)],
            market_type_codes=["MATCH_ODDS"],
        )

        catalogues = self.trading.betting.list_market_catalogue(
            filter=m_filter,
            max_results=5,
            market_projection=["RUNNER_METADATA", "EVENT", "COMPETITION"],
        )

        if not catalogues:
            print(f"No MATCH_ODDS market found for event ID {event_id}")
            return None

        # Take the main Match Winner market
        cat = catalogues[0]
        market_id = cat.market_id
        comp_name = cat.competition.name if hasattr(cat, "competition") and cat.competition else ""
        venue_name = getattr(cat.event, "venue", None) if hasattr(cat, "event") and cat.event else "N/A"
        runner_map = {r.selection_id: r.runner_name for r in cat.runners}

        # Step 2: Retrieve real-time odds price book
        books = self.trading.betting.list_market_book(
            market_ids=[market_id],
            price_projection={"priceData": ["EX_BEST_OFFERS"]},
        )

        if not books:
            return None

        book = books[0]
        runners_data = []

        for r in book.runners:
            name = runner_map.get(r.selection_id, f"Runner {r.selection_id}")
            best_back = (
                r.ex.available_to_back[0].price
                if r.ex and r.ex.available_to_back
                else None
            )
            best_back_size = (
                r.ex.available_to_back[0].size
                if r.ex and r.ex.available_to_back
                else 0.0
            )
            best_lay = (
                r.ex.available_to_lay[0].price
                if r.ex and r.ex.available_to_lay
                else None
            )
            best_lay_size = (
                r.ex.available_to_lay[0].size
                if r.ex and r.ex.available_to_lay
                else 0.0
            )

            runners_data.append(
                {
                    "selection_id": r.selection_id,
                    "runner_name": name,
                    "back_price": best_back,
                    "back_size": best_back_size,
                    "lay_price": best_lay,
                    "lay_size": best_lay_size,
                }
            )

        return {
            "market_id": str(market_id),
            "market_name": str(cat.market_name),
            "competition_name": str(comp_name),
            "venue": str(venue_name or "N/A"),
            "total_matched": float(getattr(book, "total_matched", 0.0)),
            "runners": runners_data,
        }

    def match_team_name(self, betfair_name, elo_teams, min_threshold=0.6):
        """
        Fuzzy-matches a Betfair runner/team name to closest team name in our Elo database.
        """
        if not betfair_name:
            return None, 0.0

        clean_bf = betfair_name.strip().lower()

        # Check explicit overrides dictionary first
        if clean_bf in self.TEAM_NAME_OVERRIDES:
            override_name = self.TEAM_NAME_OVERRIDES[clean_bf]
            if override_name in elo_teams:
                return override_name, 1.0

        best_match = None
        best_score = 0.0

        for team in elo_teams:
            clean_elo = team.strip().lower()

            # Exact match
            if clean_bf == clean_elo:
                return team, 1.0

            # Substring match
            if clean_bf in clean_elo or clean_elo in clean_bf:
                ratio = len(min(clean_bf, clean_elo, key=len)) / len(max(clean_bf, clean_elo, key=len))
                score = max(0.85, ratio)
            else:
                score = SequenceMatcher(None, clean_bf, clean_elo).ratio()

            if score > best_score:
                best_score = score
                best_match = team

        if best_score >= min_threshold:
            return best_match, round(best_score, 4)

        return None, 0.0

    def get_settled_runner_status(self, market_id, selection_id):
        """
        Reads terminal status of a specific runner directly from the market BOOK
        (survives catalogue purge). Returns "win", "loss", "void", or "pending".
        """
        self._ensure_login()
        books = self.trading.betting.list_market_book(market_ids=[str(market_id)])
        if not books:
            return "pending"
        book = books[0]
        if getattr(book, "status", None) != "CLOSED":
            return "pending"                      # match not settled yet

        try:
            sel = int(selection_id)
        except (ValueError, TypeError):
            return "pending"

        runner = next((r for r in book.runners if int(getattr(r, "selection_id", -1)) == sel), None)
        if runner is None:
            return "pending"                      # wrong/unknown market — do not guess

        status = getattr(runner, "status", None)
        if status == "WINNER":
            return "win"
        if status == "LOSER":
            return "loss"
        if status in ("REMOVED", "REMOVED_VACANT"):
            return "void"
        # CLOSED market but our runner is neither W/L/removed, or whole market voided:
        if not any(getattr(r, "status", None) == "WINNER" for r in book.runners):
            return "void"                         # abandoned / no-result market
        return "pending"

    def check_event_outcome(self, event_id, bet_on_team):
        """
        Checks if an event's Match Winner market is settled and determines outcome for bet_on_team.

        Returns:
            str: "win", "loss", "void", or "pending"
        """
        self._ensure_login()

        m_filter = market_filter(
            event_ids=[str(event_id)],
            market_type_codes=["MATCH_ODDS"],
        )

        catalogues = self.trading.betting.list_market_catalogue(
            filter=m_filter,
            max_results=5,
            market_projection=["RUNNER_METADATA", "EVENT"],
        )

        if not catalogues:
            return "pending"

        cat = catalogues[0]
        market_id = cat.market_id
        runner_map = {r.selection_id: r.runner_name for r in cat.runners}

        books = self.trading.betting.list_market_book(
            market_ids=[market_id],
        )

        if not books:
            return "pending"

        book = books[0]
        if book.status != "CLOSED":
            return "pending"

        winning_runner = None
        for r in book.runners:
            if getattr(r, "status", None) == "WINNER":
                winning_runner = runner_map.get(r.selection_id, "")
                break

        if not winning_runner:
            return "void"

        matched_team, score = self.match_team_name(winning_runner, {bet_on_team})
        if matched_team == bet_on_team or score >= 0.8:
            return "win"
        else:
            return "loss"

    def get_live_market_odds(self, market_id):
        """
        Fetches real-time odds for a specific market ID.
        Optimized for in-play: single API call, minimal payload.

        Returns: dict with runners and current back/lay prices and volume.
        """
        self._ensure_login()
        books = self.trading.betting.list_market_book(
            market_ids=[str(market_id)],
            price_projection={"priceData": ["EX_BEST_OFFERS"]},
        )
        if not books:
            return None

        book = books[0]
        return {
            "market_id": str(market_id),
            "status": str(getattr(book, "status", "UNKNOWN")),
            "in_play": bool(getattr(book, "inplay", False)),
            "total_matched": float(getattr(book, "total_matched", 0.0)),
            "runners": [
                {
                    "selection_id": r.selection_id,
                    "back_price": r.ex.available_to_back[0].price if r.ex and r.ex.available_to_back else None,
                    "back_size": r.ex.available_to_back[0].size if r.ex and r.ex.available_to_back else 0.0,
                    "lay_price": r.ex.available_to_lay[0].price if r.ex and r.ex.available_to_lay else None,
                    "lay_size": r.ex.available_to_lay[0].size if r.ex and r.ex.available_to_lay else 0.0,
                    "last_price_traded": getattr(r, "last_price_traded", None),
                }
                for r in book.runners
            ],
        }

    def place_bet(self, market_id, selection_id, side="BACK", price=2.0, size=2.0):
        """
        Places a limit order bet on Betfair Exchange.

        Args:
            market_id (str): Betfair market ID
            selection_id (int/str): Runner selection ID
            side (str): "BACK" or "LAY"
            price (float): Odds price
            size (float): Stake in £

        Returns: dict with bet_id, status ("SUCCESS"/"FAILURE"), size_matched, average_price_matched.
        """
        self._ensure_login()
        instruction = {
            "selectionId": int(selection_id),
            "side": str(side).upper(),
            "orderType": "LIMIT",
            "limitOrder": {
                "price": float(price),
                "size": round(float(size), 2),
                "persistenceType": "LAPSE",
            },
        }

        result = self.trading.betting.place_orders(
            market_id=str(market_id),
            instructions=[instruction],
        )

        if result and getattr(result, "status", "") == "SUCCESS":
            placed = result.place_instruction_reports[0]
            return {
                "bet_id": str(placed.bet_id),
                "status": "SUCCESS",
                "size_matched": float(placed.size_matched),
                "average_price_matched": float(placed.average_price_matched),
            }
        else:
            error_msg = getattr(result, "error_code", "UNKNOWN_ERROR")
            return {
                "bet_id": None,
                "status": "FAILURE",
                "error": str(error_msg),
            }

