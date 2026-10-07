import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.cache_manager import CacheManager
from src.data_processing.format_spec import franchise_competitions, INTL_COMPETITIONS, spec
from src.features.icc_team_rating_engine import ICCTeamRatingEngine
from src.features.icc_player_rating_engine import ICCPlayerRatingEngine, AsOfRatingIndex
from src.features.player_elo_engine import PlayerEloEngine


class FranchiseBridgeEngine:
    """
    Extends the validated international rating scale into franchise cricket, by using
    internationally-rated players present in a franchise XI as ANCHORS.

    The problem: ICCPlayerRatingEngine rates internationals only (725 of 8,547 men's t20
    players). Franchise-only players are invisible to it, and franchise teams have no rating.
    But anchors play franchise cricket too -- so a franchise-only player can be judged by how
    they perform in the same matches, against the same attacks, as players whose quality is
    already known on a trusted scale.

    Mechanism, per franchise match at date D (see markdowns/win_predictor/franchise_bridge_plan.md):
      1. Anchors = players with >= MIN_MATCHES_TO_RANK international appearances STRICTLY
         BEFORE D (AsOfRatingIndex.is_anchor -- time-aware, so a player who debuts
         internationally in 2024 is not an anchor for a 2016 IPL match).
      2. Each XI's batting/bowling strength = mean international rating of its anchors in that
         role, as of D. Bowling strength is what an opposing BATTER faced; batting strength is
         what an opposing BOWLER faced -- kept separate rather than one team number, matching
         ICC's own description ("the rating of the opposition bowlers" for batting).
      3. That anchor-mean lives on the PLAYER points scale, but _batting_points/_bowling_points
         expect a TEAM rating (~SEED_RATING-centred). The map between them is FITTED ON
         INTERNATIONAL MATCHES, where both the anchor set and the true team rating are known
         (fit_strength_map) -- then applied unchanged to franchise XIs. Fitting on the
         validated data rather than assuming a scale relationship is the whole point.
      4. Shrink the mapped strength toward SEED_RATING by anchor count (a 1-anchor estimate is
         far weaker than a 6-anchor one), same reliability-blend shape used throughout both
         engines.
      5. Score EVERY player in the match -- anchor and franchise-only alike -- with
         ICCPlayerRatingEngine's existing _batting_points/_bowling_points, unchanged. Because
         opponent strength is expressed on the gold-standard scale, the resulting points are
         too.

    Output is deliberately kept SEPARATE from the international ratings rather than merged:
    Phase 3's validation compares an anchor's bridge-derived rating against their independently
    known international rating, which is only possible while the two stay independent.

    Scope: Tier A leagues only by default (format_spec.FRANCHISE_BRIDGE_TIERS) -- the leagues
    where anchors are present on both sides in 55-98% of matches. Tier B/C need multi-hop
    anchoring and are gated on Phase 3's result.
    """

    # Anchor-count reliability: an XI's strength estimate is blended toward SEED_RATING by
    # n_anchors / (n_anchors + ANCHOR_RELIABILITY_K). At the Tier A median (~4 anchors) this
    # keeps ~57% of the estimate; a lone anchor keeps 25%. Same blend shape as
    # MIN_MATCHES_RELIABLE elsewhere, just over anchors-in-this-match rather than career games.
    ANCHOR_RELIABILITY_K = 3.0

    # An XI with fewer than this many anchors in a role contributes no strength estimate at all
    # for that role (falls back to SEED_RATING entirely). Distinct from the shrinkage above:
    # this is the floor below which the estimate is not merely weak but meaningless.
    MIN_ANCHORS_FOR_ESTIMATE = 1

    # Phase 6 (multi-hop): innings a player needs in a PREVIOUS bridge pass before they may
    # act as a second-hop anchor for a thinly-anchored league. Same bar as the international
    # anchor test (MIN_MATCHES_TO_RANK), so a second-hop anchor is at least as established
    # within its own league as a first-hop anchor is internationally. Errors compound across
    # hops -- a second-hop anchor carries the bridge's own ~0.86 (batting) / ~0.60 (bowling)
    # agreement with the true scale -- so Tier B/C ratings are inherently noisier than Tier A's
    # and are reported as such.
    SECOND_HOP_MIN_INNINGS = ICCPlayerRatingEngine.MIN_MATCHES_TO_RANK

    def __init__(self, format_type="t20", tiers=("A",), processed_path="data/processed"):
        self.format_type = format_type
        self.tiers = tuple(tiers)
        self.competitions = set(franchise_competitions(format_type, tiers))
        self.processed_path = Path(processed_path)
        tier_tag = "".join(self.tiers).lower()
        self.cache_file = self.processed_path / f"{format_type}_franchise_bridge_tier{tier_tag}.parquet"
        # The fitted strength map is persisted BESIDE the parquet rather than kept only in
        # memory: a cache-hit run() returns without fitting, and anything downstream that needs
        # the map (team_ratings_as_of) would otherwise silently get an empty one. Same class of
        # footgun as the ledger-after-cache-hit bug fixed in the international engines.
        self.strength_map_file = self.processed_path / f"{format_type}_franchise_bridge_tier{tier_tag}_strengthmap.json"
        os.makedirs(self.processed_path, exist_ok=True)

        # scoring formulas are reused wholesale from the international engine
        self._scorer = ICCPlayerRatingEngine(format_type=format_type, processed_path=processed_path)
        # (player, role) -> [(match_date, points_earned)]
        self.ledgers = {}
        # fitted player-scale -> team-scale maps, set by fit_strength_map()
        self.strength_map = {}

    # ------------------------------------------------------------------ anchors

    @staticmethod
    def _split_players(players_str) -> list:
        return [n.strip() for n in str(players_str).split(",") if n.strip()]

    def _xi_strength(self, players: list, role: str, as_of_date, index: AsOfRatingIndex,
                     extra_index: AsOfRatingIndex = None, extra_offset: float = 0.0):
        """Mean anchor rating (player scale) for this XI in `role` as of as_of_date, plus the
        anchor count backing it.

        First hop: a genuine internationally-rated anchor, always preferred when available.
        Second hop (Phase 6, only when `extra_index` is supplied): a player who is not an
        international anchor but HAS been rated by a previous bridge pass in a
        better-connected league. `extra_offset` shifts those ratings onto the international
        scale before they are mixed in -- bridged ratings sit systematically below
        international ones for the same players, so using them raw would bias every strength
        estimate downward. Both hops use STRICTLY-BEFORE lookups, so this stays leak-safe."""
        vals = []
        for p in players:
            if index.is_anchor(p, as_of_date) and index.role_innings_before(p, role, as_of_date) > 0:
                vals.append(index.effective_rating(p, role, as_of_date))
            elif extra_index is not None:
                n = extra_index.role_innings_before(p, role, as_of_date)
                if n >= self.SECOND_HOP_MIN_INNINGS:
                    vals.append(extra_index.effective_rating(p, role, as_of_date) - extra_offset)
        if len(vals) < self.MIN_ANCHORS_FOR_ESTIMATE:
            return None, 0
        return float(np.mean(vals)), len(vals)

    @staticmethod
    def measure_bridge_offset(bridge_df: pd.DataFrame, icc_player_df: pd.DataFrame,
                              role: str, as_of_date) -> float:
        """Mean (bridged - international) rating gap for players rated in both systems -- the
        systematic offset a second-hop anchor must be corrected by. Measured rather than
        hardcoded so it stays correct if the point formulas are ever re-fitted."""
        bridge = FranchiseBridgeEngine.get_effective_ratings_from_df(bridge_df, role, as_of_date=as_of_date)
        intl = ICCPlayerRatingEngine.get_effective_ratings_from_df(icc_player_df, role, as_of_date=as_of_date)
        common = sorted(set(bridge) & set(intl))
        if len(common) < 10:
            return 0.0
        return float(np.mean([bridge[p] - intl[p] for p in common]))

    # ------------------------------------------------- player-scale -> team-scale map

    def fit_strength_map(self, metadata_df: pd.DataFrame, icc_elo_df: pd.DataFrame,
                         index: AsOfRatingIndex) -> dict:
        """Fits, on INTERNATIONAL matches only, the linear map from an XI's anchor-mean player
        rating to that team's actual ICCTeamRatingEngine rating.

        Both sides are known for internationals, so this is a straight calibration against
        validated data -- no assumed scale relationship. The same anchor-subset definition is
        used here as in franchise matches, so the map transfers rather than being fitted on a
        systematically different quantity.

        Returns {role: (intercept, slope, n, r)} for role in {batter, bowler}.
        """
        intl = set(INTL_COMPETITIONS.get(self.format_type, ()))
        meta = metadata_df[
            metadata_df["competition"].isin(intl)
            & (metadata_df["team_type"] == "international")
        ].copy()
        meta["date"] = pd.to_datetime(meta["date"])

        team_elo = {}
        for row in icc_elo_df.itertuples(index=False):
            team_elo[(row.match_id, row.team_1)] = row.team_1_elo
            team_elo[(row.match_id, row.team_2)] = row.team_2_elo

        samples = {"batter": [], "bowler": []}
        for row in meta.itertuples(index=False):
            for team, players_str in ((row.team_1, row.team_1_players), (row.team_2, row.team_2_players)):
                true_rating = team_elo.get((row.match_id, team))
                if true_rating is None:
                    continue
                players = self._split_players(players_str)
                for role in ("batter", "bowler"):
                    strength, n_anchors = self._xi_strength(players, role, row.date, index)
                    if strength is None:
                        continue
                    samples[role].append((strength, true_rating, n_anchors))

        fitted = {}
        for role, rows in samples.items():
            if len(rows) < 50:
                # not enough calibration data -- fall back to identity-at-SEED_RATING, i.e.
                # every XI reads as neutral, which makes the bridge inert rather than wrong
                fitted[role] = (ICCTeamRatingEngine.SEED_RATING, 0.0, len(rows), 0.0)
                continue
            x = np.array([r[0] for r in rows], dtype=float)
            y = np.array([r[1] for r in rows], dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            r = float(np.corrcoef(x, y)[0, 1])
            fitted[role] = (float(intercept), float(slope), len(rows), r)

        self.strength_map = fitted
        return fitted

    def _mapped_strength(self, anchor_strength, n_anchors: int, role: str) -> float:
        """Anchor-mean (player scale) -> team-rating-equivalent (team scale), shrunk toward
        SEED_RATING by how many anchors backed the estimate."""
        seed = ICCTeamRatingEngine.SEED_RATING
        if anchor_strength is None or n_anchors <= 0 or role not in self.strength_map:
            return seed
        intercept, slope, _, _ = self.strength_map[role]
        mapped = intercept + slope * anchor_strength
        reliability = n_anchors / (n_anchors + self.ANCHOR_RELIABILITY_K)
        return reliability * mapped + (1.0 - reliability) * seed

    # ------------------------------------------------------------------ main pass

    def run(self, deliveries_df: pd.DataFrame, metadata_df: pd.DataFrame,
            icc_elo_df: pd.DataFrame, icc_player_df: pd.DataFrame,
            force: bool = False, second_hop_df: pd.DataFrame = None) -> pd.DataFrame:
        if not force and CacheManager.is_cache_valid(self.cache_file, [Path(__file__)]):
            print(f"Loading franchise bridge ratings from cache: {self.cache_file}")
            self._load_strength_map()
            return pd.read_parquet(self.cache_file)

        print(f"Computing franchise bridge ratings ({self.format_type}, tiers={self.tiers})...")
        self.ledgers = {}

        index = AsOfRatingIndex(icc_player_df)

        # Phase 6: an earlier bridge pass (typically Tier A) supplies second-hop anchors for
        # leagues too thin to anchor internationally. Offsets are MEASURED per role, not
        # assumed, so bridged ratings land on the international scale before being mixed in.
        extra_index, extra_offsets = None, {"batter": 0.0, "bowler": 0.0}
        if second_hop_df is not None and len(second_hop_df):
            extra_index = AsOfRatingIndex(second_hop_df)
            as_of_off = pd.to_datetime(second_hop_df["match_date"]).max() + pd.Timedelta(days=1)
            for role in ("batter", "bowler"):
                extra_offsets[role] = self.measure_bridge_offset(
                    second_hop_df, icc_player_df, role, as_of_off)
            print(f"  second-hop anchors enabled from {len(second_hop_df):,} prior-pass innings "
                  f"(offset correction: batter {extra_offsets['batter']:+.2f}, "
                  f"bowler {extra_offsets['bowler']:+.2f})")

        fitted = self.fit_strength_map(metadata_df, icc_elo_df, index)
        for role, (intercept, slope, n, r) in fitted.items():
            print(f"  strength map [{role}]: team_rating = {intercept:.2f} + {slope:.3f} * anchor_mean "
                  f"(n={n}, r={r:.3f})")

        meta = metadata_df[metadata_df["competition"].isin(self.competitions)].copy()
        meta["date"] = pd.to_datetime(meta["date"])
        meta = meta.sort_values(by=["date", "match_id"]).reset_index(drop=True)
        print(f"  {len(meta)} franchise matches in scope")

        match_ids = set(meta["match_id"])
        deliveries = deliveries_df[deliveries_df["match_id"].isin(match_ids)]
        bat_stats, bowl_stats, innings_totals = self._scorer._aggregate_match_stats(deliveries)

        innings_lookup = innings_totals.set_index(["match_id", "batting_team"])
        bat_by_match = {mid: g for mid, g in bat_stats.groupby("match_id")}
        bowl_by_match = {mid: g for mid, g in bowl_stats.groupby("match_id")}

        records = []
        for mrow in meta.itertuples(index=False):
            match_id, t1, t2 = mrow.match_id, mrow.team_1, mrow.team_2
            winner, date, comp = mrow.winner, mrow.date, mrow.competition

            xi = {
                t1: self._split_players(mrow.team_1_players),
                t2: self._split_players(mrow.team_2_players),
            }
            # per-XI strength on the team scale: bowling strength is what an opposing batter
            # faced, batting strength is what an opposing bowler faced
            strength, anchors = {}, {}
            for team, players in xi.items():
                for role in ("batter", "bowler"):
                    s, n = self._xi_strength(players, role, date, index,
                                             extra_index=extra_index,
                                             extra_offset=extra_offsets[role])
                    strength[(team, role)] = self._mapped_strength(s, n, role)
                    anchors[(team, role)] = n

            def result_for(team):
                if not winner or winner not in (t1, t2):
                    return 0.0
                return 1.0 if winner == team else -1.0

            def innings_run_rate(batting_team):
                key = (match_id, batting_team)
                if key not in innings_lookup.index:
                    return 0.0
                irow = innings_lookup.loc[key]
                return (irow["innings_runs"] / irow["innings_balls"] * 6.0) if irow["innings_balls"] > 0 else 0.0

            match_bat = bat_by_match.get(match_id)
            if match_bat is not None:
                for brow in match_bat.itertuples(index=False):
                    batter, batting_team = brow.batter, brow.batting_team
                    opponent = t2 if batting_team == t1 else t1
                    # a batter faces the opposition's BOWLING
                    opp_strength = strength.get((opponent, "bowler"), ICCTeamRatingEngine.SEED_RATING)
                    pre = self._effective_from_ledger(batter, "batter", date)
                    points = self._scorer._batting_points(
                        brow.runs, brow.balls_faced, innings_run_rate(batting_team),
                        opp_strength, result_for(batting_team),
                    )
                    records.append({
                        "match_id": match_id, "competition": comp, "player": batter,
                        "role": "batter", "team": batting_team,
                        "raw_elo": pre, "effective_elo": pre, "points_earned": points,
                        "opponent_strength": opp_strength,
                        "opponent_anchors": anchors.get((opponent, "bowler"), 0),
                        "is_anchor": index.is_anchor(batter, date),
                        "match_date": date,
                    })
                    self.ledgers.setdefault((batter, "batter"), []).append((date, points))

            match_bowl = bowl_by_match.get(match_id)
            if match_bowl is not None:
                for wrow in match_bowl.itertuples(index=False):
                    bowler, batting_team = wrow.bowler, wrow.batting_team
                    bowling_team = t2 if batting_team == t1 else t1
                    # a bowler faces the opposition's BATTING (batting_team IS the opposition here)
                    opp_strength = strength.get((batting_team, "batter"), ICCTeamRatingEngine.SEED_RATING)
                    pre = self._effective_from_ledger(bowler, "bowler", date)
                    points = self._scorer._bowling_points(
                        wrow.wickets, wrow.runs_conceded, wrow.balls_bowled,
                        innings_run_rate(batting_team), opp_strength, result_for(bowling_team),
                    )
                    records.append({
                        "match_id": match_id, "competition": comp, "player": bowler,
                        "role": "bowler", "team": bowling_team,
                        "raw_elo": pre, "effective_elo": pre, "points_earned": points,
                        "opponent_strength": opp_strength,
                        "opponent_anchors": anchors.get((batting_team, "batter"), 0),
                        "is_anchor": index.is_anchor(bowler, date),
                        "match_date": date,
                    })
                    self.ledgers.setdefault((bowler, "bowler"), []).append((date, points))

        result = pd.DataFrame(records)
        result.to_parquet(self.cache_file, compression="snappy")
        self._save_strength_map()
        CacheManager.register_cache(self.cache_file, [Path(__file__)])
        return result

    def _save_strength_map(self):
        with open(self.strength_map_file, "w", encoding="utf-8") as f:
            json.dump(self.strength_map, f, indent=2)

    def _load_strength_map(self):
        if self.strength_map_file.exists():
            with open(self.strength_map_file, encoding="utf-8") as f:
                self.strength_map = {k: tuple(v) for k, v in json.load(f).items()}
        return self.strength_map

    def _effective_from_ledger(self, player: str, role: str, as_of_date) -> float:
        """Pre-match bridge rating from what this engine has accumulated so far -- the same
        time-weighting + reliability blend the international engine uses, over the franchise
        ledger. Chronological by construction (run() processes matches in date order), so it
        only ever sees prior matches."""
        ledger = self.ledgers.get((player, role))
        if not ledger:
            return ICCPlayerRatingEngine.NEUTRAL_BASELINE

        weighted_points, weighted_matches = 0.0, 0.0
        for match_date, points in ledger:
            age_days = (as_of_date - match_date).days
            if age_days < 0:
                continue
            w = ICCPlayerRatingEngine._weight_for_age(age_days)
            if w <= 0.0:
                continue
            weighted_points += w * points
            weighted_matches += w

        if weighted_matches <= 0.0:
            return ICCPlayerRatingEngine.NEUTRAL_BASELINE
        raw = weighted_points / weighted_matches
        n = len(ledger)
        reliability = n / (n + ICCPlayerRatingEngine.MIN_MATCHES_RELIABLE)
        return reliability * raw + (1.0 - reliability) * ICCPlayerRatingEngine.NEUTRAL_BASELINE

    # ------------------------------------------------------------------ reporting

    @staticmethod
    def get_effective_ratings_from_df(bridge_df: pd.DataFrame, role: str, as_of_date=None) -> dict:
        """Current bridge-derived rating per player for `role`, with the same qualification and
        recency floors the international leaderboard uses. Reuses AsOfRatingIndex so the
        time-weighting is identical to the international side."""
        if as_of_date is None:
            as_of_date = pd.Timestamp.now()
        index = AsOfRatingIndex(bridge_df)
        role_df = bridge_df[bridge_df["role"] == role]
        counts = role_df["player"].value_counts().to_dict()
        last = role_df.groupby("player")["match_date"].max().to_dict()

        out = {}
        for player, n in counts.items():
            if n < ICCPlayerRatingEngine.MIN_MATCHES_TO_RANK:
                continue
            if (pd.Timestamp(as_of_date) - pd.Timestamp(last[player])).days > ICCPlayerRatingEngine.MAX_INACTIVITY_DAYS:
                continue
            out[player] = index.effective_rating(player, role, as_of_date)
        return out

    # ------------------------------------------------------- Phase 4: franchise team ratings

    def team_ratings_as_of(self, bridge_df: pd.DataFrame, metadata_df: pd.DataFrame,
                           as_of_date=None, lookback_days: int = 365,
                           min_matches: int = 5) -> pd.DataFrame:
        """Franchise TEAM ratings, composed from the bridge-derived player ratings.

        Deliberately compositional rather than a separate points-exchange run over league
        matches: the player ratings are already on a trusted scale, and running an independent
        team-level exchange would re-import the scale-compression problem already fought and
        fixed at international level (icc_rating_engine_plan.md).

        Per team, over the XIs it actually fielded in the lookback window:
          - batting strength = position-weighted mean of its players' bridge batting ratings
            (PlayerEloEngine._weighted_mean_by_position + W_BAT, the same weighting production
            squad-Elo uses -- openers count for more than a #11)
          - bowling strength = frontline-rank-weighted mean of bowling ratings
            (_weighted_mean_by_bowling_rank + W_BOWL -- a team is judged on its best ~6
            bowlers, not on whoever bowled one over)
        Both are then mapped onto the INTERNATIONAL TEAM rating scale via the same fitted maps
        the bridge uses for opponent strength, so a franchise team's number is directly
        comparable to a national team's.

        Players with little history are handled by the rating's own reliability shrinkage
        (AsOfRatingIndex.effective_rating blends toward NEUTRAL_BASELINE), so no arbitrary
        replacement-level constant is needed.
        """
        if as_of_date is None:
            as_of_date = pd.Timestamp.now()
        as_of_date = pd.Timestamp(as_of_date)
        if not self.strength_map:
            self._load_strength_map()
        if not self.strength_map:
            raise RuntimeError(
                "strength_map is empty -- call run() (which fits it) or ensure "
                f"{self.strength_map_file.name} exists."
            )

        index = AsOfRatingIndex(bridge_df)
        fmt = self.format_type
        w_bat = PlayerEloEngine.W_BAT.get(fmt, PlayerEloEngine.W_BAT["t20"])
        w_bowl = PlayerEloEngine.W_BOWL.get(fmt, PlayerEloEngine.W_BOWL["t20"])
        neutral = ICCPlayerRatingEngine.NEUTRAL_BASELINE

        meta = metadata_df[metadata_df["competition"].isin(self.competitions)].copy()
        meta["date"] = pd.to_datetime(meta["date"])
        window_start = as_of_date - pd.Timedelta(days=lookback_days)
        meta = meta[(meta["date"] > window_start) & (meta["date"] <= as_of_date)]

        per_team = {}
        for row in meta.itertuples(index=False):
            for team, players_str in ((row.team_1, row.team_1_players), (row.team_2, row.team_2_players)):
                players = self._split_players(players_str)
                if not players:
                    continue
                bat_map = {p: index.effective_rating(p, "batter", as_of_date) for p in players}
                bowl_map = {p: index.effective_rating(p, "bowler", as_of_date) for p in players}
                bat = PlayerEloEngine._weighted_mean_by_position(players, bat_map, neutral, w_bat)
                bowl = PlayerEloEngine._weighted_mean_by_bowling_rank(players, bowl_map, neutral, w_bowl)
                entry = per_team.setdefault(team, {"bat": [], "bowl": [], "n": 0,
                                                   "competition": row.competition})
                entry["bat"].append(bat)
                entry["bowl"].append(bowl)
                entry["n"] += 1

        rows = []
        for team, e in per_team.items():
            if e["n"] < min_matches:
                continue
            bat_mean = float(np.mean(e["bat"]))
            bowl_mean = float(np.mean(e["bowl"]))
            # map each role onto the international team-rating scale, then combine
            bat_scaled = self._mapped_strength(bat_mean, len(w_bat), "batter")
            bowl_scaled = self._mapped_strength(bowl_mean, len(w_bowl), "bowler")
            rows.append({
                "team": team,
                "competition": e["competition"],
                "matches_in_window": e["n"],
                "batting_strength": round(bat_mean, 2),
                "bowling_strength": round(bowl_mean, 2),
                "batting_rating": round(bat_scaled, 1),
                "bowling_rating": round(bowl_scaled, 1),
                "team_rating": round((bat_scaled + bowl_scaled) / 2.0, 1),
            })

        return (pd.DataFrame(rows)
                .sort_values("team_rating", ascending=False)
                .reset_index(drop=True))

    def clear_cache(self):
        CacheManager.clear_cache(self.cache_file)
