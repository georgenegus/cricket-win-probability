import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.cache_manager import CacheManager
from src.features.icc_team_rating_engine import ICCTeamRatingEngine


class ICCPlayerRatingEngine:
    """
    Reconstructs the ICC's player-ratings methodology -- batting/bowling points per
    qualifying innings, weighted moving average over a time-weighted rolling window --
    restricted to genuine internationals only, mirroring ICCTeamRatingEngine's scope.

    NOT a certified replica: see icc_team_rating_engine.py's module docstring and
    markdowns/win_predictor/icc_rating_engine_plan.md for the same caveat and sourcing.
    ICC's public description of the methodology ("runs scored, rating of the opposition
    bowlers, match result, comparison to the overall scores in the match" for batting;
    "wickets, runs conceded, match result, bonus for dismissing highly-rated batters" for
    bowling) does not disclose an exact formula. Where it doesn't, this engine makes the
    following documented, deliberately simple choices:
      - "Opposition bowling/batting rating" is approximated by the OPPONENT TEAM's
        ICCTeamRatingEngine rating, not an aggregate of the individual opposing XI (which
        would need its own chicken-and-egg squad loop) -- a coarser but directionally sound
        proxy for how strong the attack/lineup faced actually was.
      - "Comparison to the overall scores in the match" is implemented as the player's
        strike-rate/economy relative to that specific innings' own run rate -- a natural,
        leak-safe, conditions-adjusted baseline requiring no external constant.
      - Cold start blends a player's weighted-average points toward a neutral baseline of
        0.0 as they accumulate qualifying innings, the same reliability-blend shape as
        PlayerEloEngine.get_effective_elo (n / (n + min_matches_reliable)).
    This is a first-attempt reconstruction meant for cross-checking against real ICC
    rankings (scripts/measure_icc_rating_crosscheck.py), not a calibrated production
    feature -- the point-formula weights below are documented but not fitted to any target
    distribution.
    """

    # Match ICCTeamRatingEngine's time-weighted rolling window exactly, so team and player
    # ratings age out of relevance at the same rate.
    FULL_WEIGHT_DAYS = ICCTeamRatingEngine.FULL_WEIGHT_DAYS
    DROP_WEIGHT_DAYS = ICCTeamRatingEngine.DROP_WEIGHT_DAYS
    HALF_WEIGHT = ICCTeamRatingEngine.HALF_WEIGHT

    MIN_MATCHES_RELIABLE = 10
    NEUTRAL_BASELINE = 0.0

    # Hard qualification floor: a player with fewer than this many qualifying innings for a
    # role is excluded from the reported leaderboard ENTIRELY (get_effective_ratings_from_df),
    # not just shrunk toward NEUTRAL_BASELINE. Same distinction as
    # ICCTeamRatingEngine.MIN_MATCHES_TO_RANK vs MIN_MATCHES_RELIABLE. Added 2026-09-01 per
    # explicit request -- applies to both batting and bowling leaderboards.
    MIN_MATCHES_TO_RANK = 20

    # Hard recency floor: a player whose last qualifying innings for a role is older than
    # this many days before as_of_date is excluded from the reported leaderboard ENTIRELY,
    # regardless of rating or innings count. Added 2026-09-02 -- the time-weighted decay
    # alone (FULL_WEIGHT_DAYS/DROP_WEIGHT_DAYS above) still let players up to ~4 years
    # inactive rank on a "current" leaderboard at half weight (e.g. D Pretorius, last t20
    # appearance 2022-10-04, was still #1 bowler); this is a separate, harder gate on top,
    # matching FULL_WEIGHT_DAYS (only the "fully current" window counts as active).
    MAX_INACTIVITY_DAYS = FULL_WEIGHT_DAYS

    W_SR_BONUS = 0.5        # batting: credit per SR-point above the innings' own rate, scaled by balls faced
    W_ECONOMY_BONUS = 0.5   # bowling: credit per run-per-over saved below the innings' own rate, scaled by balls bowled
    # Credit for the player's side winning the match (penalty for losing). Flat, not
    # opponent-scaled -- ICC cites match result as a factor separate from opposition strength.
    #
    # FITTED 2026-09-02 (scripts/fit_result_bonus.py), and it turned out to be ROLE-DEPENDENT,
    # which a single shared constant could not express. Same exact-sweep method as WICKET_VALUE
    # (points are linear in this weight). Both objectives agreed, per role:
    #   W_RESULT:            0     5     10(old)   20     40
    #   batting gate       0.738  0.751   0.759   0.755  0.704   -> peak at 10, keep it
    #   bowling gate       0.460  0.451   0.449   0.416  0.376   -> monotonically WORSE, use 0
    #   bowling bridge rel 0.732  0.730   0.727   0.719  0.699   -> same story
    #
    # Why the split is mechanistically sensible, not just curve-fitting: a bowler's
    # contribution to a win is ALREADY captured by their wickets and economy, so adding a flat
    # team-result bonus mostly injects their BATTERS' performance into their rating as noise.
    # For a batter the result carries genuine extra context (seeing out a chase) beyond raw
    # runs, so it earns its place there.
    W_RESULT_BAT = 10.0
    W_RESULT_BOWL = 0.0
    # bowling: points per wicket taken -- i.e. the wickets-vs-economy trade-off, since only
    # this value's RATIO to W_ECONOMY_BONUS matters for ranking.
    #
    # FITTED 2026-09-02 (scripts/fit_bowling_wicket_value.py), 20.0 -> 80.0. Was previously an
    # unfitted guess, and was the last untested lead on the franchise bridge's weak bowling
    # agreement (franchise_bridge_plan.md Phase 3: batting Spearman 0.758 vs bowling 0.402).
    # _bowling_points is linear in its weights, so the components were precomputed per innings
    # and every candidate evaluated exactly (the reconstruction reproduces stored points to
    # 2.8e-14). Two independent objectives were swept and agreed:
    #   WICKET_VALUE:   0      20(old)   80(new)   400     2000
    #   intl reliability  0.617  0.809    0.821   0.821    0.821
    #   bridge gate       0.306  0.402    0.449   0.451    0.457
    # Both plateau by ~80; past it the gains are noise-level while face validity keeps eroding
    # (JJ Bumrah, an economy-AND-wickets bowler, falls #2 -> #6 -> #8 across 20/80/400 as the
    # measure narrows toward pure wicket-count). 80 takes essentially all the available gain
    # while keeping a real economy contribution.
    #
    # CAVEAT worth remembering: raising this improves self-consistency because wickets are a
    # more REPEATABLE signal, not because economy is worthless for bowling quality. This is a
    # reliability-vs-validity trade-off that was made deliberately, not a free win -- do not
    # push it higher just because the correlation curve keeps inching up.
    # RE-FITTED 2026-09-02, 80.0 -> 1.5, after two errors in the first fit were found:
    #
    #   1. PARAMETER INTERACTION. WICKET_VALUE was fitted while W_RESULT was still 10.0 for
    #      bowling; W_RESULT_BOWL was then set to 0.0 in a separate, later fit. Re-running the
    #      gate sweep afterwards reverses the ordering completely -- gate is now 0.540 at
    #      WICKET_VALUE=1.5 vs 0.460 at 80. Fitting interacting parameters one at a time
    #      (coordinate descent) walked into a bad local optimum; they needed fitting jointly.
    #   2. THE CRITERION REWARDED NARROWING. Self-consistency and cross-source agreement both
    #      rise as the rating collapses toward pure wicket-count, because wickets are a more
    #      REPEATABLE signal -- not because the rating is measuring bowling better. The
    #      caveat written here at the time named this risk exactly, and the fit walked into it
    #      anyway. Forward-predictive validity (scripts/compare_rating_systems.py) is the
    #      criterion that catches it, because it asks whether the rating anticipates real
    #      future performance rather than whether it agrees with itself.
    #
    # Forward-predictive validity, mean Spearman of rating-at-T vs the next year's actual
    # performance (international scope, 2016-2025):
    #   WICKET_VALUE:      0     1.5     2.5      5      20      80
    #   predicts wickets  0.070  0.154  0.187   0.215   0.162   0.161
    #   predicts economy  0.294  0.146  0.079  -0.029  -0.128  -0.162
    # Above ~2 the entire wicket-prediction gain is already banked while economy prediction
    # goes NEGATIVE -- i.e. the rating starts actively preferring expensive strike bowlers.
    # The two scopes disagree on the exact optimum (international favours ~1.5, franchise ~5),
    # so the shipped value maximises the WORST case across both scopes and both facets --
    # the only setting keeping all four correlations positive:
    #   WV:            1.5     3      5
    #   intl wkts     0.154  0.197  0.215
    #   intl econ     0.146  0.047 -0.029
    #   franchise wkts 0.012  0.121  0.214
    #   franchise econ 0.328  0.252  0.170
    #   WORST         0.012  0.047 -0.029
    #
    # PRECISION CAVEAT: anything in roughly 1.5-5 is defensible; the evidence supports "low
    # single digits, emphatically not 80", not this exact figure. Do not re-tune it by chasing
    # third-decimal movement on these same metrics -- that is the over-fitting failure the
    # first fit already demonstrated.
    WICKET_VALUE = 3.0

    # Strength-of-schedule fix (2026-09-02, per explicit request): opponent strength was
    # originally an ADDITIVE term (W_OPPOSITION=0.05 * (opponent_rating - SEED_RATING)), which
    # measured out to only a +/-2.4 point swing across the entire qualified t20 team-rating
    # range (52.8-149.2) -- negligible against a top player's 30-60-point rating, so beating
    # weak associate opposition earned almost the same credit as beating elite sides. Runs/
    # wickets are now MULTIPLIED by this factor instead (see _opponent_factor, used in
    # _batting_points/_bowling_points), so a spell against a team rated well below SEED_RATING
    # (100) is directly discounted rather than nudged. Diagnosed after the recency/staleness
    # fixes still left the t20 leaderboard dominated by associate players (Karanbir Singh,
    # HAM Shah, D Pretorius topping Buttler/Bumrah) -- see markdowns/win_predictor/
    # icc_rating_engine_plan.md for the measured before/after.
    def _opponent_factor(self, opponent_team_rating: float) -> float:
        return opponent_team_rating / ICCTeamRatingEngine.SEED_RATING

    def __init__(self, format_type="t20", processed_path="data/processed"):
        self.format_type = format_type
        self.processed_path = Path(processed_path)
        self.cache_file = self.processed_path / f"{format_type}_icc_player_ratings.parquet"
        os.makedirs(self.processed_path, exist_ok=True)

        # (player, role) -> list of (match_date, points_earned)
        self.ledgers = {}

    @classmethod
    def _weight_for_age(cls, age_days: float) -> float:
        if age_days < cls.FULL_WEIGHT_DAYS:
            return 1.0
        if age_days < cls.DROP_WEIGHT_DAYS:
            return cls.HALF_WEIGHT
        return 0.0

    def get_raw_rating(self, player: str, role: str, as_of_date: pd.Timestamp) -> float:
        ledger = self.ledgers.get((player, role))
        if not ledger:
            return self.NEUTRAL_BASELINE

        weighted_points = 0.0
        weighted_matches = 0.0
        for match_date, points in ledger:
            age_days = (as_of_date - match_date).days
            if age_days < 0:
                continue
            w = self._weight_for_age(age_days)
            if w <= 0.0:
                continue
            weighted_points += w * points
            weighted_matches += w

        if weighted_matches <= 0.0:
            return self.NEUTRAL_BASELINE
        return weighted_points / weighted_matches

    def get_match_count(self, player: str, role: str) -> int:
        return len(self.ledgers.get((player, role), []))

    def get_effective_rating(self, player: str, role: str, as_of_date: pd.Timestamp) -> float:
        """Reliability-blended rating -- mirrors PlayerEloEngine.get_effective_elo's
        cold-start shape, blending a sparse player's raw rating toward a neutral baseline."""
        raw = self.get_raw_rating(player, role, as_of_date)
        n = self.get_match_count(player, role)
        reliability = n / (n + self.MIN_MATCHES_RELIABLE)
        return reliability * raw + (1.0 - reliability) * self.NEUTRAL_BASELINE

    @staticmethod
    def _aggregate_match_stats(deliveries_df: pd.DataFrame):
        """Per-match batting/bowling aggregates, plus each innings' own run-rate context.
        Column shape mirrors PlayerEloEngine._aggregate_match_stats for consistency."""
        df = deliveries_df
        if "legal_delivery" not in df.columns:
            df = df.copy()
            df["legal_delivery"] = ((df["is_wide"] == 0) & (df["is_noball"] == 0)).astype(int)

        bat_stats = (
            df.groupby(["match_id", "batting_team", "batter"])
            .agg(runs=("runs_off_bat", "sum"), balls_faced=("legal_delivery", "sum"))
            .reset_index()
        )

        bowl_stats = (
            df.groupby(["match_id", "batting_team", "bowler"])
            .agg(runs_conceded=("total_runs", "sum"), balls_bowled=("legal_delivery", "sum"), wickets=("wicket", "sum"))
            .reset_index()
        )

        innings_totals = (
            df.groupby(["match_id", "batting_team"])
            .agg(innings_runs=("total_runs", "sum"), innings_balls=("legal_delivery", "sum"))
            .reset_index()
        )

        return bat_stats, bowl_stats, innings_totals

    def _batting_points(self, runs, balls_faced, innings_run_rate, opponent_team_rating, result_indicator) -> float:
        if balls_faced <= 0:
            return 0.0
        sr = runs / balls_faced * 100.0
        innings_sr = innings_run_rate * 100.0 / 6.0
        opponent_factor = self._opponent_factor(opponent_team_rating)
        return (
            runs * opponent_factor
            + self.W_SR_BONUS * balls_faced * (sr - innings_sr) / 100.0 * opponent_factor
            + self.W_RESULT_BAT * result_indicator
        )

    def _bowling_points(self, wickets, runs_conceded, balls_bowled, innings_run_rate, opponent_team_rating, result_indicator) -> float:
        if balls_bowled <= 0:
            return 0.0
        rpo = runs_conceded / balls_bowled * 6.0
        opponent_factor = self._opponent_factor(opponent_team_rating)
        return (
            self.WICKET_VALUE * wickets * opponent_factor
            + self.W_ECONOMY_BONUS * balls_bowled * (innings_run_rate - rpo) / 6.0 * opponent_factor
            + self.W_RESULT_BOWL * result_indicator
        )

    def run(self, deliveries_df: pd.DataFrame, metadata_df: pd.DataFrame, icc_elo_df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        if not force and CacheManager.is_cache_valid(self.cache_file, [Path(__file__)]):
            print(f"Loading ICC player ratings from cache: {self.cache_file}")
            return pd.read_parquet(self.cache_file)

        print(f"Computing ICC-style international player ratings ({self.format_type})...")
        self.ledgers = {}

        intl_match_ids = set(icc_elo_df["match_id"])
        deliveries_intl = deliveries_df[deliveries_df["match_id"].isin(intl_match_ids)]
        bat_stats, bowl_stats, innings_totals = self._aggregate_match_stats(deliveries_intl)

        meta = metadata_df[metadata_df["match_id"].isin(intl_match_ids)][
            ["match_id", "team_1", "team_2", "winner", "date"]
        ].copy()
        meta["date"] = pd.to_datetime(meta["date"])
        meta = meta.sort_values(by=["date", "match_id"]).reset_index(drop=True)

        elo_lookup = icc_elo_df.set_index("match_id")[["team_1", "team_2", "team_1_elo", "team_2_elo"]]
        innings_lookup = innings_totals.set_index(["match_id", "batting_team"])
        bat_by_match = {mid: g for mid, g in bat_stats.groupby("match_id")}
        bowl_by_match = {mid: g for mid, g in bowl_stats.groupby("match_id")}

        records = []
        for _, mrow in meta.iterrows():
            match_id, t1, t2, winner, date = mrow["match_id"], mrow["team_1"], mrow["team_2"], mrow["winner"], mrow["date"]
            if match_id not in elo_lookup.index:
                continue
            erow = elo_lookup.loc[match_id]
            team_rating = {erow["team_1"]: erow["team_1_elo"], erow["team_2"]: erow["team_2_elo"]}

            def result_for(team, _winner=winner, _t1=t1, _t2=t2):
                if not _winner or _winner not in (_t1, _t2):
                    return 0.0
                return 1.0 if _winner == team else -1.0

            def innings_run_rate(batting_team, _match_id=match_id):
                key = (_match_id, batting_team)
                if key not in innings_lookup.index:
                    return 0.0
                irow = innings_lookup.loc[key]
                return (irow["innings_runs"] / irow["innings_balls"] * 6.0) if irow["innings_balls"] > 0 else 0.0

            match_bat = bat_by_match.get(match_id)
            if match_bat is not None:
                for _, brow in match_bat.iterrows():
                    batter, batting_team = brow["batter"], brow["batting_team"]
                    opponent_team = t2 if batting_team == t1 else t1
                    opponent_rating = team_rating.get(opponent_team, ICCTeamRatingEngine.SEED_RATING)

                    pre_rating = self.get_effective_rating(batter, "batter", date)
                    points = self._batting_points(
                        brow["runs"], brow["balls_faced"], innings_run_rate(batting_team),
                        opponent_rating, result_for(batting_team),
                    )
                    # points_earned is what THIS innings contributed to the player's ledger --
                    # stored so get_current_ratings_from_df can reconstruct the full ledger
                    # from this DataFrame alone and decay it forward to "now" (or any
                    # as_of_date), rather than only being able to read back the pre-match
                    # raw_elo/effective_elo snapshot, which is frozen at whatever this
                    # innings' own date was. See icc_rating_engine_plan.md's "current vs
                    # all-time" note -- found via a long-retired player (AM Rahane, last
                    # T20I 2016) still topping the "current" batting leaderboard.
                    records.append({
                        "match_id": match_id, "player": batter, "role": "batter",
                        "raw_elo": pre_rating, "effective_elo": pre_rating,
                        "points_earned": points, "match_date": date,
                    })
                    self.ledgers.setdefault((batter, "batter"), []).append((date, points))

            match_bowl = bowl_by_match.get(match_id)
            if match_bowl is not None:
                for _, wrow in match_bowl.iterrows():
                    bowler, batting_team = wrow["bowler"], wrow["batting_team"]
                    bowling_team = t2 if batting_team == t1 else t1
                    opponent_rating = team_rating.get(batting_team, ICCTeamRatingEngine.SEED_RATING)

                    pre_rating = self.get_effective_rating(bowler, "bowler", date)
                    points = self._bowling_points(
                        wrow["wickets"], wrow["runs_conceded"], wrow["balls_bowled"],
                        innings_run_rate(batting_team), opponent_rating, result_for(bowling_team),
                    )
                    records.append({
                        "match_id": match_id, "player": bowler, "role": "bowler",
                        "raw_elo": pre_rating, "effective_elo": pre_rating,
                        "points_earned": points, "match_date": date,
                    })
                    self.ledgers.setdefault((bowler, "bowler"), []).append((date, points))

        result = pd.DataFrame(records)
        result.to_parquet(self.cache_file, compression="snappy")
        CacheManager.register_cache(self.cache_file, [Path(__file__)])
        return result

    @staticmethod
    def get_ledger_from_df(icc_player_df: pd.DataFrame, role: str) -> dict:
        """Reconstructs every (player, role) full (match_date, points_earned) ledger from
        run()'s output DataFrame -- works regardless of whether run() computed fresh or
        loaded from cache (unlike self.ledgers, which is only populated by an in-process,
        cache-missed run())."""
        ledgers = {}
        role_df = icc_player_df[icc_player_df["role"] == role]
        for row in role_df.itertuples(index=False):
            ledgers.setdefault(row.player, []).append((row.match_date, row.points_earned))
        return ledgers

    @classmethod
    def get_current_ratings_from_df(cls, icc_player_df: pd.DataFrame, role: str, as_of_date: pd.Timestamp = None) -> dict:
        """Raw (unshrunk) time-weighted rating per player for `role`, AS OF as_of_date
        (defaults to now).

        Correctly decays a player who hasn't appeared recently toward NEUTRAL_BASELINE --
        rebuilds each player's full ledger from the DataFrame (get_ledger_from_df) and
        re-runs the same time-weighting get_raw_rating() uses internally, just referenced
        against as_of_date instead of each innings' own date. The prior version of this
        method took the last row's PRE-match effective_elo verbatim, which is frozen at
        whatever that innings' date was -- so a long-retired player (e.g. AM Rahane, last
        T20I 2016) topped the "current" leaderboard with a decade-stale snapshot. See
        markdowns/win_predictor/icc_rating_engine_plan.md's "current vs all-time" note."""
        if as_of_date is None:
            as_of_date = pd.Timestamp.now()
        ledgers = cls.get_ledger_from_df(icc_player_df, role)

        ratings = {}
        for player, ledger in ledgers.items():
            weighted_points, weighted_matches = 0.0, 0.0
            for match_date, points in ledger:
                age_days = (as_of_date - match_date).days
                if age_days < 0:
                    continue
                w = cls._weight_for_age(age_days)
                if w <= 0.0:
                    continue
                weighted_points += w * points
                weighted_matches += w
            ratings[player] = (weighted_points / weighted_matches) if weighted_matches > 0.0 else cls.NEUTRAL_BASELINE
        return ratings

    @classmethod
    def get_effective_ratings_from_df(cls, icc_player_df: pd.DataFrame, role: str, as_of_date: pd.Timestamp = None) -> dict:
        """Reliability-blended rating per player for `role`, AS OF as_of_date -- the raw
        ratings from get_current_ratings_from_df, shrunk toward NEUTRAL_BASELINE for players
        with few qualifying innings in this pool (n / (n + MIN_MATCHES_RELIABLE), same shape
        as get_effective_rating / PlayerEloEngine.get_effective_elo). Use this for any
        external ranking/leaderboard.

        NOTE: n is each player's TOTAL lifetime innings count in this pool (not just innings
        within the decay window) -- a player with 200 career innings who's been retired for
        years is not a thin sample, they're inactive, which get_current_ratings_from_df
        already decays toward NEUTRAL_BASELINE for.

        Players with fewer than MIN_MATCHES_TO_RANK (20) lifetime qualifying innings for
        `role`, OR whose last qualifying innings is more than MAX_INACTIVITY_DAYS before
        as_of_date, are dropped from the returned dict entirely, not just shrunk -- two
        separate hard qualification floors, same as ICC's own player rankings require a
        minimum sample and recent activity before a player is published at all."""
        if as_of_date is None:
            as_of_date = pd.Timestamp.now()
        raw_ratings = cls.get_current_ratings_from_df(icc_player_df, role, as_of_date=as_of_date)
        role_df = icc_player_df[icc_player_df["role"] == role]
        innings_counts = role_df["player"].value_counts().to_dict()
        last_match_date = role_df.groupby("player")["match_date"].max().to_dict()

        effective = {}
        for player, raw in raw_ratings.items():
            n = innings_counts.get(player, 0)
            if n < cls.MIN_MATCHES_TO_RANK:
                continue
            inactive_days = (as_of_date - last_match_date[player]).days
            if inactive_days > cls.MAX_INACTIVITY_DAYS:
                continue
            reliability = n / (n + cls.MIN_MATCHES_RELIABLE)
            effective[player] = reliability * raw + (1.0 - reliability) * cls.NEUTRAL_BASELINE
        return effective

    def clear_cache(self):
        CacheManager.clear_cache(self.cache_file)


class AsOfRatingIndex:
    """O(log n) as-of-date lookups over an ICCPlayerRatingEngine output DataFrame.

    ICCPlayerRatingEngine.get_raw_rating/get_effective_rating rescan a player's whole ledger
    on every call, which is fine for one "current" snapshot but not for the franchise-bridge
    workload: a rating for every anchor in every XI of every franchise match (~3,000 matches
    x ~8 anchors for Tier A alone). This precomputes, per (player, role), the sorted appearance
    dates plus a prefix sum of points earned, so each query is a few binary searches instead of
    a full scan.

    Exact by construction, not an approximation: the time-weighting is piecewise-constant
    (full weight inside FULL_WEIGHT_DAYS, HALF_WEIGHT out to DROP_WEIGHT_DAYS, zero beyond), so
    a windowed weighted mean is just two prefix-sum differences. Parity against the original
    per-call implementation is asserted in tests/test_asof_rating_index.py.

    See markdowns/win_predictor/franchise_bridge_plan.md Phase 1.
    """

    def __init__(self, icc_player_df: pd.DataFrame, engine_cls=ICCPlayerRatingEngine):
        self.cls = engine_cls
        self._dates = {}    # (player, role) -> np.ndarray[datetime64[ns]], ascending
        self._prefix = {}   # (player, role) -> np.ndarray[float], prefix sum of points_earned
        self._appearances = {}  # player -> np.ndarray[datetime64[ns]] of unique match dates

        df = icc_player_df.copy()
        df["match_date"] = pd.to_datetime(df["match_date"])
        df = df.sort_values("match_date")

        for (player, role), g in df.groupby(["player", "role"], sort=False):
            dates = g["match_date"].to_numpy()
            points = g["points_earned"].to_numpy(dtype=float)
            self._dates[(player, role)] = dates
            self._prefix[(player, role)] = np.concatenate([[0.0], np.cumsum(points)])

        for player, g in df[["player", "match_date"]].drop_duplicates().groupby("player", sort=False):
            self._appearances[player] = np.sort(g["match_date"].to_numpy())

    def _windowed_mean(self, player: str, role: str, as_of_date) -> float:
        dates = self._dates.get((player, role))
        if dates is None or len(dates) == 0:
            return self.cls.NEUTRAL_BASELINE
        prefix = self._prefix[(player, role)]

        as_of = np.datetime64(pd.Timestamp(as_of_date))
        # Boundaries mirror get_raw_rating's `(as_of - match_date).days` comparisons exactly:
        # full weight while age_days < FULL_WEIGHT_DAYS, half weight while < DROP_WEIGHT_DAYS.
        full_cutoff = as_of - np.timedelta64(int(self.cls.FULL_WEIGHT_DAYS) - 1, "D")
        drop_cutoff = as_of - np.timedelta64(int(self.cls.DROP_WEIGHT_DAYS) - 1, "D")

        # STRICTLY BEFORE as_of: a franchise match on date D must only ever see internationals
        # played before D, never one played the same day. Consistent with role_innings_before/
        # appearances_before below, and with run()'s own chronological usage (where the ledger
        # only ever holds prior matches at the moment of the call).
        i_future = np.searchsorted(dates, as_of, side="left")
        i_full = np.searchsorted(dates, full_cutoff, side="left")
        i_drop = np.searchsorted(dates, drop_cutoff, side="left")

        full_n = i_future - i_full
        half_n = i_full - i_drop
        if full_n <= 0 and half_n <= 0:
            return self.cls.NEUTRAL_BASELINE

        full_sum = prefix[i_future] - prefix[i_full]
        half_sum = prefix[i_full] - prefix[i_drop]

        weighted_points = full_sum + self.cls.HALF_WEIGHT * half_sum
        weighted_matches = full_n + self.cls.HALF_WEIGHT * half_n
        if weighted_matches <= 0.0:
            return self.cls.NEUTRAL_BASELINE
        return float(weighted_points / weighted_matches)

    def raw_rating(self, player: str, role: str, as_of_date) -> float:
        """Time-weighted mean points using only innings STRICTLY BEFORE as_of_date -- matches
        ICCPlayerRatingEngine.get_raw_rating called on a ledger truncated to that point, which
        is exactly the state run() holds when it computes a pre-match rating."""
        return self._windowed_mean(player, role, as_of_date)

    def effective_rating(self, player: str, role: str, as_of_date) -> float:
        """Reliability-blended rating as of as_of_date -- matches
        ICCPlayerRatingEngine.get_effective_rating, using appearances BEFORE as_of_date for n
        (so it is leak-safe when queried at a future franchise match's date)."""
        raw = self._windowed_mean(player, role, as_of_date)
        n = self.role_innings_before(player, role, as_of_date)
        reliability = n / (n + self.cls.MIN_MATCHES_RELIABLE)
        return reliability * raw + (1.0 - reliability) * self.cls.NEUTRAL_BASELINE

    def role_innings_before(self, player: str, role: str, as_of_date) -> int:
        dates = self._dates.get((player, role))
        if dates is None:
            return 0
        return int(np.searchsorted(dates, np.datetime64(pd.Timestamp(as_of_date)), side="left"))

    def appearances_before(self, player: str, as_of_date) -> int:
        """Unique international match appearances strictly before as_of_date (deduped across
        roles) -- the anchor-qualification count, matching how
        scripts/measure_franchise_bridge_density.py defines it."""
        dates = self._appearances.get(player)
        if dates is None:
            return 0
        return int(np.searchsorted(dates, np.datetime64(pd.Timestamp(as_of_date)), side="left"))

    def is_anchor(self, player: str, as_of_date, min_appearances: int = None) -> bool:
        """Whether this player was already a gold-standard rated international as of the given
        date -- the time-aware anchor test the franchise bridge is built on."""
        if min_appearances is None:
            min_appearances = self.cls.MIN_MATCHES_TO_RANK
        return self.appearances_before(player, as_of_date) >= min_appearances
