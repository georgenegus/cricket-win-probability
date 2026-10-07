import os
from pathlib import Path

import pandas as pd

from src.utils.cache_manager import CacheManager
from src.data_processing.format_spec import INTL_COMPETITIONS


class ICCTeamRatingEngine:
    """
    Reconstructs the ICC's team-ratings methodology -- points-exchange per match, averaged
    over a time-weighted rolling window -- restricted to genuine internationals only
    (INTL_COMPETITIONS), as opposed to elo_engine.py's per-competition logistic Elo pooled
    across every league too.

    NOT a certified replica: ICC does not publish its exact formula, historical weighting,
    or qualification thresholds (icc-cricket.com/rankings/about is qualitative only). This is
    a documented community reconstruction (the "Kendix system", as consistently described
    across Wikipedia's ICC rankings pages) -- see markdowns/win_predictor/icc_rating_engine_plan.md
    for the sourcing and the exact formula choices made here where the source material was
    ambiguous. Use for rank-correlation / face-validity cross-checks, not exact validation.

    Unlike EloEngine, a team's rating is NOT a running scalar updated match-by-match -- ICC
    computes rating = cumulative points / cumulative matches within a rolling weighted window,
    so this engine keeps a per-team match ledger (date, points-earned) and recomputes the
    weighted average at each query point. Fine at this scale (a few thousand matches/format).
    """

    # ICC's real team ratings sit roughly in the 0-300 range (see the fixture snapshot).
    # A team with no international history yet is seeded at a neutral mid-table rating rather
    # than 0 (which would read as "worst team ever") or the leaders' level (which would read as
    # undefeated-champion) -- this is a deliberate, documented choice, not an arbitrary default.
    SEED_RATING = 100.0

    # Time-weighting of the rolling window: 0-12 months old counts fully, 12-24 months counts
    # at half weight, beyond ~4 years is dropped entirely. ICC re-derives this once a year every
    # May (a hard cliff); this engine uses a continuous piecewise function of match age instead
    # -- a deliberate simplification, not an attempt at ICC's exact annual-roll-off mechanics.
    FULL_WEIGHT_DAYS = 365
    DROP_WEIGHT_DAYS = 365 * 4
    HALF_WEIGHT = 0.5

    # Below this rating gap, a match is "closely matched"; at or above it, "mismatched".
    CLOSE_GAP_THRESHOLD = 40.0
    CLOSE_MATCH_BONUS = 50.0
    EXPECTED_RESULT_BONUS = 10.0
    UPSET_BONUS = 90.0

    # Reliability-blend shrinkage for teams with a thin international sample -- same shape
    # as PlayerEloEngine.get_effective_elo (n / (n + min_matches_reliable) toward a neutral
    # baseline), applied only to the "effective" rating exposed for external reporting, never
    # to the raw weighted-average rating the match-to-match points exchange itself uses (same
    # raw-vs-effective split PlayerEloEngine already makes). Added after Phase 3's crosscheck
    # found Italy/Uganda (2 matches, od) and Barbados/Trinidad and Tobago (2-3 matches, odw)
    # rocketing to the top of their rankings off a couple of upset-bonus wins with no volume
    # regularization -- see markdowns/win_predictor/icc_rating_engine_plan.md.
    MIN_MATCHES_RELIABLE = 10

    # Hard qualification floor: a team with fewer than this many matches in the pool is
    # excluded from the reported ranking ENTIRELY (get_effective_ratings_from_df), not just
    # shrunk toward SEED_RATING. Distinct from MIN_MATCHES_RELIABLE above, which controls how
    # much a QUALIFYING team's rating is blended toward neutral -- this controls whether a
    # team appears on the leaderboard at all, mirroring ICC's own qualification threshold
    # before a team appears on its published table. Added 2026-09-01 per explicit request.
    MIN_MATCHES_TO_RANK = 20

    # Hard recency floor: a team whose last match in this pool is older than this many days
    # before as_of_date is excluded from the reported ranking ENTIRELY, regardless of rating
    # or match count. Added 2026-09-02 -- the time-weighted decay alone (FULL_WEIGHT_DAYS/
    # DROP_WEIGHT_DAYS above) still let teams/players up to ~4 years inactive rank on a
    # "current" leaderboard at half weight; this is a separate, harder gate on top, matching
    # FULL_WEIGHT_DAYS (only the "fully current" window counts as active).
    MAX_INACTIVITY_DAYS = FULL_WEIGHT_DAYS

    # Post-hoc affine stretch (2026-09-02, per explicit request): the chronological points-
    # exchange dynamics (_match_points, CLOSE_GAP_THRESHOLD) are validated against real ICC
    # rankings (0.79-0.97 Spearman across formats) and are left COMPLETELY UNCHANGED here --
    # this only widens the rating VALUES exposed externally (run()'s team_{1,2}_elo output
    # columns, get_current_ratings_from_df, get_effective_ratings_from_df), never the raw
    # values _match_points()/the ledger accumulation use internally. Reason for not baking
    # the stretch into the dynamics themselves (unlike PlayerEloEngine's *_ELO_STRETCH, which
    # DOES scale its own K-factor/divisor): this engine's points exchange has a discrete
    # gap-threshold branch whose self-referential equilibrium is sensitive to the scale it
    # operates on (verified: naively raising/lowering CLOSE_GAP_THRESHOLD shifts where that
    # equilibrium settles, in a hard-to-predict direction) -- stretching internally would risk
    # destabilizing the very correlations this was validated against. A pure external stretch
    # is a strictly monotonic transform, so it cannot change any Spearman correlation.
    #
    # Motivation: qualified team ratings were compressed relative to ICC's real scale (e.g.
    # t20's raw spread across its fixture-matched teams was 30.0 points vs ICC's real 66),
    # which meant ICCPlayerRatingEngine's opponent-strength discount (_opponent_factor, itself
    # added 2026-09-02 for the same root cause) could only discount as much as the gap it was
    # given -- beating a weak associate side barely looked different from beating a strong one.
    #
    # Fitted per format as (ICC's own real rating spread across the fixture-matched teams) /
    # (this engine's RAW unstretched spread over the same teams), measured 2026-09-02 via
    # tests/fixtures/icc_rankings_snapshot_2026.json. Not re-derived per query -- a fixed,
    # documented constant, same convention as PlayerEloEngine's *_ELO_STRETCH constants.
    TEAM_ELO_STRETCH = {
        "t20": 2.20,   # ICC spread 66 / raw spread 30.0, n=11 matched teams
        "od": 1.24,    # ICC spread 39 / raw spread 31.4, n=9 matched teams
        "t20w": 2.53,  # ICC spread 95 / raw spread 37.5, n=10 matched teams
        "odw": 2.23,   # ICC spread 117 / raw spread 52.4, n=10 matched teams
    }

    def __init__(self, format_type="t20", processed_path="data/processed"):
        self.format_type = format_type
        self.processed_path = Path(processed_path)
        self.intl_competitions = set(INTL_COMPETITIONS.get(format_type, ()))
        self.cache_file = self.processed_path / f"{format_type}_icc_team_ratings.parquet"
        os.makedirs(self.processed_path, exist_ok=True)

        # team -> list of (match_date: pd.Timestamp, points_earned: float)
        self.ledgers = {}

    @classmethod
    def _weight_for_age(cls, age_days: float) -> float:
        if age_days < cls.FULL_WEIGHT_DAYS:
            return 1.0
        if age_days < cls.DROP_WEIGHT_DAYS:
            return cls.HALF_WEIGHT
        return 0.0

    def get_rating(self, team: str, as_of_date: pd.Timestamp) -> float:
        """Time-weighted average rating for `team` using only matches strictly before
        as_of_date -- the pre-match rating a caller would have seen at that point."""
        ledger = self.ledgers.get(team)
        if not ledger:
            return self.SEED_RATING

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
            return self.SEED_RATING
        return weighted_points / weighted_matches

    def get_match_count(self, team: str) -> int:
        return len(self.ledgers.get(team, []))

    @classmethod
    def stretch_rating(cls, raw_rating: float, format_type: str) -> float:
        """Affine hold-SEED_RATING-fixed stretch (see TEAM_ELO_STRETCH) -- widens a raw
        rating for external consumption without touching the internal points-exchange
        dynamics. Strictly monotonic, so it never changes any Spearman rank correlation."""
        stretch = cls.TEAM_ELO_STRETCH.get(format_type, 1.0)
        return cls.SEED_RATING + (raw_rating - cls.SEED_RATING) * stretch

    def _match_points(self, rating_winner_pre: float, rating_loser_pre: float):
        """Points earned by the winner/loser of a match, given their PRE-match ratings.
        See the module docstring / build plan for the sourcing and the "opponent's rating
        + bonus" framing chosen where the source material was ambiguous."""
        gap = abs(rating_winner_pre - rating_loser_pre)
        if gap < self.CLOSE_GAP_THRESHOLD:
            bonus = self.CLOSE_MATCH_BONUS
        elif rating_winner_pre >= rating_loser_pre:
            bonus = self.EXPECTED_RESULT_BONUS
        else:
            bonus = self.UPSET_BONUS

        winner_points = rating_loser_pre + bonus
        loser_points = rating_winner_pre - bonus
        return winner_points, loser_points

    def run(self, metadata_df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        if not force and CacheManager.is_cache_valid(self.cache_file, [Path(__file__)]):
            print(f"Loading ICC team ratings from cache: {self.cache_file}")
            return pd.read_parquet(self.cache_file)

        print(f"Computing ICC-style international team ratings ({self.format_type})...")
        self.ledgers = {}

        # competition folder alone is NOT a clean international filter -- e.g. t20/t20s mixes
        # in 936/4655 (20%) domestic club matches (India's Syed Mushtaq Ali Trophy state
        # teams), t20w/t20s mixes in 197/2233 (8.8%); od/odw's folders are ~clean (6/2564,
        # 8/638). Cricsheet's own team_type field is the authoritative signal -- found
        # 2026-09-01 when domestic Indian teams (Tamil Nadu, Mumbai) showed up in the t20
        # team-ratings top 10. See markdowns/win_predictor/icc_rating_engine_plan.md.
        df = metadata_df[
            metadata_df["competition"].isin(self.intl_competitions)
            & (metadata_df["team_type"] == "international")
        ].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["date", "match_id"]).reset_index(drop=True)

        records = []
        for _, row in df.iterrows():
            match_id = row["match_id"]
            t1, t2 = row["team_1"], row["team_2"]
            winner = row["winner"]
            date = row["date"]

            r1 = self.get_rating(t1, date)
            r2 = self.get_rating(t2, date)

            if winner and winner in (t1, t2):
                if winner == t1:
                    winner_points, loser_points = self._match_points(r1, r2)
                    t1_points, t2_points = winner_points, loser_points
                else:
                    winner_points, loser_points = self._match_points(r2, r1)
                    t1_points, t2_points = loser_points, winner_points
                self.ledgers.setdefault(t1, []).append((date, t1_points))
                self.ledgers.setdefault(t2, []).append((date, t2_points))
            else:
                # Tie / no result: each side's match points = the opponent's pre-match
                # rating, unchanged -- no swing either way.
                t1_points, t2_points = r2, r1
                self.ledgers.setdefault(t1, []).append((date, t1_points))
                self.ledgers.setdefault(t2, []).append((date, t2_points))

            # team_{1,2}_points is what THIS match contributed to that team's ledger --
            # stored so get_current_ratings_from_df can reconstruct the full ledger from
            # this DataFrame alone and decay it forward to "now" (or any as_of_date),
            # rather than only being able to read back the pre-match team_{1,2}_elo
            # snapshot, which is frozen at whatever this match's own date was. See
            # markdowns/win_predictor/icc_rating_engine_plan.md's "current vs all-time" note.
            #
            # team_{1,2}_elo is the STRETCHED rating (see TEAM_ELO_STRETCH / stretch_rating)
            # -- external consumers (the player engine's opponent-strength lookup, any
            # reporting) get the widened value; t1_points/t2_points above (the ledger this
            # match feeds) were computed from the RAW r1/r2, so the internal points-exchange
            # dynamics are completely unaffected by the stretch.
            records.append(
                {
                    "match_id": match_id,
                    "team_1": t1,
                    "team_2": t2,
                    "team_1_elo": self.stretch_rating(r1, self.format_type),
                    "team_2_elo": self.stretch_rating(r2, self.format_type),
                    "team_1_points": t1_points,
                    "team_2_points": t2_points,
                    "match_date": date,
                }
            )

        result = pd.DataFrame(records)
        result.to_parquet(self.cache_file, compression="snappy")
        CacheManager.register_cache(self.cache_file, [Path(__file__)])
        return result

    @staticmethod
    def get_ledger_from_df(icc_elo_df: pd.DataFrame) -> dict:
        """Reconstructs every team's full (match_date, points_earned) ledger from run()'s
        output DataFrame -- works regardless of whether run() computed fresh or loaded from
        cache (unlike self.ledgers, which is only populated by an in-process, cache-missed
        run())."""
        ledgers = {}
        for row in icc_elo_df.itertuples(index=False):
            ledgers.setdefault(row.team_1, []).append((row.match_date, row.team_1_points))
            ledgers.setdefault(row.team_2, []).append((row.match_date, row.team_2_points))
        return ledgers

    @classmethod
    def get_current_ratings_from_df(cls, icc_elo_df: pd.DataFrame, as_of_date: pd.Timestamp = None) -> dict:
        """Raw (unshrunk) time-weighted rating per team AS OF as_of_date (defaults to now).

        Correctly decays an inactive team toward SEED_RATING as time passes since its last
        match -- rebuilds each team's full ledger from the DataFrame (get_ledger_from_df)
        and re-runs the same time-weighting get_rating() uses internally, just referenced
        against as_of_date instead of each match's own date. Earlier versions of this method
        took the last row's PRE-match team_{1,2}_elo verbatim, which is frozen at whatever
        that match's date was -- fine for a team that's still active, silently wrong for one
        that hasn't played in years (found 2026-09-01 checking a player-engine equivalent of
        this exact bug against real retirement dates -- see the build plan's "current vs
        all-time" note)."""
        if as_of_date is None:
            as_of_date = pd.Timestamp.now()
        ledgers = cls.get_ledger_from_df(icc_elo_df)

        ratings = {}
        for team, ledger in ledgers.items():
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
            ratings[team] = (weighted_points / weighted_matches) if weighted_matches > 0.0 else cls.SEED_RATING
        return ratings

    @classmethod
    def get_effective_ratings_from_df(cls, icc_elo_df: pd.DataFrame, format_type: str, as_of_date: pd.Timestamp = None) -> dict:
        """Reliability-blended, STRETCHED rating per team AS OF as_of_date -- the raw ratings
        from get_current_ratings_from_df, shrunk toward SEED_RATING for teams with few
        matches in this pool (n / (n + MIN_MATCHES_RELIABLE), same shape as
        PlayerEloEngine.get_effective_elo), then widened via stretch_rating(format_type) (see
        TEAM_ELO_STRETCH). Use this for any external ranking/leaderboard;
        get_current_ratings_from_df's raw values are what the points-exchange dynamics
        themselves use and are neither shrinkage-adjusted nor stretched.

        Teams with fewer than MIN_MATCHES_TO_RANK (20) lifetime matches, OR whose last match
        is more than MAX_INACTIVITY_DAYS before as_of_date, are dropped from the returned
        dict entirely, not just shrunk -- two separate hard qualification floors, same as
        ICC's own rankings require a minimum sample and (implicitly, via annual re-derivation)
        recent activity before a team is published at all.

        NOTE: match_counts is each team's TOTAL lifetime match count in this pool (not just
        matches within the decay window) -- a team that played 200 matches then went quiet
        is caught by the recency floor, not treated as a thin sample."""
        if as_of_date is None:
            as_of_date = pd.Timestamp.now()
        raw_ratings = cls.get_current_ratings_from_df(icc_elo_df, as_of_date=as_of_date)
        match_counts = pd.concat(
            [icc_elo_df["team_1"], icc_elo_df["team_2"]]
        ).value_counts().to_dict()
        last_match_date = pd.concat(
            [
                icc_elo_df[["team_1", "match_date"]].rename(columns={"team_1": "team"}),
                icc_elo_df[["team_2", "match_date"]].rename(columns={"team_2": "team"}),
            ]
        ).groupby("team")["match_date"].max().to_dict()

        effective = {}
        for team, raw in raw_ratings.items():
            n = match_counts.get(team, 0)
            if n < cls.MIN_MATCHES_TO_RANK:
                continue
            inactive_days = (as_of_date - last_match_date[team]).days
            if inactive_days > cls.MAX_INACTIVITY_DAYS:
                continue
            reliability = n / (n + cls.MIN_MATCHES_RELIABLE)
            blended = reliability * raw + (1.0 - reliability) * cls.SEED_RATING
            effective[team] = cls.stretch_rating(blended, format_type)
        return effective

    def clear_cache(self):
        CacheManager.clear_cache(self.cache_file)
