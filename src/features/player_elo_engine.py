import os
import math
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from typing import Dict


from src.utils.cache_manager import CacheManager
from src.data_processing.format_spec import SEED_SOURCE


class PlayerEloEngine:
    """
    Computes chronological rolling Elo ratings for individual batters and bowlers.
    Incorporates Bayesian shrinkage toward team Elo to handle sparse/new players.
    """

    # Phase 1 (t20_player_elo_revamp_plan.md): hold-center affine stretch, t20 only --
    # fitted in Phase 0 from the real career-peak distribution (batter p99 1655.5 -> 2000).
    # od/hnd must keep D=400/K untouched, so this constant is never applied to them.
    # Phase 6: re-fitted against the POST-Phase-4 distribution (Phases 2-4's non-affine
    # dynamics changes had drifted p99 to 2157.3) to land the final numbers precisely --
    # 3.22 -> 2.48 (measured p99=1999.8 vs target 2000.0; see scripts/measure_player_elo_distribution.py).
    # RE-FITTED 2026-08-17 (player_elo_stretch_recalibration_plan.md): 2.48 -> 3.29. NOT a
    # dynamics change -- the constant went stale because the distribution it is fitted against
    # moved underneath it. Player Elo anchors on team Elo (opposing bowling-squad Elo, its
    # _team_to_player_scale fallback, and the Phase-4 opponent-quality weight), and
    # _team_to_player_scale amplifies every team-Elo point BY THIS CONSTANT -- so the
    # 2026-08-06 stale-team-Elo-cache rebuild (t20 team Elo moved on 3,697/10,634 matches by
    # up to ~400 pts; see the memory note / that fix's write-up) compressed the player tail
    # with no code change at all: batter career-peak p99 drifted 1999.8 -> 1865.3, Buttler
    # 1983.6 -> 1806.9, Kohli 1962.1 -> 1762.1, while p50 barely moved (1691.7 -> 1679.5).
    # Re-pinned with scripts/fit_player_elo_stretch.py (>=50-match cut, target 2000):
    # 2.48 (p99 1865.3) -> 3.3948 (2015.2) -> 3.2944 (1998.8), shipped at 3.29.
    # LESSON: this constant is downstream of team Elo. Any change to EloEngine, its cache, or
    # the squad-Elo aggregation decalibrates it silently -- re-run the fit script and check,
    # don't assume a shipped stretch stays fitted.
    T20_ELO_STRETCH = 3.29

    # Phase 1 (od_player_elo_revamp_plan.md): hold-center affine stretch, od only -- fitted
    # against the POST-Phase-3 distribution (batter p99 1921.1 -> 2000; Phase 3's dynamic-K
    # shipped first for od, unlike t20 where scale was Phase 1). t20/hnd keep this at 1.0
    # (never applied to them). Much smaller than T20_ELO_STRETCH because od was already close
    # to target -- see the plan's Phase 0 finding. Re-fitted precisely in od's own Phase 6.
    OD_ELO_STRETCH = 1.19

    # Phase 1 (hnd_player_elo_revamp_plan.md): hold-center affine stretch, hnd only -- fitted
    # from hnd's own career-peak distribution (batter p99 1679.9 -> 2000, >=20-match cut,
    # parity with T20/OD's own min-matches choice; see scripts/measure_player_elo_distribution.py
    # --format hnd --min-matches 20). t20/od keep their own independently-fitted stretches above;
    # this constant is never applied to them. Re-fitted precisely in this plan's own Phase 4 once
    # Phases 2-3's dynamics (if kept) have reshaped the tail, mirroring T20 Phase 6's final trim.
    # RE-FITTED 2026-08-17 (player_elo_stretch_recalibration_plan.md): 2.78 -> 4.00, alongside
    # t20's own re-fit. Same root cause (see T20_ELO_STRETCH above -- the 2026-08-06 stale-
    # team-Elo-cache rebuild also moved hnd's team Elo, on 163/167 matches by up to ~98 pts),
    # PLUS a second, hnd-specific compounding factor: the frozen hnd<-t20 seed snapshot was
    # still dated 2026-08-05, i.e. it predated the cache fix, so hnd was warm-starting every
    # seeded player from pre-fix t20 ratings. Refreshing that snapshot first (this plan's
    # Phase 2) recovered most of the drift on its own -- measured p99 1653.8 off the old
    # production parquet vs 1847.2 recomputed on the refreshed seed -- and the re-fit closes
    # the rest: 2.78 (p99 1847.2) -> 4.0038 (2000.0), shipped at 4.00.
    # CAVEAT on this number's stability: hnd's >=20-match batter population is only n=42, so
    # the p99 anchor interpolates between the top one or two players and is far more
    # single-player-sensitive than t20's n=786. That was equally true when 2.78 was fitted, so
    # it is not a new weakness -- but treat a future hnd re-fit as noisier than t20's, and
    # re-fit only when the measured gap is large (as here), never to chase a few points.
    HND_ELO_STRETCH = 4.00

    # Phase 1 (effective_game_elo_plan.md): empirical share of team legal balls faced by
    # batting position (index 0 = position 1 = first-listed), measured per format from the
    # deliveries corpus (Phase 0). team_X_players lists ARE batting order -- Phase 0 confirmed
    # Spearman(list order, actual first-appearance order) median 1.000 for all three formats
    # (t20 n=2003 mean .993, od n=2105 mean .997, hnd n=277 mean .992) -- so weighting the
    # squad-average by list position is valid everywhere. Replaces the flat np.mean, which gave
    # a #11 (~1% of team balls faced) the same weight as an opener (~17%). Unlike
    # T20_ELO_STRETCH/OD_ELO_STRETCH this is NOT frozen for cross-format-consistency reasons --
    # it's a general aggregation fix, gated per format purely by its own keep-or-revert Brier
    # result (no cross-format freeze fixture; see the plan's Phase-0 invariant).
    #
    # Phase 1 keep-or-revert (full-corpus win-prob Brier vs the locked baseline; tolerance
    # overall +0.0010 / phase +0.0025):
    #   t20: 0.127552 vs 0.127562 baseline (-0.000010)            -> PASS, kept weighted.
    #   od:  0.134374 vs 0.134128 baseline (+0.000246)             -> PASS, kept weighted.
    #   hnd: 0.164365 vs 0.139453 baseline (+0.024912, all 3 phases
    #        failed by 10-25x tolerance)                            -> FAIL, REVERTED to uniform.
    # hnd's corpus (162 matches / 15,010 states) is almost certainly too small to support the
    # added variance from position-weighting without overfitting. Uniform weights make
    # _weighted_mean_by_position() arithmetically identical to the pre-Phase-1 flat np.mean
    # (equal weight renormalizes to 1/N for any squad length) -- this is a byte-exact revert,
    # not an approximation. Confirmed empirically: re-eval after reverting reproduced hnd's
    # exact Phase-0 baseline Brier (0.139453, delta 0.000000).
    # Per-format cherry-pick (post-Phase-2): briefly shipped, then DELIBERATELY UNDONE by the
    # user's explicit call -- despite t20 (batting-only) and od (bowling-only) each measuring
    # marginally better in isolation, the user judged that having BOTH batting- and
    # bowling-weighted squad Elo available for BOTH formats is more valuable long-term than the
    # small Brier gain from cherry-picking (a product/optionality decision, not a metric-driven
    # one -- the cherry-picked deltas were noise-adjacent to begin with, see the Invariants
    # callouts). W_BAT/W_BOWL below are back to "both phases, both formats" (t20 + od each
    # fully weighted on both sides; hnd untouched, still fully reverted -- its corpus never
    # supported either weighting). See the Invariants section for the full history (both
    # bugs, the cumulative-drift check, the cherry-pick, and this reversal) and the tracker's
    # "cherry-pick reverted" row for the restoration mechanics.
    # A TRUE (unbounded) flat-mean revert -- used wherever a format/side has been reverted to
    # "uniform" -- must have MORE slots than any real squad list can ever have, or
    # _weighted_mean_by_position/_weighted_mean_by_bowling_rank's top-len(weights) truncation
    # silently drops the lowest-ranked name instead of averaging everyone. Found the hard way:
    # t20 team_1_players/team_2_players actually run up to 13 names (491+477 matches list 12,
    # 2+5 list 13 -- some league's Impact-Player/substitute rule), od up to 12 (124 team-
    # entries); a naive 11-slot uniform vector truncated those, producing a real (if small,
    # ~0.0001 Brier) gap from the true old flat-mean behavior it was supposed to reproduce
    # exactly. 20 is comfortably above every observed length (max 13) in all three formats.
    UNIFORM_FLAT_MEAN = [1.0 / 20.0] * 20

    # t20w: UNIFORM_FLAT_MEAN, not t20's own measured positional weights -- t20's W_BAT vector
    # was empirically fit to the MEN's batting-order distribution (effective_game_elo_plan.md
    # Phase 0); reusing it for the women's population without its own equivalent measurement
    # would be exactly the kind of unvalidated cross-population borrowing that plan's Invariants
    # warn against for the Elo stretch constants. Uniform is the same conservative choice hnd
    # already uses for the same reason (its corpus was too small to validate weighting; t20w's
    # corpus hasn't been measured for its own weights yet, even though it's large enough to).
    # Revisit with a dedicated Phase-0-style measurement (batting-order Spearman + w_bat share by
    # position, t20w's own deliveries) before ever weighting this format's squad aggregation.
    W_BAT = {
        "t20": [0.17257, 0.17055, 0.16282, 0.14871, 0.11879, 0.08718, 0.05949, 0.03809, 0.02315, 0.01309, 0.00556],
        "od":  [0.14898, 0.14974, 0.15719, 0.14523, 0.12207, 0.09537, 0.06976, 0.04861, 0.03246, 0.02070, 0.00988],
        "hnd": UNIFORM_FLAT_MEAN,
        "t20w": UNIFORM_FLAT_MEAN,
        # odw/hndw: same reasoning as t20w above -- od's/hnd's measured vectors are men's-
        # population fits, not yet re-derived for the women's corpora. Uniform until a
        # dedicated measurement.
        "odw": UNIFORM_FLAT_MEAN,
        "hndw": UNIFORM_FLAT_MEAN,
    }

    # Phase 2 (effective_game_elo_plan.md): frontline-weighted bowling squad Elo, K=6. Phase 0
    # measured, per format, the share of team overs bowled by each bowling-Elo rank (1=highest
    # pre-match bowling effective_elo in the squad): the top 6 ranks bowl ~96-97% of overs in
    # every format (t20 cumulative .967, od .956, hnd .972 at rank 6; mean distinct
    # bowlers/innings ~5.9-6.1) -- ranks 7-11 combine for ~0. Weights below are each format's
    # raw top-6 share renormalized to sum to 1.0 (e.g. t20's raw top-6 sum 0.96709 -> divided
    # through), replacing the old flat 11-player mean that diluted the attack with ~5 players
    # who effectively don't bowl. Unlike W_BAT this is rank-based, not list-position-based --
    # see _weighted_mean_by_bowling_rank(). Same per-format keep-or-revert discipline as
    # Phase 1 (no cross-format freeze); starts from Phase 1's per-format outcome (t20/od
    # weighted batting kept, hnd batting reverted to uniform).
    #
    # Phase 2 keep-or-revert (full-corpus win-prob Brier vs the PHASE-1 outcome, since this
    # stacks on batting; same tolerance overall +0.0010 / phase +0.0025):
    #   t20: 0.127643 vs 0.127552 Phase-1 (+0.000091)  -> PASS, kept K=6 weighted.
    #   od:  0.134050 vs 0.134374 Phase-1 (-0.000324)  -> PASS (improved), kept K=6 weighted.
    #   hnd: 0.169494 vs 0.139453 Phase-1 (+0.030041, every phase failed 10-20x tolerance,
    #        worse even than hnd's Phase-1 batting failure) -> FAIL, REVERTED.
    # Same root cause as Phase 1: hnd's 162-match/15,010-state corpus can't support the added
    # variance from rank-weighting. hnd's revert uses 11 uniform slots (NOT 6) -- ranking
    # doesn't change a uniform average's value, and _weighted_mean_by_bowling_rank truncates to
    # len(weights), so 11 slots means no top-K truncation occurs and this is arithmetically
    # identical to the pre-Phase-2 flat np.mean over all listed players regardless of bowling
    # rank. Confirmed byte-exact empirically (see Phase 1's revert methodology): re-eval of the
    # restored pre-Phase-2 pkl against states rebuilt from the reverted parquet reproduced
    # hnd's Phase-1 outcome (== Phase-0 baseline) exactly, delta 0.000000.
    # Cherry-pick reverted (see the W_BAT comment above): t20's bowling goes back to its real
    # K=6 weighting alongside od's, per the user's explicit "keep both signals available for
    # both formats" call.
    # t20w: UNIFORM_FLAT_MEAN -- same reasoning as W_BAT above, t20's rank-weighted vector is a
    # men's-population measurement, not yet re-derived for t20w.
    W_BOWL = {
        "t20": [0.18197, 0.17691, 0.17480, 0.17141, 0.16506, 0.12984],
        "od":  [0.18399, 0.17779, 0.17375, 0.16859, 0.16277, 0.13311],
        "hnd": UNIFORM_FLAT_MEAN,
        "t20w": UNIFORM_FLAT_MEAN,
        "odw": UNIFORM_FLAT_MEAN,
        "hndw": UNIFORM_FLAT_MEAN,
    }

    def __init__(
        self,
        k_runs=None,
        k_wickets=None,
        decay_days=180.0,
        decay_rate=0.95,
        default_elo=1500.0,
        min_matches_reliable=10,
        processed_path="data/processed",
        format_type="t20",
        form_window=5,
    ):
        self.format_type = format_type
        # odw is also a 50-over format -- step-size regime is tied to per-innings ball
        # volume (a structural fact), not a men's-population fit, so it's safe to widen
        # rather than let odw fall through to t20's 0.3/3.0.
        if k_runs is None:
            k_runs = 0.8 if format_type in ("od", "odw") else 0.3
        if k_wickets is None:
            k_wickets = 12.0 if format_type in ("od", "odw") else 3.0

        # elo_divisor and K are stretched by the SAME factor -- together they form one
        # affine map (elo' = 1500 + a*(elo-1500)) that must not change model predictions.
        if format_type == "t20":
            stretch = self.T20_ELO_STRETCH
        elif format_type == "od":
            stretch = self.OD_ELO_STRETCH
        elif format_type == "hnd":
            stretch = self.HND_ELO_STRETCH
        else:
            stretch = 1.0
        self.k_runs = float(k_runs) * stretch
        self.k_wickets = float(k_wickets) * stretch
        self.elo_divisor = 400.0 * stretch
        self.decay_days = float(decay_days)
        self.decay_rate = float(decay_rate)
        self.default_elo = float(default_elo)
        self.min_matches_reliable = float(min_matches_reliable)
        # Phase 3 (t20_player_elo_revamp_plan.md, t20; od_player_elo_revamp_plan.md, od):
        # experience-graded K. Multiplier on k_runs/k_wickets that starts high for newcomers
        # (fast convergence) and decays to 1.0 (today's fixed K, unchanged) as matches_played
        # grows -- same functional form as EloEngine.compute_dynamic_k. t20 reuses EloEngine's
        # own t20 max/min ratio (50/20=2.5); od reuses EloEngine's own od ratio (35/25=1.4) --
        # NOT t20's 2.5, since od's team-level engine was already independently tuned to a
        # gentler ratio. k_scale=30 kept for od too: measured od_player_elo_revamp_plan.md
        # Phase 3 per-player appearance-count profiles are near-identical to t20's through p75
        # (both p50=7, p75~19-20; only the elite p99 tail differs, 140 vs 231), i.e.
        # statistically indistinguishable over the region k_scale actually shapes.
        # Phase 2 (hnd_player_elo_revamp_plan.md) -- TRIED, REVERTED. Ported od's max_mult=1.4
        # (conservative choice, k_scale=30 unchanged -- hnd's own measured appearance profile
        # has batter p50=7, matching t20/od's own p50 exactly; only the upper tail is compressed,
        # a genuine fact about the competition's youth (started 2021, max appearances=43), not a
        # k_scale mismatch). Gate result (unpooled hnd win-prob Brier, scripts/hnd_winprob_
        # phase_gate.py -- see the plan's Phase 2 tracker row): 0.166382 (Phase 1) -> 0.167464,
        # Delta +0.001082 -- a regression exceeding the +-0.0010 overall tolerance (worse than
        # T20's own analogous Phase 2 veteran-de-shrink miss of +0.000167, which itself reverted
        # for the same reason: no independent justification to keep a phase whose only stated
        # purpose is a Brier gain it didn't deliver). Per-phase: death improved (-0.001393) but
        # middle (+0.001700) and powerplay (+0.001675) both worsened, consistent with hnd's small
        # corpus (162 matches) not supporting the added variance -- the same failure mode as
        # effective_game_elo_plan.md's position/bowling squad-weighting experiments on hnd.
        # hnd stays at the exact 1.0 no-op alongside t20w below.
        if format_type == "t20":
            self.dynamic_k_max_mult = 2.5
        elif format_type == "od":
            self.dynamic_k_max_mult = 1.4
        else:
            self.dynamic_k_max_mult = 1.0
        self.dynamic_k_min_mult = 1.0
        self.dynamic_k_scale = 30.0
        # Phase 4 (t20_player_elo_revamp_plan.md): opponent-quality weight, t20 only.
        # Multiplies the update magnitude by a bounded, saturating (tanh) function of
        # ABSOLUTE opponent quality (opp squad Elo's deviation from 1500) -- runs/wickets
        # against a stronger opponent move the rating more than the same surprise against a
        # weaker one, beyond what the already opponent-adjusted expected-runs/wickets term
        # captures. weight_scale=300 calibrated from the real post-Phase-3 squad-Elo
        # deviation distribution (median ~100-130, p90 ~300-380, p99 ~555-690), so typical
        # matchups get a modest adjustment and only genuinely extreme mismatches approach
        # the +/-weight_coef bound (1.5x/0.5x at weight_coef=0.5). od: weight_coef=0.0 -- never
        # shipped (od_player_elo_revamp_plan.md flagged it "speculative", deprioritized after
        # dynamic K proved the bigger win, not gate-failed).
        # Phase 3 (hnd_player_elo_revamp_plan.md) -- TRIED, REVERTED. weight_coef=0.5/
        # weight_scale=300, the latter measured (not guessed) from hnd's own post-Phase-1
        # opponent-Elo deviation distribution (median 113.7, p90 297.9, p99 387.0 -- closely
        # matched the T20 distribution weight_scale=300 was fitted from). Gate result (unpooled
        # hnd win-prob Brier, scripts/hnd_winprob_phase_gate.py -- see the plan's Phase 3 tracker
        # row): Phase 1 0.166382 -> 0.166768, Delta +0.000386 -- inside the +-0.0010 tolerance
        # band but a genuine worsening, not an improvement. Reverted per the same bar T20's own
        # Phase 2 (veteran de-shrink) was reverted under: a phase whose sole justification is a
        # Brier gain must actually show one, "within tolerance" is not "earned its place". hnd
        # stays at the exact 1.0 no-op alongside od (never shipped) and t20w (not yet measured).
        self.opp_weight_coef = 0.5 if format_type == "t20" else 0.0
        self.opp_weight_scale = 300.0
        # Fallback is UNIFORM_FLAT_MEAN, not t20's measured positional weights -- a missing
        # key (a format with no entry yet) must never silently borrow a men's-population-fit
        # vector. Byte-identical for t20/od/hnd/t20w, which all have explicit entries above.
        self.w_bat = self.W_BAT.get(format_type, self.UNIFORM_FLAT_MEAN)
        self.w_bowl = self.W_BOWL.get(format_type, self.UNIFORM_FLAT_MEAN)
        self.processed_path = Path(processed_path)
        self.form_window = int(form_window)
        self.cache_file = self.processed_path / f"{format_type}_player_elo_ratings.parquet"
        os.makedirs(self.processed_path, exist_ok=True)

        # Base format averages -- the expected-runs/expected-wickets baseline every player's
        # Elo update is measured against. Must be measured per format's own corpus, not
        # borrowed: t20w's actual scoring rate is genuinely lower than men's t20 (measured
        # 2026-08-05, legal balls only: t20w 0.9785 runs/ball vs men's t20 1.2154 -- consistent
        # with t20w's lower measured par score, format_spec.py's t20w default_venue_score
        # 125.0 vs t20's 160.0). Reusing t20's 1.25 here would systematically bias every t20w
        # player's expected performance too high (batters look like they underperform, bowlers
        # look like they overperform, relative to a bar that's set too high). Wicket rate is
        # nearly gender-invariant (0.0562 vs 0.0568), unlike run rate.
        # odw/hndw measured 2026-08-06, legal balls only, full corpus (929 / 155 matches):
        # odw 0.7280 runs/ball, 0.0289 wickets/ball; hndw 1.1942 runs/ball, 0.0609 wickets/ball.
        # Consistent with the same men's-vs-women's reduction pattern t20w showed (odw is
        # ~14% below od's 0.85, same direction/magnitude as t20w's ~22% reduction from t20).
        if format_type == "od":
            self.avg_runs_per_ball = 0.85
            self.avg_wickets_per_ball = 0.026
        elif format_type == "odw":
            self.avg_runs_per_ball = 0.7280
            self.avg_wickets_per_ball = 0.0289
        elif format_type == "t20w":
            self.avg_runs_per_ball = 0.9785
            self.avg_wickets_per_ball = 0.0562
        elif format_type == "hndw":
            self.avg_runs_per_ball = 1.1942
            self.avg_wickets_per_ball = 0.0609
        else:  # t20, hnd
            self.avg_runs_per_ball = 1.25
            self.avg_wickets_per_ball = 0.054

        # State tracking: ratings and match counts
        self.batter_ratings = {}       # player -> elo
        self.bowler_ratings = {}       # player -> elo
        self.batter_match_counts = {}  # player -> count
        self.bowler_match_counts = {}  # player -> count
        self.batter_last_dates = {}    # player -> date
        self.bowler_last_dates = {}    # player -> date

        # Phase 2 (frontline_bowler_eligibility_plan.md): player -> deque of 0/1 flags, one per
        # named appearance, tracking whether they bowled -- trailing FRONTLINE_BOWL_WINDOW only.
        # Read via get_recent_bowl_rate(); NOT yet consumed by the rank key (that's Phase 3) --
        # this phase only populates it, so it can be verified leakage-free (byte-identical Brier
        # against the Phase-0 baseline) before anything downstream starts reading it.
        self.player_recent_bowled = {}  # player -> deque([0/1], maxlen=FRONTLINE_BOWL_WINDOW)

        # Recent deltas tracking for form features (Proposal 5)
        self.batter_recent_deltas = {}  # player -> list of Elo updates
        self.bowler_recent_deltas = {}  # player -> list of Elo updates

    def get_raw_elo(self, player: str, role: str) -> float:
        if role == "batter":
            return self.batter_ratings.setdefault(player, self.default_elo)
        else:
            return self.bowler_ratings.setdefault(player, self.default_elo)

    def get_match_count(self, player: str, role: str) -> int:
        if role == "batter":
            return self.batter_match_counts.setdefault(player, 0)
        else:
            return self.bowler_match_counts.setdefault(player, 0)

    def get_last_date(self, player: str, role: str):
        if role == "batter":
            return self.batter_last_dates.get(player, None)
        else:
            return self.bowler_last_dates.get(player, None)

    def get_recent_bowl_rate(self, player: str) -> float:
        """Share of `player`'s trailing FRONTLINE_BOWL_WINDOW named appearances (batting or
        bowling) in which they actually bowled, as of strictly before the match currently being
        processed -- see the deque populated at the end of each per-match iteration in run().

        A player absent from player_recent_bowled (never named in a processed match yet) gets
        rate 0.0, matching bowling_appearance_rates()'s serving-side "missing means not
        eligible" contract -- never "unknown, assume eligible".
        """
        d = self.player_recent_bowled.get(player)
        return (sum(d) / len(d)) if d else 0.0

    def _bowler_rank_key(self, players: list, eff_elos: dict) -> dict:
        """Phase 3 (frontline_bowler_eligibility_plan.md): zeroes any player whose recent
        bowling-appearance rate is below FRONTLINE_BOWL_MIN_RATE, not merely anyone with a
        bowling appearance ever. The old bar -- get_match_count(p, "bowler") > 0 -- let a
        batter's stale, shrinkage-inflated bowling Elo from a handful of long-ago overs rank as
        a frontline bowler forever. Observed on Surrey vs Kent 2026-08-05: DP Sibley (3 career
        bowling appearances, none since 2018) ranked as Surrey's TOP bowling option ahead of two
        bowlers who'd bowled in 10 of their last 10 appearances.

        get_recent_bowl_rate() is leakage-safe by construction (Phase 2: only reflects matches
        strictly before this one). Deliberately no known_bowler ground-truth escape hatch here,
        unlike the serving-side frontline_bowl_sort_key -- extract_match_squads' use of in-match
        truth is explicitly leakage-fine for simulation, but this IS the training loop. A
        debutant bowler therefore still ranks 0.0 on their debut, identical to today's
        behaviour (their get_match_count was also 0 then).
        """
        return {
            p: (eff_elos[p] if self.get_recent_bowl_rate(p) >= FRONTLINE_BOWL_MIN_RATE else 0.0)
            for p in players
        }

    def _team_to_player_scale(self, team_elo: float) -> float:
        """
        Maps a team-Elo value (EloEngine's fixed 1500-centred scale) onto this engine's
        player scale. t20's, od's and hnd's player scales are each stretched (their own Phase 1
        hold-center rescale, hnd_player_elo_revamp_plan.md for hnd); t20w shares the team scale
        unchanged, so this is a no-op for it. Every site using team Elo as a player-scale proxy
        must route through here -- see the plans' "scale-mixing" invariant. OD/HND-specific:
        this includes the opp_elo used directly in their expected-runs term (od/hnd have no
        squad-average opponent proxy like t20's -- see run()'s opp_eff_elo selection), which
        the od/hnd revamp plans' Phase 1 invariant calls out as the #1 risk unique to them --
        miss it and the affine map is broken.
        """
        if self.format_type == "t20":
            return 1500.0 + (team_elo - 1500.0) * self.T20_ELO_STRETCH
        elif self.format_type == "od":
            return 1500.0 + (team_elo - 1500.0) * self.OD_ELO_STRETCH
        elif self.format_type == "hnd":
            return 1500.0 + (team_elo - 1500.0) * self.HND_ELO_STRETCH
        return team_elo

    @staticmethod
    def _weighted_mean_by_position(players: list, elo_map: dict, fallback: float, weights: list) -> float:
        """
        Weighted mean of `players`' Elo (looked up in `elo_map`, falling back to `fallback` for
        a player missing this match's snapshot -- e.g. no bowling record this game) using
        per-position `weights` (index 0 = first-listed player). Two edge cases handled so this
        never silently misweights or crashes:
          - Squad list shorter than len(weights) (rare/incomplete records): renormalizes over
            the weights actually used, so the result stays a proper convex combination instead
            of implicitly losing probability mass to absent trailing positions.
          - Squad list longer than len(weights) (e.g. an impact-sub name appended): extra
            players reuse the last (smallest) weight rather than being dropped or crashing.
          - Empty list: returns `fallback` directly (mirrors the old flat-mean's empty-list
            branch).
        """
        if not players:
            return fallback
        last_w = weights[-1]
        total_w = 0.0
        total = 0.0
        for i, p in enumerate(players):
            w = weights[i] if i < len(weights) else last_w
            total += w * elo_map.get(p, fallback)
            total_w += w
        return total / total_w if total_w > 0 else fallback

    @staticmethod
    def _weighted_mean_by_bowling_rank(players: list, elo_map: dict, fallback: float, weights: list,
                                        rank_key_map: dict = None) -> float:
        """
        Frontline-weighted mean of `players`' bowling Elo (Phase 2): ranks the squad (descending)
        then takes a weighted mean of the top len(weights) ranks using `weights` (index 0 =
        strongest bowler in the squad). Unlike _weighted_mean_by_position this is rank-based,
        not identity/list-position-based.

        `rank_key_map` (optional, defaults to `elo_map`) determines RANKING ORDER ONLY -- the
        VALUE blended into the weighted mean always comes from `elo_map`. This split matters
        because `elo_map` here is typically each player's bowling `effective_elo`, which for a
        player with NO prior bowling history is a Bayesian-shrinkage blend toward the TEAM's
        overall Elo (see get_effective_elo) -- for a strong team this can EXCEED genuine
        specialist bowlers' real ratings (e.g. a ~2070 team Elo t20 side shrinkage-anchors its
        non-bowlers near ~2930 on the player scale, well above most real bowling Elos), which
        would rank pure batters as "frontline bowlers" ahead of the team's actual attack --
        exactly the trap match_simulator.py's own (in-match, leakage-fine-for-simulation)
        attack-selection already guards against by zeroing non-bowlers for ranking purposes
        only. Callers building a pre-match (leakage-free) rank key should zero out players with
        zero PRIOR bowling appearances (e.g. via get_match_count(p, "bowler") == 0), not rely on
        elo_map directly.

        Same edge-case handling as _weighted_mean_by_position: a squad with fewer than
        len(weights) players renormalizes over the ranks that exist; empty -> fallback.
        """
        if not players:
            return fallback
        rank_source = rank_key_map if rank_key_map is not None else elo_map
        ranked_players = sorted(players, key=lambda p: rank_source.get(p, fallback), reverse=True)
        total_w = 0.0
        total = 0.0
        for i, p in enumerate(ranked_players[:len(weights)]):
            w = weights[i]
            total += w * elo_map.get(p, fallback)
            total_w += w
        return total / total_w if total_w > 0 else fallback

    def get_effective_elo(self, player: str, role: str, team_elo: float) -> float:
        """
        Bayesian shrinkage toward team Elo for players with limited match experience.
        """
        raw_elo = self.get_raw_elo(player, role)
        n_matches = self.get_match_count(player, role)

        # Blend factor (Sigmoid-like blend)
        reliability = n_matches / (n_matches + self.min_matches_reliable)
        team_elo_scaled = self._team_to_player_scale(team_elo)
        return reliability * raw_elo + (1.0 - reliability) * team_elo_scaled

    def _dynamic_k_multiplier(self, n_matches: float) -> float:
        """
        Phase 3: multiplier on k_runs/k_wickets for the update at this player's n_matches-th
        appearance. Same shape as EloEngine.compute_dynamic_k; floor (dynamic_k_min_mult=1.0)
        reproduces today's fixed K exactly as matches_played grows, so only newcomers'
        updates move faster -- established players' dynamics are untouched. Active for t20
        (max_mult=2.5) and od (max_mult=1.4); exactly 1.0 for hnd (tried and reverted,
        hnd_player_elo_revamp_plan.md Phase 2 -- see its tracker row) and t20w (max_mult==
        min_mult for both).
        """
        return self.dynamic_k_min_mult + (self.dynamic_k_max_mult - self.dynamic_k_min_mult) / (
            1.0 + n_matches / self.dynamic_k_scale
        )

    def _opponent_quality_weight(self, opp_elo: float) -> float:
        """
        Phase 4: multiplicative weight on the update magnitude, based on absolute opponent
        quality (opp_elo's deviation from 1500). Symmetric on the update's sign -- a
        stronger opponent makes ANY surprise (over- or under-performance) more informative,
        weighted equally. Bounded/saturating (tanh), so a genuinely extreme mismatch can't
        blow the update up unboundedly. weight(1500) == 1.0 exactly (today's behavior).
        Active for t20 only; exactly 1.0 for od (never shipped), hnd (tried and reverted,
        hnd_player_elo_revamp_plan.md Phase 3 -- see its tracker row), and t20w (not yet
        measured) -- opp_weight_coef=0.0 for all three.
        """
        return 1.0 + self.opp_weight_coef * math.tanh((opp_elo - 1500.0) / self.opp_weight_scale)

    def calculate_decayed_elo(self, elo: float, last_date, current_date) -> float:
        if not last_date or not current_date:
            return elo
        try:
            d1 = last_date if isinstance(last_date, pd.Timestamp) else pd.to_datetime(last_date)
            d2 = current_date if isinstance(current_date, pd.Timestamp) else pd.to_datetime(current_date)
            days = (d2 - d1).days
            if days > self.decay_days:
                periods = (days - self.decay_days) / self.decay_days
                decay_factor = self.decay_rate ** periods
                return self.default_elo + (elo - self.default_elo) * decay_factor
        except Exception:
            pass
        return elo

    def _aggregate_match_stats(self, deliveries_df: pd.DataFrame):
        """
        Aggregates deliveries into match-level batter and bowler statistics.
        """
        # Create legal delivery flag if not present
        if "legal_delivery" not in deliveries_df.columns:
            deliveries_df = deliveries_df.copy()
            deliveries_df["legal_delivery"] = (
                (deliveries_df["is_wide"] == 0) & (deliveries_df["is_noball"] == 0)
            ).astype(int)

        # Batting stats
        bat_stats = (
            deliveries_df.groupby(["match_id", "batting_team", "batter"])
            .agg(
                runs=("runs_off_bat", "sum"),
                balls_faced=("legal_delivery", "sum"),
                dismissals=("wicket", "sum"),
            )
            .reset_index()
        )
        # Cap dismissals at 1 per innings
        bat_stats["dismissals"] = np.minimum(bat_stats["dismissals"], 1)

        # Bowling stats
        # Opposing team is the one batting (batting_team).
        # Cast batting_team to string first to avoid pandas categorical list aggregation error
        batting_teams_str = deliveries_df[["match_id", "batting_team"]].copy()
        batting_teams_str["batting_team"] = batting_teams_str["batting_team"].astype(str)
        bowling_teams = (
            batting_teams_str.drop_duplicates()
            .groupby("match_id")
            .agg(teams=("batting_team", list))
            .reset_index()
        )

        bowl_stats = (
            deliveries_df.groupby(["match_id", "batting_team", "bowler"])
            .agg(
                runs_conceded=("total_runs", "sum"),
                balls_bowled=("legal_delivery", "sum"),
                wickets=("wicket", "sum"),
            )
            .reset_index()
        )

        return bat_stats, bowl_stats, bowling_teams

    def run(self, deliveries_df: pd.DataFrame, elo_df: pd.DataFrame, force=False, precomputed_stats=None, seed_ratings=None) -> pd.DataFrame:
        """
        Computes rolling player Elo ratings chronologically.

        Args:
            deliveries_df: Raw ball-by-ball deliveries DataFrame.
            elo_df: Team Elo ratings DataFrame.
            force: If True, recompute even if cache exists.
            precomputed_stats: Optional tuple of (bat_stats, bowl_stats, bowling_teams)
                from a prior call to _aggregate_match_stats(). Avoids re-aggregating
                deliveries in sweep scripts where the raw data doesn't change.
            seed_ratings: Optional dict with 'batter' and 'bowler' keys mapping player -> elo
                (e.g., warm-started T20 ratings for Phase 4b transfer learning).
        """
        sources = [Path(__file__)]
        registry_file = Path(__file__).parent / "build_player_registry.py"
        if registry_file.exists():
            sources.append(registry_file)

        if not force and CacheManager.is_cache_valid(self.cache_file, sources):
            print(f"Loading player Elo ratings from cache: {self.cache_file}")
            return pd.read_parquet(self.cache_file)


        print("Computing rolling player Elo ratings...")
        # Reset internal states to ensure a clean run
        if seed_ratings:
            batter_seeds = seed_ratings.get("batter", {})
            bowler_seeds = seed_ratings.get("bowler", {})
            if self.format_type in SEED_SOURCE:
                # seed_ratings comes from SEED_SOURCE[self.format_type]'s effective_elo
                # (cricsheet_importer.retrain_full / hundred_validation.py). Re-map onto this
                # format's OWN stretched scale (not the source's, and not an unstretched 1:1)
                # -- hnd_player_elo_revamp_plan.md Phase 1: since elo_divisor/k_runs/k_wickets
                # operate on THIS format's own *_ELO_STRETCH-wide scale, a seed left on the
                # source's scale would start every seeded player off-scale relative to how
                # their rating evolves from here. The relative-skill mapping is the ratio of
                # the two formats' own stretches -- generalises hnd<-t20's original remap
                # (2.78/2.48) to hndw<-t20w (1.0/1.0, a no-op, both unstretched).
                stretch_map = {"t20": self.T20_ELO_STRETCH, "od": self.OD_ELO_STRETCH, "hnd": self.HND_ELO_STRETCH}
                source_format = SEED_SOURCE[self.format_type]
                remap = stretch_map.get(self.format_type, 1.0) / stretch_map.get(source_format, 1.0)
                batter_seeds = {k: 1500.0 + (float(v) - 1500.0) * remap for k, v in batter_seeds.items()}
                bowler_seeds = {k: 1500.0 + (float(v) - 1500.0) * remap for k, v in bowler_seeds.items()}
            self.batter_ratings = {k: float(v) for k, v in batter_seeds.items()}
            self.bowler_ratings = {k: float(v) for k, v in bowler_seeds.items()}
            print(f"  Warm-seeded {len(self.batter_ratings)} batters and {len(self.bowler_ratings)} bowlers from prior ratings.")
        else:
            self.batter_ratings = {}
            self.bowler_ratings = {}
        self.batter_match_counts = {}
        self.bowler_match_counts = {}
        self.batter_last_dates = {}
        self.bowler_last_dates = {}
        self.batter_recent_deltas = {}
        self.bowler_recent_deltas = {}

        # Pre-aggregate match stats (or use precomputed)
        if precomputed_stats is not None:
            bat_stats, bowl_stats, bowling_teams = precomputed_stats
        else:
            bat_stats, bowl_stats, bowling_teams = self._aggregate_match_stats(deliveries_df)

        # Convert to list of dict records grouped by match_id for O(1) loop lookups
        bat_records = bat_stats.to_dict("records")
        bat_groups = {}
        for r in bat_records:
            bat_groups.setdefault(r["match_id"], []).append(r)

        bowl_records = bowl_stats.to_dict("records")
        bowl_groups = {}
        for r in bowl_records:
            bowl_groups.setdefault(r["match_id"], []).append(r)

        # Index bowling teams for fast lookup
        opp_team_map = {}
        for _, row in bowling_teams.iterrows():
            m_id = row["match_id"]
            teams = row["teams"]
            if len(teams) == 2:
                opp_team_map[m_id] = {teams[0]: teams[1], teams[1]: teams[0]}
            else:
                opp_team_map[m_id] = {}

        # Map match_id to pre-match team Elo ratings
        cols = ["team_1", "team_2", "team_1_elo", "team_2_elo"]
        if "team_1_players" in elo_df.columns:
            cols.extend(["team_1_players", "team_2_players"])
        team_elo_map = elo_df.set_index("match_id")[cols].to_dict("index")

        # Sort matches chronologically by match_date
        sorted_matches = elo_df.sort_values(by=["match_date", "match_id"])[["match_id", "match_date"]].drop_duplicates()

        player_elo_records = []

        for _, match_row in sorted_matches.iterrows():
            match_id = match_row["match_id"]
            match_date = match_row["match_date"]

            t_elos = team_elo_map.get(match_id, {})
            if not t_elos:
                continue

            t1, t2 = t_elos["team_1"], t_elos["team_2"]
            t1_elo, t2_elo = t_elos["team_1_elo"], t_elos["team_2_elo"]

            # Dict for opposition team Elo lookup
            team_strength = {t1: t1_elo, t2: t2_elo}

            # Parse squad player lists
            t1_players_str = t_elos.get("team_1_players", "")
            t2_players_str = t_elos.get("team_2_players", "")
            t1_players = [p.strip() for p in t1_players_str.split(",") if p.strip()] if isinstance(t1_players_str, str) else []
            t2_players = [p.strip() for p in t2_players_str.split(",") if p.strip()] if isinstance(t2_players_str, str) else []

            # Pre-decay playing XI players to match_date and build lookup maps of pre-match effective Elo
            match_batter_eff_elos = {}
            match_bowler_eff_elos = {}

            for p in t1_players:
                b_elo = self.get_raw_elo(p, "batter")
                b_elo = self.calculate_decayed_elo(b_elo, self.get_last_date(p, "batter"), match_date)
                self.batter_ratings[p] = b_elo
                match_batter_eff_elos[p] = self.get_effective_elo(p, "batter", t1_elo)

                bowl_elo = self.get_raw_elo(p, "bowler")
                bowl_elo = self.calculate_decayed_elo(bowl_elo, self.get_last_date(p, "bowler"), match_date)
                self.bowler_ratings[p] = bowl_elo
                match_bowler_eff_elos[p] = self.get_effective_elo(p, "bowler", t1_elo)

            for p in t2_players:
                b_elo = self.get_raw_elo(p, "batter")
                b_elo = self.calculate_decayed_elo(b_elo, self.get_last_date(p, "batter"), match_date)
                self.batter_ratings[p] = b_elo
                match_batter_eff_elos[p] = self.get_effective_elo(p, "batter", t2_elo)

                bowl_elo = self.get_raw_elo(p, "bowler")
                bowl_elo = self.calculate_decayed_elo(bowl_elo, self.get_last_date(p, "bowler"), match_date)
                self.bowler_ratings[p] = bowl_elo
                match_bowler_eff_elos[p] = self.get_effective_elo(p, "bowler", t2_elo)

            # Compute squad Elo averages (using pre-match effective Elo ratings). Fallbacks
            # here use team Elo as a player-scale proxy -- route through
            # _team_to_player_scale (scale-mixing invariant; no-op for hnd).
            #
            # BATTING (Phase 1, effective_game_elo_plan.md): position-weighted by self.w_bat --
            # t1_players/t2_players are listed in batting order (Phase-0 confirmed), so this
            # reflects that an opener facing ~17% of team balls matters far more to the squad's
            # batting strength than a #11 facing ~1%.
            #
            # BOWLING (Phase 2): frontline-weighted by self.w_bowl (rank-based, top K=6 by
            # bowling Elo) instead of a flat mean over all 11 -- Phase 0 measured the top 6
            # bowling-Elo ranks bowl ~96-97% of a team's overs in every format, so the old flat
            # mean diluted the attack with ~5 players who barely bowl.
            #
            # Bug fix (post-Phase-2 review): match_bowler_eff_elos[p] is populated for EVERY
            # listed player, not just those who bowl -- for a player with zero prior bowling
            # appearances this is a pure Bayesian-shrinkage blend toward the TEAM's overall Elo
            # (get_effective_elo, reliability=0), which _team_to_player_scale can push WELL
            # above genuine specialist bowlers' real ratings for a strong team (observed: a
            # ~2076 team Elo t20 side's non-bowlers shrinkage-anchor near ~2929, vs its actual
            # best bowler's real ~2255 -- ranking on raw elo_map values let 5 pure batters
            # outrank the entire attack, producing a nonsensical squad average >2600). Rank on
            # a separate key instead -- see _bowler_rank_key (Phase 3,
            # frontline_bowler_eligibility_plan.md: recent-bowling-rate gated, not merely
            # "bowled at least once ever", which mirrors match_simulator.py's own
            # attack-selection guard against exactly this trap far more closely than the
            # original "any prior appearance" bar did).
            t1_bowler_rank_key = self._bowler_rank_key(t1_players, match_bowler_eff_elos)
            t2_bowler_rank_key = self._bowler_rank_key(t2_players, match_bowler_eff_elos)

            t1_bat_squad = self._weighted_mean_by_position(
                t1_players, match_batter_eff_elos, self._team_to_player_scale(t1_elo), self.w_bat
            )
            t1_bowl_squad = self._weighted_mean_by_bowling_rank(
                t1_players, match_bowler_eff_elos, self._team_to_player_scale(t1_elo), self.w_bowl,
                rank_key_map=t1_bowler_rank_key
            )

            t2_bat_squad = self._weighted_mean_by_position(
                t2_players, match_batter_eff_elos, self._team_to_player_scale(t2_elo), self.w_bat
            )
            t2_bowl_squad = self._weighted_mean_by_bowling_rank(
                t2_players, match_bowler_eff_elos, self._team_to_player_scale(t2_elo), self.w_bowl,
                rank_key_map=t2_bowler_rank_key
            )

            # 1. Batters
            match_batters = bat_groups.get(match_id, [])
            for row in match_batters:
                batter = row["batter"]
                batting_team = row["batting_team"]
                opp_team = opp_team_map.get(match_id, {}).get(batting_team, None)
                opp_elo = team_strength.get(opp_team, 1500.0)

                # Get pre-match Elo (already decayed at loop start)
                b_elo = self.batter_ratings.get(batter, self.default_elo)

                # Get pre-match form (Proposal 5)
                b_deltas = self.batter_recent_deltas.get(batter, [])
                b_form = float(np.mean(b_deltas)) if b_deltas else 0.0

                # Record pre-match ratings
                player_elo_records.append({
                    "match_id": match_id,
                    "player": batter,
                    "role": "batter",
                    "raw_elo": b_elo,
                    "effective_elo": match_batter_eff_elos.get(batter, self.get_effective_elo(batter, "batter", team_strength.get(batting_team, 1500.0))),
                    "recent_form": b_form,
                    "team_1_batting_squad_elo": t1_bat_squad,
                    "team_1_bowling_squad_elo": t1_bowl_squad,
                    "team_2_batting_squad_elo": t2_bat_squad,
                    "team_2_bowling_squad_elo": t2_bowl_squad,
                })

                # Compute updates
                balls = row["balls_faced"]
                if balls > 0:
                    # Opposing bowling quality is squad bowling Elo for T20, team Elo for ODI.
                    # od_player_elo_revamp_plan.md Phase 1: team Elo is NOT on the player
                    # scale (unlike t20's squad averages, which already are), so it must route
                    # through _team_to_player_scale -- a no-op pre-Phase-1/for hnd.
                    # t20w uses the squad-based proxy too (not just t20): it shares t20's exact
                    # squad-list/batting-order structure and PlayerEloEngine.run() already
                    # computes t1/t2_bowl_squad unconditionally for every format -- this is a
                    # more precise data source, not a calibrated men's-specific constant, so
                    # there's no cross-population borrowing risk in extending it. hndw included
                    # too: it seeds+pools from t20w, so its dynamics should match t20w's rather
                    # than falling to the team-Elo path od/hnd use. odw stays on the team-Elo
                    # path, mirroring od.
                    opp_eff_elo = (t2_bowl_squad if batting_team == t1 else t1_bowl_squad) if self.format_type in ("t20", "t20w", "hndw") else self._team_to_player_scale(opp_elo)

                    exp_runs_per_ball = np.clip(
                        self.avg_runs_per_ball * (10.0 ** ((b_elo - opp_eff_elo) / self.elo_divisor)), 0.1, 3.0
                    )
                    exp_wickets_per_ball = np.clip(
                        self.avg_wickets_per_ball * (10.0 ** ((opp_eff_elo - b_elo) / self.elo_divisor)), 0.001, 0.5
                    )

                    expected_runs = exp_runs_per_ball * balls
                    expected_dismissals = exp_wickets_per_ball * balls

                    delta_runs = row["runs"] - expected_runs
                    delta_wickets = row["dismissals"] - expected_dismissals

                    # Phase 3: experience-graded K -- faster convergence for newcomers,
                    # decaying to today's fixed K as a floor (t20+od active; no-op for hnd).
                    k_mult = self._dynamic_k_multiplier(self.get_match_count(batter, "batter"))
                    # Phase 4: stronger opponents make the same surprise more informative
                    # (no-op for od/hnd).
                    opp_w = self._opponent_quality_weight(opp_eff_elo)
                    new_elo = (
                        b_elo
                        + self.k_runs * k_mult * opp_w * delta_runs
                        - self.k_wickets * k_mult * opp_w * delta_wickets
                    )
                    self.batter_ratings[batter] = new_elo

                    # Track delta for recent form (Proposal 5)
                    self.batter_recent_deltas.setdefault(batter, []).append(new_elo - b_elo)
                    if len(self.batter_recent_deltas[batter]) > self.form_window:
                        self.batter_recent_deltas[batter].pop(0)

                self.batter_match_counts[batter] = self.get_match_count(batter, "batter") + 1
                self.batter_last_dates[batter] = match_date

            # 2. Bowlers
            match_bowlers = bowl_groups.get(match_id, [])
            for row in match_bowlers:
                bowler = row["bowler"]
                # Determine bowling team and opposing batting team
                opp_team = row["batting_team"]
                bowl_team = opp_team_map.get(match_id, {}).get(opp_team, None)

                opp_elo = team_strength.get(opp_team, 1500.0)
                bowl_team_elo = team_strength.get(bowl_team, 1500.0)

                # Get pre-match Elo (already decayed at loop start)
                bowl_elo = self.bowler_ratings.get(bowler, self.default_elo)

                # Get pre-match form (Proposal 5)
                bowl_deltas = self.bowler_recent_deltas.get(bowler, [])
                bowl_form = float(np.mean(bowl_deltas)) if bowl_deltas else 0.0

                # Record pre-match ratings
                player_elo_records.append({
                    "match_id": match_id,
                    "player": bowler,
                    "role": "bowler",
                    "raw_elo": bowl_elo,
                    "effective_elo": match_bowler_eff_elos.get(bowler, self.get_effective_elo(bowler, "bowler", bowl_team_elo)),
                    "recent_form": bowl_form,
                    "team_1_batting_squad_elo": t1_bat_squad,
                    "team_1_bowling_squad_elo": t1_bowl_squad,
                    "team_2_batting_squad_elo": t2_bat_squad,
                    "team_2_bowling_squad_elo": t2_bowl_squad,
                })

                # Compute updates
                balls = row["balls_bowled"]
                if balls > 0:
                    # Opposing batting quality is squad batting Elo for T20, team Elo for ODI.
                    # od_player_elo_revamp_plan.md Phase 1: same team-scale fix as the batter
                    # site above -- route through _team_to_player_scale (no-op pre-Phase-1/hnd).
                    # See the batter-side comment above -- same reasoning, t20w/hndw included.
                    opp_eff_elo = (t2_bat_squad if bowl_team == t1 else t1_bat_squad) if self.format_type in ("t20", "t20w", "hndw") else self._team_to_player_scale(opp_elo)

                    exp_runs_per_ball = np.clip(
                        self.avg_runs_per_ball * (10.0 ** ((opp_eff_elo - bowl_elo) / self.elo_divisor)), 0.1, 3.0
                    )
                    exp_wickets_per_ball = np.clip(
                        self.avg_wickets_per_ball * (10.0 ** ((bowl_elo - opp_eff_elo) / self.elo_divisor)), 0.001, 0.5
                    )

                    expected_runs = exp_runs_per_ball * balls
                    expected_wickets = exp_wickets_per_ball * balls

                    delta_runs = row["runs_conceded"] - expected_runs
                    delta_wickets = row["wickets"] - expected_wickets

                    # Phase 3: experience-graded K -- faster convergence for newcomers,
                    # decaying to today's fixed K as a floor (t20+od active; no-op for hnd).
                    k_mult = self._dynamic_k_multiplier(self.get_match_count(bowler, "bowler"))
                    # Phase 4: stronger opponents make the same surprise more informative
                    # (no-op for od/hnd).
                    opp_w = self._opponent_quality_weight(opp_eff_elo)
                    new_elo = (
                        bowl_elo
                        - self.k_runs * k_mult * opp_w * delta_runs
                        + self.k_wickets * k_mult * opp_w * delta_wickets
                    )
                    self.bowler_ratings[bowler] = new_elo

                    # Track delta for recent form (Proposal 5)
                    self.bowler_recent_deltas.setdefault(bowler, []).append(new_elo - bowl_elo)
                    if len(self.bowler_recent_deltas[bowler]) > self.form_window:
                        self.bowler_recent_deltas[bowler].pop(0)

                self.bowler_match_counts[bowler] = self.get_match_count(bowler, "bowler") + 1
                self.bowler_last_dates[bowler] = match_date

            # Phase 2 (frontline_bowler_eligibility_plan.md): record this match's bowl/no-bowl
            # outcome for EVERY named player, not just those who bowled -- a non-bowling
            # appearance is itself the signal that keeps a specialist batter's rate low despite
            # being rated (and named) every game. Placed after both loops above, so
            # get_recent_bowl_rate() calls made while building a FUTURE match's rank key see only
            # this match's outcome and never leak it into ranking for the CURRENT match (mirrors
            # get_match_count's existing pre-match-only semantics). Not read anywhere yet --
            # Phase 3 wires it into the rank key.
            bowled_this_match = {r["bowler"] for r in match_bowlers}
            for p in t1_players + t2_players:
                self.player_recent_bowled.setdefault(
                    p, deque(maxlen=FRONTLINE_BOWL_WINDOW)
                ).append(1 if p in bowled_this_match else 0)

        df = pd.DataFrame(player_elo_records)
        print(f"Saving player Elo ratings to cache: {self.cache_file}")
        df.to_parquet(self.cache_file, index=False, compression="snappy")
        CacheManager.register_cache(self.cache_file, sources)
        return df

    def clear_cache(self):
        CacheManager.clear_cache(self.cache_file)


def _seed_snapshot_path(target_format: str, source_format: str) -> Path:
    return Path(f"data/processed/_{target_format}_seed_from_{source_format}.parquet")


# Reproduces the pre-existing hnd<-t20 snapshot filename exactly, so that frozen file is
# reused untouched by the generalisation below (no re-freeze, no refresh=False breakage).
HND_SEED_SNAPSHOT_PATH = _seed_snapshot_path("hnd", "t20")


def build_seed_ratings(target_format: str, source_format: str = None, refresh: bool = False):
    """
    Returns the {'batter': {...}, 'bowler': {...}} seed dict `target_format` warm-starts
    from (PlayerEloEngine.run(..., seed_ratings=...)), sourced from a FROZEN snapshot of
    `source_format`'s effective_elo (defaults to SEED_SOURCE[target_format]) -- never the
    live `{source_format}_player_elo_ratings.parquet` directly.

    Why frozen, not live: a source format's own dynamics can be under active, individually
    Brier-gated experimentation (some changes get reverted) -- see t20_player_elo_revamp_plan.md.
    The target must stay frozen throughout that process, but an experiment's rescale isn't
    always affine/invertible, so there's no way to "map back" a reverted or in-progress
    change. Freezing the seed at snapshot time and re-reading that frozen copy is the only
    way to guarantee the target doesn't silently drift while the source experiments.
    refresh=True deliberately rebuilds the snapshot from the source's CURRENT state -- only
    do this once a phase is fully decided (e.g. a plan's ship step), never as a side effect
    of an in-progress experiment.

    Pool/seed within gender only -- see format_spec.py's SEED_SOURCE gender assertion.
    """
    if source_format is None:
        source_format = SEED_SOURCE[target_format]
    snapshot_path = _seed_snapshot_path(target_format, source_format)
    source_player_elo_path = f"data/processed/{source_format}_player_elo_ratings.parquet"

    if refresh or not snapshot_path.exists():
        source_pe = pd.read_parquet(source_player_elo_path)
        latest_source = source_pe.sort_values("match_id").groupby(["player", "role"]).last().reset_index()
        seed_df = latest_source[["player", "role", "effective_elo"]].copy()
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        seed_df.to_parquet(snapshot_path, index=False)
    else:
        seed_df = pd.read_parquet(snapshot_path)

    b_seeds = seed_df[seed_df["role"] == "batter"].set_index("player")["effective_elo"].to_dict()
    w_seeds = seed_df[seed_df["role"] == "bowler"].set_index("player")["effective_elo"].to_dict()
    return {"batter": b_seeds, "bowler": w_seeds}


def build_hnd_seed_ratings(t20_player_elo_path="data/processed/t20_player_elo_ratings.parquet", refresh=False):
    """Back-compat shim -- see build_seed_ratings. Kept for existing callers
    (regression_test.py, hundred_validation.py, regression_snapshot_player_elo.py)."""
    return build_seed_ratings("hnd", "t20", refresh=refresh)


# Phase 3 (effective_game_elo_plan.md): calibrated in Phase 0 against the real team-scale
# deviation distribution ((squad_now - trailing_baseline)/stretch: p5/p95 +/-48, mean|dev|
# 21.8) -- gain=1.0 means a team fielding its typical/usual XI reads effective_elo ~= team_elo
# without any further rescaling.
EFFECTIVE_ELO_GAIN = 1.0


def compute_effective_elo(team_elo, batting_squad_elo, bowling_squad_elo, squad_baseline,
                           format_type, gain=EFFECTIVE_ELO_GAIN):
    """
    Phase 3 (effective_game_elo_plan.md) -- the "effective game Elo" surfaced in the UI:

        effective_elo = team_elo + gain * (squad_now - squad_baseline) / stretch

        squad_now      = mean of (weighted) batting_squad_elo & bowling_squad_elo for THIS XI
        squad_baseline = the TEAM's OWN trailing-N-match mean of that same quantity (its
                          typical recent squad strength) -- deliberately NOT team_elo mapped to
                          player scale via _team_to_player_scale, which would algebraically
                          cancel team_elo out of the formula entirely (squad_now - team_to_
                          player_scale(team_elo) loses the W/L signal, clustering every side at
                          roughly the same team-scale figure). The team-relative trailing
                          baseline is what keeps BOTH signals: start from the team's actual W/L
                          Elo, then nudge by whether today's XI is stronger/weaker than what
                          this team usually fields.
        stretch        = the format's player-scale stretch (PlayerEloEngine.T20_ELO_STRETCH /
                          OD_ELO_STRETCH; 1.0 for hnd), so the (player-scale) squad deviation is
                          expressed in team-scale points before being added to team_elo.

    squad_baseline=None (or NaN -- e.g. a team's first few matches, before enough trailing
    history exists) means "we don't know whether this XI is unusual" -- returns team_elo
    unchanged rather than guessing.

    Design note (flagged in the plan, not resolved by this formula): this is TEAM-RELATIVE, not
    an absolute cross-team strength comparison -- a team whose typical XI is already far
    stronger than another team's typical XI will still show a higher effective_elo even when
    both field entirely ordinary sides. It answers "is this team fielding a stronger/weaker XI
    than usual," not "whose current XI is objectively stronger" (that second question is what
    the Monte Carlo sim's player-level ratings already answer independently).
    """
    if squad_baseline is None or pd.isna(squad_baseline):
        return float(team_elo)
    if format_type == "t20":
        stretch = PlayerEloEngine.T20_ELO_STRETCH
    elif format_type == "od":
        stretch = PlayerEloEngine.OD_ELO_STRETCH
    elif format_type == "hnd":
        stretch = PlayerEloEngine.HND_ELO_STRETCH
    else:
        stretch = 1.0
    squad_now = (float(batting_squad_elo) + float(bowling_squad_elo)) / 2.0
    return float(team_elo) + gain * (squad_now - float(squad_baseline)) / stretch


# --- Frontline-bowler eligibility ------------------------------------------------------
# A player earns a role="bowler" row ONLY for matches in which they actually bowled a ball
# (PlayerEloEngine builds bowler rows by aggregating deliveries), so the share of a player's
# recent appearances carrying a bowler row is a true "does this person bowl when they play"
# signal. The mere EXISTENCE of a bowling Elo is not: it persists forever after a single
# long-ago cameo, and Bayesian shrinkage (get_effective_elo, low reliability) anchors that
# stale rating near the team's own strength -- which outranks genuine frontline bowlers rated
# below 1500. Observed on Surrey vs Kent 2026-08-05: DP Sibley (3 career bowling appearances,
# none since 2018, bowling Elo shrunk to ~1570) ranked as Surrey's TOP bowling option ahead of
# DJ Worrall and JPA Taylor, who each bowled in 10 of their last 10 appearances.
#
# Threshold measured against the od corpus: among players who have ever bowled, career bowling
# share has p10 = 0.228 and median 0.52, so a 0.2 floor sits just under the genuine-bowler
# population while excluding the cameo class (1,110 of 3,135 ever-bowled players have <= 3
# career bowling appearances; 147 of those are established batters with >= 10 innings).
#
# frontline_bowler_eligibility_plan.md Phase 0: a single shared MIN_RATE/WINDOW, not a
# per-format dict, is deliberate -- the threshold sweep (rate in {.1,.15,.2,.25,.3} x window in
# {5,10,15}) found 0.2/window=10 sits in the same flat plateau independently for BOTH t20 and od
# (od 0.15->0.20: 54.42%->54.74% squads changed; t20 0.15->0.20: 52.82%->53.17%; the real cliffs
# are 0.10->0.15 and 0.20->0.25 in both formats), so nothing here is overfit to the format it was
# originally derived from. hnd/t20w/odw/hndw's uniform W_BOWL makes eligibility a rank-order
# no-op regardless of threshold (confirmed both synthetically -- a fully reversed rank key moves
# t20 182.3 / od 182.5 Elo, those four formats exactly 0.000000 -- and on real match-data replay:
# 0.00% of squads changed, 0.000 delta, across all four).
FRONTLINE_BOWL_WINDOW = 10
FRONTLINE_BOWL_MIN_RATE = 0.2


def bowling_appearance_rates(
    p_df: pd.DataFrame,
    window: int = FRONTLINE_BOWL_WINDOW,
    before_date=None,
    players=None,
) -> Dict[str, float]:
    """Share of each player's most recent `window` appearances in which they actually bowled.

    `p_df` must carry a `match_date` column (callers map it on from the match metadata / team
    Elo table). `before_date` restricts to strictly-earlier matches, keeping the rate
    leakage-free when building a point-in-time squad for a historical fixture; `players`
    narrows the computation to the names actually needed.

    Players with no dated appearances at all are simply absent from the result -- callers treat
    a missing entry as rate 0.0 (not eligible), never as "unknown, assume yes".
    """
    if p_df is None or p_df.empty or "match_date" not in p_df.columns:
        return {}
    df = p_df
    if players is not None:
        df = df[df["player"].isin(set(players))]
    if df.empty:
        return {}

    df = df[["player", "match_id", "role", "match_date"]].copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date"])
    if before_date is not None:
        df = df[df["match_date"] < before_date]
    if df.empty:
        return {}

    df["_bowled"] = df["role"].eq("bowler")
    # One row per (player, match): did they bowl in it?
    appearances = df.groupby(["player", "match_id", "match_date"], as_index=False)["_bowled"].max()
    appearances = appearances.sort_values("match_date")
    recent = appearances.groupby("player").tail(window)
    return recent.groupby("player")["_bowled"].mean().to_dict()


_BOWLER_VOLUME_MEMO = {"key": None, "df": None}


def bowler_volume_table(format_type: str, processed_dir="data/processed") -> pd.DataFrame:
    """Per (player, match_id, match_date) count of legal+illegal balls bowled, built from the
    deliveries corpus and cached as a small derived parquet (~1-2 MB vs. the multi-million-row
    deliveries file it is summarised from).

    Volume is what actually distinguishes a frontline bowler from a fifth/part-time option --
    bowling_appearance_rates() only answers "does this person bowl at all", which a genuine
    4-over-per-innings part-timer passes just as cleanly as a 10-over specialist.
    """
    p_dir = Path(processed_dir)
    cache = p_dir / f"_{format_type}_bowler_volume.parquet"
    source = p_dir / f"{format_type}_cricket_data.parquet"
    # In-process memo on top of the on-disk cache: extract_match_squads() is called once per
    # match, so without this a backtest re-reads the same parquet hundreds of times. The key
    # MUST include the source mtime as well as the cache's -- keyed on the cache alone, the memo
    # short-circuits the staleness check below and keeps serving a summary built from a
    # superseded deliveries corpus.
    memo_key = (
        str(cache),
        cache.stat().st_mtime if cache.exists() else None,
        source.stat().st_mtime if source.exists() else None,
    )
    if _BOWLER_VOLUME_MEMO.get("key") == memo_key:
        return _BOWLER_VOLUME_MEMO["df"]
    # Rebuild whenever the deliveries corpus is newer than this summary. NOT registered with
    # CacheManager: that hashes SOURCE CODE, whereas this cache's only real dependency is a
    # multi-million-row data file whose content-hash would cost more to compute than the rebuild
    # it guards. mtime is the cheap correct check here -- and it matters, because
    # scripts/import_and_retrain.py appends new matches on a schedule, which would otherwise
    # leave every player's workload frozen at whatever it was the first time this ran.
    if cache.exists() and (not source.exists() or cache.stat().st_mtime >= source.stat().st_mtime):
        df = pd.read_parquet(cache)
        _BOWLER_VOLUME_MEMO.update(key=memo_key, df=df)
        return df

    deliveries = pd.read_parquet(source, columns=["bowler", "match_id"])
    vol = deliveries.groupby(["bowler", "match_id"]).size().reset_index(name="balls")
    vol = vol.rename(columns={"bowler": "player"})

    elo_path = p_dir / f"{format_type}_elo_ratings.parquet"
    if elo_path.exists():
        edf = pd.read_parquet(elo_path, columns=["match_id", "match_date"])
        date_map = dict(zip(edf["match_id"].astype(str), pd.to_datetime(edf["match_date"], errors="coerce")))
        vol["match_date"] = vol["match_id"].astype(str).map(date_map)
    else:
        vol["match_date"] = pd.NaT

    vol.to_parquet(cache, index=False, compression="snappy")
    # re-key on the file we just wrote, so the memo matches what a later call would load
    _BOWLER_VOLUME_MEMO.update(
        key=(str(cache), cache.stat().st_mtime, source.stat().st_mtime if source.exists() else None),
        df=vol,
    )
    return vol


def bowling_volume_rates(
    p_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    window: int = FRONTLINE_BOWL_WINDOW,
    before_date=None,
    players=None,
) -> Dict[str, float]:
    """Mean balls bowled per recent appearance (NOT per match-in-which-they-bowled) -- a
    non-bowling appearance counts as a genuine zero, so this measures expected workload rather
    than "how many overs when they do come on".

    Same point-in-time contract as bowling_appearance_rates: `before_date` restricts to strictly
    earlier matches, and a player with no dated appearances is simply absent (callers treat that
    as 0.0).
    """
    if p_df is None or p_df.empty or "match_date" not in p_df.columns:
        return {}
    df = p_df
    if players is not None:
        df = df[df["player"].isin(set(players))]
    if df.empty:
        return {}

    df = df[["player", "match_id", "match_date"]].copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date"])
    if before_date is not None:
        df = df[df["match_date"] < before_date]
    if df.empty:
        return {}

    appearances = df.drop_duplicates(["player", "match_id"])
    balls = vol_df[["player", "match_id", "balls"]].copy()
    balls["match_id"] = balls["match_id"].astype(str)
    appearances = appearances.assign(match_id=appearances["match_id"].astype(str))
    merged = appearances.merge(balls, on=["player", "match_id"], how="left")
    merged["balls"] = merged["balls"].fillna(0.0)

    merged = merged.sort_values("match_date")
    recent = merged.groupby("player").tail(window)
    return recent.groupby("player")["balls"].mean().to_dict()


def frontline_bowl_sort_key(name, elo_value, rates, known_bowler=False, volumes=None):
    """Sort key for attack selection: (is_frontline, volume, elo). Descending sort puts every
    established/known bowler above every non-bowler; WITHIN that tier it ranks by expected
    bowling workload, and only uses Elo to break volume ties.

    Why volume outranks Elo here: this key selects WHO BOWLS, not who bowls well. Elo measures
    quality, which is close to irrelevant for the selection question -- a well-rated part-timer
    who sends down 4 overs is not a better pick for a frontline slot than a modestly-rated
    specialist who sends down 10. Measured on od: ranking the eligible tier by volume rather
    than Elo changes top-5 membership in 83.4% of squads that have a real choice, and lifts the
    mean workload of the selected five from 39.7 to 46.2 balls/appearance (e.g. Yuvraj Singh at
    ~8-21 balls displaced by Kumble/Nehra/Agarkar at 48-60).

    `known_bowler=True` marks ground truth that this player bowled in the match being built
    (available for historical fixtures, where it beats any recency prior).

    `volumes=None` (or a player missing from it) degrades to volume 0.0, which reproduces the
    previous pure-Elo ordering within the tier -- so callers without a volume table keep working
    unchanged.

    Returning a tuple rather than filtering keeps the "always produce a full attack" contract:
    if a squad has fewer eligible bowlers than slots, the remaining slots still fill from the
    ineligible pool instead of coming back short.
    """
    eligible = bool(known_bowler) or float(rates.get(name, 0.0)) >= FRONTLINE_BOWL_MIN_RATE
    volume = float((volumes or {}).get(name, 0.0))
    return (1 if eligible else 0, volume, elo_value)


_MATCH_SQUAD_ELO_CACHE: Dict[str, pd.DataFrame] = {}

_SQUAD_ELO_COLS = [
    "team_1_batting_squad_elo", "team_1_bowling_squad_elo",
    "team_2_batting_squad_elo", "team_2_bowling_squad_elo",
]


def get_match_squad_elo(fmt: str, match_id, processed_dir="data/processed"):
    """The (Phase-1/2-weighted) batting/bowling squad Elo the win-prob model actually trained
    on for this match -- straight off {fmt}_player_elo_ratings.parquet, exact/authoritative
    rather than recomputed. Single source of truth for both app.py's Lineups & Elo display and
    PreMatchPredictor's optional squad-Elo-aware prediction (frontline_bowler_eligibility_plan.md
    follow-up: PreMatchPredictor.predict() otherwise falls back to team-level Elo, which is a
    real train/serve skew for any caller who DOES know the match's actual XI).

    Returns (t1_batting, t1_bowling, t2_batting, t2_bowling) or None if the parquet, its
    squad-Elo columns, or this match_id aren't available.
    """
    key = f"{processed_dir}/{fmt}"
    pdf = _MATCH_SQUAD_ELO_CACHE.get(key)
    if pdf is None:
        path = Path(processed_dir) / f"{fmt}_player_elo_ratings.parquet"
        pdf = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        _MATCH_SQUAD_ELO_CACHE[key] = pdf
    if pdf.empty or not all(c in pdf.columns for c in _SQUAD_ELO_COLS):
        return None
    row = pdf[pdf["match_id"].astype(str) == str(match_id)]
    if row.empty:
        return None
    r = row.iloc[0]
    return tuple(float(r[c]) for c in _SQUAD_ELO_COLS)

