import numpy as np
import pandas as pd

from src.models.delivery_outcome_model import FEATURES, LEGAL_CLASSES

# Runs credited for each legal-outcome class, index-aligned with LEGAL_CLASSES
# ([0, 1, 2, 3, 4, 6, "W"]). The wicket class credits 0 runs off the bat, matching how
# Phase 1's training labels collapse wicket deliveries (see build_training_frame()).
WICKET_CLASS_IDX = LEGAL_CLASSES.index("W")
LEGAL_RUN_VALUES = np.array([0 if c == "W" else c for c in LEGAL_CLASSES], dtype=np.int64)

# Real cricket law: max overs per bowler = total_overs / 5 (4 in a T20, 10 in an ODI).
# A round-robin over exactly 5 bowlers saturates that limit exactly and never repeats a
# bowler on consecutive overs, so it satisfies both bookkeeping rules with no extra logic.
MIN_BOWLERS_REQUIRED = 5


def _phase_encoded_from_over(over_idx: np.ndarray) -> np.ndarray:
    """
    Buckets a 0-indexed over number into powerplay(0)/middle(1)/death(2). Bin edges must
    exactly match build_training_frame()'s pd.cut(bins=[-1, 5, 14, 999]) — any drift here
    silently skews every prediction the simulator makes.
    """
    return np.select([over_idx <= 5, over_idx <= 14], [0, 1], default=2).astype(np.int64)


def _sample_categorical(prob_matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorised multinomial draw: one class index per row via inverse-CDF."""
    cdf = np.cumsum(prob_matrix, axis=1)
    u = rng.random((prob_matrix.shape[0], 1))
    return (u < cdf).argmax(axis=1)


def _lineup_arrays(lineup):
    """lineup: ordered list (batting order) of dicts with 'effective_elo' / 'recent_form'."""
    if len(lineup) < 11:
        raise ValueError(f"lineup must contain at least 11 batters — got {len(lineup)}.")
    elo = np.array([float(p.get("effective_elo", 1500.0)) for p in lineup], dtype=np.float64)
    form = np.array([float(p.get("recent_form", 0.0)) for p in lineup], dtype=np.float64)
    return elo, form


def _attack_arrays(attack):
    """attack: list (bowling rotation order) of dicts with 'effective_elo' / 'recent_form'."""
    if len(attack) < MIN_BOWLERS_REQUIRED:
        raise ValueError(
            f"attack must contain at least {MIN_BOWLERS_REQUIRED} bowlers to legally cover an "
            f"innings under the max-overs-per-bowler law (total_overs/5) — got {len(attack)}."
        )
    elo = np.array([float(p.get("effective_elo", 1500.0)) for p in attack], dtype=np.float64)
    form = np.array([float(p.get("recent_form", 0.0)) for p in attack], dtype=np.float64)
    return elo, form


def _assemble_state_matrix(idx, score, wickets, legal_balls, striker,
                           lineup_elo, lineup_form, attack_elo, attack_form,
                           venue_avg_score, innings, total_balls) -> pd.DataFrame:
    """Builds the FEATURES-column state matrix for the live sims in `idx`, one row each."""
    lb = legal_balls[idx]
    over_idx = lb // 6
    ball_in_over = lb % 6
    striker_idx = striker[idx]
    bowler_idx = over_idx % attack_elo.shape[0]   # round-robin bowling rotation

    state = pd.DataFrame({
        "over_float": over_idx.astype(np.float64) + ball_in_over.astype(np.float64) / 6.0,
        "balls_remaining": (total_balls - lb).astype(np.float64),
        "phase_encoded": _phase_encoded_from_over(over_idx),
        "innings": np.full(idx.size, innings, dtype=np.int64),
        "wickets_lost": wickets[idx].astype(np.float64),
        "batter_elo": lineup_elo[striker_idx],
        "bowler_elo": attack_elo[bowler_idx],
        "matchup_elo_diff": lineup_elo[striker_idx] - attack_elo[bowler_idx],
        "venue_avg_first_innings_score": np.full(idx.size, float(venue_avg_score)),
        "recent_form_batter": lineup_form[striker_idx],
        "recent_form_bowler": attack_form[bowler_idx],
    })
    return state[FEATURES]


def _swap(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> None:
    """In-place swap of a[idx] <-> b[idx]. Fancy indexing already copies, so no aliasing risk."""
    if idx.size:
        a[idx], b[idx] = b[idx], a[idx]


def _apply_legal_outcome(legal_idx, class_idx, score, wickets, striker, non_striker,
                         next_bat, per_batter, legal_balls) -> None:
    """
    Applies sampled legal-ball outcomes in-place to the live simulation-state arrays.

    Order of operations (matches the build plan's bookkeeping table exactly):
      1. Consume the legal ball.
      2. Scored deliveries: add runs, credit the striker, rotate strike on odd runs.
      3. Wicket deliveries: fall a wicket, promote the next batter into the striker's slot
         (non-striker is untouched — non-striker run-outs are out of scope, matching how
         Phase 1's training labels never distinguish them).
      4. End-of-over strike swap, applied uniformly to every sim that just completed an over —
         this composes with (2)'s swap rather than replacing it (see the WARNING in the plan:
         an odd-run shot on the last ball of an over nets to NO swap, since the two cancel).
    """
    legal_balls[legal_idx] += 1

    is_wicket = class_idx == WICKET_CLASS_IDX
    scored_idx = legal_idx[~is_wicket]
    wicket_idx = legal_idx[is_wicket]

    if scored_idx.size:
        runs = LEGAL_RUN_VALUES[class_idx[~is_wicket]]
        score[scored_idx] += runs
        per_batter[scored_idx, striker[scored_idx]] += runs
        odd_idx = scored_idx[(runs % 2) == 1]
        _swap(striker, non_striker, odd_idx)

    if wicket_idx.size:
        wickets[wicket_idx] += 1
        striker[wicket_idx] = next_bat[wicket_idx]
        next_bat[wicket_idx] += 1

    end_of_over_idx = legal_idx[(legal_balls[legal_idx] % 6) == 0]
    _swap(striker, non_striker, end_of_over_idx)


def _update_live(live, wickets, legal_balls, total_balls, score, target, innings) -> np.ndarray:
    """A sim goes dead on all-out, overs completed, or (innings 2) target reached."""
    live = live & (wickets < 10) & (legal_balls < total_balls)
    if innings == 2 and target is not None:
        target_arr = np.broadcast_to(np.asarray(target, dtype=np.float64), score.shape)
        live = live & (score < target_arr)
    return live


def simulate_innings(model, lineup, attack, venue_avg_score, n_sims=10000,
                     total_overs=20.0, innings=1, target=None, rng=None):
    """
    Runs n_sims independent innings in lockstep, one delivery attempt at a time, fully
    vectorised across sims. One batched model call per delivery attempt (not per sim) is
    what keeps 10,000 sims tractable — see the build plan's Phase 3 performance note.

    Args:
        model: a fitted DeliveryOutcomeModel.
        lineup: ordered (batting order) list of >=11 dicts with 'effective_elo'/'recent_form'.
        attack: list of >=5 dicts with 'effective_elo'/'recent_form' (bowling rotation order).
        venue_avg_score: float, this match's venue_avg_first_innings_score.
        target: None (innings 1), or a scalar/length-n_sims array of chase targets (innings 2).
                A sim's chase ends the instant its score reaches its own target.
        rng: an np.random.Generator; pass one seeded upstream (Phase 3) for reproducibility.

    Returns dict of length-n_sims arrays: final_score, wickets, legal_balls, and
    per_batter_runs (n_sims x len(lineup)).
    """
    rng = rng if rng is not None else np.random.default_rng()
    total_balls = int(round(total_overs * 6))
    n_batters = len(lineup)

    lineup_elo, lineup_form = _lineup_arrays(lineup)
    attack_elo, attack_form = _attack_arrays(attack)

    score = np.zeros(n_sims, dtype=np.int32)
    wickets = np.zeros(n_sims, dtype=np.int8)
    striker = np.zeros(n_sims, dtype=np.int64)
    non_striker = np.ones(n_sims, dtype=np.int64)
    next_bat = np.full(n_sims, 2, dtype=np.int64)
    legal_balls = np.zeros(n_sims, dtype=np.int32)
    per_batter = np.zeros((n_sims, n_batters), dtype=np.int32)
    live = np.ones(n_sims, dtype=bool)

    # Safety cap against a pathological/degenerate model (e.g. near-zero P(legal)) looping
    # indefinitely. Extras run ~4% historically, so a real model needs well under 2x total_balls.
    max_deliveries = total_balls * 3
    attempts = 0

    while live.any():
        attempts += 1
        if attempts > max_deliveries:
            raise RuntimeError(
                f"Innings simulation exceeded {max_deliveries} delivery attempts without "
                f"finishing ({int(live.sum())} sims still live) — the outcome model's extras "
                f"probabilities are likely degenerate."
            )

        idx = np.where(live)[0]
        state = _assemble_state_matrix(idx, score, wickets, legal_balls, striker,
                                       lineup_elo, lineup_form, attack_elo, attack_form,
                                       venue_avg_score, innings, total_balls)
        probs = model.predict_proba_batch(state)

        extra_class = _sample_categorical(probs["extras"], rng)   # 0 legal / 1 wide / 2 noball
        is_extra = extra_class > 0
        extra_idx = idx[is_extra]
        if extra_idx.size:
            # Flat +1 run, ball re-bowled: no legal ball consumed, no strike change. This is a
            # deliberate simplification inherited from Phase 1's two-stage label design, which
            # does not model multi-run wides/no-balls or bat-runs-off-a-no-ball.
            score[extra_idx] += 1

        legal_idx = idx[~is_extra]
        if legal_idx.size:
            class_idx = _sample_categorical(probs["legal"][~is_extra], rng)
            _apply_legal_outcome(legal_idx, class_idx, score, wickets, striker,
                                 non_striker, next_bat, per_batter, legal_balls)

        live = _update_live(live, wickets, legal_balls, total_balls, score, target, innings)

    return {
        "final_score": score,
        "wickets": wickets,
        "legal_balls": legal_balls,
        "per_batter_runs": per_batter,
    }
