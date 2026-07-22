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

# Empirical extras distributions for stochastic sampling (avoids ML overfitting on noisy tails)
EMPIRICAL_WIDE_RUNS = np.array([1, 2, 3, 4, 5], dtype=np.int32)
EMPIRICAL_WIDE_PROBS = np.array([0.9062, 0.0500, 0.0119, 0.0013, 0.0306])

EMPIRICAL_NOBALL_RUNS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
EMPIRICAL_NOBALL_PROBS = np.array([0.3776, 0.3256, 0.0970, 0.0097, 0.1197, 0.0181, 0.0523])

EMPIRICAL_BYES_RUNS = np.array([0, 1, 2, 3, 4], dtype=np.int32)
EMPIRICAL_BYES_PROBS = np.array([0.9485, 0.0427, 0.0034, 0.0003, 0.0051])


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


def _assemble_state_matrix(idx, score, wickets, legal_balls, striker, non_striker,
                           lineup_elo, lineup_form, attack_elo, attack_form,
                           venue_avg_score, innings, total_balls, per_batter_balls, target, format_type="t20") -> pd.DataFrame:
    """Builds the FEATURES-column state matrix for the live sims in `idx`, one row each."""
    lb = legal_balls[idx]
    over_idx = lb // 6
    ball_in_over = lb % 6
    striker_idx = striker[idx]
    non_striker_idx = non_striker[idx]
    bowler_idx = over_idx % attack_elo.shape[0]   # round-robin bowling rotation
    
    # -1 to match build_training_frame's convention: balls_remaining is measured AFTER the
    # current delivery is bowled (training ball 1 -> 119, not 120). This also feeds required_run_rate
    # below, keeping the denominator identical to training.
    balls_rem = total_balls - lb - 1

    runs_req = np.zeros(idx.size, dtype=np.float64)
    rrr = np.zeros(idx.size, dtype=np.float64)
    if innings == 2 and target is not None:
        target_arr = np.broadcast_to(np.asarray(target, dtype=np.float64), score.shape)[idx]
        runs_req = target_arr - score[idx]
        rrr_denom = np.where(balls_rem == 0, 1.0, balls_rem) / 6.0
        rrr = np.clip(np.clip(runs_req, 0.0, None) / rrr_denom, 0.0, 36.0)

    state = pd.DataFrame({
        "over_float": over_idx.astype(np.float64) + ball_in_over.astype(np.float64) / 6.0,
        "balls_remaining": balls_rem.astype(np.float64),
        "phase_encoded": _phase_encoded_from_over(over_idx),
        "innings": np.full(idx.size, innings, dtype=np.int64),
        "wickets_lost": wickets[idx].astype(np.float64),
        "batter_elo": lineup_elo[striker_idx],
        "bowler_elo": attack_elo[bowler_idx],
        "matchup_elo_diff": lineup_elo[striker_idx] - attack_elo[bowler_idx],
        "venue_avg_first_innings_score": np.full(idx.size, float(venue_avg_score)),
        "recent_form_batter": lineup_form[striker_idx],
        "recent_form_bowler": attack_form[bowler_idx],
        "current_team_score": score[idx].astype(np.float64),
        # -1: per_batter_balls was already incremented for the current ball in the main loop, but
        # training's cumcount() counts only PRECEDING balls (ball 1 -> 0). The current striker's
        # count is always >= 1 here, so this cannot go negative.
        "striker_balls_faced": (per_batter_balls[idx, striker_idx] - 1).astype(np.float64),
        "required_run_rate": rrr,
        "non_striker_elo": lineup_elo[non_striker_idx],
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
    per_batter_balls = np.zeros((n_sims, n_batters), dtype=np.int32)
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
        per_batter_balls[idx, striker[idx]] += 1
        
        format_type = getattr(model, "format_type", "t20")
        state = _assemble_state_matrix(idx, score, wickets, legal_balls, striker, non_striker,
                                       lineup_elo, lineup_form, attack_elo, attack_form,
                                       venue_avg_score, innings, total_balls, per_batter_balls, target,
                                       format_type=format_type)
        probs = model.predict_proba_batch(state)

        extra_class = _sample_categorical(probs["extras"], rng)   # 0 legal / 1 wide / 2 noball
        is_extra = extra_class > 0
        extra_idx = idx[is_extra]
        if extra_idx.size:
            is_wide = extra_class[is_extra] == 1
            is_noball = extra_class[is_extra] == 2
            
            runs = np.zeros(extra_idx.size, dtype=np.int32)
            batter_runs = np.zeros(extra_idx.size, dtype=np.int32)
            
            if is_wide.any():
                runs[is_wide] = rng.choice(EMPIRICAL_WIDE_RUNS, p=EMPIRICAL_WIDE_PROBS, size=is_wide.sum())
            if is_noball.any():
                nb_runs = rng.choice(EMPIRICAL_NOBALL_RUNS, p=EMPIRICAL_NOBALL_PROBS, size=is_noball.sum())
                runs[is_noball] = nb_runs
                batter_runs[is_noball] = nb_runs - 1  # 1 extra, rest off bat
                
            score[extra_idx] += runs
            
            if is_noball.any():
                nb_idx = extra_idx[is_noball]
                b_runs = batter_runs[is_noball]
                b_mask = b_runs > 0
                if b_mask.any():
                    scored_nb = nb_idx[b_mask]
                    sb_runs = b_runs[b_mask]
                    per_batter[scored_nb, striker[scored_nb]] += sb_runs
                    
            # Rotate strike on odd run-runs for wides (subtracting the 1 penalty run),
            # or odd batted runs for noballs
            odd_swap_idx = np.empty(0, dtype=np.int64)
            if is_wide.any():
                w_runs = runs[is_wide]
                odd_w = extra_idx[is_wide][((w_runs - 1) % 2) == 1]
                odd_swap_idx = np.concatenate([odd_swap_idx, odd_w])
            if is_noball.any():
                b_runs = batter_runs[is_noball]
                odd_nb = extra_idx[is_noball][(b_runs % 2) == 1]
                odd_swap_idx = np.concatenate([odd_swap_idx, odd_nb])
                
            _swap(striker, non_striker, odd_swap_idx)

        legal_idx = idx[~is_extra]
        if legal_idx.size:
            class_idx = _sample_categorical(probs["legal"][~is_extra], rng)
            
            # Sample byes/legbyes for dot balls (class 0)
            dot_mask = class_idx == 0
            if dot_mask.any():
                byes = rng.choice(EMPIRICAL_BYES_RUNS, p=EMPIRICAL_BYES_PROBS, size=dot_mask.sum())
                has_byes = byes > 0
                if has_byes.any():
                    bye_idx = legal_idx[dot_mask][has_byes]
                    bye_runs = byes[has_byes]
                    score[bye_idx] += bye_runs
                    odd_bye = bye_idx[(bye_runs % 2) == 1]
                    _swap(striker, non_striker, odd_bye)

            _apply_legal_outcome(legal_idx, class_idx, score, wickets, striker,
                                 non_striker, next_bat, per_batter, legal_balls)

        live = _update_live(live, wickets, legal_balls, total_balls, score, target, innings)

    return {
        "final_score": score,
        "wickets": wickets,
        "legal_balls": legal_balls,
        "per_batter_runs": per_batter,
    }
