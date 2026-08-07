"""
Per-player, per-phase (powerplay/middle/death) Elo tracking.

Phase 5 of markdowns/win_predictor/t20_player_elo_revamp_plan.md -- T20 ONLY. od/hnd never
call this; the whole engine and its consumers are t20-gated at the feature-wiring level
(xgb_win_probability.py), not internally, so there is no rescale-inverse-map or hnd-seed-
freeze concern here at all -- od/hnd simply never invoke it.

Design: rather than reimplementing PlayerEloEngine's already-validated update dynamics
(Phase 1's hold-center rescale, Phase 3's dynamic K, Phase 4's opponent-quality weighting)
a second time, this runs THREE independent PlayerEloEngine instances, each on a
phase-filtered slice of deliveries. This gets the full stack of prior work for free, with
zero new update-mechanics code -- and the squad-average opponent-strength computed INSIDE
each phase-filtered instance is automatically phase-specific too (a batter's death-overs
expected-runs baseline is computed against the opponent's death-overs bowling squad
strength, not their overall career strength), an emergent property of filtering the input
rather than something coded explicitly.

Phase boundary: matches PhaseEloEngine's (team-level) own fixed format_spec-derived
boundary, NOT build_states.py's adaptive/rain-shortened-match boundary -- consistent with
the team-level sibling this engine parallels. A small, documented simplification for
rain-affected matches (rare in the corpus), not a correctness bug.
"""

import numpy as np
import pandas as pd

from src.data_processing.format_spec import spec
from src.features.player_elo_engine import PlayerEloEngine

PHASES = ["powerplay", "middle", "death"]


def _tag_phase(deliveries_df: pd.DataFrame, format_type: str) -> pd.DataFrame:
    """Tags each delivery with its phase, using format_spec's canonical ball boundaries --
    identical definition to PhaseEloEngine's (team-level) phase split."""
    fmt_spec = spec(format_type)
    bpo = float(fmt_spec["balls_per_over"])
    pp_end = fmt_spec["pp_end_ball"] / bpo
    death_start = fmt_spec["death_start_ball"] / bpo

    df = deliveries_df.copy()
    over_float = df["over"].astype(float) + (df["ball"].astype(float) - 1.0) / bpo
    df["_phase"] = np.where(
        over_float < pp_end, "powerplay",
        np.where(over_float < death_start, "middle", "death")
    )
    return df


def build_player_phase_elo(
    deliveries_df: pd.DataFrame,
    elo_df: pd.DataFrame,
    format_type: str = "t20",
    form_window: int = 15,
    force: bool = False,
) -> pd.DataFrame:
    """
    Returns one row per (match_id, player, role), with columns
    {phase}_effective_elo for phase in powerplay/middle/death -- the rating this player
    carried into that phase at that match (1500.0-filled for players who never appeared in
    that phase by that match, matched to the same neutral default used everywhere else).
    """
    tagged = _tag_phase(deliveries_df, format_type)

    phase_frames = []
    for phase in PHASES:
        phase_deliveries = tagged[tagged["_phase"] == phase].drop(columns=["_phase"])
        engine = PlayerEloEngine(format_type=format_type, form_window=form_window)
        # Each phase gets its own dedicated cache file -- independent state, independent
        # CacheManager entry, so a phase's cache invalidates/rebuilds on its own.
        engine.cache_file = engine.processed_path / f"{format_type}_player_phase_elo_{phase}.parquet"
        phase_df = engine.run(phase_deliveries, elo_df, force=force)
        phase_df = phase_df[["match_id", "player", "role", "effective_elo"]].rename(
            columns={"effective_elo": f"{phase}_effective_elo"}
        )
        phase_frames.append(phase_df)

    combined = phase_frames[0]
    for pf in phase_frames[1:]:
        combined = combined.merge(pf, on=["match_id", "player", "role"], how="outer")

    for phase in PHASES:
        combined[f"{phase}_effective_elo"] = combined[f"{phase}_effective_elo"].fillna(1500.0)

    return combined
