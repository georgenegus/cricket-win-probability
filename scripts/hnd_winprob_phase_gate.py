"""
Phase gate for markdowns/win_predictor/hnd_player_elo_revamp_plan.md.

UNPOOLED by design (deviation from the T20/OD plans' methodology, documented in the plan's
Invariants section): this machine has ~7.7GB RAM and repeatedly hit ArrowMemoryError building
T20's full second-innings pool (production's retrain_full('hnd') pools T20 states into hnd's
win-prob training -- see cricsheet_importer.py's Phase 5/6 pooling step). Rebuilding that pool
for every one of Phases 1-3's gate checks isn't viable here, so each PHASE gate compares
UNPOOLED hnd-only Brier against the PREVIOUS phase's unpooled number (apples-to-apples relative
deltas -- exactly what keep-or-revert needs). The real pooled, production-faithful number is
only computed once, at Phase 4's final ship retrain (via retrain_full('hnd')), compared against
the originally-locked pooled baseline (0.139453).

Usage:
    py scripts/hnd_winprob_phase_gate.py                  # evaluate CURRENT code's hnd Brier
    py scripts/hnd_winprob_phase_gate.py --phase0-ref      # force HND_ELO_STRETCH=1.0 (reproduces
                                                            # pre-Phase-1 behavior exactly) to
                                                            # (re-)establish the unpooled Phase 0
                                                            # reference point once
Both modes write ratings to a SCRATCH cache (never the production hnd_player_elo_ratings.parquet)
and fit the classifier to a SCRATCH cache (never production hnd_xgb_models.pkl) -- this script
is read-only w.r.t. production artifacts. Run cricsheet_importer.retrain_full('hnd') separately
once a phase is kept, to actually update production.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.format_spec import spec
from src.data_processing.build_states import StateBuilder
from src.features.elo_engine import EloEngine
from src.features.player_elo_engine import PlayerEloEngine, build_hnd_seed_ratings
from src.features.phase_elo_engine import PhaseEloEngine
from src.features.venue_engine import VenueEngine
from src.models.xgb_win_probability import XGBWinProbabilityModel

FORMAT = "hnd"
SCRATCH_ELO_PATH = PROJECT_ROOT / "data" / "processed" / "_scratch_hnd_player_elo.parquet"
SCRATCH_XGB_PATH = PROJECT_ROOT / "data" / "processed" / "_scratch_hnd_xgb_models.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-ref", action="store_true",
                     help="Monkeypatch HND_ELO_STRETCH=1.0 to reproduce pre-Phase-1 behavior exactly.")
    args = ap.parse_args()

    if args.phase0_ref:
        PlayerEloEngine.HND_ELO_STRETCH = 1.0
        print("[REF MODE] HND_ELO_STRETCH monkeypatched to 1.0 (reproduces Phase 0 exactly).")

    t_start = time.time()
    fmt_spec = spec(FORMAT)

    loader = CricketDataLoader(str(PROJECT_ROOT / "data" / "raw"), format_type=FORMAT)
    deliveries = loader.load_all()

    meta_extractor = MatchMetadataExtractor(str(PROJECT_ROOT / "data" / "raw"), format_type=FORMAT)
    meta_df = meta_extractor.extract_all()
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == fmt_spec["gender"]].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=FORMAT)
    elo_df = elo_engine.run(meta_df)

    print("[HND] Computing player Elo ratings (force=True, SCRATCH cache -- production untouched)...")
    seed_ratings = build_hnd_seed_ratings()
    player_elo_engine = PlayerEloEngine(format_type=FORMAT, form_window=15)
    player_elo_engine.cache_file = SCRATCH_ELO_PATH
    player_elo_df = player_elo_engine.run(deliveries, elo_df, force=True, seed_ratings=seed_ratings)

    phase_elo_engine = PhaseEloEngine(format_type=FORMAT)
    phase_elo_df = phase_elo_engine.run(deliveries, elo_df)

    venue_engine = VenueEngine(format_type=FORMAT)
    venue_df = venue_engine.run(meta_df, deliveries)
    elo_df = elo_df.merge(venue_df, on="match_id", how="left")

    states = StateBuilder(
        deliveries, elo_df, player_elo_df=player_elo_df, phase_elo_df=phase_elo_df,
        total_overs=fmt_spec["total_overs"], format_type=FORMAT,
        balls_per_over=fmt_spec["balls_per_over"],
    ).build()

    second_innings = states[states["innings"] == 2].copy()

    if SCRATCH_XGB_PATH.exists():
        SCRATCH_XGB_PATH.unlink()

    print(f"[HND] Fitting 5-Fold XGBoost Win Probability Classifier (UNPOOLED, scratch cache, n={len(second_innings):,})...")
    xgb_model = XGBWinProbabilityModel(format_type=FORMAT)
    xgb_model.fit(second_innings, cache_path=SCRATCH_XGB_PATH)

    batch_probs = xgb_model.predict_batch(second_innings)
    second_innings["pred_prob"] = batch_probs
    second_innings["error"] = (second_innings["pred_prob"] - second_innings["won_match"]) ** 2

    overall_brier = float(second_innings["error"].mean())
    label = "PHASE-0 REFERENCE (unpooled)" if args.phase0_ref else "CURRENT CODE (unpooled)"
    print(f"\n{'='*60}")
    print(f"HND WIN-PROB BRIER [{label}], full corpus, n={len(second_innings):,}: {overall_brier:.6f}")
    for phase_name, group in second_innings.groupby("phase"):
        print(f"  Phase '{phase_name:<9}': {group['error'].mean():.6f}  (n={len(group):,})")
    print(f"{'='*60}")
    print(f"Elapsed: {time.time() - t_start:.1f}s")

    for p in (SCRATCH_XGB_PATH, SCRATCH_ELO_PATH):
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    main()
