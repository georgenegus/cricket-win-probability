"""
Phase 5 of markdowns/win_predictor/rating_source_toggle_plan.md: trains a SEPARATE model set
per qualifying format, using the ICC-style combined ratings (Phase 2's
scripts/build_combined_ratings.py output) in place of Elo features, then locks their own
baselines.

Why separate models are required rather than a feature swap at serving time: the deployed
{fmt}_xgb_models.pkl etc. were trained on Elo-scaled features. Feeding them ICC-scale values is
out-of-distribution and silently degrades output (constraint 2 in the plan). These artifacts
are `_icc`-suffixed and never read by the deployed Elo path.

Pipeline mirrors src/data_processing/cricsheet_importer.py::retrain_full() almost exactly, with
one substitution: team/player ratings come from src/features/rating_source.py's
load_team_ratings/load_player_ratings(fmt, "icc") -- the Phase 2 combined artifacts -- instead
of EloEngine/PlayerEloEngine. PhaseEloEngine and VenueEngine stay IDENTICAL to the Elo pipeline
(format-structural only; PhaseEloEngine.run only ever reads match_id/match_date/team_1/team_2
off its elo_df argument, never a rating VALUE), the same "shared, identical in both arms" design
scripts/measure_rating_swap_brier.py already validated the swap on.

No POOL_SOURCE pooling (hnd<-t20, hndw<-t20w): neither hnd nor hndw is a Phase-1-qualifying icc
format, so retrain_full's pooling branch has nothing to trigger for any format this script runs.

After training, evaluates each format via tests/regression_test.py::evaluate_format_models and
tests/sim_baseline_metrics.py::evaluate_sim_model -- both made source-aware for this phase,
default-"elo" behaviour on every existing call site untouched -- and locks the results to
data/processed/baseline_metrics_icc.json and data/processed/sim_baseline_metrics_icc.json
(same schema as their Elo counterparts, so a future gate script can reuse
regression_test.py's comparison logic against them unchanged).

Usage:
    py scripts/train_rating_source_models.py --format t20
    py scripts/train_rating_source_models.py --formats t20 t20w od odw
    py scripts/train_rating_source_models.py --formats t20 --force   # ignore existing _icc caches
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import gc
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.build_states import StateBuilder
from src.data_processing.format_spec import spec, RATING_SOURCES
from src.features.phase_elo_engine import PhaseEloEngine
from src.features.venue_engine import VenueEngine
from src.features import rating_source
from src.models.first_innings_predictor import FirstInningsPredictor
from src.models.xgb_win_probability import XGBWinProbabilityModel
from src.models.delivery_outcome_model import DeliveryOutcomeModel, build_training_frame

SOURCE = "icc"
PROC = PROJECT_ROOT / "data" / "processed"


def train_for_format(fmt: str, force: bool = False):
    if not rating_source.is_supported(fmt, SOURCE):
        raise ValueError(
            f"'{fmt}' has no '{SOURCE}' rating coverage (RATING_SOURCES['{SOURCE}']['formats'] "
            f"= {RATING_SOURCES[SOURCE]['formats']}). Run scripts/build_combined_ratings.py "
            f"--format {fmt} first if this format was recently gated in (Phase 1)."
        )

    fmt_spec = spec(fmt)
    reg_path = PROC / f"{fmt}_{SOURCE}_first_innings_regressor.pkl"
    xgb_path = PROC / f"{fmt}_{SOURCE}_xgb_models.pkl"
    dom_path = PROC / f"{fmt}_{SOURCE}_delivery_outcome_model.pkl"
    if force:
        for p in (reg_path, xgb_path, dom_path):
            p.unlink(missing_ok=True)

    print(f"\n{'='*70}")
    print(f"  RATING-SOURCE MODEL TRAIN: {fmt.upper()} (source={SOURCE})")
    print(f"{'='*70}\n")

    raw = str(PROJECT_ROOT / "data" / "raw")
    deliveries = CricketDataLoader(raw, format_type=fmt).load_all()
    meta_df = MatchMetadataExtractor(raw, format_type=fmt).extract_all()
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == fmt_spec["gender"]].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    print(f"[{fmt.upper()}] Loading '{SOURCE}' combined ratings (Phase 2 artifact)...")
    team_df = rating_source.load_team_ratings(fmt, SOURCE)
    player_df = rating_source.load_player_ratings(fmt, SOURCE)

    print(f"[{fmt.upper()}] Computing phase Elo ratings (shared, format-structural only)...")
    phase_elo_df = PhaseEloEngine(format_type=fmt).run(deliveries, team_df)

    print(f"[{fmt.upper()}] Computing venue features (shared)...")
    venue_df = VenueEngine(format_type=fmt).run(meta_df, deliveries)

    team_df = team_df.merge(venue_df, on="match_id", how="left")

    print(f"[{fmt.upper()}] Building game states...")
    states = StateBuilder(
        deliveries, team_df, player_elo_df=player_df, phase_elo_df=phase_elo_df,
        total_overs=fmt_spec["total_overs"], format_type=fmt,
        balls_per_over=fmt_spec["balls_per_over"],
    ).build()

    print(f"[{fmt.upper()}] Fitting First Innings Regressor -> {reg_path.name}")
    reg_model = FirstInningsPredictor(format_type=fmt)
    reg_model.fit(states, cache_path=reg_path)

    print(f"[{fmt.upper()}] Fitting 5-Fold XGBoost Win Probability Classifier -> {xgb_path.name}")
    xgb_model = XGBWinProbabilityModel(format_type=fmt)
    xgb_model.fit(states[states["innings"] == 2].copy(), cache_path=xgb_path)

    print(f"[{fmt.upper()}] Building delivery-outcome training frame...")
    training_frame = build_training_frame(deliveries, player_df, venue_df, format_type=fmt)
    print(f"[{fmt.upper()}] Fitting Delivery Outcome Model -> {dom_path.name}")
    dom_model = DeliveryOutcomeModel(format_type=fmt)
    dom_model.fit(training_frame, cache_path=dom_path)

    del deliveries, meta_df, team_df, player_df, phase_elo_df, venue_df, states, training_frame
    del reg_model, xgb_model, dom_model
    gc.collect()

    print(f"\n[OK] {fmt.upper()} training complete -> {reg_path.name}, {xgb_path.name}, {dom_path.name}")


def lock_baselines(fmt: str):
    """Evaluates the just-trained models and (re)locks this format's entry in
    baseline_metrics_icc.json / sim_baseline_metrics_icc.json -- preserving every other
    format's existing entry, matching regression_test.py --generate-format's convention."""
    from tests.regression_test import evaluate_format_models, DEFAULT_TOLERANCE
    from tests.sim_baseline_metrics import evaluate_sim_model, DEFAULT_TOLERANCE as SIM_TOLERANCE

    print(f"\n[{fmt.upper()}] Locking win-probability / first-innings baseline (source={SOURCE})...")
    t0 = time.time()
    metrics = evaluate_format_models(fmt, evaluate_full=True, source=SOURCE)
    print(f"  Win Probability Brier : {metrics['win_probability']['overall_brier']:.6f} "
          f"(n={metrics['win_probability']['sample_size']:,})")
    print(f"  First Innings MAE     : {metrics['first_innings']['overall_mae']:.2f} runs "
          f"(n={metrics['first_innings']['sample_size']:,})")
    print(f"  Completed in {time.time() - t0:.1f}s")

    wp_path = PROC / f"baseline_metrics_{SOURCE}.json"
    baseline = json.loads(wp_path.read_text()) if wp_path.exists() else {"tolerance": DEFAULT_TOLERANCE}
    baseline[fmt] = metrics
    baseline["tolerance"] = baseline.get("tolerance", DEFAULT_TOLERANCE)
    wp_path.write_text(json.dumps(baseline, indent=2))
    print(f"  Wrote {wp_path}")

    print(f"\n[{fmt.upper()}] Locking delivery-outcome (simulator) baseline (source={SOURCE})...")
    t0 = time.time()
    sim_metrics = evaluate_sim_model(fmt, source=SOURCE)
    print(f"  extras_log_loss={sim_metrics['extras_log_loss']:.6f}  "
          f"legal_log_loss={sim_metrics['legal_log_loss']:.6f}  "
          f"n_deliveries={sim_metrics['n_deliveries']:,}")
    print(f"  Completed in {time.time() - t0:.1f}s")

    sim_path = PROC / f"sim_baseline_metrics_{SOURCE}.json"
    sim_baseline = json.loads(sim_path.read_text()) if sim_path.exists() else {}
    sim_baseline[fmt] = sim_metrics
    sim_baseline.setdefault("tolerance", SIM_TOLERANCE)
    sim_path.write_text(json.dumps(sim_baseline, indent=2))
    print(f"  Wrote {sim_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", default=None)
    p.add_argument("--formats", nargs="+", default=None)
    p.add_argument("--force", action="store_true", help="Ignore existing _icc model caches and retrain.")
    p.add_argument("--skip-lock", action="store_true", help="Train only; skip baseline evaluation/locking.")
    args = p.parse_args()
    formats = args.formats or ([args.format] if args.format else list(RATING_SOURCES[SOURCE]["formats"]))

    for fmt in formats:
        train_for_format(fmt, force=args.force)
        if not args.skip_lock:
            lock_baselines(fmt)


if __name__ == "__main__":
    main()
