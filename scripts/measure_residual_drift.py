"""
Phase 3 of mc_runrate_feedback_plan.md -- re-measures state drift now that Phase 1's damping is
live (od SCORE_FEEDBACK_SHRINK=0.30, t20=1.0), and characterises whatever bias remains.

Ships nothing; this is pure measurement, reusing the same methodology as Phase 0's match-level
paired drift test (scratchpad/state_drift_paired.py, referenced by this plan's Phase 3 spec) --
match as the unit of independence, |t| > 2 as the significance bar at this sample size (see the
NOTE that test's own history earned: pooling deliveries once turned a 1-SE difference into an
apparent 5.8-SE finding).

Three questions:

  1. DID PHASE 1 DO WHAT IT CLAIMS? Re-run the original 6-feature paired drift test, now against
     the LIVE damped simulator. current_team_score is captured from _assemble_state_matrix's
     OUTPUT (the FED/damped value, exactly what the model sees) -- so a collapsed drift here is
     direct evidence the damping mechanism works, not just that the final-score bias improved.

  2. HOW MUCH OF THE REMAINING BIAS IS "LEFT ON THE TABLE" BY CHOOSING shrink=0.30 OVER shrink=0.0
     (full damping), VS GENUINELY UNEXPLAINED? Phase 1 deliberately chose the smallest
     intervention that clears the gate, not the lowest bias -- so some loop-driven bias remains
     by design. Packs THREE shrink candidates (1.0 baseline / production / 0.0 full-damping)
     into one simulate_innings() call per match (same technique as tests/score_feedback_shrink_
     sweep.py) to decompose today's bias into "loop, left un-damped on purpose" vs "residual
     that damping cannot reach at all".

  3. WHERE DOES THE IRREDUCIBLE RESIDUAL LIVE? Phase 0 found no single tracked feature's MARGINAL
     mean was significant beyond current_team_score -- it must live in a joint distribution or in
     sequencing. Compares the JOINT (wickets_lost, over) relationship at several fixed over
     checkpoints (real vs simulated, at the PRODUCTION shrink), which a single innings-long
     marginal mean cannot reveal: a wicket cluster that shifts EARLY vs LATE nets to the same
     average wickets_lost but a different checkpoint-by-checkpoint shape.

Usage:
    py scripts/measure_residual_drift.py --formats od t20
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import json
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

import numpy as np
import pandas as pd

from src.data_processing.format_spec import spec
from src.models.delivery_outcome_model import DeliveryOutcomeModel
from src.simulation import innings_simulator as isim
from src.simulation.match_simulator import extract_match_squads
from tests.sim_baseline_metrics import build_frame

TRACK_FEATURES = ["wickets_lost", "current_team_score", "batter_elo", "bowler_elo",
                  "striker_balls_faced", "matchup_elo_diff"]

# Over checkpoints per format for the joint (wickets_lost, over) comparison -- spread across
# powerplay/middle/death so a shift in WHEN wickets cluster shows up even if the innings-long
# average doesn't move.
OVER_CHECKPOINTS = {
    "od": [10, 20, 30, 40],
    "t20": [5, 10, 15],
}

N_SIMS_PER_BLOCK = 150  # per shrink candidate, per match -- matches Phase 0's original script


def _select_matches(format_type: str, n_matches: int) -> pd.DataFrame:
    fmt_spec = spec(format_type)
    total_balls, bpo = fmt_spec["total_balls"], fmt_spec["balls_per_over"]
    meta = pd.read_parquet(PROJECT_ROOT / f"data/processed/{format_type}_match_metadata.parquet")
    d = pd.read_parquet(PROJECT_ROOT / f"data/processed/{format_type}_cricket_data.parquet",
                        columns=["match_id", "innings", "total_runs", "wicket", "is_wide", "is_noball"])
    d["legal"] = ((d["is_wide"] == 0) & (d["is_noball"] == 0)).astype(int)
    agg = d[d["innings"] == 1].groupby("match_id").agg(
        runs=("total_runs", "sum"), lb=("legal", "sum"), w=("wicket", "sum"))
    full = agg[(agg["lb"] >= total_balls - bpo) | (agg["w"] == 10)].reset_index()
    merged = meta.merge(full, on="match_id").sort_values("match_id")
    idx = np.linspace(0, len(merged) - 1, min(n_matches, len(merged)), dtype=int)
    return merged.iloc[idx].reset_index(drop=True)


def _real_checkpoint_wickets(real_match_rows: pd.DataFrame, checkpoints, bpo) -> dict:
    """real_match_rows: this match's innings-1 legal deliveries (from build_frame), indexed by
    legal_balls_before. Returns {checkpoint_over: wickets_lost_at_that_ball_or_nan}."""
    out = {}
    idx_col = real_match_rows.set_index("legal_balls_before")["wickets_lost"]
    for over in checkpoints:
        ball = int(round(over * bpo))
        if ball in idx_col.index:
            out[over] = float(idx_col.loc[ball] if np.isscalar(idx_col.loc[ball]) else idx_col.loc[ball].iloc[0])
        else:
            # Innings ended before this ball (all out or, rarely, target reached) -- use the
            # last available wickets_lost as the "still batting" censor is not applicable here
            # (real matches don't reach dead balls), so this checkpoint is simply unobserved.
            out[over] = np.nan
    return out


def run_format(fmt: str, n_matches: int, seed: int = 42) -> dict:
    print("=" * 78)
    print(f"RESIDUAL DRIFT RE-MEASUREMENT -- {fmt.upper()}  (post-Phase-1)")
    print("=" * 78, flush=True)

    fmt_spec = spec(fmt)
    bpo = fmt_spec["balls_per_over"]
    total_overs = fmt_spec["total_overs"]
    checkpoints = OVER_CHECKPOINTS[fmt]
    checkpoint_balls = [int(round(o * bpo)) for o in checkpoints]

    prod_shrink = isim.SCORE_FEEDBACK_SHRINK.get(fmt, 1.0)
    # Three candidates packed as contiguous blocks: [no-damping baseline, PRODUCTION, full-damping]
    shrink_candidates = sorted({1.0, float(prod_shrink), 0.0})
    prod_block = shrink_candidates.index(float(prod_shrink))
    print(f"  production shrink = {prod_shrink}  |  candidates packed = {shrink_candidates}  "
          f"(production is block {prod_block})")

    model = DeliveryOutcomeModel(format_type=fmt)
    model.fit(None, cache_path=PROJECT_ROOT / f"data/processed/{fmt}_delivery_outcome_model.pkl")

    sample = _select_matches(fmt, n_matches)
    print(f"  {len(sample)} sample matches selected", flush=True)

    frame = build_frame(fmt)
    real_all = frame[(frame["extra_class"] == "legal") & (frame["innings"] == 1)].copy()
    real_all["match_id"] = real_all["match_id"].astype(str)
    is_legal = pd.Series(1, index=real_all.index)
    real_all["legal_balls_before"] = (
        is_legal.groupby([real_all["match_id"]]).cumsum() - 1
    ).astype(int)
    real_by_match = {mid: g for mid, g in real_all.groupby("match_id")}

    # --- spy setup: capture (a) mean/sum of TRACK_FEATURES per shrink block (feature drift),
    # (b) wickets_lost at each checkpoint ball, per sim, per shrink block (joint distribution) ---
    n_sims_total = len(shrink_candidates) * N_SIMS_PER_BLOCK
    block_of_sim = np.repeat(np.arange(len(shrink_candidates)), N_SIMS_PER_BLOCK)

    cur = {}

    def _reset_capture():
        cur["n_by_block"] = np.zeros(len(shrink_candidates))
        cur["sums_by_block"] = {f: np.zeros(len(shrink_candidates)) for f in TRACK_FEATURES}
        # sim_ckpt[block][checkpoint] -> list of wickets_lost values (one per sim that reached it)
        cur["sim_ckpt"] = {b: {c: [] for c in checkpoint_balls} for b in range(len(shrink_candidates))}
        # A sim can be spied on MULTIPLE times at the same lb (wide/no-ball redos before the
        # legal ball that finally advances it) -- without dedup, wickets_lost at that checkpoint
        # would be recorded once per redo, over-weighting sims with more extras. Tracked per
        # checkpoint only (not per block) since a sim index maps to exactly one block.
        cur["ckpt_captured"] = {c: set() for c in checkpoint_balls}

    _orig = isim._assemble_state_matrix

    def _spy(*a, **k):
        st = _orig(*a, **k)
        idx = np.asarray(a[0])
        blocks = block_of_sim[idx]
        st_vals = {f: st[f].to_numpy() for f in TRACK_FEATURES if f in st.columns}
        for b in range(len(shrink_candidates)):
            m = blocks == b
            if not m.any():
                continue
            cur["n_by_block"][b] += int(m.sum())
            for f, vals in st_vals.items():
                cur["sums_by_block"][f][b] += float(vals[m].sum())

        # Checkpoint capture: lb is this delivery's "before-ball" legal-ball count, computed the
        # same way _assemble_state_matrix does internally (4th positional arg = legal_balls).
        lb = np.asarray(a[3])[idx]
        wk = np.asarray(a[2])[idx]  # wickets BEFORE this delivery -- matches wickets_lost's own convention
        for ckpt_ball in checkpoint_balls:
            hit = lb == ckpt_ball
            if not hit.any():
                continue
            hit_sims = idx[hit]
            already = cur["ckpt_captured"][ckpt_ball]
            unseen = np.array([s not in already for s in hit_sims])
            if not unseen.any():
                continue
            new_sims = hit_sims[unseen]
            new_wk = wk[hit][unseen]
            already.update(int(s) for s in new_sims)
            new_blocks = block_of_sim[new_sims]
            for b in range(len(shrink_candidates)):
                bm = new_blocks == b
                if bm.any():
                    cur["sim_ckpt"][b][ckpt_ball].extend(new_wk[bm].tolist())
        return st

    isim._assemble_state_matrix = _spy

    recs = []
    ckpt_records = {b: {c: [] for c in checkpoint_balls} for b in range(len(shrink_candidates))}
    t0 = time.time()
    n_ok = 0
    for i, row in sample.iterrows():
        mid = str(row["match_id"])
        rg = real_by_match.get(mid)
        if rg is None or len(rg) == 0:
            continue
        try:
            t1, t2, venue = extract_match_squads(mid, fmt)
        except Exception:
            continue

        _reset_capture()
        shrink_arr = np.repeat(np.array(shrink_candidates, dtype=np.float64), N_SIMS_PER_BLOCK)
        res = isim.simulate_innings(
            model=model, lineup=t1["lineup"], attack=t2["attack"], venue_avg_score=venue,
            n_sims=n_sims_total, total_overs=total_overs, balls_per_over=bpo,
            format_type=fmt, innings=1, target=None,
            rng=np.random.default_rng(int(mid) % 100000 if mid.isdigit() else seed + i),
            score_feedback_shrink=shrink_arr,
        )
        if cur["n_by_block"].sum() == 0:
            continue

        rec = {"match_id": mid, "a_runs": float(row["runs"])}
        scores = res["final_score"]
        for b, s in enumerate(shrink_candidates):
            bm = block_of_sim == b
            rec[f"s_runs_shrink{s}"] = float(scores[bm].mean())
            n_b = cur["n_by_block"][b]
            if n_b > 0:
                for f in TRACK_FEATURES:
                    rec[f"sim_{f}_shrink{s}"] = cur["sums_by_block"][f][b] / n_b
        for f in TRACK_FEATURES:
            if f in rg.columns:
                rec[f"real_{f}"] = float(rg[f].mean())
        recs.append(rec)

        real_ckpt = _real_checkpoint_wickets(rg, checkpoints, bpo)
        rec["_real_ckpt"] = real_ckpt
        for b in range(len(shrink_candidates)):
            for c_over, c_ball in zip(checkpoints, checkpoint_balls):
                vals = cur["sim_ckpt"][b][c_ball]
                if vals:
                    ckpt_records[b][c_ball].append({"match_id": mid, "real": real_ckpt.get(c_over, np.nan),
                                                    "sim_mean": float(np.mean(vals)), "sim_vals": vals})

        n_ok += 1
        if n_ok % 25 == 0:
            elapsed = time.time() - t0
            print(f"  {n_ok}/{len(sample)} matches ({elapsed/60:.1f}m elapsed, "
                  f"~{elapsed/n_ok*(len(sample)-n_ok)/60:.1f}m remaining)", flush=True)

    isim._assemble_state_matrix = _orig

    df = pd.DataFrame(recs)
    n = len(df)
    print(f"\n=== {fmt}: match-level paired drift, n={n} matches ===\n")

    # --- Question 1: feature drift at PRODUCTION shrink ---
    prod_s = shrink_candidates[prod_block]
    print(f"[Q1] Feature drift at PRODUCTION shrink={prod_s} (current_team_score is the FED/damped value):")
    print(f"{'feature':24s} {'real':>10s} {'sim':>10s} {'diff':>10s} {'SE':>8s} {'t':>7s}")
    feature_drift = {}
    for f in TRACK_FEATURES:
        rc, sc = f"real_{f}", f"sim_{f}_shrink{prod_s}"
        if rc not in df.columns or sc not in df.columns:
            continue
        diff = df[sc] - df[rc]
        se = diff.std(ddof=1) / np.sqrt(n)
        t = diff.mean() / se if se > 0 else float("nan")
        feature_drift[f] = {"real": float(df[rc].mean()), "sim": float(df[sc].mean()),
                            "diff": float(diff.mean()), "se": float(se), "t": float(t)}
        print(f"{f:24s} {df[rc].mean():10.3f} {df[sc].mean():10.3f} {diff.mean():+10.3f} "
              f"{se:8.3f} {t:+7.2f}")

    # --- Question 2: bias decomposition across the 3 shrink candidates ---
    print(f"\n[Q2] Innings-total bias decomposition (real mean = {df['a_runs'].mean():.1f}):")
    print(f"{'shrink':>8s} {'sim mean':>10s} {'bias':>8s} {'SE':>7s} {'t':>7s}  note")
    bias_decomp = {}
    for s in shrink_candidates:
        col = f"s_runs_shrink{s}"
        d_runs = df[col] - df["a_runs"]
        se = d_runs.std(ddof=1) / np.sqrt(n)
        t = d_runs.mean() / se if se > 0 else float("nan")
        note = ("no-damping baseline" if s == 1.0 else
                "PRODUCTION" if s == prod_s else
                "full damping" if s == 0.0 else "")
        bias_decomp[s] = {"sim_mean": float(df[col].mean()), "bias": float(d_runs.mean()),
                          "se": float(se), "t": float(t)}
        print(f"{s:>8.2f} {df[col].mean():>10.1f} {d_runs.mean():>+8.2f} {se:>7.2f} {t:>+7.2f}  {note}")

    decomposition = {}
    if 1.0 in bias_decomp and 0.0 in bias_decomp:
        loop_explained = bias_decomp[1.0]["bias"] - bias_decomp[0.0]["bias"]
        left_on_table = bias_decomp[1.0]["bias"] - bias_decomp.get(prod_s, bias_decomp[0.0])["bias"]
        residual_at_full_damping = bias_decomp[0.0]["bias"]
        pct_removed = f"{100*left_on_table/loop_explained:.0f}%" if loop_explained else "n/a"
        pct_residual = (f"{100*abs(residual_at_full_damping)/abs(bias_decomp[1.0]['bias']):.0f}%"
                        if bias_decomp[1.0]["bias"] else "n/a")
        print(f"\n  loop's full explainable bias (shrink 1.0 -> 0.0): {loop_explained:+.2f} runs")
        print(f"  removed by PRODUCTION shrink={prod_s}:              {left_on_table:+.2f} runs "
              f"({pct_removed} of the loop's explainable share)")
        print(f"  residual bias remaining even at FULL damping:      {residual_at_full_damping:+.2f} runs "
              f"({pct_residual} of the original bias -- genuinely unexplained by this loop)")
        decomposition = {
            "loop_explainable_bias": float(loop_explained),
            "removed_by_production": float(left_on_table),
            "residual_at_full_damping": float(residual_at_full_damping),
        }

    # --- Question 3: joint (wickets_lost, over) comparison at PRODUCTION shrink ---
    print(f"\n[Q3] Joint (wickets_lost, over) comparison at PRODUCTION shrink={prod_s}:")
    print(f"{'over':>6s} {'n':>5s} {'real mean':>10s} {'sim mean':>9s} {'diff':>8s} {'SE':>7s} {'t':>7s}  "
          f"real_p(>=3)  sim_p(>=3)")
    ckpt_results = {}
    for c_over, c_ball in zip(checkpoints, checkpoint_balls):
        recs_c = ckpt_records[prod_block][c_ball]
        recs_c = [r for r in recs_c if not np.isnan(r["real"])]
        if len(recs_c) < 20:
            print(f"{c_over:>6d}  -- too few matches with data at this checkpoint ({len(recs_c)}), skipped")
            continue
        real_vals = np.array([r["real"] for r in recs_c])
        sim_means = np.array([r["sim_mean"] for r in recs_c])
        all_sim_vals = np.concatenate([r["sim_vals"] for r in recs_c])
        diff = sim_means - real_vals
        se = diff.std(ddof=1) / np.sqrt(len(recs_c))
        t = diff.mean() / se if se > 0 else float("nan")
        real_p3 = float((real_vals >= 3).mean())
        sim_p3 = float((all_sim_vals >= 3).mean())
        ckpt_results[c_over] = {
            "n_matches": len(recs_c), "real_mean": float(real_vals.mean()),
            "sim_mean": float(sim_means.mean()), "diff": float(diff.mean()),
            "se": float(se), "t": float(t), "real_p_ge3": real_p3, "sim_p_ge3": sim_p3,
        }
        print(f"{c_over:>6d} {len(recs_c):>5d} {real_vals.mean():>10.2f} {sim_means.mean():>9.2f} "
              f"{diff.mean():>+8.2f} {se:>7.2f} {t:>+7.2f}  {real_p3:>10.1%} {sim_p3:>10.1%}")

    out = {
        "format": fmt, "n_matches": n, "production_shrink": prod_s,
        "shrink_candidates": shrink_candidates,
        "feature_drift_at_production": feature_drift,
        "bias_decomposition": bias_decomp,
        "decomposition_summary": decomposition,
        "checkpoint_wickets_at_production": ckpt_results,
    }
    out_path = PROJECT_ROOT / "outputs" / f"residual_drift_{fmt}.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {out_path}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--formats", nargs="+", default=["od", "t20"], choices=["t20", "od"])
    p.add_argument("--matches", type=int, default=200,
                   help="Match sample size -- the plan's own bar is n>=200 for |t|>2 to mean anything.")
    args = p.parse_args()

    for fmt in args.formats:
        run_format(fmt, args.matches)


if __name__ == "__main__":
    main()
