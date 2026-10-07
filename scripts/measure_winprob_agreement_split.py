"""
Diagnoses the MC calibration battery's Test 2 ("win-prob agreement") failure on all three
women's formats (t20w 22.3%, hndw 19.2%, odw 13.8% mean abs diff vs a <=12% gate, while t20
10.6% / od 9.7% / hnd 9.4% all pass).

THE PROBLEM WITH THE GATE ITSELF: Test 2 measures mean |sim_prob - xgb_prob|, a SYMMETRIC
agreement metric. It says the two models disagree; it cannot say which one is wrong, or whether
either is. This script resolves that by scoring BOTH against the one thing that is not an
opinion -- the actual match winner:

  - Brier(sim vs actual)  -- the Monte Carlo simulator's P(first-batting side wins)
  - Brier(xgb vs actual)  -- PreMatchPredictor's P(team_a wins)
  - the distribution of each (mean, sd, min, max), because a large mean |diff| can come from
    one side being CONFIDENT and the other COMPRESSED toward 0.5 without either being
    "wrong" in a Brier sense -- a failure mode the agreement metric cannot distinguish.

Also reports the base rate (how often the first-batting side actually won in the sample), since
a model that just predicts the base rate every time is the natural no-skill reference.

ALREADY RULED OUT before writing this (measured, not assumed): a bat-first conditioning
mismatch. Test 2 simulates with bat_first="a" (so sim_prob is conditioned on team_a batting
first) while PreMatchPredictor is toss-agnostic by construction -- it returns
0.5*P(A chases) + 0.5*(1 - P(B chases)). That IS a real apples-to-oranges element, but the
bat-first advantage measured across the full corpus is only -3.2%..+0.1% per format, and is
SMALLER for t20w (+0.1%) than t20 (-2.5%) -- so it cannot explain a 22% vs 10% split.

Mirrors Test 2's sample selection, squad-Elo passing and simulate_match call exactly, so the
numbers are comparable to the gate's own. Uses fewer sims by default (win probability is a mean
over sims and converges far faster than the tail quantiles Test 1/3 need).

Ships nothing -- pure measurement.

Run:
    py scripts/measure_winprob_agreement_split.py
    py scripts/measure_winprob_agreement_split.py --formats t20 t20w --n-sims 1000
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

from src.data_processing.format_spec import spec, ACTIVE_FORMATS
from src.models.delivery_outcome_model import DeliveryOutcomeModel
from src.simulation.match_simulator import simulate_match, extract_match_squads
from src.betting.pre_match_predictor import PreMatchPredictor
from src.features.player_elo_engine import get_match_squad_elo


def _sample(format_type, n_matches=100, full=False):
    """Byte-identical population + linspace selection to generate_mc_calibration_metrics.py.

    full=True returns the whole 100-match sample -- what the re-specified Test 2 now scores
    (it was merged into Test 3's loop on 2026-08-13). full=False keeps the OLD first-40% slice,
    retained only for reproducing this script's original n=40 pilot numbers.
    """
    fs = spec(format_type)
    total_balls, bpo = fs["total_balls"], fs["balls_per_over"]
    meta = pd.read_parquet(PROJECT_ROOT / f"data/processed/{format_type}_match_metadata.parquet")
    data = pd.read_parquet(PROJECT_ROOT / f"data/processed/{format_type}_cricket_data.parquet",
                           columns=["match_id", "innings", "total_runs", "wicket", "is_wide"])
    gender = fs["gender"]
    if "gender" in meta.columns:
        meta = meta[meta["gender"] == gender]
    if "gender" in data.columns:
        data = data[data["gender"] == gender]

    agg = data[data["innings"] == 1].groupby("match_id").agg(
        actual_total=("total_runs", "sum"),
        legal_balls=("is_wide", lambda x: (x == 0).sum()),
        wkts=("wicket", "sum"))
    full_len = agg[(agg["legal_balls"] >= total_balls - bpo) | (agg["wkts"] == 10)].reset_index()
    merged = meta.merge(full_len, on="match_id").sort_values("match_id")
    idx = np.linspace(0, len(merged) - 1, min(n_matches, len(merged)), dtype=int)
    sample_df = merged.iloc[idx]
    if full:
        return sample_df
    return sample_df.iloc[:max(1, int(round(len(sample_df) * 0.4)))]  # legacy Test 2 slice


def _skill_stats(sim, act):
    """Brier skill vs a base-rate-only forecast, WITH its paired sampling SE.

    skill = mean[(base - y)^2] - mean[(p - y)^2] = mean[d_i], where the per-match contribution
    d_i = (base - y_i)^2 - (p_i - y_i)^2. Because it is a mean of per-match terms, its SE is
    just sd(d_i)/sqrt(n) -- the MATCH is the unit of independence, this repo's standing rule.

    Computing it this way (paired, per-match) rather than differencing two independently
    estimated Brier scores matters: the two Brier scores are computed on the SAME matches and
    are strongly correlated, so treating them as independent would badly overstate the SE.

    Caveat: `base` is estimated from the same sample it scores, so d_i carries a small
    second-order dependence. At n~100 that is negligible next to the sampling noise this is
    measuring.
    """
    base = act.mean()
    d = (base - act) ** 2 - (sim - act) ** 2
    n = len(d)
    skill = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "skill": skill, "skill_se": se, "n": int(n),
        "t": float(skill / se) if se and se > 0 else float("nan"),
        "per_match_sd": float(d.std(ddof=1)) if n > 1 else float("nan"),
        # SE the GATE will see, i.e. rescaled to its own 100-match sample.
        "se_at_n100": float(d.std(ddof=1) / np.sqrt(100)) if n > 1 else float("nan"),
    }


def run_format(fmt, n_sims, full_sample=True, rows_out=None):
    fs = spec(fmt)
    model = DeliveryOutcomeModel(format_type=fmt)
    model.fit(None, cache_path=PROJECT_ROOT / f"data/processed/{fmt}_delivery_outcome_model.pkl")
    predictor = PreMatchPredictor()
    fi_sample = _sample(fmt, full=full_sample)

    rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(fi_sample.iterrows()):
        m_id = row["match_id"]
        try:
            t1, t2, venue_score = extract_match_squads(m_id, format_type=fmt)
            squad_elo = get_match_squad_elo(fmt, m_id)
            squad_kwargs = {}
            if squad_elo is not None:
                a_bat, a_bowl, b_bat, b_bowl = squad_elo
                squad_kwargs = dict(team_a_batting_squad_elo=a_bat, team_a_bowling_squad_elo=a_bowl,
                                    team_b_batting_squad_elo=b_bat, team_b_bowling_squad_elo=b_bowl)
            pm_res = predictor.predict(team_a=t1["name"], team_b=t2["name"], venue=row.get("venue"),
                                       format_type=fmt, **squad_kwargs)
            match_sim = simulate_match(
                model=model, team_a=t1, team_b=t2, venue_avg_score=venue_score,
                n_sims=n_sims, total_overs=float(fs["total_overs"]),
                balls_per_over=float(fs["balls_per_over"]),
                format_type=fmt, bat_first="a", seed=int(m_id) % 100000,
            )
        except Exception as e:
            print(f"    [skip] {m_id}: {e}")
            continue

        winner = row.get("winner")
        if not isinstance(winner, str) or not winner:
            continue  # no-result / tie -- no ground truth to score against
        rows.append({
            "match_id": str(m_id),
            "sim_prob": float(match_sim["first_batting_won"].mean()),
            "xgb_prob": float(pm_res["prob_a"]),
            # team_1 is the innings-1 batting side in every format's corpus (verified 100%),
            # and extract_match_squads returns it as t1/team_a -- so this is P(bat-first wins)
            # on the simulator side and the actual bat-first outcome here.
            "actual": 1.0 if winner == t1["name"] else 0.0,
        })
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(fi_sample)} ({(time.time()-t0)/60:.1f}m)", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        return None
    sim, xgb, act = df["sim_prob"].values, df["xgb_prob"].values, df["actual"].values
    base = act.mean()
    if rows_out is not None:
        df.insert(0, "format", fmt)
        rows_out.append(df)
    return {
        "n": int(len(df)),
        "mean_abs_diff": float(np.abs(sim - xgb).mean()),
        "base_rate": float(base),
        "skill_stats": _skill_stats(sim, act),
        "brier_sim": float(((sim - act) ** 2).mean()),
        "brier_xgb": float(((xgb - act) ** 2).mean()),
        "brier_baserate": float(((base - act) ** 2).mean()),
        "sim_mean": float(sim.mean()), "sim_sd": float(sim.std(ddof=1)),
        "sim_min": float(sim.min()), "sim_max": float(sim.max()),
        "xgb_mean": float(xgb.mean()), "xgb_sd": float(xgb.std(ddof=1)),
        "xgb_min": float(xgb.min()), "xgb_max": float(xgb.max()),
        "corr": float(np.corrcoef(sim, xgb)[0, 1]) if len(df) > 2 else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--formats", nargs="+", default=list(ACTIVE_FORMATS), choices=list(ACTIVE_FORMATS))
    p.add_argument("--n-sims", type=int, default=1000)
    p.add_argument("--legacy-40pct-sample", action="store_true",
                   help="Score the OLD first-40%% slice instead of the full 100-match sample "
                        "the re-specified Test 2 now uses (reproduces this script's n=40 pilot).")
    args = p.parse_args()

    print("=" * 108)
    print(f"WIN-PROB SKILL + NOISE FLOOR  (n_sims={args.n_sims}, "
          f"sample={'legacy 40%' if args.legacy_40pct_sample else 'full 100'})")
    print("=" * 108, flush=True)

    out, all_rows = {}, []
    for fmt in args.formats:
        print(f"\n--- {fmt.upper()} ---", flush=True)
        r = run_format(fmt, args.n_sims, full_sample=not args.legacy_40pct_sample, rows_out=all_rows)
        if r is None:
            print("  (no scorable matches)"); continue
        out[fmt] = r
        ss = r["skill_stats"]
        print(f"  n={r['n']}  mean|sim-xgb|={r['mean_abs_diff']:.1%}  base rate={r['base_rate']:.1%}")
        print(f"  Brier: sim={r['brier_sim']:.4f}  xgb={r['brier_xgb']:.4f}  base-rate-only={r['brier_baserate']:.4f}")
        print(f"  SKILL {ss['skill']:+.4f} +/- {ss['skill_se']:.4f} SE  (t={ss['t']:+.2f}, "
              f"per-match sd={ss['per_match_sd']:.4f}, SE at n=100 {ss['se_at_n100']:.4f})")

    print("\n" + "=" * 108)
    print(f"{'fmt':6s} {'n':>4s} {'skill':>9s} {'SE':>8s} {'t':>6s} {'SE@n100':>8s} "
          f"{'Brier_sim':>10s} {'Brier_xgb':>10s} {'Brier_base':>11s}")
    for fmt, r in out.items():
        ss = r["skill_stats"]
        print(f"{fmt:6s} {r['n']:>4d} {ss['skill']:>+9.4f} {ss['skill_se']:>8.4f} {ss['t']:>+6.2f} "
              f"{ss['se_at_n100']:>8.4f} {r['brier_sim']:>10.4f} {r['brier_xgb']:>10.4f} {r['brier_baserate']:>11.4f}")

    if out:
        worst = min(out.items(), key=lambda kv: kv[1]["skill_stats"]["skill"])
        max_se = max(r["skill_stats"]["se_at_n100"] for r in out.values())
        print(f"\nWeakest skill: {worst[0]} {worst[1]['skill_stats']['skill']:+.4f}")
        print(f"Largest per-format SE at the gate's n=100: {max_se:.4f}")
        print(f"=> a gate floor of -(1 SE) would be about {-max_se:.4f}; -(2 SE) about {-2*max_se:.4f}")
        print("   (a floor below every format's noise band cannot flap; it still catches real breakage,")
        print("    which shows up as a LARGE negative skill, not a marginal one)")

    path = PROJECT_ROOT / "outputs" / "winprob_agreement_split.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {path}")
    if all_rows:
        rows_path = PROJECT_ROOT / "outputs" / "winprob_per_match_rows.csv"
        pd.concat(all_rows, ignore_index=True).to_csv(rows_path, index=False)
        print(f"Wrote {rows_path}  (per-match rows -- lets the SE be recomputed without re-simulating)")


if __name__ == "__main__":
    main()
