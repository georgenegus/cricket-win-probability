# Competition-Tier Calibration — Why Associate Players Top the Elo Rankings

> **Owner:** George · **Created:** 2026-09-01 · **Status:** 🔬 Analysis complete — no code changed yet
> **Purpose / Strategy:** Diagnose why the T20 player-Elo leaderboard is topped by associate-nation internationals while Kohli, Rashid Khan and de Villiers sit mid-table, and scope the fix.
> **Trigger:** User observation — _"I haven't heard of any of the players in the top 8 with less than 100 games. These top spots should have the likes of Chris Gayle, AB de Villiers, Jos Buttler, Virat Kohli."_ Hypothesis offered: _no distinction between elite franchise/international cricket and second-rate franchise/associate cricket._
> **Verdict:** **Hypothesis confirmed, and it is worse than stated.** This is a model-structure defect, not a scale/calibration one — no rescale can fix an ordering problem.
> **Scope:** Diagnosis only. No engine change is proposed for immediate implementation; remedies are scoped and ranked at the end for a separate, gated plan.

---

## Executive Summary

The player-Elo leaderboard is inverted. The top of both the batting and bowling tables is occupied almost exclusively by associate-nation internationals, while the genuinely elite sit mid-table.

| Batters (≥50 matches, n=786) | Peak | Competition mix |
|---|---:|---|
| BJ Bennett | 2090.7 | **t20s: 55 (100%)** |
| Karanbir Singh | 2055.0 | **t20s: 51 (100%)** |
| FH Allen | 2028.3 | t20s 76, mlc 33, ssm 28 |
| Abhishek Sharma | 2024.8 | ipl 89, t20s 86 |
| CO Obuya | 2004.5 | **t20s: 69 (100%)** |
| K Bhurtel | 2004.0 | t20s 77, npl 13 |
| … | | |
| **CH Gayle** | 1932.6 | rank 33 |
| **JC Buttler** | 1917.6 | **rank 50** |
| **AB de Villiers** | 1864.9 | rank 121 |
| **V Kohli** | 1858.9 | **rank 126** |

**The bowling side is worse.** Of 648 bowlers with ≥50 matches, the top 12 peaks are almost entirely Rwanda/Nepal/Kenya internationals (AR Ramjani 2105.4, Vraj Patel 2051.0, MG Erasmus 2028.0, C Rububagumya 1985.6, K Irakoze 1970.3, LN Oluoch 1968.6). For comparison:

| Bowler | Peak | Rank of 648 |
|---|---:|---:|
| JJ Bumrah | 1896.8 | 37 |
| SP Narine | 1859.5 | 71 |
| TA Boult | 1821.2 | 126 |
| MA Starc | 1816.8 | 136 |
| **Rashid Khan** | **1780.8** | **216** |

Among the elite, rank degrades monotonically with **IPL share of career**: Gayle (t20s-heavier) 33rd → Buttler 50th → de Villiers 121st → Kohli (275 of 387 matches in IPL) 126th. Playing in the strongest league in the world actively suppresses your rating.

> [!IMPORTANT]
> This is **not** fixable by changing `T20_ELO_STRETCH`. An affine rescale is monotonic — it cannot reorder anyone. The stretch recalibration (`player_elo_stretch_recalibration_plan.md`) is correct and orthogonal to this. Confirmed empirically: the batter-peak p99 is stable at ~1998–2001 across every `--min-matches` cut from 50 to 250, and the median match count of the >2000 cohort (92) is **identical** to the population median (92) — so this is not a small-sample artefact either.

---

## Root Cause: three compounding mechanisms

### 1. Team Elo is pooled **per competition**, each anchored independently at 1500

`EloEngine` keys every rating pool by competition ([elo_engine.py:197-206](../../src/features/elo_engine.py#L197)):

```python
def get_elo(self, competition: str, team: str) -> float:
    comp_pool = self.pools.setdefault(competition, {})
```

So the IPL, the Blast, the BBL and international cricket are each their own closed rating universe, every one of them centred on 1500. Measured:

| Pool | matches | mean | p95 | max | min |
|---|---:|---:|---:|---:|---:|
| t20s | 9,310 | 1538 | 1731 | **1981** | 1108 |
| ntb (Blast) | 2,978 | 1505 | 1639 | 1731 | 1254 |
| **ipl** | 2,486 | 1516 | 1625 | **1697** | 1260 |
| bbl | 1,324 | 1506 | 1637 | 1718 | 1234 |
| psl | 714 | 1507 | 1622 | 1696 | 1286 |

The best IPL side ever tops out at **1697**; an international side reaches **1981**. Not because the IPL is weaker, but because each pool is normalised to its own membership. **1500 means "average for this competition", never "average in world cricket".**

### 2. `PlayerEloEngine` has **zero** competition awareness

There is no reference to `competition` or `team_type` anywhere in `player_elo_engine.py`. It reads `team_strength.get(opp_team)` and treats every pool-local number as one global scale. The batter update anchors on the opposing **bowling squad Elo** ([player_elo_engine.py:796](../../src/features/player_elo_engine.py#L796)):

```python
exp_runs_per_ball = clip(avg_runs_per_ball * 10**((b_elo - opp_eff_elo) / elo_divisor), 0.1, 3.0)
delta_runs = row["runs"] - exp_runs_per_ball * balls
new_elo = b_elo + k_runs * k_mult * opp_w * delta_runs
```

Every term is competition-blind. The net effect, measured — mean opposing bowling-squad Elo actually faced, by competition:

| Competition | mean attack faced |
|---|---:|
| ipl | **1594** |
| t20s | **1589** |
| bpl / psl | 1584 |
| cpl | 1580 |
| bbl | 1557 |
| ntb | 1538 |

**The model believes an IPL attack and an international attack are equivalent to within 5 Elo points**, and that the entire quality span from the IPL to the county Blast is 56 points — on a scale where ~500 points separates average from world-class.

### 3. Shrinkage amplification inflates thin-sample opposition

`get_effective_elo` blends a player's raw rating toward their team's Elo by `reliability = n/(n+10)`, and the team anchor is passed through `_team_to_player_scale`, which **multiplies the team-Elo deviation by `T20_ELO_STRETCH` (3.29)**. Associate bowlers are far thinner-sampled than league bowlers:

| Competition | median prior matches | mean reliability | **% of effective_elo inherited from team Elo** |
|---|---:|---:|---:|
| t20s | 14 | 0.5 | **46.1%** |
| ctc | 16 | 0.6 | 43.8% |
| ntb | 35 | 0.7 | 30.7% |
| bbl | 40 | 0.7 | 29.8% |
| **ipl** | **48** | **0.7** | **25.8%** |
| psl | 55 | 0.7 | 27.9% |

Nearly half of an associate bowler's apparent quality is not their own record at all — it is their team's pool-local Elo, amplified 3.29×. The consequence is the single most damning number in this analysis:

> **BJ Bennett (55 caps, associate international) is credited with facing a mean bowling attack of 1793.
> Virat Kohli, across 275 IPL matches, is credited with 1611.**

The model believes Bennett faced *systematically stronger* bowling than Kohli. Kohli, Buttler, de Villiers and Gayle all sit in the 1584–1630 band; every associate in the top 5 sits at 1620–1793.

---

## Why this produces the observed inversion

```
Per-competition pools, each centred 1500
        │
        ▼
"Average IPL attack" == "average associate attack"  (1594 vs 1589)
        │
        ├──► Dominating a weak pool earns the same credit as dominating the IPL
        │
        ▼
Thin-sample associate opponents shrink toward team Elo (46%),
amplified x3.29 onto the player scale
        │
        ▼
Associate opposition scores HIGHER than IPL opposition (1793 vs 1611)
        │
        ▼
Elite league players are penalised; associate specialists top the table
```

The two roles fail identically because the update is symmetric — batters anchor on the opposing bowling squad, bowlers on the opposing batting squad. Both anchors are pool-local, so **any fix must be applied to both roles**, not just batting.

---

## Remedies (ranked, not yet implemented)

| # | Approach | Effort | Assessment |
|---|---|:---:|---|
| 1 | **Cross-pool linking via bridge players** — estimate a per-competition offset from players who appear in multiple competitions, then express all team Elo on one global scale before it reaches `PlayerEloEngine` | 🔴 High | **Principled fix.** 306 batters and 269 bowlers with ≥50 matches appear in both IPL and t20s, so the bridge sample is ample. Addresses the root cause rather than symptoms |
| 2 | **Explicit competition-strength prior** — a fitted per-competition constant added to team Elo on the player-scale mapping | 🟡 Medium | Cheaper and interpretable; risks becoming a hand-tuned fudge factor unless fitted from data, and needs refitting as leagues change |
| 3 | **Split `t20s` into tiers** — separate full-member from associate internationals using `team_type` / ICC status | 🟢 Low | `t20s` is 4,655 matches lumping India–Australia with Rwanda–Nigeria. Cheapest partial win, but leaves franchise-vs-international uncalibrated |
| 4 | **Per-competition base rates** — `avg_runs_per_ball` is a single global constant (1.25) applied to every competition | 🟢 Low | Independently worth doing; a genuine bias but smaller than the anchor problem |
| 5 | Reduce shrinkage amplification for thin-sample players | 🟡 Medium | Treats a symptom, not the cause. Would blunt mechanism 3 without fixing 1 or 2 |

> [!CAUTION]
> Any of these changes **ordering**, so unlike the stretch refit none is prediction-invariant. Each must go through the standard keep-or-revert Brier gate across all six formats, and `*_ELO_STRETCH` will need re-fitting afterwards — the stretch is downstream of team Elo, which every one of these remedies moves.

> [!NOTE]
> **Unverified detail:** V Sooryavanshi (cited as an example, Elo >2100 with a poor international record) does not appear in the current t20 batter rows under that spelling — likely a corpus cutoff or a name-normalisation difference. The mechanism above explains the reported symptom regardless, but the specific case was not reproduced.

---

## Evidence Log

| # | Finding | Method |
|---|---|---|
| 1 | `EloEngine` pools per competition, each at 1500 | Code — `elo_engine.py:197-206` |
| 2 | `PlayerEloEngine` never reads competition/team_type | Code — grep, zero matches |
| 3 | IPL pool max 1697 vs t20s max 1981; all pool means 1505–1538 | Team Elo × metadata join |
| 4 | Mean attack faced: ipl 1594 ≈ t20s 1589; full league range 56 pts | Squad-Elo columns × competition |
| 5 | Batter top 8 associate-dominated; Buttler 50th, Kohli 126th | Career-peak ranking |
| 6 | Bowler top 12 associate-dominated; Rashid Khan 216/648 | Career-peak ranking |
| 7 | Bennett faced 1793 mean attack vs Kohli 1611 | Per-player opposing squad Elo |
| 8 | t20s bowlers inherit 46% of rating from team Elo vs IPL 26% | reliability = n/(n+10) by competition |
| 9 | Not a small-sample artefact — p99 stable 1998–2001 across cuts 50→250; >2000 cohort median 92 matches == population median 92 | Sensitivity sweep |
| — | *Discarded:* mean `raw_elo` by competition for bridge players (−6.0 batters, +3.6 bowlers) — uninformative, since `raw_elo` is one continuous series per player and the average mostly reflects career timing, not competition difficulty | — |
