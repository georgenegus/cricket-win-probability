"""
Unit tests for mid-innings resume in the Monte Carlo simulator (live-simulation feature).

Uses a deterministic fake DeliveryOutcomeModel so the tests are fast and don't depend on
trained model caches — they exercise the resume bookkeeping, not model accuracy.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.innings_simulator import simulate_innings
from src.simulation.match_simulator import simulate_match
from src.data_processing.format_spec import spec


class FakeModel:
    """Returns fixed, well-formed outcome probabilities for every delivery."""
    format_type = "t20"

    def predict_proba_batch(self, state):
        n = len(state)
        extras = np.tile(np.array([0.96, 0.03, 0.01]), (n, 1))          # legal / wide / noball
        legal = np.tile(np.array([0.30, 0.30, 0.10, 0.02, 0.15, 0.10, 0.03]), (n, 1))  # 7 classes → 1.0
        return {"extras": extras, "legal": legal}


def _team(name):
    lineup = [{"name": f"{name}_{i}", "effective_elo": 1500.0, "recent_form": 0.0} for i in range(11)]
    attack = [{"name": f"{name}_bwl_{i}", "effective_elo": 1500.0, "recent_form": 0.0} for i in range(5)]
    return {"name": name, "lineup": lineup, "attack": attack}


class TestInningsResume(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.lineup = _team("A")["lineup"]
        self.attack = _team("B")["attack"]
        self.rng = lambda: np.random.default_rng(7)

    def test_resume_floors_score_and_wickets(self):
        """A resumed innings never dips below its seeded score/wickets/balls."""
        start = {"runs": 100, "wickets": 3, "legal_balls": 60}
        out = simulate_innings(self.model, self.lineup, self.attack, venue_avg_score=160.0,
                               n_sims=500, format_type="t20", innings=1,
                               rng=self.rng(), start_state=start)
        self.assertTrue((out["final_score"] >= 100).all())
        self.assertTrue((out["wickets"] >= 3).all())
        self.assertTrue((out["legal_balls"] >= 60).all())
        self.assertTrue((out["legal_balls"] <= spec("t20")["total_balls"]).all())

    def test_higher_seed_yields_higher_final(self):
        """With the same balls remaining, a higher seeded score yields a higher final total."""
        low = simulate_innings(self.model, self.lineup, self.attack, 160.0, n_sims=500,
                               format_type="t20", innings=1, rng=self.rng(),
                               start_state={"runs": 100, "wickets": 2, "legal_balls": 60})
        high = simulate_innings(self.model, self.lineup, self.attack, 160.0, n_sims=500,
                                format_type="t20", innings=1, rng=self.rng(),
                                start_state={"runs": 150, "wickets": 2, "legal_balls": 60})
        self.assertGreater(high["final_score"].mean(), low["final_score"].mean())

    def test_terminal_snapshot_stays_dead(self):
        """An already-complete snapshot (all out) does not advance."""
        out = simulate_innings(self.model, self.lineup, self.attack, 160.0, n_sims=200,
                               format_type="t20", innings=1, rng=self.rng(),
                               start_state={"runs": 175, "wickets": 10, "legal_balls": 118})
        self.assertTrue((out["final_score"] == 175).all())
        self.assertTrue((out["wickets"] == 10).all())


class TestMatchLiveState(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.team_a = _team("A")
        self.team_b = _team("B")

    def test_innings2_fixes_first_innings(self):
        """Live in innings 2: the 1st innings is fixed at target-1, chase resumes from the snapshot."""
        sim = simulate_match(self.model, self.team_a, self.team_b, venue_avg_score=160.0, n_sims=500,
                             format_type="t20", bat_first="b",
                             live_state={"innings": 2, "runs": 90, "wickets": 3,
                                         "legal_balls": 60, "target": 180})
        self.assertTrue((sim["inn1_score"] == 179).all())
        self.assertTrue((sim["inn2_score"] >= 90).all())

    def test_innings1_resumes_first_innings(self):
        """Live in innings 1: 1st innings resumes; 2nd innings is a full chase."""
        sim = simulate_match(self.model, self.team_a, self.team_b, venue_avg_score=160.0, n_sims=500,
                             format_type="t20", bat_first="a",
                             live_state={"innings": 1, "runs": 45, "wickets": 1, "legal_balls": 36})
        self.assertTrue((sim["inn1_score"] >= 45).all())

    def test_pre_match_path_unchanged(self):
        """No live_state → the standard pre-match two-innings simulation still runs."""
        sim = simulate_match(self.model, self.team_a, self.team_b, venue_avg_score=160.0, n_sims=500,
                             format_type="t20", bat_first="a")
        self.assertEqual(len(sim["inn1_score"]), 500)
        self.assertTrue((sim["inn1_score"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
