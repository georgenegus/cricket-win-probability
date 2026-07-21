import math


class ValueDetector:
    """
    Converts Betfair Exchange odds to fair probabilities,
    detects value bets, and computes Kelly criterion stakes.
    """

    def __init__(self, min_edge=0.03, kelly_fraction=0.25, commission=0.05):
        """
        Args:
            min_edge (float): Minimum edge (model_prob - fair_prob) to flag value bet. Default 3%.
            kelly_fraction (float): Fraction of Kelly criterion to stake. Default 0.25 (quarter Kelly).
            commission (float): Betfair commission on net winnings. Default 5% (0.05).
        """
        self.min_edge = float(min_edge)
        self.kelly_fraction = float(kelly_fraction)
        self.commission = float(commission)

    @staticmethod
    def odds_to_implied_prob(decimal_odds):
        """Converts decimal odds to implied probability."""
        if decimal_odds is None or decimal_odds <= 1.0:
            return 0.0
        return 1.0 / float(decimal_odds)

    @staticmethod
    def remove_overround(implied_probs):
        """
        Normalises a dictionary of {team_name: implied_prob} so they sum to 1.0.

        Example:
            {"India": 0.556, "Australia": 0.476}  # sum = 1.032
            → {"India": 0.539, "Australia": 0.461}
        """
        total = sum(implied_probs.values())
        if total <= 0:
            return implied_probs
        return {team: prob / total for team, prob in implied_probs.items()}

    def detect_value(self, model_prob, fair_market_prob):
        """
        Compares model probability against fair market probability.

        Returns: dict:
            {
                "edge": float,
                "is_value": bool,
                "confidence": str  # "strong", "moderate", "none"
            }
        """
        edge = float(model_prob - fair_market_prob)
        is_value = edge >= self.min_edge

        if edge >= 0.06:
            confidence = "strong"
        elif edge >= self.min_edge:
            confidence = "moderate"
        else:
            confidence = "none"

        return {
            "edge": round(edge, 4),
            "is_value": is_value,
            "confidence": confidence,
        }

    def kelly_stake(self, model_prob, decimal_odds):
        """
        Computes fractional Kelly criterion stake as a percentage of bankroll.
        Accounts for Betfair commission on net winnings.

        Kelly formula (commission-adjusted):
            effective_odds = 1.0 + (decimal_odds - 1.0) * (1.0 - commission)
            b = effective_odds - 1.0
            p = model_prob
            q = 1.0 - p
            kelly_fraction_full = (b * p - q) / b
            stake_pct = kelly_fraction_full * kelly_fraction

        Returns: float (stake as fraction of bankroll, e.g. 0.042 for 4.2%). Returns 0.0 if edge is negative.
        """
        if decimal_odds is None or decimal_odds <= 1.0:
            return 0.0

        p = float(model_prob)
        q = 1.0 - p

        effective_odds = 1.0 + (float(decimal_odds) - 1.0) * (1.0 - self.commission)
        b = effective_odds - 1.0

        if b <= 0:
            return 0.0

        full_kelly = (b * p - q) / b

        if full_kelly <= 0:
            return 0.0

        stake_pct = full_kelly * self.kelly_fraction
        return round(float(stake_pct), 4)

    def analyze_match(self, team_a, team_b, model_prob_a, model_prob_b, back_odds_a, back_odds_b):
        """
        Full value analysis pipeline for a single match.

        Args:
            team_a (str): Name of Team A
            team_b (str): Name of Team B
            model_prob_a (float): Model's win probability for Team A
            model_prob_b (float): Model's win probability for Team B
            back_odds_a (float): Betfair back odds for Team A
            back_odds_b (float): Betfair back odds for Team B

        Returns: dict with complete value assessment and recommended actions.
        """
        imp_a = self.odds_to_implied_prob(back_odds_a)
        imp_b = self.odds_to_implied_prob(back_odds_b)

        raw_implied = {team_a: imp_a, team_b: imp_b}
        fair_probs = self.remove_overround(raw_implied)

        fair_a = fair_probs.get(team_a, 0.5)
        fair_b = fair_probs.get(team_b, 0.5)

        overround = round(sum(raw_implied.values()), 4)

        val_a = self.detect_value(model_prob_a, fair_a)
        val_b = self.detect_value(model_prob_b, fair_b)

        kelly_a = self.kelly_stake(model_prob_a, back_odds_a) if val_a["is_value"] else 0.0
        kelly_b = self.kelly_stake(model_prob_b, back_odds_b) if val_b["is_value"] else 0.0

        recommendation = "No value detected."
        if val_a["is_value"] and val_b["is_value"]:
            if val_a["edge"] >= val_b["edge"]:
                recommendation = f"BACK {team_a} @ {back_odds_a} (edge: +{val_a['edge']:.1%}, stake: {kelly_a:.1%} bankroll)"
            else:
                recommendation = f"BACK {team_b} @ {back_odds_b} (edge: +{val_b['edge']:.1%}, stake: {kelly_b:.1%} bankroll)"
        elif val_a["is_value"]:
            recommendation = f"BACK {team_a} @ {back_odds_a} (edge: +{val_a['edge']:.1%}, stake: {kelly_a:.1%} bankroll)"
        elif val_b["is_value"]:
            recommendation = f"BACK {team_b} @ {back_odds_b} (edge: +{val_b['edge']:.1%}, stake: {kelly_b:.1%} bankroll)"

        return {
            "team_a": team_a,
            "team_b": team_b,
            "model_prob_a": round(float(model_prob_a), 4),
            "model_prob_b": round(float(model_prob_b), 4),
            "back_odds_a": back_odds_a,
            "back_odds_b": back_odds_b,
            "implied_prob_a": round(imp_a, 4),
            "implied_prob_b": round(imp_b, 4),
            "overround": overround,
            "fair_prob_a": round(fair_a, 4),
            "fair_prob_b": round(fair_b, 4),
            "edge_a": val_a["edge"],
            "edge_b": val_b["edge"],
            "is_value_a": val_a["is_value"],
            "is_value_b": val_b["is_value"],
            "kelly_stake_a": kelly_a,
            "kelly_stake_b": kelly_b,
            "recommended_action": recommendation,
        }
