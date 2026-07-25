"""
History-driven venue resolution engine.
Precomputes team venue distributions from historical Cricsheet metadata
and infers venue keys when Betfair omits ground metadata.

Concentration is measured on ALL appearances (a team's matches as either
team_1 or team_2), not team_1-only — team_1 is not a reliable proxy for
"home team" in this dataset, and restricting to it discards roughly half
the data.

A team's ground is only trusted as "concentrated" when it clears two
conditions:
  - share:  the top venue accounts for >= min_share of all appearances.
  - margin: the top venue is >= min_margin ahead of the second most
            common venue.
Share alone is not enough. Domestic county teams spread their away
fixtures across ~17 different opposition grounds, so one real home
ground clears ~45-55% easily (Lancashire 0.47, Somerset 0.49,
Nottinghamshire 0.55, Surrey 0.55 — margins 0.37-0.47). Franchise teams
concentrate their away fixtures into a handful of other league venues,
which flattens the top share even when a real home ground exists
(RCB 0.28, CSK 0.32, Mumbai Indians 0.34 — margins 0.21-0.28). A single
share threshold cannot separate a genuine-but-flattened home ground from
a genuine rotator (Argentina 0.28 share but only 0.15 margin, scattered
near-evenly across neutral grounds) — margin is what does that.
Defaults (min_share=0.35, min_margin=0.20, min_matches=10) were picked by
checking them against known rotators (Argentina) and known home-ground
teams (the county sides above) and confirming the rotators abstain while
the home-ground teams resolve.
"""

import json
from pathlib import Path
import pandas as pd
from src.features.venue_aliases import resolve_venue_alias

DEFAULT_MIN_SHARE = 0.35
DEFAULT_MIN_MARGIN = 0.20
DEFAULT_MIN_MATCHES = 10


def build_team_venue_distributions(format_type: str = "t20", processed_dir: str = "data/processed") -> pd.DataFrame:
    """
    Computes historical venue frequency distributions per team from match metadata.
    Counts a team's appearance whether it played as team_1 or team_2 — team_1-only
    counting discarded ~half the data and understated concentration for every team.
    Uses resolve_venue_alias to group raw venue strings under canonical venue keys.
    """
    p_dir = Path(processed_dir)
    meta_file = p_dir / f"{format_type}_match_metadata.parquet"
    lookup_file = p_dir / f"{format_type}_venue_lookup.parquet"

    if not meta_file.exists() or not lookup_file.exists():
        print(f"Warning: Missing metadata or lookup parquet for {format_type}. Cannot build venue dist.")
        return pd.DataFrame()

    meta_df = pd.read_parquet(meta_file)
    lookup_df = pd.read_parquet(lookup_file)
    available_venues = set(lookup_df["venue"].dropna().unique())

    valid_meta = meta_df[meta_df["venue"].notna()].copy()

    resolved_records = []
    for _, row in valid_meta.iterrows():
        raw_v = str(row["venue"]).strip()
        canonical_v = resolve_venue_alias(raw_v, available_venues)
        if not canonical_v:
            continue
        for team_col in ("team_1", "team_2"):
            team = row.get(team_col)
            if pd.notna(team):
                resolved_records.append({"team": str(team).strip(), "canonical_venue": canonical_v})

    if not resolved_records:
        return pd.DataFrame()

    df_resolved = pd.DataFrame(resolved_records)
    grouped = df_resolved.groupby(["team", "canonical_venue"]).size().reset_index(name="match_count")

    dist_records = []
    for team, group in grouped.groupby("team"):
        total_m = group["match_count"].sum()
        sorted_group = group.sort_values(by="match_count", ascending=False)
        top_row = sorted_group.iloc[0]

        top_venue = top_row["canonical_venue"]
        top_count = top_row["match_count"]
        top_share = round(float(top_count / total_m), 4)

        second_count = int(sorted_group.iloc[1]["match_count"]) if len(sorted_group) > 1 else 0
        second_share = round(float(second_count / total_m), 4)

        dist_records.append({
            "team": team,
            "primary_venue": top_venue,
            "primary_venue_share": top_share,
            "second_venue_share": second_share,
            "concentration_margin": round(top_share - second_share, 4),
            "primary_venue_matches": int(top_count),
            "total_appearances": int(total_m),
        })

    dist_df = pd.DataFrame(dist_records)
    out_file = p_dir / f"{format_type}_team_venue_dist.parquet"
    print(f"Saving {format_type.upper()} team venue distribution to: {out_file}")
    dist_df.to_parquet(out_file, index=False, compression="snappy")
    return dist_df


def validate_venue_overrides(
    overrides_path: str = "data/reference/venue_overrides.json",
    processed_dir: str = "data/processed",
) -> bool:
    """
    Validates that all target venue strings in venue_overrides.json resolve cleanly
    to valid keys in the corresponding format's venue lookup parquet.
    """
    o_path = Path(overrides_path)
    if not o_path.exists():
        print(f"No venue_overrides.json file found at {o_path}.")
        return True

    try:
        with open(o_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
    except Exception as e:
        print(f"Error reading {o_path}: {e}")
        return False

    all_valid = True
    p_dir = Path(processed_dir)

    for fmt in ["t20", "od"]:
        fmt_dict = overrides.get(fmt, {})
        lookup_file = p_dir / f"{fmt}_venue_lookup.parquet"

        if not lookup_file.exists():
            print(f"Lookup file {lookup_file} missing for validation.")
            continue

        lookup_df = pd.read_parquet(lookup_file)
        available_venues = set(lookup_df["venue"].dropna().unique())

        for team, venue_str in fmt_dict.items():
            resolved = resolve_venue_alias(venue_str, available_venues)
            if not resolved or resolved not in available_venues:
                print(f"Validation Failure [{fmt.upper()}]: Override '{team}' -> '{venue_str}' does NOT resolve to a valid lookup key.")
                all_valid = False
            else:
                print(f"Validation OK [{fmt.upper()}]: '{team}' -> '{resolved}'")

    return all_valid


class VenueHistoryResolver:
    """
    Infers the most likely venue for a fixture based on seed overrides and
    precomputed historical distributions for BOTH teams in the fixture.
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        overrides_path: str = "data/reference/venue_overrides.json",
        min_share: float = DEFAULT_MIN_SHARE,
        min_margin: float = DEFAULT_MIN_MARGIN,
        min_matches: int = DEFAULT_MIN_MATCHES,
    ):
        self.processed_dir = Path(processed_dir)
        self.overrides_path = Path(overrides_path)
        self.min_share = min_share
        self.min_margin = min_margin
        self.min_matches = min_matches
        self.overrides = self._load_overrides()
        self.t20_dist = self._load_dist("t20")
        self.od_dist = self._load_dist("od")
        self._venue_set_cache = {}

    def _load_overrides(self) -> dict:
        if not self.overrides_path.exists():
            return {"t20": {}, "od": {}}
        try:
            with open(self.overrides_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "t20": data.get("t20", {}),
                    "od": data.get("od", {}),
                }
        except Exception as e:
            print(f"Warning: Failed to load venue overrides from {self.overrides_path}: {e}")
            return {"t20": {}, "od": {}}

    def _load_dist(self, format_type: str) -> dict:
        """
        Read-only load (Phase 11 fix). Construction must not have disk side
        effects, so this no longer builds the distribution parquet on demand
        — that happens explicitly as part of the data build pipeline (see
        cricsheet_importer.retrain_lightweight / retrain_full), which call
        build_team_venue_distributions() right after VenueEngine.run().
        """
        dist_file = self.processed_dir / f"{format_type}_team_venue_dist.parquet"
        if not dist_file.exists():
            print(
                f"Warning: {dist_file} not found. History-based venue resolution is "
                f"unavailable for {format_type.upper()} until it is built — run "
                f"build_team_venue_distributions(format_type='{format_type}') "
                f"(wired into the retrain_lightweight/retrain_full build pipeline)."
            )
            return {}

        df = pd.read_parquet(dist_file)
        if df.empty:
            return {}

        return df.set_index("team").to_dict(orient="index")

    def _available_venues(self, fmt: str) -> set:
        if fmt not in self._venue_set_cache:
            lookup_file = self.processed_dir / f"{fmt}_venue_lookup.parquet"
            if lookup_file.exists():
                lookup_df = pd.read_parquet(lookup_file)
                self._venue_set_cache[fmt] = set(lookup_df["venue"].dropna().unique())
            else:
                self._venue_set_cache[fmt] = set()
        return self._venue_set_cache[fmt]

    def _resolve_override(self, team: str, fmt: str) -> str | None:
        fmt_overrides = self.overrides.get(fmt, {})
        if team in fmt_overrides:
            available = self._available_venues(fmt)
            canonical = resolve_venue_alias(fmt_overrides[team], available)
            if canonical:
                return canonical
        return None

    def _concentrated_venue(self, team: str, fmt: str) -> tuple[str, float] | None:
        """Returns (venue, share) if the team has a clearly dominant venue, else None."""
        dist = self.od_dist if fmt == "od" else self.t20_dist
        info = dist.get(team)
        if not info:
            return None

        share = float(info["primary_venue_share"])
        margin = float(info.get("concentration_margin", share - float(info.get("second_venue_share", 0.0))))
        matches = int(info["total_appearances"])

        if share >= self.min_share and margin >= self.min_margin and matches >= self.min_matches:
            return info["primary_venue"], share
        return None

    def resolve_from_history(
        self,
        team_a: str,
        team_b: str = None,
        format_type: str = "t20",
    ) -> tuple[str | None, float, dict]:
        """
        Resolves fixture (team_a v team_b) using seed overrides first, then the
        historical concentration of BOTH teams — never title order alone.

        Returns (venue, confidence, info):
          - venue: canonical venue string, or None if no confident resolution.
          - confidence: 0.0-1.0 (1.0 for an explicit override).
          - info["status"] tells the caller how the result was reached, so it
            can decide whether to log to the review queue:
              "override"             - explicit seed override, confidence 1.0.
              "single_concentrated"  - exactly one team has a dominant venue.
              "agree"                - both teams' dominant venues match.
              "tiebreak"             - both teams concentrated at DIFFERENT
                                        venues; team_a's venue (title order)
                                        is used as a last-resort tiebreak,
                                        never as the primary signal. Callers
                                        should treat this as lower-trust and
                                        may want to log it for audit.
              "no_data"              - neither team has a confident venue;
                                        this is the ordinary baseline case
                                        (no history to be ambiguous about),
                                        not something to queue.
        """
        fmt = "od" if str(format_type).lower() in ["od", "odi", "one-day"] else "t20"
        team_a_clean = str(team_a).strip()
        team_b_clean = str(team_b).strip() if team_b else ""

        # Step 1: Explicit overrides win outright (new teams / ground relocations).
        override_a = self._resolve_override(team_a_clean, fmt)
        if override_a:
            return override_a, 1.0, {"status": "override", "team": team_a_clean}

        if team_b_clean:
            override_b = self._resolve_override(team_b_clean, fmt)
            if override_b:
                return override_b, 1.0, {"status": "override", "team": team_b_clean}

        # Step 2: Historical concentration for BOTH teams.
        conc_a = self._concentrated_venue(team_a_clean, fmt)
        conc_b = self._concentrated_venue(team_b_clean, fmt) if team_b_clean else None

        if conc_a and not conc_b:
            venue, share = conc_a
            return venue, share, {"status": "single_concentrated", "team": team_a_clean}

        if conc_b and not conc_a:
            venue, share = conc_b
            return venue, share, {"status": "single_concentrated", "team": team_b_clean}

        if conc_a and conc_b:
            venue_a, share_a = conc_a
            venue_b, share_b = conc_b
            if venue_a == venue_b:
                return venue_a, max(share_a, share_b), {"status": "agree"}

            # Both teams have a real home ground and they differ — title order
            # is used only as a last-resort tiebreak (unreliable as a primary
            # signal for internationals/ICC/neutral fixtures), so trust is low.
            return venue_a, share_a, {
                "status": "tiebreak",
                "team": team_a_clean,
                "rejected_candidate": {"team": team_b_clean, "venue": venue_b, "share": share_b},
            }

        return None, 0.0, {"status": "no_data"}
