"""
Phase 2 of markdowns/win_predictor/rating_source_toggle_plan.md.

Resolves a (format, source) pair to its rating parquet paths via format_spec.RATING_SOURCES,
so existing read sites in app.py / pre_match_predictor.py take a `source` argument instead of
hardcoding "data/processed/{fmt}_elo_ratings.parquet" -- a path swap, not a logic swap. Calling
any function here with source="elo" (the default everywhere) reproduces today's exact path, so
converting a call site to go through this resolver is behaviour-preserving by construction.
"""
from pathlib import Path

import pandas as pd

from src.data_processing.format_spec import RATING_SOURCES, DEFAULT_RATING_SOURCE


def is_supported(format_type: str, source: str = DEFAULT_RATING_SOURCE) -> bool:
    """Whether `source` has rating coverage for `format_type` at all -- the toggle disables
    itself (with a tooltip) for a (format, source) pair this returns False for."""
    cfg = RATING_SOURCES.get(source)
    return cfg is not None and format_type in cfg["formats"]


def resolve_path(format_type: str, kind: str, source: str = DEFAULT_RATING_SOURCE,
                 processed_path: str = "data/processed") -> Path:
    """`kind` is "team" or "player". Raises for an unknown source or an unsupported
    (format, source) pair rather than silently falling back -- a caller that gets this wrong
    should fail loudly, not serve a prediction off the wrong scale."""
    cfg = RATING_SOURCES.get(source)
    if cfg is None:
        raise ValueError(f"Unknown rating source '{source}'. Known: {sorted(RATING_SOURCES)}")
    if format_type not in cfg["formats"]:
        raise ValueError(
            f"Rating source '{source}' has no coverage for format '{format_type}'. "
            f"Supported formats: {cfg['formats']}"
        )
    template = cfg[f"{kind}_template"]
    return Path(processed_path) / template.format(fmt=format_type)


def load_team_ratings(format_type: str, source: str = DEFAULT_RATING_SOURCE,
                      processed_path: str = "data/processed") -> pd.DataFrame:
    return pd.read_parquet(resolve_path(format_type, "team", source, processed_path))


def load_player_ratings(format_type: str, source: str = DEFAULT_RATING_SOURCE,
                        processed_path: str = "data/processed") -> pd.DataFrame:
    return pd.read_parquet(resolve_path(format_type, "player", source, processed_path))


def available_sources(format_type: str) -> tuple:
    """Source keys with coverage for `format_type`, in RATING_SOURCES' declared order -- what
    the UI toggle should offer for this format."""
    return tuple(s for s, cfg in RATING_SOURCES.items() if format_type in cfg["formats"])
