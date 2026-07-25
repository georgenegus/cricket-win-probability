"""
Format Spec - Single source of truth for per-format ball geometry and phase edges.
"""

FORMAT_SPECS = {
    "t20": {
        "format_type": "t20",
        "total_overs": 20,
        "balls_per_over": 6,
        "total_balls": 120,
        "pp_end_ball": 36,
        "death_start_ball": 90,
        "default_venue_score": 160.0,
    },
    "od": {
        "format_type": "od",
        "total_overs": 50,
        "balls_per_over": 6,
        "total_balls": 300,
        "pp_end_ball": 60,
        "death_start_ball": 240,
        "default_venue_score": 260.0,
    },
    "hnd": {
        "format_type": "hnd",
        "total_overs": 20,
        "balls_per_over": 5,
        "total_balls": 100,
        "pp_end_ball": 25,
        "death_start_ball": 75,
        "default_venue_score": 140.0,
    },
}


def spec(format_type: str) -> dict:
    """Returns the format specification dictionary for format_type ('t20', 'od', 'hnd')."""
    fmt = format_type.lower() if format_type else "t20"
    if fmt not in FORMAT_SPECS:
        raise ValueError(f"Unknown format_type '{format_type}'. Expected one of {list(FORMAT_SPECS.keys())}")
    return FORMAT_SPECS[fmt]


# Keyword lists for inferring a match format from free text (event / competition /
# market / league names). Order matters: Hundred is checked first (its venues also
# host T20s), then ODI, then T20, defaulting to T20 when nothing matches.
_HND_KEYWORDS = ["hundred", "100-ball", "100 ball", "the hundred"]
_OD_KEYWORDS = ["odi", "one-day", "one day", "50 over", "50-over", "metro bank",
                "royal london", "list a", "1-day"]
_T20_KEYWORDS = ["t20", "twenty20", "blast", "ipl", "cpl", "bbl", "psl", "sa20",
                 "ilt20", "mlc", "lpl", "bpl", "super smash"]


def infer_format(text: str) -> str:
    """Infers a modelled format code ('hnd' | 'od' | 't20') from free-form match text.

    Defaults to 't20' when no keyword matches. Shared by the pre-match scanner and
    the live-games endpoint so the classification stays consistent.
    """
    t = (text or "").lower()
    if any(kw in t for kw in _HND_KEYWORDS):
        return "hnd"
    if any(kw in t for kw in _OD_KEYWORDS):
        return "od"
    if any(kw in t for kw in _T20_KEYWORDS):
        return "t20"
    return "t20"
