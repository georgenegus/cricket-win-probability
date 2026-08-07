"""
Format Spec - Single source of truth for per-format ball geometry and phase edges.
"""

FORMAT_SPECS = {
    "t20": {
        "format_type": "t20",
        "gender": "male",
        "total_overs": 20,
        "balls_per_over": 6,
        "total_balls": 120,
        "pp_end_ball": 36,
        "death_start_ball": 90,
        "default_venue_score": 160.0,
    },
    "od": {
        "format_type": "od",
        "gender": "male",
        "total_overs": 50,
        "balls_per_over": 6,
        "total_balls": 300,
        "pp_end_ball": 60,
        "death_start_ball": 240,
        "default_venue_score": 260.0,
    },
    "hnd": {
        "format_type": "hnd",
        "gender": "male",
        "total_overs": 20,
        "balls_per_over": 5,
        "total_balls": 100,
        "pp_end_ball": 25,
        "death_start_ball": 75,
        "default_venue_score": 140.0,
    },
    "t20w": {
        # Same ball geometry as t20 (20 overs, 6 balls/over) -- only the par score
        # differs. default_venue_score measured empirically from the full t20w corpus
        # (mean first-innings total 124.98, n=3,200 matches, 2026-08-05) -- see
        # womens_games_ingestion_plan.md Phase 2. Do not guess this value for future
        # formats (odw/hndw); measure from their own corpus the same way.
        "format_type": "t20w",
        "gender": "female",
        "total_overs": 20,
        "balls_per_over": 6,
        "total_balls": 120,
        "pp_end_ball": 36,
        "death_start_ball": 90,
        "default_venue_score": 125.0,
    },
    "odw": {
        # Same ball geometry as od (50 overs, 6 balls/over). default_venue_score measured
        # empirically from the full odw corpus (mean first-innings total 220.88, n=929
        # matches, 2026-08-06, after migrating the 921 female ODIs previously misfiled in
        # data/raw/od/) -- same methodology as t20w's Phase 2 measurement (full corpus mean,
        # not the smaller >=90%-balls-filtered sample). Do not reuse od's men's par (260.0).
        "format_type": "odw",
        "gender": "female",
        "total_overs": 50,
        "balls_per_over": 6,
        "total_balls": 300,
        "pp_end_ball": 60,
        "death_start_ball": 240,
        "default_venue_score": 220.88,
    },
    "hndw": {
        # Same ball geometry as hnd (100 balls, 5/over). default_venue_score measured
        # empirically from the full hndw corpus (mean first-innings total 126.61, n=155
        # matches, 2026-08-06). Do not reuse hnd's men's par (140.0).
        "format_type": "hndw",
        "gender": "female",
        "total_overs": 20,
        "balls_per_over": 5,
        "total_balls": 100,
        "pp_end_ball": 25,
        "death_start_ball": 75,
        "default_venue_score": 126.61,
    },
}

# Formats with a trained model registry entry in app.py (win-prob / first-innings /
# delivery-outcome models all exist and are served live). A format can have a
# FORMAT_SPECS entry (its ball geometry is known, the generic feature pipeline can run
# for it) before it belongs here -- ACTIVE_FORMATS only grows once Phase 3/4 of
# womens_games_ingestion_plan.md actually trains + wires up that format's models.
# Centralised so future formats are a one-line addition instead of hunting down every
# call site that iterates "all served formats" (app.py, regression_test.py, invalidate_caches).
ACTIVE_FORMATS = ("t20", "od", "hnd", "t20w", "odw", "hndw")

# POOL_SOURCE: target format -> larger same-gender format whose second-innings states are
# pooled in as extra (train-time-only, never held out, never scored) win-prob training
# signal, for a corpus too thin to train a stable model alone -- see
# hundred_winprob_pooling_shipped.md (hnd<-t20) and womens_games_ingestion_plan.md (hndw<-t20w).
# SEED_SOURCE: target format -> the format whose frozen player-Elo snapshot the target warm-
# starts from. Kept as a SEPARATE dict from POOL_SOURCE (even though currently identical)
# because a format can pool without seeding, or vice versa -- they answer different questions.
POOL_SOURCE = {"hnd": "t20", "hndw": "t20w"}
SEED_SOURCE = {"hnd": "t20", "hndw": "t20w"}

for _target, _source in {**POOL_SOURCE, **SEED_SOURCE}.items():
    assert FORMAT_SPECS[_target]["gender"] == FORMAT_SPECS[_source]["gender"], (
        f"POOL_SOURCE/SEED_SOURCE must stay within gender: {_target} "
        f"({FORMAT_SPECS[_target]['gender']}) -> {_source} ({FORMAT_SPECS[_source]['gender']})"
    )
del _target, _source


def spec(format_type: str) -> dict:
    """Returns the format specification dictionary for format_type (e.g. 't20', 'od', 'hnd', 't20w')."""
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


# ESPN's own per-event eventType / class.generalClassCard (from the scoreboard header and
# summary APIs) is authoritative when present -- unlike infer_format()'s text-keyword guess,
# it doesn't depend on the match/league name containing a format hint (e.g. "South Africa
# Emerging Players v University Sport South Africa" gives no textual clue, but ESPN's own
# class says "ODI"/"Other OD"). Match types we don't model at all (Test, First-Class) map to
# None so callers can exclude them instead of silently defaulting to 't20' and mispricing —
# this is exactly what caused a live Test match to be offered as a T20 fixture.
_ESPN_UNSUPPORTED_CLASSES = ["test", "first-class", "first class", "fc"]
_ESPN_OD_CLASSES = ["odi", "list a", "other od", "one-day international"]
_ESPN_T20_CLASSES = ["t20", "twenty20", "other t20"]

# Gender signal for womens_games_ingestion_plan.md Phase 5 -- verified against ESPN's REAL
# live payload (2026-08-05): "The Hundred Women's Competition" events return
# class.generalClassCard = "Women's T20", a clean, distinct, gendered string. This CONTRADICTS
# classify_espn_event's older docstring claim that the field is blank for women's Hundred
# fixtures -- that may have been true once, but ESPN's API has since improved (or the original
# claim was wrong); trust the field, it's populated now. Event names also reliably suffix each
# team with "(Women)" and league names contain "Women's" -- both already flow into
# fallback_text at the call site (app.py: f"{name} {league_name}"), so no new API surface or
# call-site change was needed for this, only the check below. "wbbl"/"wpl" kept as defensive
# abbreviation backups in case some feed path omits the literal word "women".
_WOMEN_KEYWORDS = ["women", "wbbl", "wpl"]


def _is_women_event(class_card: str, fallback_text: str) -> bool:
    text = f"{class_card or ''} {fallback_text or ''}".lower()
    return any(kw in text for kw in _WOMEN_KEYWORDS)


def classify_espn_event(event_type: str, class_card: str, fallback_text: str = ""):
    """Classifies an ESPN cricket event into a modelled format code
    ('t20'|'od'|'hnd'|'t20w'|'odw'|'hndw'), or None for a match type we don't model (Test,
    First-Class) -- excluded rather than silently mispriced. Prefers ESPN's own
    eventType/class.generalClassCard; falls back to infer_format(fallback_text) only when
    that signal is missing or unrecognised.

    ESPN has no distinct class for The Hundred Men's Competition -- it labels those fixtures
    generically as plain "T20"/"Twenty20" -- so the hnd keyword check against fallback_text
    always runs first, ahead of trusting ESPN's class. (The Hundred WOMEN'S Competition is
    different: its own class_card says "Women's T20" directly -- see _WOMEN_KEYWORDS above.)

    Gender: a detected women's fixture routes to the "w"-suffixed code (t20w/odw/hndw) --
    trained models now exist for all three (womens_games_ingestion_plan.md Phases 3/6, odw/hndw
    phase). Confirmed against real ESPN data that gender routing matters: without it, a live
    Hundred Women's fixture's fallback_text contains "the hundred" (from its own league name)
    and would otherwise be classified plain "hnd" by the branch below and priced by the men's
    model.
    """
    is_women = _is_women_event(class_card, fallback_text)

    if any(kw in (fallback_text or "").lower() for kw in _HND_KEYWORDS):
        base = "hnd"
    else:
        signal = f"{event_type or ''} {class_card or ''}".strip().lower()
        if not signal:
            base = infer_format(fallback_text)
        elif any(kw in signal for kw in _ESPN_UNSUPPORTED_CLASSES):
            base = None
        elif any(kw in signal for kw in _ESPN_OD_CLASSES):
            base = "od"
        elif any(kw in signal for kw in _ESPN_T20_CLASSES):
            base = "t20"
        else:
            base = infer_format(fallback_text)

    if base is None:
        return None
    if not is_women:
        return base
    return f"{base}w"
