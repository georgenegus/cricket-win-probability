"""
Format Spec - Single source of truth for per-format ball geometry and phase edges.
"""

# `innings_total_sd` -- standard deviation of FIRST-INNINGS TOTALS, measured 2026-08-11 over each
# format's own full-length innings only (whole quota bowled, or all out -- the same population
# generate_mc_calibration_metrics.py samples). Used to scale the Monte Carlo simulator's
# absolute-run calibration gates (MAE / bias / std-dev range) across formats.
#
# Why this exists as a SEPARATE number from default_venue_score, when the gate scaling used to
# reuse that one: MAE, bias and std-dev are all DISPERSION metrics, so they scale with how widely
# innings totals spread, not with where they are centred. The two are not proportional, and the
# error is bidirectional -- t20w has a LOWER mean than t20 (125 vs 160) but a HIGHER absolute
# spread (39.7 vs 37.7), so mean-scaling made its gates ~35% too tight, while hnd/hndw's came out
# too loose. See mc_runrate_feedback_plan.md's Deferred section for the full derivation.
#
# Sample sizes: t20 n=10,013 | od n=3,768 | t20w n=3,042 | odw n=842 | hnd n=157 | hndw n=155.
# hnd/hndw's are thin (~6% relative SE on the SD itself) -- treat their derived gates as
# provisional and re-measure once those corpora grow.
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
        "innings_total_sd": 37.74,   # n=10,013
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
        "innings_total_sd": 69.89,   # n=3,768
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
        "innings_total_sd": 27.36,   # n=157
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
        "innings_total_sd": 39.65,   # n=3,042
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
        "innings_total_sd": 66.24,   # n=842
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
        "innings_total_sd": 23.06,   # n=155
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

# INTL_COMPETITIONS: format -> raw data/raw/<format>/<competition>/ folder name(s) that hold
# genuine nation-vs-nation internationals (as opposed to domestic franchise leagues). Same
# `competition` column EloEngine already pools on (match_metadata.py sets it to the raw folder
# name) -- this is just the subset that corresponds to what the ICC actually ranks. hnd/hndw have
# no entry: The Hundred is a domestic-only competition, there are no international matches to
# select. See markdowns/win_predictor/icc_rating_engine_plan.md.
INTL_COMPETITIONS = {"t20": ("t20s",), "od": ("odis",), "t20w": ("t20s",), "odw": ("odis",)}

# FRANCHISE_BRIDGE_TIERS: franchise/domestic competitions PER FORMAT, tiered by how densely
# "gold standard" internationally-rated players appear in them -- the constraint that decides
# whether a league's players can be linked to the international rating scale by observing them
# in the same matches as players whose quality is already known. See
# markdowns/win_predictor/franchise_bridge_plan.md.
#
# Density measured by scripts/measure_franchise_bridge_density.py using the strict time-aware
# definition (a player counts as an anchor for a match only if they had already reached
# ICCPlayerRatingEngine.MIN_MATCHES_TO_RANK international innings BEFORE that match's date).
# "both>=3" = share of matches where BOTH XIs contained at least 3 such anchors.
# t20 measured 2026-09-02; od/t20w/odw measured 2026-09-03.
#
#   Tier A -- directly bridgeable in a single hop (both>=3 in >=50% of matches)
#   Tier B -- thin (>=10%); usable only with wide uncertainty, or once multi-hop anchors exist
#   Tier C -- barely bridgeable directly (<10%); overwhelmingly career-domestic players
#
# Two findings worth keeping in view, because they shape what can ever be shipped per format:
#   - In t20, the competitions with the MOST players needing ratings are the hardest to anchor
#     (ntb, England's T20 Blast, is the largest league in the corpus at n=1,489 yet sits in
#     Tier C). Hence the phased build: Tier A first, validate, only then multi-hop for B/C.
#   - od and odw have NO Tier A or B competition at all. Domestic 50-over cricket is played
#     overwhelmingly by uncapped squad players -- even a single anchor appears in only 19-43%
#     of matches. Those formats cannot be bridged single-hop; see the plan's gated multi-hop
#     attempt. hnd/hndw are absent entirely: The Hundred has no international pool of its own
#     format to anchor from.
FRANCHISE_BRIDGE_TIERS = {
    # format: {tier: {competition folder: (name, n_matches, avg anchors/XI, both>=3 %)}}
    "t20": {
        "A": {
            "mlc": ("Major League Cricket (USA)", 105, 5.6, 98.1),
            "ilt": ("ILT20 (UAE)", 134, 6.5, 95.5),
            "psl": ("Pakistan Super League", 357, 4.7, 87.1),
            "lpl": ("Lanka Premier League", 121, 4.8, 83.5),
            "cpl": ("Caribbean Premier League", 411, 4.6, 74.2),
            "npl": ("Nepal Premier League", 64, 3.6, 68.8),
            "sat": ("SA20 (South Africa)", 130, 4.5, 65.4),
            "ipl": ("Indian Premier League", 1243, 3.7, 63.2),
            "bpl": ("Bangladesh Premier League", 469, 3.7, 55.0),
        },
        "B": {
            "msl": ("Mzansi Super League (South Africa)", 56, 2.5, 30.4),
            "bbl": ("Big Bash League (Australia)", 662, 2.0, 13.7),
        },
        "C": {
            "ntb": ("T20 Blast (England counties)", 1489, 1.3, 1.9),
            "ipt": ("Inter-Provincial Trophy (Ireland)", 109, 1.7, 3.7),
            "ctc": ("CSA T20 Challenge (South Africa provinces)", 314, 0.8, 1.0),
            "ssm": ("Super Smash (New Zealand)", 270, 1.0, 0.7),
            "mct": ("Major Clubs T20 (Sri Lanka)", 72, 0.7, 0.0),
        },
    },
    # t20w's marquee women's leagues anchor BETTER than the IPL -- WPL and WTC are at 100%.
    "t20w": {
        "A": {
            "wtc": ("Women's T20 Challenge (India)", 13, 7.1, 100.0),
            "wpl": ("Women's Premier League (India)", 88, 6.2, 100.0),
            "wsl": ("Women's Super League (England)", 95, 3.8, 70.5),
            "wbb": ("Women's Big Bash League (Australia)", 519, 3.7, 68.0),
        },
        "C": {
            "wtb": ("Women's T20 Blast (England)", 112, 1.5, 6.2),
            "blz": ("Women's Super Smash / Blaze (NZ)", 17, 2.3, 5.9),
            "cec": ("CSA Women's T20 (South Africa)", 124, 1.3, 0.0),
        },
    },
    "od": {
        "C": {
            "odms": ("Domestic one-day (multi-nation)", 1457, 0.8, 1.5),
            "odc": ("Domestic one-day cup", 122, 0.9, 0.8),
        },
    },
    "odw": {
        "C": {
            "wod": ("Women's domestic one-day", 105, 0.9, 1.0),
            "rhf": ("Rachael Heyhoe Flint Trophy (England)", 189, 0.6, 0.0),
        },
    },
}


def franchise_competitions(format_type: str = "t20", tiers=("A",)) -> tuple:
    """Competition folder names for the requested tiers within one format.

    Returns an empty tuple for a format/tier combination that does not exist (e.g. od has no
    Tier A), rather than raising -- callers iterate formats and an absent tier is a legitimate
    "this format has nothing bridgeable at that quality", not a programming error.
    """
    fmt_tiers = FRANCHISE_BRIDGE_TIERS.get(format_type, {})
    out = []
    for tier in tiers:
        if tier not in ("A", "B", "C"):
            raise ValueError(f"Unknown franchise bridge tier '{tier}'. Expected one of A/B/C.")
        out.extend(fmt_tiers.get(tier, {}).keys())
    return tuple(out)


def bridgeable_formats(tiers=("A",)) -> tuple:
    """Formats having at least one competition in the requested tiers -- i.e. those the
    franchise bridge can actually be run for."""
    return tuple(f for f in FRANCHISE_BRIDGE_TIERS if franchise_competitions(f, tiers))


# RATING_SOURCES: the rating-source toggle registry (markdowns/win_predictor/
# rating_source_toggle_plan.md) -- maps a source key to its parquet filename templates, a UI
# label, and the formats it actually has coverage for. "elo" is the deployed default, always
# available, and its templates are byte-identical to what every read site already hardcodes --
# resolving through this registry with source="elo" must reproduce today's path exactly.
# "icc" is the combined ICC-methodology + franchise-bridge artifact built by
# scripts/build_combined_ratings.py; its `formats` list is deliberately NOT "every format with
# an international pool" -- od/odw have one (see INTL_COMPETITIONS) but their bridge extension
# failed Phase 1's validation gate, so they are internationals-only in practice and excluded
# here rather than left to silently degrade a prediction with 32-38% neutral-rated matches.
# hnd/hndw have no international pool at all and are absent from both.
RATING_SOURCES = {
    "elo": {
        "label": "Elo",
        "team_template": "{fmt}_elo_ratings.parquet",
        "player_template": "{fmt}_player_elo_ratings.parquet",
        "formats": ACTIVE_FORMATS,
    },
    "icc": {
        "label": "ICC-style",
        "team_template": "{fmt}_icc_combined_team_ratings.parquet",
        "player_template": "{fmt}_icc_combined_player_ratings.parquet",
        "formats": ("t20", "t20w", "od", "odw"),
    },
}
DEFAULT_RATING_SOURCE = "elo"

assert all(f in ACTIVE_FORMATS for f in RATING_SOURCES["icc"]["formats"]), (
    "RATING_SOURCES['icc']['formats'] must stay within ACTIVE_FORMATS"
)

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
