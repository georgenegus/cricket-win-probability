import io
import sys
import shutil
import requests
import zipfile
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd

# Reconfigure stdout to UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from src.utils.cache_manager import CacheManager
from src.data_processing.format_spec import ACTIVE_FORMATS, spec, POOL_SOURCE, SEED_SOURCE

CRICSHEET_BASE = "https://cricsheet.org/downloads"

# Mapping from Cricsheet competition names to local subfolder names
T20_COMPETITIONS = {
    "Big Bash League": "bbl",
    "Bangladesh Premier League": "bpl",
    "Caribbean Premier League": "cpl",
    "CSA T20 Challenge": "ctc",
    "International League T20": "ilt",
    "Indian Premier League": "ipl",
    "IPL": "ipl",
    "Inter-Provincial Trophy": "ipt",
    "Cricket Ireland Inter-Provincial Twenty20 Trophy": "ipt",
    "Lanka Premier League": "lpl",
    "Men's Cricket Tournament": "mct",
    "Major League Cricket": "mlc",
    "Mzansi Super League": "msl",
    "Nepal Premier League": "npl",
    "National T20 Cup": "ntb",
    "Pakistan Super League": "psl",
    "SA20": "sat",
    "Super Smash": "ssm",
    "T20 Blast": "t20s",
    "Vitality Blast": "t20s",
    "Vitality Blast Men": "t20s",
}

OD_COMPETITIONS = {
    "One-Day Cup": "odc",
    "One Day International": "odis",
    "ODI": "odis",
    "ODM": "odms",
    "Royal London Cup": "rlc",
    "Sheffield Shield": "sft",
    "Metro Bank One Day Cup": "odc",
    "Cricket Ireland Inter-Provincial Limited Over Cup": "odc",
}

HND_COMPETITIONS = {
    "The Hundred": "hnd",
    "The Hundred Men's Competition": "hnd",
    "The Hundred Men": "hnd",
}

# Women's-only competitions (no equivalent in the men's dicts above). Competitions that
# share an identical name across genders on Cricsheet (e.g. "Super Smash") deliberately
# are NOT duplicated here -- classify_match() falls back to the men's dict first, so a
# same-named competition resolves to the same comp_dir code regardless of gender; only
# the format_dir gets the "w" suffix. Codes below match Cricsheet's own per-competition
# zip filenames (e.g. wbb.zip, cec.zip) -- same convention as the men's dicts' values.
T20W_COMPETITIONS = {
    "Women's Big Bash League": "wbb",
    "Women's Premier League": "wpl",
    "Charlotte Edwards Cup": "cec",
    "Women's Cricket Super League": "wsl",
    "Vitality Blast Women": "wtb",
    "T20 Blaze": "blz",
    "Women's T20 Blaze": "blz",
    "Women's T20 Challenge": "wtc",
    "Women's T20 Challenge Match": "wtc",
}

ODW_COMPETITIONS = {
    "Rachael Heyhoe Flint Trophy": "rhf",
    "ECB Women's One-Day Cup": "wod",
}

HNDW_COMPETITIONS = {
    "The Hundred Women's Competition": "hnd",
    "The Hundred Women": "hnd",
}

# All format_dir values this importer can produce, men's + women's. Used to size the
# existing-match dedup scan and the per-format stats dict generically (see
# import_matches_from_zip_url) -- avoids a hardcoded t20/od/hnd triple that silently
# misses the "w" formats.
ALL_FORMAT_DIRS = ("t20", "od", "hnd", "t20w", "odw", "hndw")



def download_zip(url: str) -> io.BytesIO:
    """Downloads a ZIP archive from Cricsheet into an in-memory buffer."""
    print(f"Downloading: {url}")
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    print(f"  Downloaded {len(response.content) / 1024 / 1024:.2f} MB")
    return io.BytesIO(response.content)


def classify_match(yaml_bytes: bytes) -> dict:
    """
    Parses YAML header and classifies match by format and competition folder.

    Returns dict with keys: match_type, competition, gender, format_dir, comp_dir
    Or None if unparseable, an unrecognised gender (not "male"/"female" -- e.g. rare
    "mixed" exhibition matches), or unsupported match type (Tests, Multi-day).

    Female matches route to a "w"-suffixed format_dir (t20w/odw/hndw) rather than
    being dropped -- see womens_games_ingestion_plan.md. comp_dir resolution checks
    the men's dict FIRST regardless of gender: competitions with an identical name on
    both sides of Cricsheet (e.g. "Super Smash") resolve to the same comp_dir code for
    both genders, since only format_dir needs the gender split, not comp_dir.
    """
    try:
        match = yaml.load(yaml_bytes, Loader=SafeLoader)
    except Exception:
        return None

    if not match or "info" not in match:
        return None

    info = match["info"]
    gender = info.get("gender", "male")
    if gender not in ("male", "female"):
        return None  # unrecognised gender tag (e.g. "mixed") -- not modeled, same as before
    is_female = gender == "female"

    match_type = str(info.get("match_type", "")).upper()
    competition = info.get("competition", "")
    balls_per_over = info.get("balls_per_over")

    # Determine base format (gender-agnostic: match-type/competition/ball-geometry rules)
    if (
        match_type in ("HND", "HUNDRED", "100")
        or competition in HND_COMPETITIONS
        or competition in HNDW_COMPETITIONS
        or "The Hundred" in competition
        or "Hundred" in competition
        or balls_per_over == 5
    ):
        base_format = "hnd"
        comp_dir = HND_COMPETITIONS.get(competition) or HNDW_COMPETITIONS.get(competition, "hnd")
    elif match_type in ("T20", "IT20", "T20I"):
        base_format = "t20"
        comp_dir = T20_COMPETITIONS.get(competition) or T20W_COMPETITIONS.get(competition, "t20s")
    elif match_type in ("ODI", "ODM", "OD"):
        base_format = "od"
        comp_dir = OD_COMPETITIONS.get(competition) or ODW_COMPETITIONS.get(competition, "odis")
    else:
        return None  # Skip Tests, multi-day, etc.

    format_dir = f"{base_format}w" if is_female else base_format

    return {
        "match_type": match_type,
        "competition": competition,
        "gender": gender,
        "format_dir": format_dir,
        "comp_dir": comp_dir,
    }


def _empty_import_stats() -> dict:
    stats = {f"new_{fmt}": 0 for fmt in ALL_FORMAT_DIRS}
    stats.update({"skipped": 0, "skipped_exists": 0, "errors": 0})
    return stats


def _import_matches_from_zip_buffer(
    zip_buffer,
    raw_root: Path,
    dry_run: bool = False,
) -> dict:
    """
    Shared core: classifies + writes every match in an open zip buffer into data/raw/,
    skipping any match_id already present in ANY of the 6 format dirs (t20/od/hnd +
    t20w/odw/hndw). Used by both the Cricsheet-URL importer and the local-zip importer
    (for the staged female competition archives) so dedup logic lives in exactly one place.
    """
    stats = _empty_import_stats()

    # Build set of existing match IDs across every format directory (men's + women's).
    # A combined-gender source zip (e.g. Cricsheet's single "The Hundred" archive, which
    # contains both hnd and hndw matches) relies on this covering ALL_FORMAT_DIRS -- not
    # just the format being targeted -- so already-imported matches of either gender are
    # correctly skipped rather than reprocessed.
    existing = {
        fmt: ({p.stem for p in (raw_root / fmt).glob("**/*.yaml")} if (raw_root / fmt).exists() else set())
        for fmt in ALL_FORMAT_DIRS
    }
    all_existing = set().union(*existing.values())

    with zipfile.ZipFile(zip_buffer) as zf:
        yaml_files = [f for f in zf.namelist() if f.endswith(".yaml")]
        print(f"  Found {len(yaml_files)} YAML files in archive")

        for filename in yaml_files:
            match_id = Path(filename).stem

            if match_id in all_existing:
                stats["skipped_exists"] += 1
                continue

            try:
                yaml_bytes = zf.read(filename)
                classification = classify_match(yaml_bytes)

                if classification is None:
                    stats["skipped"] += 1
                    continue

                fmt = classification["format_dir"]
                target_dir = raw_root / fmt / classification["comp_dir"]
                target_file = target_dir / f"{match_id}.yaml"

                if dry_run:
                    print(f"  [DRY RUN] Would import: {match_id} -> {fmt}/{classification['comp_dir']}")
                else:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_file.write_bytes(yaml_bytes)
                    print(f"  [+] Imported: {match_id} -> {fmt}/{classification['comp_dir']}/")

                stats[f"new_{fmt}"] += 1
                existing[fmt].add(match_id)
                all_existing.add(match_id)

            except Exception as e:
                print(f"  [-] Error processing {filename}: {e}")
                stats["errors"] += 1

    return stats


def import_matches_from_zip_url(
    url: str,
    raw_data_root: str = "data/raw",
    dry_run: bool = False,
) -> dict:
    """
    Downloads a ZIP from Cricsheet and imports any missing matches into data/raw/.
    """
    raw_root = Path(raw_data_root)

    try:
        zip_buffer = download_zip(url)
    except requests.RequestException as e:
        print(f"ERROR: Failed to download from {url}: {e}")
        return _empty_import_stats()

    return _import_matches_from_zip_buffer(zip_buffer, raw_root, dry_run=dry_run)


def import_matches_from_local_zip(
    zip_path: str,
    raw_data_root: str = "data/raw",
    dry_run: bool = False,
) -> dict:
    """
    Imports matches from a ZIP already on disk (no download) -- e.g. a manually-staged
    Cricsheet competition archive in data/raw/female/. Same classify+dedup logic as the
    URL-based importer.
    """
    raw_root = Path(raw_data_root)
    print(f"Importing local archive: {zip_path}")
    with open(zip_path, "rb") as f:
        zip_buffer = io.BytesIO(f.read())
    return _import_matches_from_zip_buffer(zip_buffer, raw_root, dry_run=dry_run)


def import_female_staging_dir(
    staging_dir: str = "data/raw/female",
    raw_data_root: str = "data/raw",
    dry_run: bool = False,
) -> dict:
    """
    Imports every ZIP in a staging folder (default data/raw/female/) -- the manually
    downloaded per-competition women's archives (WBBL, WPL, The Hundred combined
    men's+women's zip, etc.) -- aggregating stats across all of them. Safe to re-run:
    each zip goes through the same match-ID dedup as every other import path.
    """
    staging = Path(staging_dir)
    zips = sorted(staging.glob("*.zip"))
    print(f"\n--- Importing {len(zips)} staged archive(s) from {staging_dir} ---")

    total = _empty_import_stats()
    for zpath in zips:
        stats = import_matches_from_local_zip(str(zpath), raw_data_root=raw_data_root, dry_run=dry_run)
        for k, v in stats.items():
            total[k] += v
    return total


def import_recent_matches(window: str = "2", raw_data_root: str = "data/raw", dry_run: bool = False, gender: str = "male") -> dict:
    """Imports matches added in the last N days (default 2 days for regular automation)."""
    suffix = "male" if gender == "male" else "female"
    url = f"{CRICSHEET_BASE}/recently_added_{window}_{suffix}.zip"
    return import_matches_from_zip_url(url, raw_data_root=raw_data_root, dry_run=dry_run)


def import_catchup(raw_data_root: str = "data/raw", dry_run: bool = False, gender: str = "male") -> dict:
    """
    Catch-up import: Downloads the 30-day recent matches archive to bring the model
    up-to-date from previous weeks.
    """
    suffix = "male" if gender == "male" else "female"
    url = f"{CRICSHEET_BASE}/recently_added_30_{suffix}.zip"
    print(f"\n--- Running 30-Day Catch-Up Import ({suffix}) ---")
    return import_matches_from_zip_url(url, raw_data_root=raw_data_root, dry_run=dry_run)


def import_full_archive(raw_data_root: str = "data/raw", dry_run: bool = False, gender: str = "male") -> dict:
    """
    Full Archive Sync (60+ days / Complete Catch-Up):
    Downloads Cricsheet's complete all_<gender>.zip archive and checks for any missing
    matches across the entire dataset.
    """
    suffix = "male" if gender == "male" else "female"
    url = f"{CRICSHEET_BASE}/all_{suffix}.zip"
    print(f"\n--- Running Full Archive Catch-Up Import (all_{suffix}.zip) ---")
    return import_matches_from_zip_url(url, raw_data_root=raw_data_root, dry_run=dry_run)


def invalidate_caches(formats: list = None):
    """
    Invalidates parquet caches in data/processed to force a rebuild with new data.
    """
    if formats is None:
        formats = list(ACTIVE_FORMATS)

    cache_patterns = [
        "{fmt}_cricket_data.parquet",
        "{fmt}_match_metadata.parquet",
        "{fmt}_elo_ratings.parquet",
        "{fmt}_phase_elo_ratings.parquet",
        "{fmt}_player_elo_ratings.parquet",
        "{fmt}_venue_features.parquet",
        "{fmt}_venue_lookup.parquet",
        "{fmt}_team_venue_dist.parquet",
    ]

    processed = Path("data/processed")
    for fmt in formats:
        for pattern in cache_patterns:
            cache_file = processed / pattern.format(fmt=fmt)
            if cache_file.exists():
                CacheManager.clear_cache(cache_file)
                print(f"  Invalidated cache: {cache_file.name}")


def retrain_lightweight(format_type: str = "t20"):
    """
    Rebuilds delivery data, metadata, Elo ratings, and venue features for the format.
    Updates the data used by PreMatchPredictor in ~30 seconds.
    """
    from src.data_processing.load_data import CricketDataLoader
    from src.data_processing.match_metadata import MatchMetadataExtractor
    from src.features.elo_engine import EloEngine
    from src.features.venue_engine import VenueEngine
    from src.features.venue_history_resolver import build_team_venue_distributions

    print(f"\nRebuilding Elo & Venue features for {format_type.upper()}...")

    loader = CricketDataLoader("data/raw", format_type=format_type)
    df = loader.load_all()

    meta = MatchMetadataExtractor("data/raw", format_type=format_type)
    meta_df = meta.extract_all(force=True)

    elo = EloEngine(
        use_dynamic_k=True,
        use_margin=True,
        use_decay=True,
        use_home_away=False,
        format_type=format_type,
    )
    elo_df = elo.run(meta_df)

    venue = VenueEngine(format_type=format_type)
    venue.clear_cache()
    venue_df = venue.run(meta_df, df)

    # Team venue distributions feed VenueHistoryResolver (live scanner) — must
    # be rebuilt whenever venue_lookup changes, since resolution is keyed off it.
    build_team_venue_distributions(format_type=format_type)

    print(f"Lightweight retrain complete for {format_type.upper()} ({len(meta_df)} matches total).")
    print(f"Latest match date: {meta_df['date'].max()}\n")


def _build_second_innings_pool_states(format_type: str) -> pd.DataFrame:
    """
    Builds second-innings states for `format_type` via the standard cached pipeline, for use as
    pooling training signal in another format's win-probability model (Phase 5/6: hnd pools in
    T20 -- see hundred_winprob_pooling_experiment.py). Mirrors retrain_full()'s own state-building
    steps 1-3 exactly.
    """
    from src.data_processing.load_data import CricketDataLoader
    from src.data_processing.match_metadata import MatchMetadataExtractor
    from src.features.elo_engine import EloEngine
    from src.features.player_elo_engine import PlayerEloEngine
    from src.features.phase_elo_engine import PhaseEloEngine
    from src.features.venue_engine import VenueEngine
    from src.data_processing.build_states import StateBuilder
    from src.data_processing.format_spec import spec

    loader = CricketDataLoader("data/raw", format_type=format_type)
    deliveries = loader.load_all()

    meta = MatchMetadataExtractor("data/raw", format_type=format_type)
    meta_df = meta.extract_all()
    if "gender" in meta_df.columns:
        # Filter to this format's own gender (t20/od/hnd -> male, t20w/odw/hndw -> female) --
        # was hardcoded to "male", which would silently zero out an all-female corpus like t20w.
        meta_df = meta_df[meta_df["gender"] == spec(format_type)["gender"]].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=format_type)
    elo_df = elo_engine.run(meta_df)

    player_elo_engine = PlayerEloEngine(format_type=format_type, form_window=15)
    player_elo_df = player_elo_engine.run(deliveries, elo_df)

    phase_elo_engine = PhaseEloEngine(format_type=format_type)
    phase_elo_df = phase_elo_engine.run(deliveries, elo_df)

    venue_engine = VenueEngine(format_type=format_type)
    venue_df = venue_engine.run(meta_df, deliveries)
    elo_df = elo_df.merge(venue_df, on="match_id", how="left")

    fmt_spec = spec(format_type)
    states = StateBuilder(
        deliveries, elo_df, player_elo_df=player_elo_df, phase_elo_df=phase_elo_df,
        total_overs=fmt_spec["total_overs"], format_type=format_type,
        balls_per_over=fmt_spec["balls_per_over"],
    ).build()

    return states[states["innings"] == 2].copy()


def _build_delivery_outcome_pool_frame(format_type: str) -> pd.DataFrame:
    """
    Builds the delivery-outcome training frame for `format_type` via the standard cached
    pipeline, for use as pooling training signal in another format's delivery-outcome model
    (DeliveryOutcomeModel's DELIVERY_OUTCOME_POOL_SOURCE -- e.g. hnd pools in T20, re-confirmed
    2026-08-12 via tests/hundred_pooling_experiment.py). Mirrors retrain_full()'s own steps 1-2
    exactly, but stops short of state-building (StateBuilder) -- the delivery-outcome model
    trains on raw deliveries, not second-innings states, unlike _build_second_innings_pool_
    states above (that one's for the WIN-PROB pool, a different data shape).
    """
    from src.data_processing.load_data import CricketDataLoader
    from src.data_processing.match_metadata import MatchMetadataExtractor
    from src.features.elo_engine import EloEngine
    from src.features.player_elo_engine import PlayerEloEngine
    from src.features.venue_engine import VenueEngine
    from src.models.delivery_outcome_model import build_training_frame

    loader = CricketDataLoader("data/raw", format_type=format_type)
    deliveries = loader.load_all()

    meta = MatchMetadataExtractor("data/raw", format_type=format_type)
    meta_df = meta.extract_all()
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == spec(format_type)["gender"]].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=format_type)
    elo_df = elo_engine.run(meta_df)

    player_elo_engine = PlayerEloEngine(format_type=format_type, form_window=15)
    player_elo_df = player_elo_engine.run(deliveries, elo_df)

    venue_engine = VenueEngine(format_type=format_type)
    venue_df = venue_engine.run(meta_df, deliveries)

    return build_training_frame(deliveries, player_elo_df, venue_df, format_type=format_type)


def retrain_full(format_type: str = "t20"):
    """
    Full pipeline retrain including XGBoost Win Probability model and First Innings Regressor.
    Re-fits GroupKFold XGBoost classifiers and FirstInningsPredictor regressors on the latest data.
    """
    from src.data_processing.load_data import CricketDataLoader
    from src.data_processing.match_metadata import MatchMetadataExtractor
    from src.features.elo_engine import EloEngine
    from src.features.player_elo_engine import PlayerEloEngine, build_seed_ratings
    from src.features.phase_elo_engine import PhaseEloEngine
    from src.features.venue_engine import VenueEngine
    from src.features.venue_history_resolver import build_team_venue_distributions
    from src.data_processing.build_states import StateBuilder
    from src.models.first_innings_predictor import FirstInningsPredictor
    from src.models.xgb_win_probability import XGBWinProbabilityModel

    print(f"\n{'='*70}")
    print(f"  FULL MODEL RETRAIN: {format_type.upper()} (XGBoost Classifier + Regressor)")
    print(f"{'='*70}\n")

    # 1. Load Data & Metadata
    loader = CricketDataLoader("data/raw", format_type=format_type)
    deliveries = loader.load_all()

    meta = MatchMetadataExtractor("data/raw", format_type=format_type)
    meta_df = meta.extract_all(force=True)

    if "gender" in meta_df.columns:
        # Filter to this format's own gender (t20/od/hnd -> male, t20w/odw/hndw -> female) --
        # was hardcoded to "male", which would silently zero out an all-female corpus like t20w.
        meta_df = meta_df[meta_df["gender"] == spec(format_type)["gender"]].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    # 2. Compute Features
    print(f"[{format_type.upper()}] Computing Elo ratings...")
    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=format_type)
    elo_df = elo_engine.run(meta_df)

    print(f"[{format_type.upper()}] Computing player Elo ratings...")
    seed_ratings = None
    if format_type in SEED_SOURCE:
        # Frozen snapshot, not the live {source}_player_elo_ratings.parquet -- see
        # build_seed_ratings' docstring (t20_player_elo_revamp_plan.md's Phases 1-5 must not
        # silently re-seed hnd while still experimental/revertable; same discipline for hndw).
        seed_ratings = build_seed_ratings(format_type)
    player_elo_engine = PlayerEloEngine(format_type=format_type, form_window=15)
    player_elo_df = player_elo_engine.run(deliveries, elo_df, seed_ratings=seed_ratings)

    print(f"[{format_type.upper()}] Computing phase Elo ratings...")
    phase_elo_engine = PhaseEloEngine(format_type=format_type)
    phase_elo_df = phase_elo_engine.run(deliveries, elo_df)

    print(f"[{format_type.upper()}] Computing Venue features...")
    venue_engine = VenueEngine(format_type=format_type)
    venue_df = venue_engine.run(meta_df, deliveries)

    # Team venue distributions feed VenueHistoryResolver (live scanner) — must
    # be rebuilt whenever venue_lookup changes, since resolution is keyed off it.
    build_team_venue_distributions(format_type=format_type)

    elo_df = elo_df.merge(venue_df, on="match_id", how="left")

    # 3. Build States
    print(f"[{format_type.upper()}] Building game states...")
    fmt_spec = spec(format_type)
    states = StateBuilder(
        deliveries,
        elo_df,
        player_elo_df=player_elo_df,
        phase_elo_df=phase_elo_df,
        total_overs=fmt_spec["total_overs"],
        format_type=format_type,
        balls_per_over=fmt_spec["balls_per_over"],
    ).build()

    # 4. Train First Innings Regressor
    reg_path = Path("data/processed") / f"{format_type}_first_innings_regressor.pkl"
    print(f"[{format_type.upper()}] Fitting First Innings Regressor...")
    reg_model = FirstInningsPredictor(format_type=format_type)
    reg_model.fit(states, cache_path=reg_path)
    if format_type == "t20":
        shutil.copyfile(reg_path, Path("data/processed/first_innings_regressor.pkl"))

    # 5. Train XGBoost Win Probability Model
    second_innings = states[states["innings"] == 2].copy()
    xgb_path = Path("data/processed") / f"{format_type}_xgb_models.pkl"
    print(f"[{format_type.upper()}] Fitting 5-Fold XGBoost Win Probability Classifier...")
    xgb_model = XGBWinProbabilityModel(format_type=format_type)

    # Phase 5/6: pool POOL_SOURCE[format_type] in as extra (never held-out, never-scored)
    # training signal for a thin corpus -- tests/hundred_winprob_pooling_experiment.py showed
    # this beats the hnd-only model on every held-out phase; hndw mirrors the same mechanism
    # pooling t20w. t20/od/t20w/odw never reach this branch, so their retrain is unaffected.
    pool_X, pool_y = None, None
    pool_source = POOL_SOURCE.get(format_type)
    if pool_source:
        print(f"[{format_type.upper()}] Building {pool_source.upper()} pool for pooled win-probability training...")
        pool_states = _build_second_innings_pool_states(pool_source)
        pool_helper = XGBWinProbabilityModel(format_type=pool_source)
        pool_X, pool_y = pool_helper._extract_features_and_target(pool_states)
        pool_X = pool_X.copy()
        pool_X["is_hundred"] = 0.0

    xgb_model.fit(second_innings, cache_path=xgb_path, pool_X=pool_X, pool_y=pool_y)
    if format_type == "t20":
        shutil.copyfile(xgb_path, Path("data/processed/xgb_models.pkl"))

    # 6. Train Delivery Outcome Model (Strategy-4 Simulator Engine)
    from src.models.delivery_outcome_model import (
        DeliveryOutcomeModel, build_training_frame, DELIVERY_OUTCOME_POOL_SOURCE,
    )
    dom_path = Path("data/processed") / f"{format_type}_delivery_outcome_model.pkl"
    print(f"[{format_type.upper()}] Fitting Delivery Outcome Model...")
    training_frame = build_training_frame(deliveries, player_elo_df, venue_df, format_type=format_type)

    # DELIVERY_OUTCOME_POOL_SOURCE-gated (currently hnd<-t20 only, re-confirmed 2026-08-12) --
    # separate from step 5's win-prob POOL_SOURCE gate, since each pooling decision here needs
    # its own evidence (tests/hundred_pooling_experiment.py), not an assumption it transfers
    # from a sibling format's win-prob pooling result. t20/od/t20w/odw never reach this branch.
    dom_pool_frame = None
    dom_pool_source = DELIVERY_OUTCOME_POOL_SOURCE.get(format_type)
    if dom_pool_source:
        print(f"[{format_type.upper()}] Building {dom_pool_source.upper()} pool for pooled delivery-outcome training...")
        dom_pool_frame = _build_delivery_outcome_pool_frame(dom_pool_source)

    dom_model = DeliveryOutcomeModel(format_type=format_type)
    dom_model.fit(training_frame, cache_path=dom_path, pool_frame=dom_pool_frame)
    if format_type == "t20":
        shutil.copyfile(dom_path, Path("data/processed/delivery_outcome_model.pkl"))

    print(f"\n[OK] Full retrain complete for {format_type.upper()}! Saved model artifacts to {xgb_path} and {dom_path}")


def log_import(stats: dict, log_path: str = "data/betting/import_log.csv"):
    """Logs import run metrics to CSV for auditability."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    row = {
        "timestamp": datetime.now().isoformat(),
        "new_t20": stats.get("new_t20", 0),
        "new_od": stats.get("new_od", 0),
        "new_hnd": stats.get("new_hnd", 0),
        "new_t20w": stats.get("new_t20w", 0),
        "new_odw": stats.get("new_odw", 0),
        "new_hndw": stats.get("new_hndw", 0),
        "skipped": stats.get("skipped", 0),
        "skipped_exists": stats.get("skipped_exists", 0),
        "errors": stats.get("errors", 0),
        "full_retrain": stats.get("full_retrain", False),
    }

    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not file_exists, index=False)
    print(f"Import log updated: {path}")


def validate_import(format_type: str = "t20"):
    """Performs post-import sanity checks on raw dataset."""
    from src.data_processing.load_data import CricketDataLoader

    loader = CricketDataLoader("data/raw", format_type=format_type)
    yaml_files = list(loader.data_path.glob("**/*.yaml"))

    print(f"Validation ({format_type.upper()}):")
    print(f"  Total YAML files in raw dataset: {len(yaml_files)}")

    stems = [f.stem for f in yaml_files]
    dupes = len(stems) - len(set(stems))
    if dupes > 0:
        print(f"  [!] WARNING: {dupes} duplicate match IDs found across subdirectories!")
    else:
        print(f"  [OK] Zero duplicate match IDs")
