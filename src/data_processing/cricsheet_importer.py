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
    Or None if unparseable, female, or unsupported match type (Tests, Multi-day).
    """
    try:
        match = yaml.load(yaml_bytes, Loader=SafeLoader)
    except Exception:
        return None

    if not match or "info" not in match:
        return None

    info = match["info"]
    gender = info.get("gender", "male")
    if gender != "male":
        return None

    match_type = str(info.get("match_type", "")).upper()
    competition = info.get("competition", "")
    balls_per_over = info.get("balls_per_over")

    # Determine format directory
    if (
        match_type in ("HND", "HUNDRED", "100")
        or competition in HND_COMPETITIONS
        or "The Hundred" in competition
        or "Hundred" in competition
        or balls_per_over == 5
    ):
        format_dir = "hnd"
        comp_dir = HND_COMPETITIONS.get(competition, "hnd")
    elif match_type in ("T20", "IT20", "T20I"):
        format_dir = "t20"
        comp_dir = T20_COMPETITIONS.get(competition, "t20s")
    elif match_type in ("ODI", "ODM", "OD"):
        format_dir = "od"
        comp_dir = OD_COMPETITIONS.get(competition, "odis")
    else:
        return None  # Skip Tests, multi-day, etc.

    return {
        "match_type": match_type,
        "competition": competition,
        "gender": gender,
        "format_dir": format_dir,
        "comp_dir": comp_dir,
    }


def import_matches_from_zip_url(
    url: str,
    raw_data_root: str = "data/raw",
    dry_run: bool = False,
) -> dict:
    """
    Downloads a ZIP from Cricsheet and imports any missing matches into data/raw/.
    """
    raw_root = Path(raw_data_root)
    stats = {"new_t20": 0, "new_od": 0, "new_hnd": 0, "skipped": 0, "skipped_exists": 0, "errors": 0}

    try:
        zip_buffer = download_zip(url)
    except requests.RequestException as e:
        print(f"ERROR: Failed to download from {url}: {e}")
        return stats

    # Build set of existing match IDs across raw directory
    existing_t20 = {p.stem for p in (raw_root / "t20").glob("**/*.yaml")} if (raw_root / "t20").exists() else set()
    existing_od = {p.stem for p in (raw_root / "od").glob("**/*.yaml")} if (raw_root / "od").exists() else set()
    existing_hnd = {p.stem for p in (raw_root / "hnd").glob("**/*.yaml")} if (raw_root / "hnd").exists() else set()
    all_existing = existing_t20.union(existing_od).union(existing_hnd)

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

                target_dir = raw_root / classification["format_dir"] / classification["comp_dir"]
                target_file = target_dir / f"{match_id}.yaml"

                if dry_run:
                    print(f"  [DRY RUN] Would import: {match_id} -> {classification['format_dir']}/{classification['comp_dir']}")
                else:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_file.write_bytes(yaml_bytes)
                    print(f"  [+] Imported: {match_id} -> {classification['format_dir']}/{classification['comp_dir']}/")

                if classification["format_dir"] == "t20":
                    stats["new_t20"] += 1
                    existing_t20.add(match_id)
                elif classification["format_dir"] == "od":
                    stats["new_od"] += 1
                    existing_od.add(match_id)
                elif classification["format_dir"] == "hnd":
                    stats["new_hnd"] += 1
                    existing_hnd.add(match_id)

                all_existing.add(match_id)

            except Exception as e:
                print(f"  [-] Error processing {filename}: {e}")
                stats["errors"] += 1

    return stats


def import_recent_matches(window: str = "2", raw_data_root: str = "data/raw", dry_run: bool = False) -> dict:
    """Imports matches added in the last N days (default 2 days for regular automation)."""
    url = f"{CRICSHEET_BASE}/recently_added_{window}_male.zip"
    return import_matches_from_zip_url(url, raw_data_root=raw_data_root, dry_run=dry_run)


def import_catchup(raw_data_root: str = "data/raw", dry_run: bool = False) -> dict:
    """
    Catch-up import: Downloads the 30-day recent matches archive to bring the model
    up-to-date from previous weeks.
    """
    url = f"{CRICSHEET_BASE}/recently_added_30_male.zip"
    print(f"\n--- Running 30-Day Catch-Up Import ---")
    return import_matches_from_zip_url(url, raw_data_root=raw_data_root, dry_run=dry_run)


def import_full_archive(raw_data_root: str = "data/raw", dry_run: bool = False) -> dict:
    """
    Full Archive Sync (60+ days / Complete Catch-Up):
    Downloads Cricsheet's complete all_male.zip archive (~17,800+ matches)
    and checks for any missing matches across the entire dataset.
    """
    url = f"{CRICSHEET_BASE}/all_male.zip"
    print(f"\n--- Running Full Archive Catch-Up Import (all_male.zip) ---")
    return import_matches_from_zip_url(url, raw_data_root=raw_data_root, dry_run=dry_run)


def invalidate_caches(formats: list = None):
    """
    Invalidates parquet caches in data/processed to force a rebuild with new data.
    """
    if formats is None:
        formats = ["t20", "od", "hnd"]

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
        meta_df = meta_df[meta_df["gender"] == "male"].copy()
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


def retrain_full(format_type: str = "t20"):
    """
    Full pipeline retrain including XGBoost Win Probability model and First Innings Regressor.
    Re-fits GroupKFold XGBoost classifiers and FirstInningsPredictor regressors on the latest data.
    """
    from src.data_processing.load_data import CricketDataLoader
    from src.data_processing.match_metadata import MatchMetadataExtractor
    from src.features.elo_engine import EloEngine
    from src.features.player_elo_engine import PlayerEloEngine
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
        meta_df = meta_df[meta_df["gender"] == "male"].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    # 2. Compute Features
    print(f"[{format_type.upper()}] Computing Elo ratings...")
    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=format_type)
    elo_df = elo_engine.run(meta_df)

    print(f"[{format_type.upper()}] Computing player Elo ratings...")
    seed_ratings = None
    if format_type == "hnd":
        t20_pe_path = Path("data/processed/t20_player_elo_ratings.parquet")
        if t20_pe_path.exists():
            t20_pe = pd.read_parquet(t20_pe_path)
            latest_t20 = t20_pe.sort_values("match_id").groupby(["player", "role"]).last().reset_index()
            b_seeds = latest_t20[latest_t20["role"] == "batter"].set_index("player")["effective_elo"].to_dict()
            w_seeds = latest_t20[latest_t20["role"] == "bowler"].set_index("player")["effective_elo"].to_dict()
            seed_ratings = {"batter": b_seeds, "bowler": w_seeds}
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
    from src.data_processing.format_spec import spec
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

    # Phase 5/6: pool T20 in as extra (never held-out, never-scored) training signal for hnd --
    # tests/hundred_winprob_pooling_experiment.py showed this beats the hnd-only model on every
    # held-out phase. t20/od never reach this branch, so their retrain is unaffected.
    pool_X, pool_y = None, None
    if format_type == "hnd":
        print("[HND] Building T20 pool for pooled win-probability training...")
        t20_pool_states = _build_second_innings_pool_states("t20")
        t20_pool_helper = XGBWinProbabilityModel(format_type="t20")
        pool_X, pool_y = t20_pool_helper._extract_features_and_target(t20_pool_states)
        pool_X = pool_X.copy()
        pool_X["is_hundred"] = 0.0

    xgb_model.fit(second_innings, cache_path=xgb_path, pool_X=pool_X, pool_y=pool_y)
    if format_type == "t20":
        shutil.copyfile(xgb_path, Path("data/processed/xgb_models.pkl"))

    # 6. Train Delivery Outcome Model (Strategy-4 Simulator Engine)
    from src.models.delivery_outcome_model import DeliveryOutcomeModel, build_training_frame
    dom_path = Path("data/processed") / f"{format_type}_delivery_outcome_model.pkl"
    print(f"[{format_type.upper()}] Fitting Delivery Outcome Model...")
    training_frame = build_training_frame(deliveries, player_elo_df, venue_df, format_type=format_type)
    dom_model = DeliveryOutcomeModel(format_type=format_type)
    dom_model.fit(training_frame, cache_path=dom_path)
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
