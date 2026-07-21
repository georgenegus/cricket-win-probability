import hashlib
import json
import os
from pathlib import Path
from typing import List, Union


class CacheManager:
    """
    Manages cache integrity for parquet and pickle files in data/processed/.
    Tracks SHA-256 hashes of source code dependencies in .cache_manifest.json.
    Automatically invalidates stale cache files when source code changes.
    """

    MANIFEST_FILE = Path("data/processed/.cache_manifest.json")

    @classmethod
    def compute_source_hash(cls, source_files: List[Union[str, Path]]) -> str:
        hasher = hashlib.sha256()
        for sf in sorted([Path(f).resolve() for f in source_files]):
            if sf.exists():
                hasher.update(sf.name.encode("utf-8"))
                with open(sf, "rb") as f:
                    hasher.update(f.read())
        return hasher.hexdigest()

    @classmethod
    def _load_manifest(cls) -> dict:
        if cls.MANIFEST_FILE.exists():
            try:
                with open(cls.MANIFEST_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[CACHE MANAGER] Warning: Could not read manifest {cls.MANIFEST_FILE}: {e}")
        return {}

    @classmethod
    def _save_manifest(cls, manifest: dict):
        cls.MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(cls.MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def is_cache_valid(cls, cache_path: Union[str, Path], source_files: List[Union[str, Path]]) -> bool:
        cache_path = Path(cache_path)
        if not cache_path.exists():
            return False

        manifest = cls._load_manifest()
        key = cache_path.name
        cached_info = manifest.get(key, {})
        recorded_hash = cached_info.get("source_hash", None)

        if recorded_hash is None:
            # Unregistered or pre-existing cache file without hash recording
            print(f"[CACHE NOTICE] Registering baseline hash for pre-existing cache: {cache_path.name}")
            cls.register_cache(cache_path, source_files)
            return True

        current_hash = cls.compute_source_hash(source_files)
        if recorded_hash != current_hash:
            print(f"[CACHE STALE] Invalidating {cache_path.name} due to source code changes.")
            cls.clear_cache(cache_path)
            return False

        return True

    @classmethod
    def register_cache(cls, cache_path: Union[str, Path], source_files: List[Union[str, Path]]):
        cache_path = Path(cache_path)
        manifest = cls._load_manifest()
        current_hash = cls.compute_source_hash(source_files)

        manifest[cache_path.name] = {
            "source_hash": current_hash,
            "source_files": [str(Path(sf).name) for sf in source_files],
            "file_size_bytes": cache_path.stat().st_size if cache_path.exists() else 0
        }
        cls._save_manifest(manifest)

    @classmethod
    def clear_cache(cls, cache_path: Union[str, Path]):
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                cache_path.unlink()
                print(f"[CACHE CLEARED] Deleted {cache_path.name}")
            except Exception as e:
                print(f"[CACHE ERROR] Failed to delete {cache_path.name}: {e}")

        manifest = cls._load_manifest()
        if cache_path.name in manifest:
            del manifest[cache_path.name]
            cls._save_manifest(manifest)
