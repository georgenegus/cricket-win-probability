"""
Review queue for venue strings/fixtures that could not be confidently resolved.

Ambiguous or below-threshold cases are recorded here instead of being silently
discarded. An entry stays at the format baseline on every occurrence until a
human reviews it and either fixes it (adds an alias/override entry) or marks
it "baseline" (genuinely unresolvable, stop flagging it). There is no
time-based expiry — the queue is a standing to-do list, not a timer.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_LOCK = Lock()

DEFAULT_QUEUE_PATH = Path("data/reference/venue_review_queue.json")

MAX_CONTEXTS_PER_ENTRY = 20


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(queue_path: Path) -> dict:
    if not queue_path.exists():
        return {}
    try:
        with open(queue_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(queue_path: Path, data: dict) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queue_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _key(raw_identifier: str, format_type: str) -> str:
    return f"{format_type}::{str(raw_identifier).strip().lower()}"


def queue_ambiguous_venue(
    raw_identifier: str,
    format_type: str,
    reason: str,
    candidates: list | None = None,
    context: dict | None = None,
    queue_path: Path = DEFAULT_QUEUE_PATH,
) -> None:
    """
    Records an unresolved venue string or fixture for manual review. Safe to
    call repeatedly for the same identifier — occurrences are counted, not
    duplicated. Entries a reviewer has already marked "baseline" are left
    untouched (they keep baselining silently and are not re-flagged).

    raw_identifier: the raw venue string, or a fixture key (e.g. "Team A v Team B")
                    for cases where the ambiguity is between two teams' venues
                    rather than a single unresolved string.
    """
    if not raw_identifier or not str(raw_identifier).strip():
        return

    key = _key(raw_identifier, format_type)

    with _LOCK:
        data = _load(queue_path)
        entry = data.get(key)

        if entry is None:
            data[key] = {
                "raw_identifier": str(raw_identifier).strip(),
                "format": format_type,
                "reason": reason,
                "status": "pending",
                "candidates": candidates or [],
                "occurrences": 1,
                "first_seen": _now_iso(),
                "last_seen": _now_iso(),
                "contexts": [context] if context else [],
            }
        else:
            entry["occurrences"] = entry.get("occurrences", 0) + 1
            entry["last_seen"] = _now_iso()
            if candidates:
                entry["candidates"] = candidates
            if entry.get("status") != "baseline" and context:
                contexts = entry.setdefault("contexts", [])
                if len(contexts) < MAX_CONTEXTS_PER_ENTRY:
                    contexts.append(context)

        _save(queue_path, data)


def load_queue(queue_path: Path = DEFAULT_QUEUE_PATH) -> dict:
    return _load(queue_path)


def is_actioned(raw_identifier: str, format_type: str, queue_path: Path = DEFAULT_QUEUE_PATH) -> bool:
    """True if a human has already marked this entry as genuinely baseline."""
    data = _load(queue_path)
    entry = data.get(_key(raw_identifier, format_type))
    return bool(entry and entry.get("status") == "baseline")
