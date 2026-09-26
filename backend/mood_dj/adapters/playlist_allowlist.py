"""Local, untracked playlist allowlist for the library-build proof of concept.

Reads `playlists.local.json` (gitignored, never committed) if present, and restricts
library building to playlists whose names match one of the listed names, compared
case-insensitively and accent-insensitively. Names are never hardcoded here: the
file is user-provided and read at runtime only.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

from mood_dj.domain.models import PlaylistSummary

logger = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return stripped.casefold().strip()


def load_playlist_allowlist(path: str) -> list[str] | None:
    """Return the configured allowlist names, or None if absent/empty."""
    file_path = Path(path)
    if not file_path.exists():
        return None

    data = json.loads(file_path.read_text())
    names = data.get("playlist_allowlist") or []
    if not names:
        return None
    return list(names)


def filter_playlists_by_allowlist(
    playlists: list[PlaylistSummary], allowlist_names: list[str] | None
) -> tuple[list[PlaylistSummary], list[str]]:
    """Return (matching playlists, allowlist names that matched nothing).

    When `allowlist_names` is None, all playlists pass through unchanged.
    """
    if allowlist_names is None:
        return list(playlists), []

    normalized_targets = {_normalize(name): name for name in allowlist_names}
    matched_targets: set[str] = set()
    filtered: list[PlaylistSummary] = []

    for playlist in playlists:
        key = _normalize(playlist.name)
        if key in normalized_targets:
            filtered.append(playlist)
            matched_targets.add(key)

    not_found = [original for key, original in normalized_targets.items() if key not in matched_targets]
    if not_found:
        logger.warning("Playlist allowlist names not found among readable playlists: %s", not_found)

    return filtered, not_found
