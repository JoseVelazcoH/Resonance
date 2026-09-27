"""Pure predicate for recognizing playlists this app itself created.

Used in two places: filtering the app's own saved playlists out of the library
build and out of the playlist grid, so recommendations never feed on their own
output. The app was renamed from Moodify to Resonance; playlists saved under the
old name are still recognized.
"""

from __future__ import annotations

APP_DESCRIPTION_MARKER = "Created by Resonance"

_KNOWN_DESCRIPTION_MARKERS = (APP_DESCRIPTION_MARKER, "Created by Moodify")

# Covers saves named "<Brand> - <mood>" or "<Brand> · <mood>" under both brands.
_KNOWN_NAME_PREFIXES = ("resonance", "moodify")
_NAME_SEPARATORS = ("-", "·")


def is_app_created(name: str, description: str | None) -> bool:
    """True when this playlist looks like one this app created."""
    if description and any(marker in description for marker in _KNOWN_DESCRIPTION_MARKERS):
        return True

    normalized_name = (name or "").strip().lower()
    for prefix in _KNOWN_NAME_PREFIXES:
        if normalized_name.startswith(prefix):
            remainder = normalized_name[len(prefix) :].lstrip()
            if any(remainder.startswith(separator) for separator in _NAME_SEPARATORS):
                return True
    return False
