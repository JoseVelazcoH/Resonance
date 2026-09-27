"""English display labels for emotion, mood, and situation taxonomy ids.

Loads `mood_dj/data/labels_en.json`, a file kept deliberately separate from
`emotions.json` and `moods.json` so that adding or fixing an English label
never changes `mood_dj.adapters.laya_track_profiler.compute_version` (which
hashes `moods.json`), and therefore never forces a re-profiling pass over the
whole library.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def load_labels_en() -> dict[str, dict[str, str]]:
    """Load the id -> English label maps for emotions, moods, and situations."""

    package = "mood_dj.data"
    with resources.files(package).joinpath("labels_en.json").open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {
        "emotions": raw["emotions"],
        "moods": raw["moods"],
        "situations": raw["situations"],
    }


def english_emotion_label(emotion_id: str, fallback: str) -> str:
    """English label for a leaf emotion id, or `fallback` (the Spanish label) if missing."""

    return load_labels_en()["emotions"].get(emotion_id, fallback)


def english_mood_label(mood_id: str, fallback: str) -> str:
    """English label for a mood id, or `fallback` (the Spanish label) if missing."""

    return load_labels_en()["moods"].get(mood_id, fallback)


def english_situation_label(situation_id: str, fallback: str) -> str:
    """English label for a situation id, or `fallback` (the Spanish label) if missing."""

    return load_labels_en()["situations"].get(situation_id, fallback)
