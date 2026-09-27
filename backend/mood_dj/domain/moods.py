"""The flat 7-mood taxonomy used for TRACK-level lyrics profiling.

Loads `mood_dj/data/moods.json`: pure domain module, no framework or I/O
dependencies beyond reading the packaged JSON data file. See that file's
`_meta` for why tracks use a flat mood instead of the deep emotion tree in
`mood_dj.domain.taxonomy` (which prompts still use).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from mood_dj.domain.taxonomy import EmotionTree


@dataclass(frozen=True)
class Mood:
    """One of the 7 flat moods a track (or a prompt's detected family) resolves to."""

    id: str
    label: str
    criterion: str
    valence: float
    arousal: float
    polarity: str
    families: tuple[str, ...]


@dataclass(frozen=True)
class MoodCatalog:
    """All 7 moods, plus the family -> mood lookup built from them."""

    meta: dict
    moods: tuple[Mood, ...]

    def by_id(self, mood_id: str) -> Mood | None:
        for mood in self.moods:
            if mood.id == mood_id:
                return mood
        return None

    def family_to_mood_id(self, family_id: str) -> str | None:
        for mood in self.moods:
            if family_id in mood.families:
                return mood.id
        return None


def _read_json(filename: str) -> dict:
    package = "mood_dj.data"
    with resources.files(package).joinpath(filename).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_mood_catalog(raw: dict) -> MoodCatalog:
    """Build a `MoodCatalog` from an already-parsed moods JSON document."""

    moods = tuple(
        Mood(
            id=m["id"],
            label=m["label"],
            criterion=m["criterion"],
            valence=m["valence"],
            arousal=m["arousal"],
            polarity=m["polarity"],
            families=tuple(m["families"]),
        )
        for m in raw["moods"]
    )
    return MoodCatalog(meta=raw.get("_meta", {}), moods=moods)


@lru_cache(maxsize=1)
def load_moods() -> MoodCatalog:
    """Load and parse `moods.json` into a `MoodCatalog`."""

    raw = _read_json("moods.json")
    return parse_mood_catalog(raw)


def family_to_mood(catalog: MoodCatalog, family_id: str) -> str | None:
    """Resolve an `emotions.json` family id to the mood id it maps to, or None."""

    return catalog.family_to_mood_id(family_id)


def validate_families_cover_all_moods(catalog: MoodCatalog, tree: EmotionTree) -> list[str]:
    """Return a list of problems: families in `tree` with no mood, or mapped twice.

    Empty list means every family in `tree` maps to exactly one mood in `catalog`.
    Used by `tests/unit/test_moods.py` to keep `moods.json` in sync with
    `emotions.json` as either file evolves.
    """

    problems: list[str] = []
    all_family_ids = [
        family.id for polarity in tree.polarities for cluster in polarity.clusters for family in cluster.families
    ]

    seen_in_moods: dict[str, list[str]] = {}
    for mood in catalog.moods:
        for family_id in mood.families:
            seen_in_moods.setdefault(family_id, []).append(mood.id)

    for family_id in all_family_ids:
        mapped_to = seen_in_moods.get(family_id, [])
        if not mapped_to:
            problems.append(f"family {family_id!r} is not mapped to any mood")
        elif len(mapped_to) > 1:
            problems.append(f"family {family_id!r} is mapped to more than one mood: {mapped_to}")

    known_family_ids = set(all_family_ids)
    for family_id in seen_in_moods:
        if family_id not in known_family_ids:
            problems.append(f"moods.json references unknown family {family_id!r}")

    return problems
