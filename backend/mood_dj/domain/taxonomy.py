"""Emotion and situation taxonomy for Laya's calibration choice questions.

Loads the static taxonomy trees from `mood_dj/data/emotions.json` and
`mood_dj/data/situations.json`. Pure domain module: no framework or I/O
dependencies beyond reading the packaged JSON data files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

MAX_CHILDREN_PER_NODE = 10


@dataclass(frozen=True)
class Emotion:
    """A single leaf emotion with its circumplex coordinates."""

    id: str
    label: str
    valence: float
    arousal: float


@dataclass(frozen=True)
class EmotionFamily:
    """A group of closely related emotions sharing a circumplex centroid."""

    id: str
    description: str
    centroid_valence: float
    centroid_arousal: float
    emotions: tuple[Emotion, ...]


@dataclass(frozen=True)
class EmotionCluster:
    """A group of families under one polarity."""

    id: str
    description: str
    centroid_valence: float
    centroid_arousal: float
    families: tuple[EmotionFamily, ...]


@dataclass(frozen=True)
class EmotionPolarity:
    """The top-level branch of the emotion tree (positive/negative/ambivalent).

    `description` defaults to "" because the packaged `emotions.json` carries
    no polarity-level description (only bare ids); a candidate taxonomy (see
    `scripts/eval_mood.py` / `eval/emotions.candidate.json`) can add one with
    concrete bilingual cues, which the profilers prefer over the generic
    "mostly {id} polarity" fallback question wording when present.
    """

    id: str
    centroid_valence: float
    centroid_arousal: float
    clusters: tuple[EmotionCluster, ...]
    description: str = ""


@dataclass(frozen=True)
class EmotionTree:
    """The full emotion taxonomy, rooted above the polarity level."""

    meta: dict
    polarities: tuple[EmotionPolarity, ...]


@dataclass(frozen=True)
class Situation:
    """A listening context used to calibrate a Laya recommendation session."""

    id: str
    label: str
    description: str
    valence: float
    arousal: float
    related_emotions: tuple[str, ...]


@dataclass(frozen=True)
class SituationGroup:
    """A group of situations (kept at or under the max-children calibration limit)."""

    id: str
    situations: tuple[Situation, ...]


@dataclass(frozen=True)
class SituationCatalog:
    """All situations, split into calibrated choice groups."""

    meta: dict
    groups: tuple[SituationGroup, ...]


def _read_json(filename: str) -> dict:
    package = "mood_dj.data"
    with resources.files(package).joinpath(filename).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_emotion(raw: dict) -> Emotion:
    return Emotion(
        id=raw["id"],
        label=raw["label"],
        valence=raw["valence"],
        arousal=raw["arousal"],
    )


def _build_family(family_id: str, raw: dict) -> EmotionFamily:
    return EmotionFamily(
        id=family_id,
        description=raw["description"],
        centroid_valence=raw["centroid"]["valence"],
        centroid_arousal=raw["centroid"]["arousal"],
        emotions=tuple(_build_emotion(e) for e in raw["emotions"]),
    )


def _build_cluster(cluster_id: str, raw: dict) -> EmotionCluster:
    families = tuple(
        _build_family(family_id, family_raw)
        for family_id, family_raw in raw["families"].items()
    )
    return EmotionCluster(
        id=cluster_id,
        description=raw["description"],
        centroid_valence=raw["centroid"]["valence"],
        centroid_arousal=raw["centroid"]["arousal"],
        families=families,
    )


def _build_polarity(polarity_id: str, raw: dict) -> EmotionPolarity:
    clusters = tuple(
        _build_cluster(cluster_id, cluster_raw)
        for cluster_id, cluster_raw in raw["clusters"].items()
    )
    return EmotionPolarity(
        id=polarity_id,
        centroid_valence=raw["centroid"]["valence"],
        centroid_arousal=raw["centroid"]["arousal"],
        clusters=clusters,
        description=raw.get("description", ""),
    )


def parse_emotion_tree(raw: dict) -> EmotionTree:
    """Build an `EmotionTree` from an already-parsed emotions JSON document.

    Exposed separately from `load_emotion_tree` so callers (e.g. the eval
    harness in `scripts/eval_mood.py`) can build a tree from an arbitrary
    candidate taxonomy file without touching the packaged `emotions.json` or
    the cached loader below.
    """

    polarities = tuple(
        _build_polarity(polarity_id, polarity_raw)
        for polarity_id, polarity_raw in raw["polarities"].items()
    )
    return EmotionTree(meta=raw["_meta"], polarities=polarities)


@lru_cache(maxsize=1)
def load_emotion_tree() -> EmotionTree:
    """Load and parse `emotions.json` into an `EmotionTree`."""

    raw = _read_json("emotions.json")
    return parse_emotion_tree(raw)


def load_emotion_tree_from_path(path: str) -> EmotionTree:
    """Load and parse an emotions JSON document from an arbitrary file path.

    Used by the eval harness to evaluate a candidate taxonomy (e.g.
    `backend/eval/emotions.candidate.json`) without editing the packaged
    `mood_dj/data/emotions.json` (which would bump `compute_version` and
    invalidate every cached track profile).
    """

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return parse_emotion_tree(raw)


@lru_cache(maxsize=1)
def load_situations() -> SituationCatalog:
    """Load and parse `situations.json` into a `SituationCatalog`."""

    raw = _read_json("situations.json")
    groups = tuple(
        SituationGroup(
            id=group_id,
            situations=tuple(
                Situation(
                    id=s["id"],
                    label=s["label"],
                    description=s["description"],
                    valence=s["valence"],
                    arousal=s["arousal"],
                    related_emotions=tuple(s["related_emotions"]),
                )
                for s in situations_raw
            ),
        )
        for group_id, situations_raw in raw["groups"].items()
    )
    return SituationCatalog(meta=raw["_meta"], groups=groups)


def iter_all_emotions(tree: EmotionTree) -> tuple[Emotion, ...]:
    """Flatten the tree into every leaf `Emotion`, in tree order."""

    emotions: list[Emotion] = []
    for polarity in tree.polarities:
        for cluster in polarity.clusters:
            for fam in cluster.families:
                emotions.extend(fam.emotions)
    return tuple(emotions)


def find_emotion(tree: EmotionTree, emotion_id: str) -> Emotion | None:
    """Find an emotion by its ascii id, or None if not present."""

    for emotion in iter_all_emotions(tree):
        if emotion.id == emotion_id:
            return emotion
    return None


def emotion_path(tree: EmotionTree, emotion_id: str) -> tuple[str, ...] | None:
    """Return the (polarity_id, cluster_id, family_id, emotion_id) path, or None."""

    for polarity in tree.polarities:
        for cluster in polarity.clusters:
            for fam in cluster.families:
                for emotion in fam.emotions:
                    if emotion.id == emotion_id:
                        return (polarity.id, cluster.id, fam.id, emotion.id)
    return None


def list_children(node: EmotionTree | EmotionPolarity | EmotionCluster | EmotionFamily) -> tuple:
    """List the direct children of a taxonomy node, whatever level it is."""

    if isinstance(node, EmotionTree):
        return node.polarities
    if isinstance(node, EmotionPolarity):
        return node.clusters
    if isinstance(node, EmotionCluster):
        return node.families
    if isinstance(node, EmotionFamily):
        return node.emotions
    raise TypeError(f"Unsupported taxonomy node type: {type(node)!r}")


def related_families_and_clusters(
    tree: EmotionTree, related_emotion_ids: tuple[str, ...]
) -> tuple[frozenset[str], frozenset[str]]:
    """Map a situation's `related_emotions` ids to the families/clusters they live in.

    Tracks no longer carry a situation pick (see `laya_track_profiler.py`, v3),
    so ranking (`mood_dj.domain.mood_selection_policy`) matches a target situation's
    related emotions against a track's own family/cluster instead. Unknown
    emotion ids are silently skipped.
    """

    families: set[str] = set()
    clusters: set[str] = set()
    for emotion_id in related_emotion_ids:
        path = emotion_path(tree, emotion_id)
        if path is None:
            continue
        _polarity_id, cluster_id, family_id, _emotion_id = path
        families.add(family_id)
        clusters.add(cluster_id)
    return frozenset(families), frozenset(clusters)


def iter_all_situations(catalog: SituationCatalog) -> tuple[Situation, ...]:
    """Flatten the situation catalog into every `Situation`, in group order."""

    situations: list[Situation] = []
    for group in catalog.groups:
        situations.extend(group.situations)
    return tuple(situations)
