"""Tests for the emotion/situation taxonomy loader (mood_dj.domain.taxonomy)."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest

from mood_dj.domain.taxonomy import (
    MAX_CHILDREN_PER_NODE,
    EmotionCluster,
    EmotionFamily,
    EmotionPolarity,
    EmotionTree,
    find_emotion,
    emotion_path,
    iter_all_emotions,
    iter_all_situations,
    list_children,
    load_emotion_tree,
    load_situations,
)

RAW_SOURCE_PATH = (
    Path(__file__).parents[2]
    / "mood_dj"
    / "data"
    / "emotions.json"
)

FIXES = {
    "Hastió": "Hastío",
    "Apreciacion": "Apreciación",
    "Autonomia": "Autonomía",
    "Fasticio": "Fastidio",
    "Lastima": "Lástima",
    "Codescendencia": "Condescendencia",
}


def _load_expected_labels() -> set[str]:
    """Re-derive the expected, deduplicated Spanish label set from the user's raw list."""

    raw_path = Path(__file__).parents[1] / "fixtures" / "emotions_raw.txt"
    text = raw_path.read_text(encoding="utf-8")
    labels: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fixed = FIXES.get(line, line)
        labels.add(fixed)
    return labels


def _slugify(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", label)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.lower().replace(" ", "_")


class TestEmotionTreeContent:
    def test_every_raw_emotion_present_exactly_once(self):
        tree = load_emotion_tree()
        emotions = iter_all_emotions(tree)
        labels = [e.label for e in emotions]

        expected = _load_expected_labels()
        actual = set(labels)

        assert actual == expected, (
            f"missing={expected - actual} extra={actual - expected}"
        )
        # exactly once each
        assert len(labels) == len(set(labels))

    def test_emotion_ids_are_unique(self):
        tree = load_emotion_tree()
        ids = [e.id for e in iter_all_emotions(tree)]
        assert len(ids) == len(set(ids))

    def test_emotion_ids_are_ascii_slugs(self):
        tree = load_emotion_tree()
        for emotion in iter_all_emotions(tree):
            assert re.fullmatch(r"[a-z0-9_]+", emotion.id), emotion.id
            assert emotion.id.encode("ascii")

    def test_coordinates_in_range(self):
        tree = load_emotion_tree()
        for emotion in iter_all_emotions(tree):
            assert -1.0 <= emotion.valence <= 1.0
            assert -1.0 <= emotion.arousal <= 1.0
        for polarity in tree.polarities:
            assert -1.0 <= polarity.centroid_valence <= 1.0
            assert -1.0 <= polarity.centroid_arousal <= 1.0
            for cluster in polarity.clusters:
                assert -1.0 <= cluster.centroid_valence <= 1.0
                assert -1.0 <= cluster.centroid_arousal <= 1.0
                for family in cluster.families:
                    assert -1.0 <= family.centroid_valence <= 1.0
                    assert -1.0 <= family.centroid_arousal <= 1.0

    def test_every_node_has_at_most_max_children(self):
        tree = load_emotion_tree()
        assert len(tree.polarities) <= MAX_CHILDREN_PER_NODE
        for polarity in tree.polarities:
            assert len(polarity.clusters) <= MAX_CHILDREN_PER_NODE
            for cluster in polarity.clusters:
                assert len(cluster.families) <= MAX_CHILDREN_PER_NODE
                for family in cluster.families:
                    assert len(family.emotions) <= MAX_CHILDREN_PER_NODE
                    assert len(family.emotions) > 0

    def test_families_and_clusters_have_description(self):
        tree = load_emotion_tree()
        for polarity in tree.polarities:
            for cluster in polarity.clusters:
                assert cluster.description.strip()
                for family in cluster.families:
                    assert family.description.strip()

    def test_meta_documents_method_and_notes(self):
        tree = load_emotion_tree()
        assert "method" in tree.meta
        assert "russell" in tree.meta["method"].lower() or "circumplex" in tree.meta["method"].lower()
        assert "notes" in tree.meta
        assert any("Justicia" in note for note in tree.meta["notes"])


class TestSituationCatalog:
    def test_situation_count_and_group_sizes(self):
        catalog = load_situations()
        situations = iter_all_situations(catalog)
        assert len(situations) == 15
        for group in catalog.groups:
            assert len(group.situations) <= MAX_CHILDREN_PER_NODE
            assert len(group.situations) > 0

    def test_situation_ids_unique(self):
        catalog = load_situations()
        ids = [s.id for s in iter_all_situations(catalog)]
        assert len(ids) == len(set(ids))

    def test_situation_coordinates_in_range(self):
        catalog = load_situations()
        for situation in iter_all_situations(catalog):
            assert -1.0 <= situation.valence <= 1.0
            assert -1.0 <= situation.arousal <= 1.0

    def test_related_emotion_ids_exist_in_taxonomy(self):
        tree = load_emotion_tree()
        catalog = load_situations()
        known_ids = {e.id for e in iter_all_emotions(tree)}
        for situation in iter_all_situations(catalog):
            assert situation.related_emotions, situation.id
            for related_id in situation.related_emotions:
                assert related_id in known_ids, (situation.id, related_id)

    def test_situations_have_spanish_label_and_english_description(self):
        catalog = load_situations()
        for situation in iter_all_situations(catalog):
            assert situation.label.strip()
            assert situation.description.strip()


class TestLoaderRoundTrip:
    def test_load_emotion_tree_is_cached_and_consistent(self):
        first = load_emotion_tree()
        second = load_emotion_tree()
        assert first is second  # lru_cache round trip
        assert isinstance(first, EmotionTree)

    def test_load_situations_round_trip(self):
        first = load_situations()
        second = load_situations()
        assert first is second

    def test_find_emotion_returns_known_emotion(self):
        tree = load_emotion_tree()
        any_emotion = iter_all_emotions(tree)[0]
        found = find_emotion(tree, any_emotion.id)
        assert found == any_emotion

    def test_find_emotion_returns_none_for_unknown_id(self):
        tree = load_emotion_tree()
        assert find_emotion(tree, "not-a-real-emotion-id") is None

    def test_emotion_path_returns_full_path(self):
        tree = load_emotion_tree()
        any_emotion = iter_all_emotions(tree)[0]
        path = emotion_path(tree, any_emotion.id)
        assert path is not None
        assert len(path) == 4
        assert path[-1] == any_emotion.id

    def test_emotion_path_returns_none_for_unknown_id(self):
        tree = load_emotion_tree()
        assert emotion_path(tree, "not-a-real-emotion-id") is None

    def test_list_children_walks_every_level(self):
        tree = load_emotion_tree()
        polarities = list_children(tree)
        assert polarities == tree.polarities
        clusters = list_children(polarities[0])
        assert all(isinstance(c, EmotionCluster) for c in clusters)
        families = list_children(clusters[0])
        assert all(isinstance(f, EmotionFamily) for f in families)
        emotions = list_children(families[0])
        assert len(emotions) > 0

    def test_list_children_rejects_unknown_node_type(self):
        with pytest.raises(TypeError):
            list_children(object())


class TestPackagedDataFile:
    def test_emotions_json_ships_inside_the_package(self):
        assert RAW_SOURCE_PATH.exists()
        data = json.loads(RAW_SOURCE_PATH.read_text(encoding="utf-8"))
        assert "polarities" in data
