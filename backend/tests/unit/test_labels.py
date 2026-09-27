"""Every taxonomy id (emotions, moods, situations) must have an English label,
and adding labels_en.json must never change laya_track_profiler.compute_version
(it only hashes moods.json + question wording, not labels_en.json)."""

from __future__ import annotations

from mood_dj.adapters.laya_track_profiler import compute_version
from mood_dj.domain.labels import (
    english_emotion_label,
    english_mood_label,
    english_situation_label,
    load_labels_en,
)
from mood_dj.domain.taxonomy import iter_all_emotions, iter_all_situations, load_emotion_tree, load_situations

# Recorded by running `compute_version()` on main before labels_en.json existed.
COMPUTE_VERSION_BEFORE_LABELS_EN = "992080b1bfb228d0"


def test_compute_version_unchanged_by_labels_en() -> None:
    assert compute_version() == COMPUTE_VERSION_BEFORE_LABELS_EN


def test_every_emotion_id_has_english_label() -> None:
    labels = load_labels_en()["emotions"]
    tree = load_emotion_tree()
    for emotion in iter_all_emotions(tree):
        assert emotion.id in labels, f"missing English label for emotion id {emotion.id!r}"
        assert labels[emotion.id]


def test_every_mood_id_has_english_label() -> None:
    from mood_dj.domain.moods import load_moods

    labels = load_labels_en()["moods"]
    catalog = load_moods()
    for mood in catalog.moods:
        assert mood.id in labels, f"missing English label for mood id {mood.id!r}"
        assert labels[mood.id]


def test_every_situation_id_has_english_label() -> None:
    labels = load_labels_en()["situations"]
    catalog = load_situations()
    for situation in iter_all_situations(catalog):
        assert situation.id in labels, f"missing English label for situation id {situation.id!r}"
        assert labels[situation.id]


def test_english_lookup_falls_back_to_spanish_when_id_unknown() -> None:
    assert english_emotion_label("not-a-real-id", "Original") == "Original"
    assert english_mood_label("not-a-real-id", "Original") == "Original"
    assert english_situation_label("not-a-real-id", "Original") == "Original"


def test_known_translations() -> None:
    assert english_emotion_label("tristeza", "Tristeza") == "Sadness"
    assert english_mood_label("sadness", "Tristeza") == "Sadness"
    assert english_situation_label("rainy_day_home", "Día de lluvia en casa") == "Rainy Day at Home"
