"""Unit tests for the flat 7-mood taxonomy loader (`mood_dj.domain.moods`)."""

from __future__ import annotations

from mood_dj.domain.moods import (
    Mood,
    MoodCatalog,
    load_moods,
    parse_mood_catalog,
    validate_families_cover_all_moods,
)
from mood_dj.domain.taxonomy import load_emotion_tree


def test_load_moods_returns_exactly_seven_moods() -> None:
    catalog = load_moods()

    assert len(catalog.moods) == 7
    assert {m.id for m in catalog.moods} == {
        "love",
        "happiness",
        "comfort",
        "sadness",
        "loneliness",
        "anger",
        "fear",
    }


def test_every_mood_has_required_fields() -> None:
    catalog = load_moods()

    for mood in catalog.moods:
        assert mood.label
        assert mood.criterion
        assert mood.polarity in ("positive", "negative")
        assert -1.0 <= mood.valence <= 1.0
        assert -1.0 <= mood.arousal <= 1.0
        assert len(mood.families) > 0


def test_by_id_resolves_a_known_mood_and_none_for_unknown() -> None:
    catalog = load_moods()

    assert catalog.by_id("love").label == "Amor"
    assert catalog.by_id("not-a-mood") is None


def test_family_to_mood_id_resolves_a_known_family() -> None:
    catalog = load_moods()

    assert catalog.family_to_mood_id("love_affection") == "love"
    assert catalog.family_to_mood_id("anger_hostility") == "anger"


def test_family_to_mood_id_is_none_for_an_unknown_family() -> None:
    catalog = load_moods()

    assert catalog.family_to_mood_id("not-a-real-family") is None


def test_every_emotions_json_family_maps_to_exactly_one_mood() -> None:
    catalog = load_moods()
    tree = load_emotion_tree()

    problems = validate_families_cover_all_moods(catalog, tree)

    assert problems == [], f"moods.json / emotions.json family coverage problems: {problems}"


def test_validate_flags_a_family_mapped_to_two_moods() -> None:
    tree = load_emotion_tree()
    catalog = MoodCatalog(
        meta={},
        moods=(
            Mood(id="a", label="A", criterion="c", valence=0.0, arousal=0.0, polarity="positive",
                 families=("love_affection",)),
            Mood(id="b", label="B", criterion="c", valence=0.0, arousal=0.0, polarity="positive",
                 families=("love_affection",)),
        ),
    )

    problems = validate_families_cover_all_moods(catalog, tree)

    assert any("love_affection" in p and "more than one" in p for p in problems)


def test_validate_flags_an_unmapped_family() -> None:
    tree = load_emotion_tree()
    catalog = MoodCatalog(meta={}, moods=())

    problems = validate_families_cover_all_moods(catalog, tree)

    assert len(problems) == len(
        [f for p in tree.polarities for c in p.clusters for f in c.families]
    )


def test_parse_mood_catalog_from_raw_dict() -> None:
    raw = {
        "_meta": {"note": "x"},
        "moods": [
            {
                "id": "love",
                "label": "Amor",
                "criterion": "Themes of affection",
                "valence": 0.5,
                "arousal": 0.2,
                "polarity": "positive",
                "families": ["love_affection"],
            }
        ],
    }

    catalog = parse_mood_catalog(raw)

    assert catalog.meta == {"note": "x"}
    assert catalog.moods[0].id == "love"
    assert catalog.moods[0].families == ("love_affection",)
