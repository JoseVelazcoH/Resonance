"""Unit tests for SqliteMoodProfileRepository (flat-mood schema, v3 table)."""

from __future__ import annotations

from mood_dj.adapters.sqlite_mood_profile_repository import SqliteMoodProfileRepository
from mood_dj.domain.models import TrackMoodProfile


def _profile(track_id: str, version: str, mood_id: str = "love", mood_confidence: float = 0.6) -> TrackMoodProfile:
    return TrackMoodProfile(
        track_id=track_id,
        mood_id=mood_id,
        mood_confidence=mood_confidence,
        positive_probability=0.7,
        version=version,
        mood_probabilities={
            "love": mood_confidence,
            "happiness": 0.1,
            "comfort": 0.1,
            "sadness": 0.05,
            "loneliness": 0.05,
            "anger": 0.0,
            "fear": 0.0,
        },
    )


def test_get_returns_none_when_never_saved(tmp_path) -> None:
    repo = SqliteMoodProfileRepository(str(tmp_path / "app.db"))
    assert repo.get("t1", "v1") is None


def test_save_then_get_round_trips_all_fields(tmp_path) -> None:
    repo = SqliteMoodProfileRepository(str(tmp_path / "app.db"))
    profile = _profile("t1", "v1")

    repo.save(profile)

    assert repo.get("t1", "v1") == profile


def test_save_then_get_round_trips_the_full_probability_distribution(tmp_path) -> None:
    repo = SqliteMoodProfileRepository(str(tmp_path / "app.db"))
    profile = _profile("t1", "v1")

    repo.save(profile)

    round_tripped = repo.get("t1", "v1")
    assert round_tripped.mood_probabilities == profile.mood_probabilities


def test_get_is_scoped_to_the_exact_version(tmp_path) -> None:
    repo = SqliteMoodProfileRepository(str(tmp_path / "app.db"))
    repo.save(_profile("t1", "v1"))

    assert repo.get("t1", "v2") is None


def test_save_upserts_on_the_same_track_and_version(tmp_path) -> None:
    repo = SqliteMoodProfileRepository(str(tmp_path / "app.db"))
    repo.save(_profile("t1", "v1", mood_id="love"))
    repo.save(_profile("t1", "v1", mood_id="anger"))

    assert repo.get("t1", "v1").mood_id == "anger"


def test_reopening_the_same_db_file_keeps_saved_profiles(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    SqliteMoodProfileRepository(db_path).save(_profile("t1", "v1"))

    reopened = SqliteMoodProfileRepository(db_path)

    assert reopened.get("t1", "v1") is not None


def test_get_all_returns_only_rows_at_the_requested_version(tmp_path) -> None:
    repo = SqliteMoodProfileRepository(str(tmp_path / "app.db"))
    repo.save(_profile("t1", "v1"))
    repo.save(_profile("t2", "v1"))
    repo.save(_profile("t3", "v2"))

    rows = repo.get_all("v1")

    assert {p.track_id for p in rows} == {"t1", "t2"}


def test_get_all_is_empty_for_an_unknown_version(tmp_path) -> None:
    repo = SqliteMoodProfileRepository(str(tmp_path / "app.db"))
    repo.save(_profile("t1", "v1"))

    assert repo.get_all("does-not-exist") == []
