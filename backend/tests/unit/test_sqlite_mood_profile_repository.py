"""Unit tests for SqliteMoodProfileRepository."""

from __future__ import annotations

from mood_dj.adapters.sqlite_mood_profile_repository import SqliteMoodProfileRepository
from mood_dj.domain.models import TrackMoodProfile


def _profile(track_id: str, version: str, valence: float = 0.5) -> TrackMoodProfile:
    return TrackMoodProfile(
        track_id=track_id,
        valence=valence,
        arousal=0.1,
        polarity_id="positive",
        cluster_id="core_positive",
        family_id="joy_elation",
        emotion_id="alegria",
        emotion_confidence=0.9,
        situation_id="party",
        situation_confidence=0.7,
        version=version,
    )


def test_get_returns_none_when_never_saved(tmp_path) -> None:
    repo = SqliteMoodProfileRepository(str(tmp_path / "app.db"))
    assert repo.get("t1", "v1") is None


def test_save_then_get_round_trips_all_fields(tmp_path) -> None:
    repo = SqliteMoodProfileRepository(str(tmp_path / "app.db"))
    profile = _profile("t1", "v1")

    repo.save(profile)

    assert repo.get("t1", "v1") == profile


def test_get_is_scoped_to_the_exact_version(tmp_path) -> None:
    repo = SqliteMoodProfileRepository(str(tmp_path / "app.db"))
    repo.save(_profile("t1", "v1"))

    assert repo.get("t1", "v2") is None


def test_save_upserts_on_the_same_track_and_version(tmp_path) -> None:
    repo = SqliteMoodProfileRepository(str(tmp_path / "app.db"))
    repo.save(_profile("t1", "v1", valence=0.1))
    repo.save(_profile("t1", "v1", valence=0.9))

    assert repo.get("t1", "v1").valence == 0.9


def test_reopening_the_same_db_file_keeps_saved_profiles(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    SqliteMoodProfileRepository(db_path).save(_profile("t1", "v1"))

    reopened = SqliteMoodProfileRepository(db_path)

    assert reopened.get("t1", "v1") is not None
