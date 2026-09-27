"""Unit tests for scripts/profile_diagnostics.py, always against a tmp_path DB.

Never runs against the real backend/data/app.db.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_ROOT / "scripts"))

from profile_diagnostics import build_report, format_report  # noqa: E402

from mood_dj.adapters.sqlite_lyrics_repository import SqliteLyricsRepository
from mood_dj.adapters.sqlite_mood_profile_repository import SqliteMoodProfileRepository
from mood_dj.domain.models import LyricsEntry, LyricsStatus, TrackMoodProfile


def _db(tmp_path) -> str:
    return str(tmp_path / "app.db")


def _profile(track_id: str, mood_id: str, mood_confidence: float, positive_probability: float, version: str = "v1") -> TrackMoodProfile:
    return TrackMoodProfile(
        track_id=track_id, mood_id=mood_id, mood_confidence=mood_confidence,
        positive_probability=positive_probability, version=version,
    )


def test_build_report_never_writes_to_the_database(tmp_path) -> None:
    db_path = _db(tmp_path)
    lyrics_repo = SqliteLyricsRepository(db_path)
    lyrics_repo.save(LyricsEntry("t1", LyricsStatus.LYRICS, "la la", "2024-01-01T00:00:00+00:00"))
    profile_repo = SqliteMoodProfileRepository(db_path)
    profile_repo.save(_profile("t1", "happiness", 0.8, 0.9))

    before = Path(db_path).read_bytes()
    build_report(db_path, "v1")
    after = Path(db_path).read_bytes()

    assert before == after


def test_build_report_aggregates_by_mood(tmp_path) -> None:
    db_path = _db(tmp_path)
    lyrics_repo = SqliteLyricsRepository(db_path)
    lyrics_repo.save(LyricsEntry("t1", LyricsStatus.LYRICS, "la", "2024-01-01T00:00:00+00:00"))
    lyrics_repo.save(LyricsEntry("t2", LyricsStatus.LYRICS, "la", "2024-01-01T00:00:00+00:00"))
    lyrics_repo.save(LyricsEntry("t3", LyricsStatus.MISSING, None, "2024-01-01T00:00:00+00:00"))
    profile_repo = SqliteMoodProfileRepository(db_path)
    profile_repo.save(_profile("t1", "happiness", 0.8, 0.9))
    profile_repo.save(_profile("t2", "anger", 0.7, 0.1))

    report = build_report(db_path, "v1")

    assert report.total_tracks == 2  # only lyrics-bearing tracks count
    assert report.profiled_count == 2
    assert {m.mood_id for m in report.moods} == {"happiness", "anger"}


def test_build_report_ignores_profiles_at_a_different_version(tmp_path) -> None:
    db_path = _db(tmp_path)
    lyrics_repo = SqliteLyricsRepository(db_path)
    lyrics_repo.save(LyricsEntry("t1", LyricsStatus.LYRICS, "la", "2024-01-01T00:00:00+00:00"))
    profile_repo = SqliteMoodProfileRepository(db_path)
    profile_repo.save(_profile("t1", "happiness", 0.8, 0.9, version="stale-version"))

    report = build_report(db_path, "v1")

    assert report.profiled_count == 0


def test_format_report_includes_mood_rows_and_overall_row(tmp_path) -> None:
    db_path = _db(tmp_path)
    lyrics_repo = SqliteLyricsRepository(db_path)
    lyrics_repo.save(LyricsEntry("t1", LyricsStatus.LYRICS, "la", "2024-01-01T00:00:00+00:00"))
    profile_repo = SqliteMoodProfileRepository(db_path)
    profile_repo.save(_profile("t1", "happiness", 0.8, 0.9))

    report = build_report(db_path, "v1")
    text = format_report(report)

    assert "happiness" in text
    assert "OVERALL" in text
