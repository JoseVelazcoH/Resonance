"""MoodProfileRepository adapter backed by stdlib sqlite3, mirroring SqliteLyricsRepository.

`track_mood_profiles_v3` is a new table (flat mood schema + full probability
distribution, see `mood_dj.domain.models.TrackMoodProfile`); older
`track_mood_profiles`/`track_mood_profiles_v2` tables are simply left in place
and never read again -- profiles are keyed by `(track_id, version)` and the
version bump in `laya_track_profiler.compute_version` means old rows are never
looked up.
"""

from __future__ import annotations

import json
from pathlib import Path

from mood_dj.adapters.sqlite_connection import open_connection
from mood_dj.domain.models import TrackMoodProfile

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS track_mood_profiles_v3 (
    track_id TEXT NOT NULL,
    version TEXT NOT NULL,
    mood_id TEXT NOT NULL,
    mood_confidence REAL NOT NULL,
    positive_probability REAL NOT NULL,
    mood_probabilities TEXT NOT NULL,
    PRIMARY KEY (track_id, version)
)
"""


class SqliteMoodProfileRepository:
    """Stores per-track mood profiles in a local SQLite database."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(_CREATE_TABLE)

    def _connect(self):
        return open_connection(self._db_path)

    def _row_to_profile(self, row) -> TrackMoodProfile:
        return TrackMoodProfile(
            track_id=row[0],
            mood_id=row[1],
            mood_confidence=row[2],
            positive_probability=row[3],
            mood_probabilities=json.loads(row[4]) if row[4] else {},
            version=row[5],
        )

    def get(self, track_id: str, version: str) -> TrackMoodProfile | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT track_id, mood_id, mood_confidence, positive_probability, mood_probabilities, version
                FROM track_mood_profiles_v3
                WHERE track_id = ? AND version = ?
                """,
                (track_id, version),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_profile(row)

    def get_all(self, version: str) -> list[TrackMoodProfile]:
        """Return every cached profile at exactly this version. Read-only, for diagnostics."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT track_id, mood_id, mood_confidence, positive_probability, mood_probabilities, version
                FROM track_mood_profiles_v3
                WHERE version = ?
                """,
                (version,),
            ).fetchall()
        return [self._row_to_profile(row) for row in rows]

    def save(self, profile: TrackMoodProfile) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO track_mood_profiles_v3 (
                    track_id, version, mood_id, mood_confidence, positive_probability, mood_probabilities
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_id, version) DO UPDATE SET
                    mood_id = excluded.mood_id,
                    mood_confidence = excluded.mood_confidence,
                    positive_probability = excluded.positive_probability,
                    mood_probabilities = excluded.mood_probabilities
                """,
                (
                    profile.track_id,
                    profile.version,
                    profile.mood_id,
                    profile.mood_confidence,
                    profile.positive_probability,
                    json.dumps(profile.mood_probabilities),
                ),
            )
