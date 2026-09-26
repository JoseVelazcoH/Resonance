"""MoodProfileRepository adapter backed by stdlib sqlite3, mirroring SqliteLyricsRepository."""

from __future__ import annotations

from pathlib import Path

from mood_dj.adapters.sqlite_connection import open_connection
from mood_dj.domain.models import TrackMoodProfile

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS track_mood_profiles (
    track_id TEXT NOT NULL,
    version TEXT NOT NULL,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    polarity_id TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    family_id TEXT NOT NULL,
    emotion_id TEXT,
    emotion_confidence REAL,
    situation_id TEXT,
    situation_confidence REAL,
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

    def get(self, track_id: str, version: str) -> TrackMoodProfile | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT track_id, valence, arousal, polarity_id, cluster_id, family_id,
                       emotion_id, emotion_confidence, situation_id, situation_confidence, version
                FROM track_mood_profiles
                WHERE track_id = ? AND version = ?
                """,
                (track_id, version),
            ).fetchone()
        if row is None:
            return None
        return TrackMoodProfile(
            track_id=row[0],
            valence=row[1],
            arousal=row[2],
            polarity_id=row[3],
            cluster_id=row[4],
            family_id=row[5],
            emotion_id=row[6],
            emotion_confidence=row[7],
            situation_id=row[8],
            situation_confidence=row[9],
            version=row[10],
        )

    def save(self, profile: TrackMoodProfile) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO track_mood_profiles (
                    track_id, version, valence, arousal, polarity_id, cluster_id,
                    family_id, emotion_id, emotion_confidence, situation_id, situation_confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_id, version) DO UPDATE SET
                    valence = excluded.valence,
                    arousal = excluded.arousal,
                    polarity_id = excluded.polarity_id,
                    cluster_id = excluded.cluster_id,
                    family_id = excluded.family_id,
                    emotion_id = excluded.emotion_id,
                    emotion_confidence = excluded.emotion_confidence,
                    situation_id = excluded.situation_id,
                    situation_confidence = excluded.situation_confidence
                """,
                (
                    profile.track_id,
                    profile.version,
                    profile.valence,
                    profile.arousal,
                    profile.polarity_id,
                    profile.cluster_id,
                    profile.family_id,
                    profile.emotion_id,
                    profile.emotion_confidence,
                    profile.situation_id,
                    profile.situation_confidence,
                ),
            )
