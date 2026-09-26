"""Port for persisting per-track mood profiles, keyed by (track_id, version)."""

from __future__ import annotations

from typing import Protocol

from mood_dj.domain.models import TrackMoodProfile


class MoodProfileRepository(Protocol):
    """Stores and retrieves cached `TrackMoodProfile` rows."""

    def get(self, track_id: str, version: str) -> TrackMoodProfile | None:
        """Return the cached profile for a track at exactly this version, or None."""
        ...

    def save(self, profile: TrackMoodProfile) -> None:
        """Persist (insert or replace) a mood profile."""
        ...
