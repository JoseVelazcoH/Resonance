"""Use case: read-only mood calibration diagnostics for a session's prepared library.

Combines the session's library size (`LibraryStore`, in-memory, session-scoped)
with every cached `TrackMoodProfile` at the current profiler version
(`MoodProfileRepository`, SQLite, global) into a `MoodDiagnosticsReport`. Never
writes anything.
"""

from __future__ import annotations

from mood_dj.application.library_store import LibraryStore
from mood_dj.domain.mood_diagnostics import MoodDiagnosticsReport, compute_diagnostics
from mood_dj.ports.mood_profile_repository import MoodProfileRepository


class LibraryDiagnosticsUseCase:
    """Builds a `MoodDiagnosticsReport` for a session's currently prepared library."""

    def __init__(self, library_store: LibraryStore, mood_profile_repository: MoodProfileRepository, profile_version: str) -> None:
        self._library_store = library_store
        self._mood_profile_repository = mood_profile_repository
        self._profile_version = profile_version

    def run(self, session_id: str) -> MoodDiagnosticsReport:
        library = self._library_store.get(session_id)
        total_tracks = len(library.tracks) if library is not None else 0

        track_ids = {track.id for track in library.tracks} if library is not None else set()
        all_profiles = self._mood_profile_repository.get_all(self._profile_version)
        profiles = [p for p in all_profiles if p.track_id in track_ids] if track_ids else all_profiles

        return compute_diagnostics(profiles, total_tracks)
