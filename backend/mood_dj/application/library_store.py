"""In-memory cache of each session's union track library, keyed by playlist snapshots.

Rebuilding the library means calling Spotify for every playlist's tracks, which is
expensive. This store lets `PrepareLibraryUseCase` skip that work when the caller's
playlist snapshot ids have not changed since the last successful prepare.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from mood_dj.domain.models import PlaylistSummary, PlaylistTrack


@dataclass(frozen=True)
class LibrarySnapshot:
    """A cached union track library plus the playlist snapshot ids it was built from.

    `playlist_order` preserves the user's playlist order so mood analysis can walk
    playlists one at a time; `tracks_by_playlist` maps each playlist id to the track
    ids it contains (a track may appear under more than one playlist); `playlists`
    keeps each playlist's display metadata (name, image, track count) for reporting
    per-playlist contribution in a recommendation.
    """

    snapshot_map: dict[str, str]
    tracks: list[PlaylistTrack]
    playlist_order: list[str] = field(default_factory=list)
    tracks_by_playlist: dict[str, list[str]] = field(default_factory=dict)
    playlists: list[PlaylistSummary] = field(default_factory=list)


class LibraryStore:
    """Thread-safe in-memory store of one `LibrarySnapshot` per session id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, LibrarySnapshot] = {}

    def get(self, session_id: str) -> LibrarySnapshot | None:
        with self._lock:
            return self._data.get(session_id)

    def save(
        self,
        session_id: str,
        snapshot_map: dict[str, str],
        tracks: list[PlaylistTrack],
        playlist_order: list[str] | None = None,
        tracks_by_playlist: dict[str, list[str]] | None = None,
        playlists: list[PlaylistSummary] | None = None,
    ) -> None:
        with self._lock:
            self._data[session_id] = LibrarySnapshot(
                snapshot_map=dict(snapshot_map),
                tracks=list(tracks),
                playlist_order=list(playlist_order or []),
                tracks_by_playlist={pid: list(track_ids) for pid, track_ids in (tracks_by_playlist or {}).items()},
                playlists=list(playlists or []),
            )
