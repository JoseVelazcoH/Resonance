"""Per-track lyrics-download status, for the UI's live status table.

`PrepareLibraryUseCase` reports progress in aggregate (counts, phase). This store
keeps the per-track detail (`downloaded` / `downloading` / `pending` / `missing` /
`instrumental`) needed to render a table of the library's tracks, with paging so the
UI does not have to pull the whole library at once.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from mood_dj.domain.models import LibraryLyricsRow, LyricsEntry, LyricsStatus, PlaylistTrack

DEFAULT_WINDOW_SIZE = 50

_STATUS_BY_LYRICS_STATUS = {
    LyricsStatus.LYRICS: "downloaded",
    LyricsStatus.INSTRUMENTAL: "instrumental",
    LyricsStatus.MISSING: "missing",
}


def _status_for_entry(entry: LyricsEntry | None) -> str:
    if entry is None:
        return "pending"
    return _STATUS_BY_LYRICS_STATUS[entry.status]


@dataclass(frozen=True)
class LibraryLyricsSummary:
    total: int
    with_lyrics: int
    instrumental: int
    missing: int
    pending: int
    all_cached: bool


class LibraryLyricsStatusStore:
    """Thread-safe per-session table of `LibraryLyricsRow`, one row per library track."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, list[LibraryLyricsRow]] = {}

    def init_rows(self, session_id: str, tracks: list[PlaylistTrack], lyrics_repository) -> None:
        rows = [
            LibraryLyricsRow(
                index=index,
                track_id=track.id,
                name=track.name,
                artist=track.artist,
                status=_status_for_entry(lyrics_repository.get(track.id)),
            )
            for index, track in enumerate(tracks)
        ]
        with self._lock:
            self._rows[session_id] = rows

    def mark_downloading(self, session_id: str, track_id: str) -> None:
        self._set_status(session_id, track_id, "downloading")

    def mark_result(self, session_id: str, track_id: str, status: str) -> None:
        self._set_status(session_id, track_id, status)

    def _set_status(self, session_id: str, track_id: str, status: str) -> None:
        with self._lock:
            for row in self._rows.get(session_id, []):
                if row.track_id == track_id:
                    row.status = status
                    return

    def get_window(
        self, session_id: str, offset: int | None = None, limit: int = DEFAULT_WINDOW_SIZE
    ) -> tuple[list[LibraryLyricsRow], int]:
        with self._lock:
            rows = list(self._rows.get(session_id, []))
        total = len(rows)
        if offset is None:
            focus = next((row.index for row in rows if row.status == "downloading"), None)
            if focus is None:
                focus = next((row.index for row in rows if row.status == "pending"), 0)
            offset = max(0, focus - limit // 2)
        return rows[offset : offset + limit], total

    def summary(self, session_id: str) -> LibraryLyricsSummary:
        with self._lock:
            rows = list(self._rows.get(session_id, []))
        total = len(rows)
        with_lyrics = sum(1 for row in rows if row.status == "downloaded")
        instrumental = sum(1 for row in rows if row.status == "instrumental")
        missing = sum(1 for row in rows if row.status == "missing")
        pending = sum(1 for row in rows if row.status in ("pending", "downloading"))
        return LibraryLyricsSummary(
            total=total,
            with_lyrics=with_lyrics,
            instrumental=instrumental,
            missing=missing,
            pending=pending,
            all_cached=total > 0 and pending == 0,
        )
