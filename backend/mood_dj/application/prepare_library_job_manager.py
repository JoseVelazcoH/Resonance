"""Runs PrepareLibrary as a background job, keeping one job per session."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Protocol

from mood_dj.application.library_lyrics_status import (
    DEFAULT_WINDOW_SIZE,
    LibraryLyricsStatusStore,
    LibraryLyricsSummary,
)
from mood_dj.domain.models import LibraryLyricsRow, LibraryPrepareProgress, LibraryPrepareState

logger = logging.getLogger(__name__)


class PreparableLibraryUseCase(Protocol):
    def run(self, session_id: str, access_token: str, on_progress=None) -> LibraryPrepareProgress: ...


class PrepareLibraryJobManager:
    """Starts and tracks background PrepareLibrary jobs, one per session id."""

    def __init__(
        self,
        use_case_factory: Callable[[], PreparableLibraryUseCase],
        lyrics_status_store: LibraryLyricsStatusStore | None = None,
    ) -> None:
        self._use_case_factory = use_case_factory
        self._lyrics_status_store = lyrics_status_store
        self._lock = threading.Lock()
        self._progress: dict[str, LibraryPrepareProgress] = {}
        self._running: set[str] = set()

    def lyrics_rows(
        self, session_id: str, offset: int | None = None, limit: int = DEFAULT_WINDOW_SIZE
    ) -> tuple[list[LibraryLyricsRow], int]:
        if self._lyrics_status_store is None:
            return [], 0
        return self._lyrics_status_store.get_window(session_id, offset, limit)

    def lyrics_summary(self, session_id: str) -> LibraryLyricsSummary | None:
        if self._lyrics_status_store is None:
            return None
        return self._lyrics_status_store.summary(session_id)

    def start(self, session_id: str, access_token: str) -> bool:
        """Start a job for `session_id` unless one is already running. Returns True if started."""
        with self._lock:
            if session_id in self._running:
                return False
            self._running.add(session_id)
            self._progress[session_id] = LibraryPrepareProgress(state=LibraryPrepareState.RUNNING)

        thread = threading.Thread(target=self._run, args=(session_id, access_token), daemon=True)
        thread.start()
        return True

    def status(self, session_id: str) -> LibraryPrepareProgress:
        with self._lock:
            return self._progress.get(session_id, LibraryPrepareProgress(state=LibraryPrepareState.IDLE))

    def _run(self, session_id: str, access_token: str) -> None:
        use_case = self._use_case_factory()

        def on_progress(progress: LibraryPrepareProgress) -> None:
            with self._lock:
                self._progress[session_id] = progress

        try:
            use_case.run(session_id, access_token, on_progress=on_progress)
        except Exception as error:  # noqa: BLE001 - reported via job status, not re-raised
            logger.warning("Library prepare job failed for session %s", session_id, exc_info=True)
            with self._lock:
                self._progress[session_id] = LibraryPrepareProgress(state=LibraryPrepareState.ERROR, error=str(error))
        finally:
            with self._lock:
                self._running.discard(session_id)
