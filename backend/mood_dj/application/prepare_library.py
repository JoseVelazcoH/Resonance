"""Use case: build the union track library across all readable playlists and fetch lyrics.

Two phases, run in order and reported through `on_progress`:

1. "reading playlists": list the user's playlists and fetch each one's tracks (skipped
   entirely, using the cached library, when every playlist's snapshot id is unchanged
   since the last successful prepare for this session).
2. "fetching lyrics": fetch lyrics for any library track not yet cached, bounded
   concurrency, reusing the LRCLIB provider and SQLite lyrics repository. Tracks
   already cached from a previous run are counted immediately, so progress never
   appears to restart from zero.

Lyrics fetching is deliberately polite: a low worker count, a shared minimum
interval between requests, and a circuit breaker that pauses the whole job when
LRCLIB is clearly struggling (many consecutive transient failures in a row),
instead of burning through the remaining queue at full speed while every request
fails. Tracks that stay unresolved after the first pass get one retry pass, after a
pause, before the job gives up on them; a job only finishes as `DONE` when nothing
is left `pending` - otherwise it finishes as `PARTIAL`, and a later call to `run`
resumes just the unresolved tracks (already true via the SQLite cache skip).

Mood (tone/fit) is intentionally NOT computed here: it is prompt-dependent work best
done playlist by playlist inside `RecommendFromLibraryUseCase`.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from typing import Callable

from mood_dj.adapters.playlist_allowlist import filter_playlists_by_allowlist, load_playlist_allowlist
from mood_dj.application.library_lyrics_status import LibraryLyricsStatusStore
from mood_dj.application.library_store import LibraryStore
from mood_dj.domain.models import (
    LibraryPhase,
    LibraryPrepareProgress,
    LibraryPrepareState,
    LyricsEntry,
    LyricsStatus,
    PlaylistTrack,
)
from mood_dj.ports.lyrics_provider import LyricsProvider
from mood_dj.ports.lyrics_repository import LyricsRepository
from mood_dj.ports.mood_profile_repository import MoodProfileRepository
from mood_dj.ports.spotify_playlists import SpotifyPlaylistsClient
from mood_dj.ports.track_profiler import TrackForProfiling, TrackProfiler

DEFAULT_MOOD_BATCH_SIZE = 16

DEFAULT_MAX_LYRICS_CONCURRENCY = 2
DEFAULT_MIN_REQUEST_INTERVAL_S = 0.25
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 8
DEFAULT_CIRCUIT_BREAKER_PAUSE_S = 60.0
DEFAULT_RETRY_PASS_PAUSE_S = 30.0

ProgressCallback = Callable[[LibraryPrepareProgress], None]

PARTIAL_ERROR_TEMPLATE = "LRCLIB is unavailable right now, {count} tracks will be retried"


def _run_with_daemon_workers(items: list, worker_fn: Callable[[object], None], max_workers: int) -> None:
    """Run `worker_fn` over `items` using a small pool of daemon threads.

    Deliberately reimplemented instead of `ThreadPoolExecutor`: the stdlib
    registers an `atexit` hook that joins every worker thread ever created by any
    `ThreadPoolExecutor`, which can hang process shutdown (e.g. Ctrl+C on uvicorn)
    if a worker is mid-retry-sleep. `ThreadPoolExecutor` also does not expose a way
    to make its worker threads daemons. Plain daemon threads are killed outright at
    interpreter exit instead of being joined, so shutdown never blocks on them.
    """
    if not items:
        return

    work_queue: queue.Queue = queue.Queue()
    for item in items:
        work_queue.put(item)

    def run() -> None:
        while True:
            try:
                item = work_queue.get_nowait()
            except queue.Empty:
                return
            worker_fn(item)

    worker_count = max(1, min(max_workers, len(items)))
    threads = [threading.Thread(target=run, daemon=True) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


class _RateLimiter:
    """Enforces a minimum interval between calls, shared across threads."""

    def __init__(self, min_interval_s: float, sleep_fn: Callable[[float], None], now_fn: Callable[[], float]) -> None:
        self._min_interval_s = min_interval_s
        self._sleep_fn = sleep_fn
        self._now_fn = now_fn
        self._lock = threading.Lock()
        self._last_call: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = self._now_fn()
            if self._last_call is not None:
                elapsed = now - self._last_call
                remaining = self._min_interval_s - elapsed
                if remaining > 0:
                    self._sleep_fn(remaining)
                    now = self._now_fn()
            self._last_call = now


class PrepareLibraryUseCase:
    """Builds a session's union track library and fetches lyrics for every track."""

    def __init__(
        self,
        playlists_client: SpotifyPlaylistsClient,
        lyrics_repository: LyricsRepository,
        lyrics_provider: LyricsProvider,
        library_store: LibraryStore,
        lyrics_status_store: LibraryLyricsStatusStore | None = None,
        track_profiler: TrackProfiler | None = None,
        mood_profile_repository: MoodProfileRepository | None = None,
        mood_batch_size: int = DEFAULT_MOOD_BATCH_SIZE,
        max_lyrics_concurrency: int = DEFAULT_MAX_LYRICS_CONCURRENCY,
        min_request_interval_s: float = DEFAULT_MIN_REQUEST_INTERVAL_S,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        circuit_breaker_pause_s: float = DEFAULT_CIRCUIT_BREAKER_PAUSE_S,
        retry_pass_pause_s: float = DEFAULT_RETRY_PASS_PAUSE_S,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.monotonic,
        playlist_allowlist_file: str | None = None,
    ) -> None:
        self._playlists_client = playlists_client
        self._lyrics_repository = lyrics_repository
        self._lyrics_provider = lyrics_provider
        self._library_store = library_store
        self._lyrics_status_store = lyrics_status_store
        self._track_profiler = track_profiler
        self._mood_profile_repository = mood_profile_repository
        self._mood_batch_size = mood_batch_size
        self._playlist_allowlist_file = playlist_allowlist_file
        self._max_lyrics_concurrency = max_lyrics_concurrency
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._circuit_breaker_pause_s = circuit_breaker_pause_s
        self._retry_pass_pause_s = retry_pass_pause_s
        self._sleep_fn = sleep_fn
        self._rate_limiter = _RateLimiter(min_request_interval_s, sleep_fn, now_fn)

    def run(
        self,
        session_id: str,
        access_token: str,
        on_progress: ProgressCallback | None = None,
        stop_event: threading.Event | None = None,
    ) -> LibraryPrepareProgress:
        progress = LibraryPrepareProgress(state=LibraryPrepareState.RUNNING, phase=LibraryPhase.READING_PLAYLISTS.value)
        lock = threading.Lock()

        def emit() -> None:
            if on_progress is not None:
                on_progress(progress)

        emit()
        tracks = self._read_playlists(session_id, access_token, progress, emit)

        progress.phase = LibraryPhase.FETCHING_LYRICS.value
        cached_count = sum(1 for track in tracks if self._lyrics_repository.get(track.id) is not None)
        progress.total = len(tracks)
        progress.cached = cached_count
        progress.processed = cached_count
        progress.tracks_total = len(tracks)
        progress.tracks_processed = cached_count
        if self._lyrics_status_store is not None:
            self._lyrics_status_store.init_rows(session_id, tracks, self._lyrics_repository)
        emit()
        pending = self._fetch_lyrics(session_id, tracks, progress, lock, emit, stop_event)

        if self._track_profiler is not None and self._mood_profile_repository is not None:
            progress.phase = LibraryPhase.READING_MOOD.value
            emit()
            self._profile_moods(tracks, progress, emit, stop_event)

        progress.pending = len(pending)
        progress.failed_transient = len(pending)
        if pending:
            progress.state = LibraryPrepareState.PARTIAL
            progress.error = PARTIAL_ERROR_TEMPLATE.format(count=len(pending))
        else:
            progress.state = LibraryPrepareState.DONE
        emit()
        return progress

    def _profile_moods(
        self,
        tracks: list[PlaylistTrack],
        progress: LibraryPrepareProgress,
        emit: Callable[[], None],
        stop_event: threading.Event | None,
    ) -> None:
        """Compute and persist mood profiles for lyrics-bearing tracks lacking a cached one.

        Persists each batch immediately as it comes back from the profiler, so a
        job interrupted mid-phase resumes cleanly: already-profiled tracks are
        skipped on the next `run` via the cache lookup below.
        """
        version = self._track_profiler.version
        lyrics_by_track: dict[str, str] = {}
        candidates: list[PlaylistTrack] = []
        already_profiled = 0
        for track in tracks:
            cached_lyrics = self._lyrics_repository.get(track.id)
            if cached_lyrics is None or cached_lyrics.status != LyricsStatus.LYRICS or not cached_lyrics.text:
                continue
            if self._mood_profile_repository.get(track.id, version) is not None:
                already_profiled += 1
                continue
            lyrics_by_track[track.id] = cached_lyrics.text
            candidates.append(track)

        progress.profiles_total = already_profiled + len(candidates)
        progress.profiles_processed = already_profiled
        emit()

        if not candidates:
            return

        for start in range(0, len(candidates), self._mood_batch_size):
            if stop_event is not None and stop_event.is_set():
                return
            chunk = candidates[start : start + self._mood_batch_size]
            profiling_inputs = [
                TrackForProfiling(
                    track_id=track.id,
                    artist=track.artist,
                    title=track.name,
                    lyrics=lyrics_by_track[track.id],
                )
                for track in chunk
            ]

            def on_batch(batch_profiles) -> None:
                for profile in batch_profiles:
                    self._mood_profile_repository.save(profile)
                progress.profiles_processed += len(batch_profiles)
                emit()

            self._track_profiler.profile(profiling_inputs, on_progress=on_batch)

    def _read_playlists(
        self,
        session_id: str,
        access_token: str,
        progress: LibraryPrepareProgress,
        emit: Callable[[], None],
    ) -> list[PlaylistTrack]:
        playlists = self._playlists_client.list_playlists(access_token)
        if self._playlist_allowlist_file is not None:
            allowlist_names = load_playlist_allowlist(self._playlist_allowlist_file)
            playlists, _not_found = filter_playlists_by_allowlist(playlists, allowlist_names)
        snapshot_map = {playlist.id: playlist.snapshot_id for playlist in playlists}
        progress.playlists = len(playlists)
        playlist_order = [playlist.id for playlist in playlists]

        cached = self._library_store.get(session_id)
        if cached is not None and cached.snapshot_map == snapshot_map:
            progress.processed = len(playlists)
            progress.total = len(playlists)
            progress.playlists_processed = len(playlists)
            progress.playlists_total = len(playlists)
            emit()
            return cached.tracks

        progress.total = len(playlists)
        progress.playlists_total = len(playlists)
        by_id: dict[str, PlaylistTrack] = {}
        tracks_by_playlist: dict[str, list[str]] = {}
        for index, playlist in enumerate(playlists):
            playlist_tracks = self._playlists_client.get_playlist_tracks(playlist.id, access_token)
            tracks_by_playlist[playlist.id] = [track.id for track in playlist_tracks]
            for track in playlist_tracks:
                existing = by_id.get(track.id)
                by_id[track.id] = self._merge_track(existing, track)
            progress.processed = index + 1
            progress.playlists_processed = index + 1
            emit()

        tracks = list(by_id.values())
        progress.tracks = len(tracks)
        self._library_store.save(
            session_id,
            snapshot_map,
            tracks,
            playlist_order=playlist_order,
            tracks_by_playlist=tracks_by_playlist,
        )
        return tracks

    def _merge_track(self, existing: PlaylistTrack | None, incoming: PlaylistTrack) -> PlaylistTrack:
        if existing is None:
            return incoming
        return PlaylistTrack(
            id=existing.id,
            name=existing.name,
            artist=existing.artist,
            album=existing.album,
            duration_s=existing.duration_s,
            cover_url=existing.cover_url or incoming.cover_url,
            external_url=existing.external_url or incoming.external_url,
        )

    def _fetch_lyrics(
        self,
        session_id: str,
        tracks: list[PlaylistTrack],
        progress: LibraryPrepareProgress,
        lock: threading.Lock,
        emit: Callable[[], None],
        stop_event: threading.Event | None,
    ) -> list[PlaylistTrack]:
        pending: list[PlaylistTrack] = []
        for track in tracks:
            cached = self._lyrics_repository.get(track.id)
            if cached is None:
                pending.append(track)
            else:
                with lock:
                    self._count_status(cached.status.value, progress)

        unresolved = self._run_fetch_pass(session_id, pending, progress, lock, emit, stop_event)

        if unresolved and (stop_event is None or not stop_event.is_set()):
            self._sleep_fn(self._retry_pass_pause_s)
            unresolved = self._run_fetch_pass(session_id, unresolved, progress, lock, emit, stop_event)

        return unresolved

    def _run_fetch_pass(
        self,
        session_id: str,
        pending: list[PlaylistTrack],
        progress: LibraryPrepareProgress,
        lock: threading.Lock,
        emit: Callable[[], None],
        stop_event: threading.Event | None,
    ) -> list[PlaylistTrack]:
        if not pending:
            return []

        unresolved: list[PlaylistTrack] = []
        consecutive_failures = [0]

        def process(track: PlaylistTrack) -> None:
            if stop_event is not None and stop_event.is_set():
                unresolved.append(track)
                return

            if self._lyrics_status_store is not None:
                self._lyrics_status_store.mark_downloading(session_id, track.id)

            self._rate_limiter.wait()
            result = self._lyrics_provider.fetch(track.artist, track.name, track.album, track.duration_s)

            if result.status is None:
                unresolved.append(track)
                should_pause = False
                with lock:
                    consecutive_failures[0] += 1
                    if consecutive_failures[0] >= self._circuit_breaker_threshold:
                        consecutive_failures[0] = 0
                        should_pause = True
                if should_pause:
                    self._sleep_fn(self._circuit_breaker_pause_s)
                emit()
                return

            with lock:
                consecutive_failures[0] = 0

            self._lyrics_repository.save(
                LyricsEntry(
                    track_id=track.id,
                    status=result.status,
                    text=result.text,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            with lock:
                progress.processed += 1
                progress.tracks_processed += 1
                self._count_status(result.status.value, progress)
            if self._lyrics_status_store is not None:
                row_status = "downloaded" if result.status.value == "lyrics" else result.status.value
                self._lyrics_status_store.mark_result(session_id, track.id, row_status)
            emit()

        _run_with_daemon_workers(pending, process, self._max_lyrics_concurrency)

        return unresolved

    def _count_status(self, status_value: str, progress: LibraryPrepareProgress) -> None:
        if status_value == "lyrics":
            progress.with_lyrics += 1
        elif status_value == "instrumental":
            progress.instrumental += 1
        elif status_value == "missing":
            progress.missing += 1
