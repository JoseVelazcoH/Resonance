"""Unit tests for PrepareLibraryUseCase, using in-memory fakes."""

from __future__ import annotations

import threading

from mood_dj.application.library_store import LibraryStore
from mood_dj.application.prepare_library import PrepareLibraryUseCase
from mood_dj.domain.models import (
    LibraryPhase,
    LibraryPrepareState,
    LyricsEntry,
    LyricsStatus,
    PlaylistSummary,
    PlaylistTrack,
)
from mood_dj.ports.lyrics_provider import LyricsLookupResult


def _summary(pid: str, snapshot: str) -> PlaylistSummary:
    return PlaylistSummary(id=pid, name=pid, image_url=None, track_count=1, snapshot_id=snapshot)


def _track(track_id: str, cover_url: str | None = None, external_url: str | None = None) -> PlaylistTrack:
    return PlaylistTrack(
        id=track_id, name=f"Song {track_id}", artist="Artist", album="Album", duration_s=200.0,
        cover_url=cover_url, external_url=external_url,
    )


class FakePlaylistsClient:
    def __init__(self, playlists: list[PlaylistSummary], tracks_by_playlist: dict[str, list[PlaylistTrack]]) -> None:
        self.playlists = playlists
        self.tracks_by_playlist = tracks_by_playlist
        self.track_calls: list[str] = []

    def list_playlists(self, access_token: str) -> list[PlaylistSummary]:
        return self.playlists

    def get_playlist_tracks(self, playlist_id: str, access_token: str) -> list[PlaylistTrack]:
        self.track_calls.append(playlist_id)
        return self.tracks_by_playlist[playlist_id]


class FakeLyricsRepository:
    def __init__(self, existing: dict[str, LyricsEntry] | None = None) -> None:
        self._store = dict(existing or {})

    def get(self, track_id: str) -> LyricsEntry | None:
        return self._store.get(track_id)

    def save(self, entry: LyricsEntry) -> None:
        self._store[entry.track_id] = entry


class FakeLyricsProvider:
    def __init__(self, results: dict[str, LyricsLookupResult]) -> None:
        self.results = results
        self.calls: list[str] = []

    def fetch(self, artist: str, title: str, album: str, duration_s: float) -> LyricsLookupResult:
        track_id = title.split(" ")[-1]
        self.calls.append(track_id)
        return self.results[track_id]


class SequencedLyricsProvider:
    """Fake provider returning the next queued result per track id, each call."""

    def __init__(self, results_by_track: dict[str, list[LyricsLookupResult]]) -> None:
        self.results_by_track = {track_id: list(results) for track_id, results in results_by_track.items()}
        self.calls: list[str] = []

    def fetch(self, artist: str, title: str, album: str, duration_s: float) -> LyricsLookupResult:
        track_id = title.split(" ")[-1]
        self.calls.append(track_id)
        return self.results_by_track[track_id].pop(0)


def _use_case(playlists_client, lyrics_repository, lyrics_provider, judge_tones=None, library_store=None, **kwargs):
    kwargs.setdefault("min_request_interval_s", 0.0)
    kwargs.setdefault("sleep_fn", lambda seconds: None)
    return PrepareLibraryUseCase(
        playlists_client=playlists_client,
        lyrics_repository=lyrics_repository,
        lyrics_provider=lyrics_provider,
        library_store=library_store or LibraryStore(),
        **kwargs,
    )


def test_run_builds_deduplicated_union_library_across_playlists() -> None:
    playlists = [_summary("p1", "s1"), _summary("p2", "s2")]
    tracks_by_playlist = {
        "p1": [_track("t1"), _track("t2", cover_url="cover2")],
        "p2": [_track("t2"), _track("t3")],
    }
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository()
    provider = FakeLyricsProvider(
        {"t1": LyricsLookupResult(LyricsStatus.LYRICS, "la"),
         "t2": LyricsLookupResult(LyricsStatus.LYRICS, "la"),
         "t3": LyricsLookupResult(LyricsStatus.INSTRUMENTAL)}
    )
    use_case = _use_case(client, repo, provider, {"t1": 0.5, "t2": 0.5})

    progress = use_case.run("session-1", "token")

    assert progress.state is LibraryPrepareState.DONE
    assert progress.playlists == 2
    assert progress.tracks == 3
    assert progress.with_lyrics == 2
    assert progress.instrumental == 1


def test_run_keeps_a_non_null_cover_when_deduplicating() -> None:
    playlists = [_summary("p1", "s1"), _summary("p2", "s2")]
    tracks_by_playlist = {
        "p1": [_track("t1", cover_url=None)],
        "p2": [_track("t1", cover_url="cover1")],
    }
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository(existing={"t1": LyricsEntry("t1", LyricsStatus.LYRICS, "la", "now")})
    provider = FakeLyricsProvider({})
    store = LibraryStore()
    use_case = _use_case(client, repo, provider, {"t1": 0.5}, library_store=store)

    use_case.run("session-1", "token")

    saved = store.get("session-1")
    assert saved.tracks[0].cover_url == "cover1"


def test_run_skips_refetching_playlist_tracks_when_snapshots_unchanged() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository(existing={"t1": LyricsEntry("t1", LyricsStatus.LYRICS, "la", "now")})
    provider = FakeLyricsProvider({})
    store = LibraryStore()
    store.save("session-1", snapshot_map={"p1": "s1"}, tracks=[_track("t1")])
    use_case = _use_case(client, repo, provider, {"t1": 0.5}, library_store=store)

    use_case.run("session-1", "token")

    assert client.track_calls == []


def test_run_refetches_when_a_playlist_snapshot_changed() -> None:
    playlists = [_summary("p1", "s2")]
    tracks_by_playlist = {"p1": [_track("t1")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository(existing={"t1": LyricsEntry("t1", LyricsStatus.LYRICS, "la", "now")})
    provider = FakeLyricsProvider({})
    store = LibraryStore()
    store.save("session-1", snapshot_map={"p1": "s1"}, tracks=[_track("t1")])
    use_case = _use_case(client, repo, provider, {"t1": 0.5}, library_store=store)

    use_case.run("session-1", "token")

    assert client.track_calls == ["p1"]


def test_run_reports_phase_transitions_via_progress_callback() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository()
    provider = FakeLyricsProvider({"t1": LyricsLookupResult(LyricsStatus.LYRICS, "la")})
    use_case = _use_case(client, repo, provider)
    phases_seen = []

    use_case.run("session-1", "token", on_progress=lambda p: phases_seen.append(p.phase))

    assert LibraryPhase.READING_PLAYLISTS.value in phases_seen
    assert LibraryPhase.FETCHING_LYRICS.value in phases_seen


def test_progress_reports_cached_and_starts_processed_from_cached_count() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1"), _track("t2")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    # t1 is already cached; only t2 needs fetching.
    repo = FakeLyricsRepository(existing={"t1": LyricsEntry("t1", LyricsStatus.LYRICS, "la", "now")})
    provider = FakeLyricsProvider({"t2": LyricsLookupResult(LyricsStatus.LYRICS, "la")})
    use_case = _use_case(client, repo, provider)
    first_fetch_progress = []

    def on_progress(p):
        if p.phase == LibraryPhase.FETCHING_LYRICS.value:
            first_fetch_progress.append((p.cached, p.processed, p.total))

    progress = use_case.run("session-1", "token", on_progress=on_progress)

    assert progress.cached == 1
    # The very first fetch-phase poll already reflects the cached track, not zero.
    assert first_fetch_progress[0] == (1, 1, 2)


def test_second_run_fetches_nothing_when_everything_is_cached() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1"), _track("t2")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository(
        existing={
            "t1": LyricsEntry("t1", LyricsStatus.LYRICS, "la", "now"),
            "t2": LyricsEntry("t2", LyricsStatus.INSTRUMENTAL, None, "now"),
        }
    )
    provider = FakeLyricsProvider({})
    store = LibraryStore()
    store.save(
        "session-1", snapshot_map={"p1": "s1"}, tracks=[_track("t1"), _track("t2")],
        playlist_order=["p1"], tracks_by_playlist={"p1": ["t1", "t2"]},
    )
    use_case = _use_case(client, repo, provider, library_store=store)

    progress = use_case.run("session-1", "token")

    assert progress.cached == 2
    assert progress.processed == 2
    assert progress.with_lyrics == 1
    assert progress.instrumental == 1


def test_interrupted_run_resumes_with_only_the_missing_tracks() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1"), _track("t2"), _track("t3")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    # Simulates a prior run that fetched t1 before being interrupted; t2 and t3 remain.
    repo = FakeLyricsRepository(existing={"t1": LyricsEntry("t1", LyricsStatus.LYRICS, "la", "now")})
    provider = FakeLyricsProvider(
        {"t2": LyricsLookupResult(LyricsStatus.LYRICS, "la"), "t3": LyricsLookupResult(LyricsStatus.MISSING)}
    )
    use_case = _use_case(client, repo, provider)

    progress = use_case.run("session-1", "token")

    assert progress.cached == 1
    assert progress.processed == 3
    assert progress.with_lyrics == 2
    assert progress.missing == 1
    assert repo.get("t1").text == "la"
    assert repo.get("t2") is not None
    assert repo.get("t3") is not None


def test_default_concurrency_is_two() -> None:
    use_case = _use_case(FakePlaylistsClient([], {}), FakeLyricsRepository(), FakeLyricsProvider({}))
    assert use_case._max_lyrics_concurrency == 2


def test_rate_limiter_enforces_minimum_interval_between_requests() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1"), _track("t2")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository()
    provider = FakeLyricsProvider(
        {"t1": LyricsLookupResult(LyricsStatus.LYRICS, "la"), "t2": LyricsLookupResult(LyricsStatus.LYRICS, "la")}
    )
    sleeps: list[float] = []
    times = iter([0.0, 0.0, 0.1, 0.1, 0.1, 0.1])
    use_case = _use_case(
        client, repo, provider,
        max_lyrics_concurrency=1,
        min_request_interval_s=0.25,
        sleep_fn=lambda seconds: sleeps.append(seconds),
        now_fn=lambda: next(times, 0.1),
    )

    use_case.run("session-1", "token")

    assert any(s > 0 for s in sleeps)


def test_circuit_breaker_pauses_after_consecutive_transient_failures() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track(f"t{i}") for i in range(3)]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository()
    provider = FakeLyricsProvider({f"t{i}": LyricsLookupResult(None) for i in range(3)})
    sleeps: list[float] = []
    use_case = _use_case(
        client, repo, provider,
        max_lyrics_concurrency=1,
        circuit_breaker_threshold=2,
        circuit_breaker_pause_s=60.0,
        retry_pass_pause_s=0.0,
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    use_case.run("session-1", "token")

    assert 60.0 in sleeps


def test_unresolved_tracks_get_a_second_retry_pass_after_a_pause() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository()
    provider = SequencedLyricsProvider({"t1": [LyricsLookupResult(None), LyricsLookupResult(LyricsStatus.LYRICS, "la")]})
    sleeps: list[float] = []
    use_case = _use_case(
        client, repo, provider,
        retry_pass_pause_s=30.0,
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    progress = use_case.run("session-1", "token")

    assert provider.calls == ["t1", "t1"]
    assert 30.0 in sleeps
    assert progress.state is LibraryPrepareState.DONE
    assert progress.pending == 0


def test_job_finishes_partial_when_tracks_stay_unresolved_after_retry_pass() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1"), _track("t2")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository()
    provider = FakeLyricsProvider(
        {"t1": LyricsLookupResult(LyricsStatus.LYRICS, "la"), "t2": LyricsLookupResult(None)}
    )
    use_case = _use_case(client, repo, provider, retry_pass_pause_s=0.0)

    progress = use_case.run("session-1", "token")

    assert progress.state is LibraryPrepareState.PARTIAL
    assert progress.pending == 1
    assert progress.failed_transient == 1
    assert progress.error is not None
    assert "1 tracks will be retried" in progress.error
    # The unresolved track was never cached, so a later run would retry it.
    assert repo.get("t2") is None


def test_stop_event_leaves_remaining_tracks_pending_without_processing_them() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1"), _track("t2")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository()
    provider = FakeLyricsProvider(
        {"t1": LyricsLookupResult(LyricsStatus.LYRICS, "la"), "t2": LyricsLookupResult(LyricsStatus.LYRICS, "la")}
    )
    stop_event = threading.Event()
    stop_event.set()
    use_case = _use_case(client, repo, provider, max_lyrics_concurrency=1)

    progress = use_case.run("session-1", "token", stop_event=stop_event)

    assert progress.state is LibraryPrepareState.PARTIAL
    assert progress.pending == 2


def test_playlists_and_tracks_progress_fields_are_populated_independently() -> None:
    playlists = [_summary("p1", "s1"), _summary("p2", "s2")]
    tracks_by_playlist = {"p1": [_track("t1")], "p2": [_track("t2")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository()
    provider = FakeLyricsProvider(
        {"t1": LyricsLookupResult(LyricsStatus.LYRICS, "la"), "t2": LyricsLookupResult(LyricsStatus.LYRICS, "la")}
    )
    use_case = _use_case(client, repo, provider)
    snapshots = []

    def on_progress(p):
        snapshots.append((p.phase, p.playlists_processed, p.playlists_total, p.tracks_processed, p.tracks_total))

    use_case.run("session-1", "token", on_progress=on_progress)

    reading_snapshots = [s for s in snapshots if s[0] == LibraryPhase.READING_PLAYLISTS.value]
    fetching_snapshots = [s for s in snapshots if s[0] == LibraryPhase.FETCHING_LYRICS.value]
    # During "reading playlists", tracks_total must not be confused with playlists_total.
    assert reading_snapshots[-1][1] == 2 and reading_snapshots[-1][2] == 2
    # By the end of "fetching lyrics", both playlist and track counters are complete.
    assert fetching_snapshots[-1][3] == 2 and fetching_snapshots[-1][4] == 2


def test_fetch_lyrics_worker_threads_are_daemon_threads() -> None:
    from mood_dj.application.prepare_library import _run_with_daemon_workers

    seen_daemon_flags: list[bool] = []

    def worker(item: str) -> None:
        seen_daemon_flags.append(threading.current_thread().daemon)

    _run_with_daemon_workers(["a", "b", "c"], worker, max_workers=2)

    assert seen_daemon_flags
    assert all(seen_daemon_flags)
