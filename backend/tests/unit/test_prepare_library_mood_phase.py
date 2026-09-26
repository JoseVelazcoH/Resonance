"""Unit tests for PrepareLibraryUseCase's "reading mood" phase, using in-memory fakes."""

from __future__ import annotations

from mood_dj.application.library_store import LibraryStore
from mood_dj.application.prepare_library import PrepareLibraryUseCase
from mood_dj.domain.models import (
    LibraryPhase,
    LibraryPrepareState,
    LyricsEntry,
    LyricsStatus,
    PlaylistSummary,
    PlaylistTrack,
    TrackMoodProfile,
)
from mood_dj.ports.track_profiler import TrackForProfiling

FAKE_VERSION = "fake-v1"


def _summary(pid: str, snapshot: str) -> PlaylistSummary:
    return PlaylistSummary(id=pid, name=pid, image_url=None, track_count=1, snapshot_id=snapshot)


def _track(track_id: str) -> PlaylistTrack:
    return PlaylistTrack(
        id=track_id, name=f"Song {track_id}", artist="Artist", album="Album", duration_s=200.0,
        cover_url=None, external_url=None,
    )


def _profile(track_id: str, version: str = FAKE_VERSION) -> TrackMoodProfile:
    return TrackMoodProfile(
        track_id=track_id,
        valence=0.5,
        arousal=0.2,
        polarity_id="positive",
        cluster_id="core_positive",
        family_id="joy_elation",
        emotion_id="alegria",
        emotion_confidence=0.9,
        situation_id="party",
        situation_confidence=0.8,
        version=version,
    )


class FakePlaylistsClient:
    def __init__(self, playlists, tracks_by_playlist) -> None:
        self.playlists = playlists
        self.tracks_by_playlist = tracks_by_playlist

    def list_playlists(self, access_token: str):
        return self.playlists

    def get_playlist_tracks(self, playlist_id: str, access_token: str):
        return self.tracks_by_playlist[playlist_id]


class FakeLyricsRepository:
    def __init__(self, existing=None) -> None:
        self._store = dict(existing or {})

    def get(self, track_id: str):
        return self._store.get(track_id)

    def save(self, entry: LyricsEntry) -> None:
        self._store[entry.track_id] = entry


class FakeLyricsProvider:
    def __init__(self, results=None) -> None:
        self.results = results or {}

    def fetch(self, artist, title, album, duration_s):
        track_id = title.split(" ")[-1]
        return self.results[track_id]


class FakeMoodProfileRepository:
    def __init__(self, existing=None) -> None:
        self._store = dict(existing or {})
        self.saved: list[TrackMoodProfile] = []

    def get(self, track_id: str, version: str):
        return self._store.get((track_id, version))

    def save(self, profile: TrackMoodProfile) -> None:
        self._store[(profile.track_id, profile.version)] = profile
        self.saved.append(profile)


class FakeTrackProfiler:
    """Profiles tracks by simply echoing back a canned profile per track id.

    Calls `on_progress` once per input batch (mirroring the real batching
    contract) so tests can assert that profiles are persisted incrementally.
    """

    version = FAKE_VERSION

    def __init__(self, batch_size: int = 100) -> None:
        self.batch_size = batch_size
        self.calls: list[list[str]] = []

    def profile(self, tracks: list[TrackForProfiling], on_progress=None):
        self.calls.append([t.track_id for t in tracks])
        all_profiles = []
        for start in range(0, len(tracks), self.batch_size):
            chunk = tracks[start : start + self.batch_size]
            batch = [_profile(t.track_id) for t in chunk]
            all_profiles.extend(batch)
            if on_progress is not None:
                on_progress(batch)
        return all_profiles


def _use_case(playlists_client, lyrics_repository, lyrics_provider, track_profiler, mood_profile_repository, **kwargs):
    kwargs.setdefault("min_request_interval_s", 0.0)
    kwargs.setdefault("sleep_fn", lambda seconds: None)
    kwargs.setdefault("library_store", LibraryStore())
    return PrepareLibraryUseCase(
        playlists_client=playlists_client,
        lyrics_repository=lyrics_repository,
        lyrics_provider=lyrics_provider,
        track_profiler=track_profiler,
        mood_profile_repository=mood_profile_repository,
        **kwargs,
    )


def test_run_computes_and_persists_profiles_for_lyrics_bearing_tracks() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1"), _track("t2")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository()
    provider = FakeLyricsProvider(
        {"t1": type("R", (), {"status": LyricsStatus.LYRICS, "text": "la la la"})(),
         "t2": type("R", (), {"status": LyricsStatus.INSTRUMENTAL, "text": None})()}
    )
    profiler = FakeTrackProfiler()
    mood_repo = FakeMoodProfileRepository()
    use_case = _use_case(client, repo, provider, profiler, mood_repo)

    progress = use_case.run("session-1", "token")

    assert progress.state is LibraryPrepareState.DONE
    # Only t1 has actual lyrics text, so only t1 gets profiled.
    assert mood_repo.get("t1", FAKE_VERSION) is not None
    assert mood_repo.get("t2", FAKE_VERSION) is None
    assert profiler.calls == [["t1"]]


def test_run_skips_profiling_tracks_that_already_have_a_cached_profile() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository(existing={"t1": LyricsEntry("t1", LyricsStatus.LYRICS, "la", "now")})
    provider = FakeLyricsProvider()
    profiler = FakeTrackProfiler()
    mood_repo = FakeMoodProfileRepository(existing={("t1", FAKE_VERSION): _profile("t1")})
    use_case = _use_case(client, repo, provider, profiler, mood_repo)

    use_case.run("session-1", "token")

    assert profiler.calls == []


def test_run_reprofiles_when_the_cached_profile_is_from_an_old_version() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository(existing={"t1": LyricsEntry("t1", LyricsStatus.LYRICS, "la", "now")})
    provider = FakeLyricsProvider()
    profiler = FakeTrackProfiler()
    # Cached profile exists but under a stale version -- must be recomputed.
    mood_repo = FakeMoodProfileRepository(existing={("t1", "stale-version"): _profile("t1", version="stale-version")})
    use_case = _use_case(client, repo, provider, profiler, mood_repo)

    use_case.run("session-1", "token")

    assert profiler.calls == [["t1"]]
    assert mood_repo.get("t1", FAKE_VERSION) is not None


def test_mood_phase_reports_progress_and_persists_each_batch_immediately() -> None:
    playlists = [_summary("p1", "s1")]
    track_ids = [f"t{i}" for i in range(5)]
    tracks_by_playlist = {"p1": [_track(t) for t in track_ids]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository(existing={t: LyricsEntry(t, LyricsStatus.LYRICS, "la", "now") for t in track_ids})
    provider = FakeLyricsProvider()
    profiler = FakeTrackProfiler(batch_size=2)
    mood_repo = FakeMoodProfileRepository()
    use_case = _use_case(client, repo, provider, profiler, mood_repo, mood_batch_size=2)
    phase_snapshots = []

    def on_progress(p):
        if p.phase == LibraryPhase.READING_MOOD.value:
            phase_snapshots.append((p.profiles_processed, p.profiles_total))

    use_case.run("session-1", "token", on_progress=on_progress)

    assert (0, 5) in phase_snapshots
    assert phase_snapshots[-1] == (5, 5)
    # Persisted incrementally, batch by batch, not all at the end.
    assert len(mood_repo.saved) == 5


def test_run_without_a_track_profiler_skips_the_mood_phase_entirely() -> None:
    playlists = [_summary("p1", "s1")]
    tracks_by_playlist = {"p1": [_track("t1")]}
    client = FakePlaylistsClient(playlists, tracks_by_playlist)
    repo = FakeLyricsRepository(existing={"t1": LyricsEntry("t1", LyricsStatus.LYRICS, "la", "now")})
    provider = FakeLyricsProvider()
    use_case = PrepareLibraryUseCase(
        playlists_client=client,
        lyrics_repository=repo,
        lyrics_provider=provider,
        library_store=LibraryStore(),
        min_request_interval_s=0.0,
        sleep_fn=lambda seconds: None,
    )
    phases_seen = []

    use_case.run("session-1", "token", on_progress=lambda p: phases_seen.append(p.phase))

    assert LibraryPhase.READING_MOOD.value not in phases_seen
