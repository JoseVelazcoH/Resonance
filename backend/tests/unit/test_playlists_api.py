"""Unit tests for the /playlists endpoints, with adapters replaced by fakes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mood_dj.api.deps import get_current_tokens, get_playlists_client, get_recommend_job_manager, get_session_id, get_settings
from mood_dj.api.main import app
from mood_dj.config import Settings
from mood_dj.application.recommend_from_library import (
    DecisionsSnapshot,
    Detected,
    DetectedEmotion,
    DetectedSituation,
    DetectedTarget,
    LibraryTrackSummary,
    PlaylistRecommendation,
    PlaylistStage,
    RankedLibraryTrack,
    RankedTrack,
)
from mood_dj.application.recommend_job_manager import RecommendJobProgress, RecommendJobState
from mood_dj.domain.models import PlaylistSummary, PlaylistTrack, SpotifyTokens, Strategy
from mood_dj.ports.spotify_playlists import SpotifyApiError


class FakePlaylistsClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_playlists(self, access_token: str):
        self.calls.append(access_token)
        return [PlaylistSummary(id="pl1", name="My Playlist", image_url=None, track_count=10, snapshot_id="s")]

    def get_playlist_tracks(self, playlist_id: str, access_token: str):
        return []


class FakeSavePlaylistsClient:
    def __init__(self, error: SpotifyApiError | None = None) -> None:
        self.error = error
        self.created: list[tuple[str, str]] = []
        self.added: list[tuple[str, list[str]]] = []

    def get_current_user_id(self, access_token: str) -> str:
        return "me"

    def create_playlist(self, user_id: str, name: str, access_token: str) -> str:
        if self.error is not None:
            raise self.error
        self.created.append((user_id, name))
        return "new-pl"

    def add_tracks(self, playlist_id: str, track_ids: list[str], access_token: str) -> None:
        self.added.append((playlist_id, track_ids))


def teardown_function() -> None:
    app.dependency_overrides = {}


def test_list_playlists_requires_session() -> None:
    client = TestClient(app)

    response = client.get("/playlists")

    assert response.status_code == 401


def _no_allowlist_settings() -> Settings:
    return Settings(
        spotify_client_id="id",
        spotify_client_secret="secret",
        dataset_path="data/tracks.parquet",
        app_db_path=":memory:",
        frontend_url="http://127.0.0.1:5173",
        spotify_redirect_uri="http://127.0.0.1:8000/auth/callback",
        playlist_allowlist_file="does-not-exist.json",
    )


def test_list_playlists_returns_mapped_summaries() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    app.dependency_overrides[get_playlists_client] = lambda: FakePlaylistsClient()
    app.dependency_overrides[get_settings] = _no_allowlist_settings
    client = TestClient(app)

    response = client.get("/playlists")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["id"] == "pl1"
    assert body[0]["name"] == "My Playlist"


def test_list_playlists_applies_allowlist_filter_when_configured(tmp_path) -> None:
    allowlist_file = tmp_path / "playlists.local.json"
    allowlist_file.write_text('{"playlist_allowlist": ["Other Playlist"]}')

    def settings_with_allowlist() -> Settings:
        return Settings(
            spotify_client_id="id",
            spotify_client_secret="secret",
            dataset_path="data/tracks.parquet",
            app_db_path=":memory:",
            frontend_url="http://127.0.0.1:5173",
            spotify_redirect_uri="http://127.0.0.1:8000/auth/callback",
            playlist_allowlist_file=str(allowlist_file),
        )

    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    app.dependency_overrides[get_playlists_client] = lambda: FakePlaylistsClient()
    app.dependency_overrides[get_settings] = settings_with_allowlist
    client = TestClient(app)

    response = client.get("/playlists")

    assert response.status_code == 200
    assert response.json() == []


class FakeRecommendJobManager:
    def __init__(self, job: RecommendJobProgress | None = None) -> None:
        self.job = job
        self.started: list[tuple[str, str]] = []
        self.status_calls: list[tuple[str, str]] = []

    def start(self, session_id, prompt) -> str:
        self.started.append((session_id, prompt))
        return "job-1"

    def status(self, job_id: str, session_id: str) -> RecommendJobProgress | None:
        self.status_calls.append((job_id, session_id))
        if job_id != "job-1":
            return None
        return self.job


def _sample_recommendation() -> PlaylistRecommendation:
    track = PlaylistTrack(
        id="t1", name="Song", artist="Artist", album="Album", duration_s=200.0,
        cover_url="https://cover", external_url="https://open",
    )
    return PlaylistRecommendation(
        strategy=Strategy.ACCOMPANY,
        signal_probabilities={"feels_bad": 0.9, "wants_change": 0.1, "wants_energy": 0.1, "wants_rest": 0.1},
        stages=[PlaylistStage(name="session", tracks=[RankedTrack(track=track, similarity=0.8)])],
        detected=Detected(
            emotion=DetectedEmotion(id="tristeza", label="Tristeza", confidence=0.8),
            family_id="melancolia",
            situation=DetectedSituation(id="rainy_day_home", label="Día de lluvia en casa", confidence=0.6),
            target=DetectedTarget(valence=-0.3, arousal=-0.2),
        ),
        excluded_no_lyrics=2,
        excluded_instrumental=1,
        excluded_no_profile=0,
        ranked_tracks=[
            RankedLibraryTrack(
                id="t1", name="Song", artist="Artist", cover_url="https://cover", similarity=0.8, selected=True
            ),
            RankedLibraryTrack(id="t2", name="Other", artist="Artist", cover_url=None, similarity=0.1, selected=False),
        ],
    )


def test_recommend_job_status_requires_session() -> None:
    client = TestClient(app)

    response = client.get("/recommend-jobs/job-1")

    assert response.status_code == 401


def test_recommend_job_status_returns_404_for_unknown_job() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    app.dependency_overrides[get_recommend_job_manager] = lambda: FakeRecommendJobManager(job=None)
    app.dependency_overrides[get_session_id] = lambda: "session-1"
    client = TestClient(app)

    response = client.get("/recommend-jobs/unknown")

    assert response.status_code == 404


def test_recommend_job_status_reports_running_progress() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    job = RecommendJobProgress(state=RecommendJobState.RUNNING, phase="judging lyrics", processed=3, total=10)
    app.dependency_overrides[get_recommend_job_manager] = lambda: FakeRecommendJobManager(job=job)
    app.dependency_overrides[get_session_id] = lambda: "session-1"
    client = TestClient(app)

    response = client.get("/recommend-jobs/job-1")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "running"
    assert body["phase"] == "judging lyrics"
    assert body["processed"] == 3
    assert body["total"] == 10
    assert body["result"] is None


def test_recommend_job_status_exposes_decisions_before_result_exists() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    decisions = DecisionsSnapshot(
        strategy=Strategy.ACCOMPANY,
        signal_probabilities={"feels_bad": 0.9, "wants_change": 0.1, "wants_energy": 0.1, "wants_rest": 0.1},
        detected=Detected(
            emotion=DetectedEmotion(id="tristeza", label="Tristeza", confidence=0.8),
            family_id="melancolia",
            situation=DetectedSituation(id="rainy_day_home", label="Día de lluvia en casa", confidence=0.6),
            target=DetectedTarget(valence=-0.3, arousal=-0.2),
        ),
    )
    job = RecommendJobProgress(
        state=RecommendJobState.RUNNING, phase="ranking tracks", processed=1, total=10, decisions=decisions
    )
    app.dependency_overrides[get_recommend_job_manager] = lambda: FakeRecommendJobManager(job=job)
    app.dependency_overrides[get_session_id] = lambda: "session-1"
    client = TestClient(app)

    response = client.get("/recommend-jobs/job-1")

    assert response.status_code == 200
    body = response.json()
    assert body["result"] is None
    assert body["decisions"]["strategy"] == "accompany"
    assert body["decisions"]["detected"]["emotion"]["label"] == "Sadness"
    assert body["decisions"]["signals"]["feels_bad"] == 0.9


def test_recommend_job_status_returns_mapped_result_when_done() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    job = RecommendJobProgress(
        state=RecommendJobState.DONE, phase="building playlist", processed=1, total=1, result=_sample_recommendation()
    )
    app.dependency_overrides[get_recommend_job_manager] = lambda: FakeRecommendJobManager(job=job)
    app.dependency_overrides[get_session_id] = lambda: "session-1"
    client = TestClient(app)

    response = client.get("/recommend-jobs/job-1")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "done"
    assert body["result"]["strategy"] == "accompany"
    assert body["result"]["stages"][0]["tracks"][0]["keep_probability"] == 0.8
    assert body["result"]["stages"][0]["tracks"][0]["duration_s"] == 200.0
    assert body["result"]["ranked_tracks"][0] == {
        "id": "t1", "name": "Song", "artist": "Artist", "cover_url": "https://cover",
        "similarity": 0.8, "selected": True,
    }
    assert body["result"]["ranked_tracks"][1]["selected"] is False
    assert body["result"]["ranked_tracks"][1]["similarity"] == 0.1


def test_recommend_job_status_exposes_library_tracks_while_running() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    job = RecommendJobProgress(
        state=RecommendJobState.RUNNING,
        phase="ranking tracks",
        processed=1,
        total=10,
        library_tracks=[LibraryTrackSummary(id="t1", name="Song", artist="Artist", cover_url="https://cover")],
    )
    app.dependency_overrides[get_recommend_job_manager] = lambda: FakeRecommendJobManager(job=job)
    app.dependency_overrides[get_session_id] = lambda: "session-1"
    client = TestClient(app)

    response = client.get("/recommend-jobs/job-1")

    assert response.status_code == 200
    body = response.json()
    assert body["result"] is None
    assert body["library_tracks"] == [{"id": "t1", "name": "Song", "artist": "Artist", "cover_url": "https://cover"}]


def test_recommend_job_status_reports_error() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    job = RecommendJobProgress(state=RecommendJobState.ERROR, error="boom")
    app.dependency_overrides[get_recommend_job_manager] = lambda: FakeRecommendJobManager(job=job)
    app.dependency_overrides[get_session_id] = lambda: "session-1"
    client = TestClient(app)

    response = client.get("/recommend-jobs/job-1")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "error"
    assert body["error"] == "boom"


def test_recommend_job_status_is_scoped_to_the_caller_session() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    app.dependency_overrides[get_session_id] = lambda: "session-7"
    job_manager = FakeRecommendJobManager(job=None)
    app.dependency_overrides[get_recommend_job_manager] = lambda: job_manager
    client = TestClient(app)

    response = client.get("/recommend-jobs/job-1")

    assert response.status_code == 404
    assert job_manager.status_calls == [("job-1", "session-7")]


def test_save_playlist_requires_session() -> None:
    client = TestClient(app)

    response = client.post("/playlists/save", json={"name": "My Mood", "track_ids": ["t1"]})

    assert response.status_code == 401


def test_save_playlist_creates_playlist_and_adds_tracks() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    fake_client = FakeSavePlaylistsClient()
    app.dependency_overrides[get_playlists_client] = lambda: fake_client
    client = TestClient(app)

    response = client.post("/playlists/save", json={"name": "My Mood", "track_ids": ["t1", "t2"]})

    assert response.status_code == 201
    assert response.json() == {"playlist_id": "new-pl"}
    assert fake_client.created == [("me", "My Mood")]
    assert fake_client.added == [("new-pl", ["t1", "t2"])]


def test_save_playlist_returns_403_when_missing_scope() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    fake_client = FakeSavePlaylistsClient(error=SpotifyApiError(403, {"error": "no permission"}))
    app.dependency_overrides[get_playlists_client] = lambda: fake_client
    client = TestClient(app)

    response = client.post("/playlists/save", json={"name": "My Mood", "track_ids": []})

    assert response.status_code == 403
    assert "reconnect" in response.json()["detail"].lower()


def test_save_playlist_returns_502_on_other_spotify_failures() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    fake_client = FakeSavePlaylistsClient(error=SpotifyApiError(500, {}))
    app.dependency_overrides[get_playlists_client] = lambda: fake_client
    client = TestClient(app)

    response = client.post("/playlists/save", json={"name": "My Mood", "track_ids": []})

    assert response.status_code == 502
