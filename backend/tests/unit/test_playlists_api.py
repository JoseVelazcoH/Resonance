"""Unit tests for the /playlists endpoints, with adapters replaced by fakes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mood_dj.api.deps import get_current_tokens, get_playlists_client, get_recommend_job_manager, get_session_id
from mood_dj.api.main import app
from mood_dj.application.recommend_from_library import (
    Detected,
    DetectedEmotion,
    DetectedSituation,
    DetectedTarget,
    PlaylistRecommendation,
    PlaylistStage,
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


def test_list_playlists_returns_mapped_summaries() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    app.dependency_overrides[get_playlists_client] = lambda: FakePlaylistsClient()
    client = TestClient(app)

    response = client.get("/playlists")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["id"] == "pl1"
    assert body[0]["name"] == "My Playlist"


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
