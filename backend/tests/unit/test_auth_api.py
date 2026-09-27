"""Unit tests for the /auth endpoints, with all adapters replaced by fakes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mood_dj.api.deps import (
    get_auth_client,
    get_auth_state_store,
    get_playlists_client,
    get_session_store,
    get_settings,
)
from mood_dj.api.main import app
from mood_dj.config import Settings
from mood_dj.domain.models import SpotifyTokens


class FakeAuthStateStore:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def save(self, state: str, code_verifier: str) -> None:
        self._store[state] = code_verifier

    def pop(self, state: str) -> str | None:
        return self._store.pop(state, None)


class FakeSessionStore:
    def __init__(self) -> None:
        self._store: dict[str, SpotifyTokens] = {}

    def save(self, session_id: str, tokens: SpotifyTokens) -> None:
        self._store[session_id] = tokens

    def get(self, session_id: str) -> SpotifyTokens | None:
        return self._store.get(session_id)

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class FakeAuthClient:
    def __init__(self) -> None:
        self.exchanged: list[dict] = []

    def exchange_code(self, code: str, redirect_uri: str, code_verifier: str) -> SpotifyTokens:
        self.exchanged.append({"code": code, "redirect_uri": redirect_uri, "code_verifier": code_verifier})
        return SpotifyTokens(access_token="access", refresh_token="refresh", expires_at=9999999999.0)


class FakePlaylistsClient:
    def __init__(self, display_name: str | None = "user") -> None:
        self.display_name = display_name

    def list_playlists(self, access_token: str):
        return []

    def get_playlist_tracks(self, playlist_id: str, access_token: str):
        return []

    def get_display_name(self, access_token: str) -> str | None:
        return self.display_name


def _settings() -> Settings:
    return Settings(
        spotify_client_id="client123",
        spotify_client_secret="secret",
        dataset_path="data/tracks.parquet",
        app_db_path="data/app.db",
        frontend_url="http://127.0.0.1:5173",
        spotify_redirect_uri="http://127.0.0.1:8000/auth/callback",
    )


def _override(auth_state_store=None, session_store=None, auth_client=None, playlists_client=None):
    app.dependency_overrides[get_settings] = _settings
    app.dependency_overrides[get_auth_state_store] = lambda: (auth_state_store or FakeAuthStateStore())
    app.dependency_overrides[get_session_store] = lambda: (session_store or FakeSessionStore())
    app.dependency_overrides[get_auth_client] = lambda: (auth_client or FakeAuthClient())
    app.dependency_overrides[get_playlists_client] = lambda: (playlists_client or FakePlaylistsClient())


def teardown_function() -> None:
    app.dependency_overrides = {}


def test_login_redirects_to_spotify_authorize_with_exact_redirect_uri() -> None:
    _override()
    client = TestClient(app, follow_redirects=False)

    response = client.get("/auth/login")

    assert response.status_code == 307
    location = response.headers["location"]
    assert location.startswith("https://accounts.spotify.com/authorize")
    assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A8000%2Fauth%2Fcallback" in location


def test_me_reports_logged_out_without_cookie() -> None:
    _override()
    client = TestClient(app)

    response = client.get("/auth/me")

    assert response.status_code == 200
    assert response.json() == {"logged_in": False, "display_name": None}


def test_callback_exchanges_code_and_sets_session_cookie() -> None:
    state_store = FakeAuthStateStore()
    state_store.save("state123", "verifier123")
    session_store = FakeSessionStore()
    auth_client = FakeAuthClient()
    _override(auth_state_store=state_store, session_store=session_store, auth_client=auth_client)
    client = TestClient(app, follow_redirects=False)

    response = client.get("/auth/callback", params={"code": "abc", "state": "state123"})

    assert response.status_code == 307
    assert response.headers["location"] == "http://127.0.0.1:5173"
    assert "session_id" in response.cookies
    assert auth_client.exchanged[0]["code"] == "abc"
    assert auth_client.exchanged[0]["code_verifier"] == "verifier123"


def test_callback_rejects_unknown_state() -> None:
    _override()
    client = TestClient(app, follow_redirects=False)

    response = client.get("/auth/callback", params={"code": "abc", "state": "unknown"})

    assert response.status_code == 400


def test_callback_rejects_spotify_error() -> None:
    _override()
    client = TestClient(app, follow_redirects=False)

    response = client.get("/auth/callback", params={"error": "access_denied", "state": "x"})

    assert response.status_code == 400


def test_me_reports_logged_in_after_callback() -> None:
    state_store = FakeAuthStateStore()
    state_store.save("state123", "verifier123")
    session_store = FakeSessionStore()
    _override(auth_state_store=state_store, session_store=session_store, playlists_client=FakePlaylistsClient("Jose"))
    client = TestClient(app, follow_redirects=False)

    client.get("/auth/callback", params={"code": "abc", "state": "state123"})
    response = client.get("/auth/me")

    assert response.json() == {"logged_in": True, "display_name": "Jose"}


def test_me_reports_none_display_name_when_spotify_profile_unavailable() -> None:
    state_store = FakeAuthStateStore()
    state_store.save("state123", "verifier123")
    session_store = FakeSessionStore()
    _override(auth_state_store=state_store, session_store=session_store, playlists_client=FakePlaylistsClient(None))
    client = TestClient(app, follow_redirects=False)

    client.get("/auth/callback", params={"code": "abc", "state": "state123"})
    response = client.get("/auth/me")

    assert response.json() == {"logged_in": True, "display_name": None}


def test_logout_clears_session() -> None:
    session_store = FakeSessionStore()
    session_store.save("sess1", SpotifyTokens(access_token="a", refresh_token="r", expires_at=99999999999.0))
    _override(session_store=session_store)
    client = TestClient(app)
    client.cookies.set("session_id", "sess1")

    response = client.post("/auth/logout")

    assert response.status_code == 200
    assert session_store.get("sess1") is None


def test_token_requires_login() -> None:
    _override()
    client = TestClient(app)

    response = client.get("/auth/token")

    assert response.status_code == 401


def test_token_returns_access_token_without_refresh_token() -> None:
    session_store = FakeSessionStore()
    session_store.save("sess1", SpotifyTokens(access_token="access-abc", refresh_token="refresh-xyz", expires_at=9999999999.0))
    _override(session_store=session_store)
    client = TestClient(app)
    client.cookies.set("session_id", "sess1")

    response = client.get("/auth/token")

    assert response.status_code == 200
    body = response.json()
    assert body == {"access_token": "access-abc", "expires_at": 9999999999.0}
    assert "refresh-xyz" not in response.text
    assert response.headers["cache-control"] == "no-store"
