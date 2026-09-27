"""Unit tests for the /player endpoints, with the SpotifyPlayerClient replaced by a fake."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mood_dj.api.deps import get_current_tokens, get_player_client
from mood_dj.api.main import app
from mood_dj.domain.models import SpotifyTokens
from mood_dj.ports.spotify_playlists import SpotifyApiError


class FakePlayerClient:
    def __init__(self, error: SpotifyApiError | None = None) -> None:
        self.error = error
        self.calls: list[dict] = []

    def play(self, device_id: str, track_uris: list[str], offset_index: int, access_token: str) -> None:
        self.calls.append(
            {"device_id": device_id, "track_uris": track_uris, "offset_index": offset_index, "access_token": access_token}
        )
        if self.error is not None:
            raise self.error


def _override(player_client=None) -> FakePlayerClient:
    fake = player_client or FakePlayerClient()
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="access-token", refresh_token="refresh", expires_at=9999999999.0
    )
    app.dependency_overrides[get_player_client] = lambda: fake
    return fake


def teardown_function() -> None:
    app.dependency_overrides = {}


def test_play_starts_playback_with_uris_and_offset() -> None:
    fake = _override()
    client = TestClient(app)

    response = client.put(
        "/player/play",
        json={"device_id": "device1", "track_ids": ["t1", "t2", "t3"], "offset_track_id": "t2"},
    )

    assert response.status_code == 200
    assert response.json() == {"playing": True}
    call = fake.calls[0]
    assert call["device_id"] == "device1"
    assert call["track_uris"] == ["spotify:track:t1", "spotify:track:t2", "spotify:track:t3"]
    assert call["offset_index"] == 1
    assert call["access_token"] == "access-token"


def test_play_defaults_offset_to_zero_without_offset_track_id() -> None:
    fake = _override()
    client = TestClient(app)

    response = client.put("/player/play", json={"device_id": "device1", "track_ids": ["t1", "t2"]})

    assert response.status_code == 200
    assert fake.calls[0]["offset_index"] == 0


def test_play_returns_404_when_device_not_ready() -> None:
    _override(FakePlayerClient(error=SpotifyApiError(404, {"error": {"message": "Device not found"}})))
    client = TestClient(app)

    response = client.put("/player/play", json={"device_id": "device1", "track_ids": ["t1"]})

    assert response.status_code == 404


def test_play_returns_403_when_missing_scope_or_not_premium() -> None:
    _override(FakePlayerClient(error=SpotifyApiError(403, {"error": {"message": "Premium required"}})))
    client = TestClient(app)

    response = client.put("/player/play", json={"device_id": "device1", "track_ids": ["t1"]})

    assert response.status_code == 403


def test_play_requires_login() -> None:
    app.dependency_overrides = {}
    client = TestClient(app)

    response = client.put("/player/play", json={"device_id": "device1", "track_ids": ["t1"]})

    assert response.status_code == 401
