"""Unit tests for the SpotifyPlayerClient adapter, with the HTTP client faked."""

from __future__ import annotations

from mood_dj.adapters.spotify_player import PLAYER_PLAY_URL, HttpxSpotifyPlayerHttpClient, SpotifyPlayerClient
from mood_dj.ports.spotify_playlists import SpotifyApiError


class FakeHttpClient:
    def __init__(self, status: int = 204, body: dict | None = None) -> None:
        self.status = status
        self.body = body or {}
        self.calls: list[dict] = []

    def put(self, url: str, access_token: str, payload: dict):
        self.calls.append({"url": url, "access_token": access_token, "payload": payload})
        return self.status, self.body


def test_play_puts_uris_and_offset_with_device_id_query_param() -> None:
    http_client = FakeHttpClient()
    client = SpotifyPlayerClient(http_client)

    client.play(
        device_id="device1",
        track_uris=["spotify:track:t1", "spotify:track:t2"],
        offset_index=1,
        access_token="token",
    )

    call = http_client.calls[0]
    assert call["url"] == f"{PLAYER_PLAY_URL}?device_id=device1"
    assert call["access_token"] == "token"
    assert call["payload"] == {"uris": ["spotify:track:t1", "spotify:track:t2"], "offset": {"position": 1}}


def test_play_raises_spotify_api_error_on_404_device_not_ready() -> None:
    http_client = FakeHttpClient(status=404, body={"error": {"message": "Device not found"}})
    client = SpotifyPlayerClient(http_client)

    try:
        client.play(device_id="device1", track_uris=["spotify:track:t1"], offset_index=0, access_token="token")
        assert False, "expected SpotifyApiError"
    except SpotifyApiError as exc:
        assert exc.status_code == 404


def test_play_raises_spotify_api_error_on_403_missing_scope_or_not_premium() -> None:
    http_client = FakeHttpClient(status=403, body={"error": {"message": "Premium required"}})
    client = SpotifyPlayerClient(http_client)

    try:
        client.play(device_id="device1", track_uris=["spotify:track:t1"], offset_index=0, access_token="token")
        assert False, "expected SpotifyApiError"
    except SpotifyApiError as exc:
        assert exc.status_code == 403


def test_httpx_client_can_be_constructed() -> None:
    assert HttpxSpotifyPlayerHttpClient() is not None
