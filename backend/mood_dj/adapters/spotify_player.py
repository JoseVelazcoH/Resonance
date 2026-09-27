"""SpotifyPlayerClient adapter backed by the Spotify Web API "Start/Resume Playback"
endpoint: PUT /v1/me/player/play?device_id=... with body {"uris": [...], "offset": {...}}.
Reference: https://developer.spotify.com/documentation/web-api/reference/start-a-users-playback
"""

from __future__ import annotations

from typing import Protocol

import httpx

from mood_dj.ports.spotify_playlists import SpotifyApiError

PLAYER_PLAY_URL = "https://api.spotify.com/v1/me/player/play"
REQUEST_TIMEOUT = 10.0


class SpotifyPlayerHttpClient(Protocol):
    """Thin boundary around the Spotify player HTTP call this adapter needs."""

    def put(self, url: str, access_token: str, payload: dict) -> tuple[int, dict]:
        """Perform an authenticated PUT and return (status_code, parsed JSON body)."""
        ...


class HttpxSpotifyPlayerHttpClient:
    """Real Spotify HTTP client, built on httpx."""

    def put(self, url: str, access_token: str, payload: dict) -> tuple[int, dict]:
        response = httpx.put(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 204 or not response.content:
            return response.status_code, {}
        try:
            body = response.json()
        except ValueError:
            body = {}
        return response.status_code, body


class SpotifyPlayerClient:
    """Starts playback on a Web Playback SDK device via the Spotify Web API."""

    def __init__(self, http_client: SpotifyPlayerHttpClient | None = None) -> None:
        self._http_client = http_client or HttpxSpotifyPlayerHttpClient()

    def play(self, device_id: str, track_uris: list[str], offset_index: int, access_token: str) -> None:
        payload = {"uris": track_uris, "offset": {"position": offset_index}}
        url = f"{PLAYER_PLAY_URL}?device_id={device_id}"
        status, body = self._http_client.put(url, access_token, payload)
        if status >= 400:
            raise SpotifyApiError(status, body)
