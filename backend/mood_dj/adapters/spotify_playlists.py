"""SpotifyPlaylistsClient adapter backed by the Spotify Web API (user token).

Uses `GET /v1/me/playlists` and `GET /v1/playlists/{id}/items`, following the `next`
link for pagination. Each playlist item holds the track under the `item` key, with
`track` as a fallback for older API responses. Null, local and episode entries are
skipped. NOTE: `GET /v1/playlists/{id}/tracks` answers 403 for this app tier, so it
must not be used; `/items` is the working equivalent.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from mood_dj.domain.models import PlaylistSummary, PlaylistTrack
from mood_dj.ports.spotify_playlists import SpotifyApiError

PLAYLISTS_URL = "https://api.spotify.com/v1/me/playlists"
CURRENT_USER_URL = "https://api.spotify.com/v1/me"
PLAYLISTS_PAGE_LIMIT = 50
ITEMS_PAGE_LIMIT = 50
REQUEST_TIMEOUT = 10.0


def playlist_items_url(playlist_id: str) -> str:
    return f"https://api.spotify.com/v1/playlists/{playlist_id}/items"


def playlist_tracks_url(playlist_id: str) -> str:
    return f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"


def user_playlists_url(user_id: str) -> str:
    return f"https://api.spotify.com/v1/users/{user_id}/playlists"


class SpotifyPlaylistsHttpClient(Protocol):
    """Thin boundary around the Spotify HTTP calls this adapter needs."""

    def get(self, url: str, access_token: str) -> dict:
        """Perform an authenticated GET and return the parsed JSON body."""
        ...

    def post(self, url: str, access_token: str, payload: dict) -> tuple[int, dict]:
        """Perform an authenticated POST and return (status_code, parsed JSON body)."""
        ...


class HttpxSpotifyPlaylistsHttpClient:
    """Real Spotify HTTP client, built on httpx."""

    def get(self, url: str, access_token: str) -> dict:
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def post(self, url: str, access_token: str, payload: dict) -> tuple[int, dict]:
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        try:
            body = response.json()
        except ValueError:
            body = {}
        return response.status_code, body


class SpotifyPlaylistsClient:
    """Reads a user's playlists and playlist tracks from the Spotify Web API."""

    def __init__(self, http_client: SpotifyPlaylistsHttpClient | None = None) -> None:
        self._http_client = http_client or HttpxSpotifyPlaylistsHttpClient()

    def list_playlists(self, access_token: str) -> list[PlaylistSummary]:
        user_id = self._http_client.get(CURRENT_USER_URL, access_token).get("id")
        playlists: list[PlaylistSummary] = []
        url: str | None = f"{PLAYLISTS_URL}?limit={PLAYLISTS_PAGE_LIMIT}"
        while url:
            page = self._http_client.get(url, access_token)
            for item in page.get("items", []):
                if self._is_readable(item, user_id):
                    playlists.append(self._map_playlist(item))
            url = page.get("next")
        return playlists

    def _is_readable(self, item: dict, user_id: str | None) -> bool:
        # Spotify answers 403 on the tracks of playlists the user only follows.
        owner_id = (item.get("owner") or {}).get("id")
        return owner_id is None or owner_id == user_id or bool(item.get("collaborative"))

    def get_playlist_tracks(self, playlist_id: str, access_token: str) -> list[PlaylistTrack]:
        tracks: list[PlaylistTrack] = []
        url: str | None = f"{playlist_items_url(playlist_id)}?limit={ITEMS_PAGE_LIMIT}"
        while url:
            page = self._http_client.get(url, access_token)
            for entry in page.get("items", []):
                track = self._map_track(entry)
                if track is not None:
                    tracks.append(track)
            url = page.get("next")
        return tracks

    def get_display_name(self, access_token: str) -> str | None:
        profile = self._http_client.get(CURRENT_USER_URL, access_token)
        return profile.get("display_name")

    def get_current_user_id(self, access_token: str) -> str:
        profile = self._http_client.get(CURRENT_USER_URL, access_token)
        return profile["id"]

    def create_playlist(self, user_id: str, name: str, access_token: str) -> str:
        payload = {"name": name, "public": False}
        status, body = self._http_client.post(PLAYLISTS_URL, access_token, payload)
        if status == 404:
            status, body = self._http_client.post(user_playlists_url(user_id), access_token, payload)
        if status >= 400:
            raise SpotifyApiError(status, body)
        return body["id"]

    def add_tracks(self, playlist_id: str, track_ids: list[str], access_token: str) -> None:
        if not track_ids:
            return
        payload = {"uris": [f"spotify:track:{track_id}" for track_id in track_ids]}
        status, body = self._http_client.post(playlist_items_url(playlist_id), access_token, payload)
        if status == 404:
            status, body = self._http_client.post(playlist_tracks_url(playlist_id), access_token, payload)
        if status >= 400:
            raise SpotifyApiError(status, body)

    def _map_playlist(self, item: dict) -> PlaylistSummary:
        images = item.get("images") or []
        return PlaylistSummary(
            id=item["id"],
            name=item["name"],
            image_url=images[0]["url"] if images else None,
            track_count=(item.get("items") or item.get("tracks") or {}).get("total", 0),
            snapshot_id=item.get("snapshot_id", ""),
        )

    def _map_track(self, entry: dict) -> PlaylistTrack | None:
        track = entry.get("item") or entry.get("track")
        if track is None:
            return None
        if track.get("is_local"):
            return None
        if track.get("type") == "episode":
            return None

        artists = track.get("artists") or []
        album = track.get("album") or {}
        album_images = album.get("images") or []
        return PlaylistTrack(
            id=track["id"],
            name=track.get("name", ""),
            artist=artists[0]["name"] if artists else "",
            album=album.get("name", ""),
            duration_s=track.get("duration_ms", 0) / 1000,
            cover_url=album_images[0]["url"] if album_images else None,
            external_url=track.get("external_urls", {}).get("spotify"),
        )
