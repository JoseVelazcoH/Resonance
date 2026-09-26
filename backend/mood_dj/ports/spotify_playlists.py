"""Port for reading a user's playlists and playlist tracks from Spotify."""

from __future__ import annotations

from typing import Protocol

from mood_dj.domain.models import PlaylistSummary, PlaylistTrack


class SpotifyApiError(Exception):
    """Raised when a Spotify Web API call fails, carrying the HTTP status code."""

    def __init__(self, status_code: int, body: dict) -> None:
        super().__init__(f"Spotify API request failed with status {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class SpotifyPlaylistsClient(Protocol):
    """Reads playlists and playlist tracks on behalf of an authenticated user."""

    def list_playlists(self, access_token: str) -> list[PlaylistSummary]:
        """Return all of the user's playlists, following pagination."""
        ...

    def get_playlist_tracks(self, playlist_id: str, access_token: str) -> list[PlaylistTrack]:
        """Return all tracks in a playlist, following pagination."""
        ...

    def get_display_name(self, access_token: str) -> str | None:
        """Return the authenticated user's Spotify display name, or None if unset."""
        ...

    def get_current_user_id(self, access_token: str) -> str:
        """Return the authenticated user's Spotify id."""
        ...

    def create_playlist(self, user_id: str, name: str, access_token: str) -> str:
        """Create a private playlist for the user and return its id."""
        ...

    def add_tracks(self, playlist_id: str, track_ids: list[str], access_token: str) -> None:
        """Add tracks to a playlist by Spotify track id."""
        ...
