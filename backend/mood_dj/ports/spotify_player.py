"""Port for starting playback on a Spotify Web Playback SDK device."""

from __future__ import annotations

from typing import Protocol


class SpotifyPlayerClient(Protocol):
    """Starts playback on a specific Spotify Connect device (the SDK player)."""

    def play(self, device_id: str, track_uris: list[str], offset_index: int, access_token: str) -> None:
        """Start playback of track_uris on device_id, beginning at offset_index.

        Raises SpotifyApiError (see mood_dj.ports.spotify_playlists) on failure, notably
        404 when the device is not (yet) known to Spotify and 403 when the required
        scope is missing or the account is not Premium.
        """
        ...
