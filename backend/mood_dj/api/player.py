"""Start playback on the Spotify Web Playback SDK device, requiring a logged-in session."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from mood_dj.api.deps import get_current_tokens, get_player_client
from mood_dj.api.schemas import PlayerPlayRequest
from mood_dj.domain.models import SpotifyTokens
from mood_dj.ports.spotify_player import SpotifyPlayerClient
from mood_dj.ports.spotify_playlists import SpotifyApiError

PLAYER_DEVICE_NOT_READY_DETAIL = "Spotify player is not ready yet. Please wait a moment and try again."
PLAYER_PERMISSION_ERROR_DETAIL = (
    "Could not start playback. This requires Spotify Premium and the streaming permission; "
    "please reconnect your Spotify account."
)

router = APIRouter(prefix="/player", tags=["player"])


@router.put("/play")
def play(
    request: PlayerPlayRequest,
    tokens: SpotifyTokens = Depends(get_current_tokens),
    player_client: SpotifyPlayerClient = Depends(get_player_client),
):
    track_uris = [f"spotify:track:{track_id}" for track_id in request.track_ids]
    offset_index = 0
    if request.offset_track_id is not None and request.offset_track_id in request.track_ids:
        offset_index = request.track_ids.index(request.offset_track_id)

    try:
        player_client.play(
            device_id=request.device_id,
            track_uris=track_uris,
            offset_index=offset_index,
            access_token=tokens.access_token,
        )
    except SpotifyApiError as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail=PLAYER_DEVICE_NOT_READY_DETAIL) from exc
        if exc.status_code == 403:
            raise HTTPException(status_code=403, detail=PLAYER_PERMISSION_ERROR_DETAIL) from exc
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"playing": True}
