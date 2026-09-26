"""Playlist listing, saving, and recommend-job status endpoints, requiring a logged-in session."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from mood_dj.api.deps import get_current_tokens, get_playlists_client, get_recommend_job_manager, get_session_id
from mood_dj.api.schemas import (
    DetectedEmotionResponse,
    DetectedResponse,
    DetectedSituationResponse,
    DetectedTargetResponse,
    ExcludedResponse,
    PlaylistRecommendResponse,
    PlaylistStageResponse,
    PlaylistSummaryResponse,
    PlaylistTrackResponse,
    RecommendJobStatusResponse,
    SavePlaylistRequest,
    SavePlaylistResponse,
)
from mood_dj.application.recommend_from_library import PlaylistRecommendation
from mood_dj.application.recommend_job_manager import RecommendJobManager
from mood_dj.domain.models import SpotifyTokens
from mood_dj.ports.spotify_playlists import SpotifyApiError, SpotifyPlaylistsClient

SPOTIFY_PERMISSION_ERROR_DETAIL = (
    "Missing permission to create playlists on Spotify. Please reconnect your Spotify account."
)

router = APIRouter(prefix="/playlists", tags=["playlists"])
recommend_jobs_router = APIRouter(prefix="/recommend-jobs", tags=["playlists"])


def _to_response(recommendation: PlaylistRecommendation) -> PlaylistRecommendResponse:
    return PlaylistRecommendResponse(
        strategy=recommendation.strategy.value,
        signals=recommendation.signal_probabilities,
        stages=[
            PlaylistStageResponse(
                name=stage.name,
                tracks=[
                    PlaylistTrackResponse(
                        id=ranked.track.id,
                        name=ranked.track.name,
                        artist=ranked.track.artist,
                        album=ranked.track.album,
                        cover_url=ranked.track.cover_url,
                        external_url=ranked.track.external_url,
                        keep_probability=ranked.similarity,
                    )
                    for ranked in stage.tracks
                ],
            )
            for stage in recommendation.stages
        ],
        detected=DetectedResponse(
            emotion=DetectedEmotionResponse(
                id=recommendation.detected.emotion.id,
                label=recommendation.detected.emotion.label,
                confidence=recommendation.detected.emotion.confidence,
            ),
            family_id=recommendation.detected.family_id,
            situation=DetectedSituationResponse(
                id=recommendation.detected.situation.id,
                label=recommendation.detected.situation.label,
                confidence=recommendation.detected.situation.confidence,
            ),
            target=DetectedTargetResponse(
                valence=recommendation.detected.target.valence,
                arousal=recommendation.detected.target.arousal,
            ),
        ),
        excluded=ExcludedResponse(
            no_lyrics=recommendation.excluded_no_lyrics,
            instrumental=recommendation.excluded_instrumental,
            no_profile=recommendation.excluded_no_profile,
        ),
        qualifying_count=recommendation.qualifying_count,
        threshold=recommendation.threshold,
    )


@router.get("", response_model=list[PlaylistSummaryResponse])
def list_playlists(
    tokens: SpotifyTokens = Depends(get_current_tokens),
    playlists_client: SpotifyPlaylistsClient = Depends(get_playlists_client),
):
    playlists = playlists_client.list_playlists(tokens.access_token)
    return [
        PlaylistSummaryResponse(
            id=p.id, name=p.name, image_url=p.image_url, track_count=p.track_count, snapshot_id=p.snapshot_id
        )
        for p in playlists
    ]


@router.post("/save", response_model=SavePlaylistResponse, status_code=201)
def save_playlist(
    request: SavePlaylistRequest,
    tokens: SpotifyTokens = Depends(get_current_tokens),
    playlists_client: SpotifyPlaylistsClient = Depends(get_playlists_client),
):
    try:
        user_id = playlists_client.get_current_user_id(tokens.access_token)
        playlist_id = playlists_client.create_playlist(user_id, request.name, tokens.access_token)
        playlists_client.add_tracks(playlist_id, request.track_ids, tokens.access_token)
    except SpotifyApiError as exc:
        if exc.status_code in (401, 403):
            raise HTTPException(status_code=exc.status_code, detail=SPOTIFY_PERMISSION_ERROR_DETAIL) from exc
        raise HTTPException(status_code=502, detail="Spotify request failed.") from exc

    return SavePlaylistResponse(playlist_id=playlist_id)


@recommend_jobs_router.get("/{job_id}", response_model=RecommendJobStatusResponse)
def recommend_job_status(
    job_id: str,
    tokens: SpotifyTokens = Depends(get_current_tokens),
    session_id: str = Depends(get_session_id),
    recommend_job_manager: RecommendJobManager = Depends(get_recommend_job_manager),
):
    job = recommend_job_manager.status(job_id, session_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Recommend job not found.")

    return RecommendJobStatusResponse(
        state=job.state.value,
        phase=job.phase,
        processed=job.processed,
        total=job.total,
        result=_to_response(job.result) if job.result is not None else None,
        error=job.error,
    )
