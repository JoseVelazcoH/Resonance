"""Playlist listing, saving, and recommend-job status endpoints, requiring a logged-in session."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from mood_dj.adapters.playlist_allowlist import filter_playlists_by_allowlist, load_playlist_allowlist
from mood_dj.api.deps import (
    get_current_tokens,
    get_playlists_client,
    get_recommend_job_manager,
    get_session_id,
    get_settings,
)
from mood_dj.api.schemas import (
    DecisionsResponse,
    DetectedEmotionResponse,
    DetectedMoodResponse,
    DetectedResponse,
    DetectedSituationResponse,
    DetectedTargetResponse,
    ExcludedResponse,
    LibraryTrackResponse,
    PlaylistContributionResponse,
    PlaylistRecommendResponse,
    PlaylistStageResponse,
    PlaylistSummaryResponse,
    PlaylistTrackResponse,
    RankedTrackResponse,
    RecommendJobStatusResponse,
    SavePlaylistRequest,
    SavePlaylistResponse,
)
from mood_dj.application.recommend_from_library import (
    DecisionsSnapshot,
    Detected,
    LibraryTrackSummary,
    PlaylistRecommendation,
)
from mood_dj.application.recommend_job_manager import RecommendJobManager
from mood_dj.config import Settings
from mood_dj.domain.labels import english_emotion_label, english_mood_label, english_situation_label
from mood_dj.domain.models import SpotifyTokens
from mood_dj.ports.spotify_playlists import SpotifyApiError, SpotifyPlaylistsClient

SPOTIFY_PERMISSION_ERROR_DETAIL = (
    "Missing permission to create playlists on Spotify. Please reconnect your Spotify account."
)

router = APIRouter(prefix="/playlists", tags=["playlists"])
recommend_jobs_router = APIRouter(prefix="/recommend-jobs", tags=["playlists"])


def _detected_to_response(detected: Detected) -> DetectedResponse:
    # Labels are translated to English at this API boundary via a separate
    # labels_en.json lookup (see mood_dj.domain.labels), so the UI is always
    # in English without editing emotions.json/moods.json (which would bump
    # laya_track_profiler.compute_version and force a full library re-profile).
    return DetectedResponse(
        emotion=DetectedEmotionResponse(
            id=detected.emotion.id,
            label=english_emotion_label(detected.emotion.id, detected.emotion.label),
            confidence=detected.emotion.confidence,
        ),
        family_id=detected.family_id,
        situation=DetectedSituationResponse(
            id=detected.situation.id,
            label=english_situation_label(detected.situation.id, detected.situation.label),
            confidence=detected.situation.confidence,
        ),
        target=DetectedTargetResponse(
            valence=detected.target.valence,
            arousal=detected.target.arousal,
        ),
        mood=DetectedMoodResponse(
            id=detected.mood.id,
            label=english_mood_label(detected.mood.id, detected.mood.label),
        ),
    )


def _library_tracks_to_response(tracks: list[LibraryTrackSummary]) -> list[LibraryTrackResponse]:
    return [
        LibraryTrackResponse(id=track.id, name=track.name, artist=track.artist, cover_url=track.cover_url)
        for track in tracks
    ]


def _decisions_to_response(decisions: DecisionsSnapshot) -> DecisionsResponse:
    return DecisionsResponse(
        strategy=decisions.strategy.value,
        signals=decisions.signal_probabilities,
        detected=_detected_to_response(decisions.detected),
    )


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
                        duration_s=ranked.track.duration_s,
                        cover_url=ranked.track.cover_url,
                        external_url=ranked.track.external_url,
                        keep_probability=ranked.similarity,
                    )
                    for ranked in stage.tracks
                ],
            )
            for stage in recommendation.stages
        ],
        detected=_detected_to_response(recommendation.detected),
        excluded=ExcludedResponse(
            no_lyrics=recommendation.excluded_no_lyrics,
            instrumental=recommendation.excluded_instrumental,
            no_profile=recommendation.excluded_no_profile,
        ),
        qualifying_count=recommendation.qualifying_count,
        threshold=recommendation.threshold,
        playlist_contributions=[
            PlaylistContributionResponse(
                playlist_id=contribution.playlist_id,
                name=contribution.name,
                image_url=contribution.image_url,
                track_count=contribution.track_count,
                contributed=contribution.contributed,
            )
            for contribution in recommendation.playlist_contributions
        ],
        ranked_tracks=[
            RankedTrackResponse(
                id=ranked.id,
                name=ranked.name,
                artist=ranked.artist,
                cover_url=ranked.cover_url,
                similarity=ranked.similarity,
                selected=ranked.selected,
            )
            for ranked in recommendation.ranked_tracks
        ],
    )


@router.get("", response_model=list[PlaylistSummaryResponse])
def list_playlists(
    tokens: SpotifyTokens = Depends(get_current_tokens),
    playlists_client: SpotifyPlaylistsClient = Depends(get_playlists_client),
    settings: Settings = Depends(get_settings),
):
    playlists = playlists_client.list_playlists(tokens.access_token)
    allowlist_names = load_playlist_allowlist(settings.playlist_allowlist_file)
    playlists, _not_found = filter_playlists_by_allowlist(playlists, allowlist_names)
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
        decisions=_decisions_to_response(job.decisions) if job.decisions is not None else None,
        library_tracks=_library_tracks_to_response(job.library_tracks) if job.library_tracks is not None else None,
    )
