"""Whole-library prepare and recommend endpoints, requiring a logged-in session."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from mood_dj.api.deps import (
    get_current_tokens,
    get_library_diagnostics_use_case,
    get_library_job_manager,
    get_recommend_job_manager,
    get_session_id,
)
from mood_dj.api.schemas import (
    LibraryDiagnosticsResponse,
    LibraryLyricsRowResponse,
    LibraryLyricsStatusResponse,
    LibraryLyricsSummaryResponse,
    LibraryPrepareStartedResponse,
    LibraryRecommendRequest,
    LibraryStatusResponse,
    MoodDiagnosticsResponse,
    RecommendJobStartedResponse,
)
from mood_dj.application.library_diagnostics import LibraryDiagnosticsUseCase
from mood_dj.application.prepare_library_job_manager import PrepareLibraryJobManager
from mood_dj.application.recommend_job_manager import RecommendJobManager
from mood_dj.domain.models import LibraryPrepareState, SpotifyTokens

NOT_PREPARED_DETAIL = "Library not prepared yet. Call POST /library/prepare first."

router = APIRouter(prefix="/library", tags=["library"])


@router.post("/prepare", response_model=LibraryPrepareStartedResponse)
def prepare_library(
    tokens: SpotifyTokens = Depends(get_current_tokens),
    session_id: str = Depends(get_session_id),
    job_manager: PrepareLibraryJobManager = Depends(get_library_job_manager),
):
    started = job_manager.start(session_id, tokens.access_token)
    return LibraryPrepareStartedResponse(started=started)


@router.get("/status", response_model=LibraryStatusResponse)
def library_status(
    session_id: str = Depends(get_session_id),
    job_manager: PrepareLibraryJobManager = Depends(get_library_job_manager),
):
    progress = job_manager.status(session_id)
    return LibraryStatusResponse(
        state=progress.state.value,
        phase=progress.phase,
        processed=progress.processed,
        total=progress.total,
        cached=progress.cached,
        playlists=progress.playlists,
        tracks=progress.tracks,
        with_lyrics=progress.with_lyrics,
        instrumental=progress.instrumental,
        missing=progress.missing,
        playlists_processed=progress.playlists_processed,
        playlists_total=progress.playlists_total,
        tracks_processed=progress.tracks_processed,
        tracks_total=progress.tracks_total,
        profiles_processed=progress.profiles_processed,
        profiles_total=progress.profiles_total,
        pending=progress.pending,
        failed_transient=progress.failed_transient,
        error=progress.error,
    )


@router.get("/lyrics/status", response_model=LibraryLyricsStatusResponse)
def lyrics_status(
    offset: int | None = None,
    limit: int = 50,
    session_id: str = Depends(get_session_id),
    job_manager: PrepareLibraryJobManager = Depends(get_library_job_manager),
):
    progress = job_manager.status(session_id)
    rows, _total_rows = job_manager.lyrics_rows(session_id, offset, limit)
    summary = job_manager.lyrics_summary(session_id)

    return LibraryLyricsStatusResponse(
        state=progress.state.value,
        phase=progress.phase,
        total=progress.total,
        processed=progress.processed,
        with_lyrics=summary.with_lyrics if summary is not None else progress.with_lyrics,
        instrumental=summary.instrumental if summary is not None else progress.instrumental,
        missing=summary.missing if summary is not None else progress.missing,
        pending=summary.pending if summary is not None else progress.pending,
        playlists_processed=progress.playlists_processed,
        playlists_total=progress.playlists_total,
        tracks_processed=progress.tracks_processed,
        tracks_total=progress.tracks_total,
        failed_transient=progress.failed_transient,
        rows=[
            LibraryLyricsRowResponse(index=row.index, track_id=row.track_id, name=row.name, artist=row.artist, status=row.status)
            for row in rows
        ],
    )


@router.get("/lyrics/summary", response_model=LibraryLyricsSummaryResponse)
def lyrics_summary(
    session_id: str = Depends(get_session_id),
    job_manager: PrepareLibraryJobManager = Depends(get_library_job_manager),
):
    summary = job_manager.lyrics_summary(session_id)
    return LibraryLyricsSummaryResponse(all_cached=summary.all_cached if summary is not None else False)


@router.post("/recommend", response_model=RecommendJobStartedResponse, status_code=202)
def recommend_from_library(
    request: LibraryRecommendRequest,
    tokens: SpotifyTokens = Depends(get_current_tokens),
    session_id: str = Depends(get_session_id),
    job_manager: PrepareLibraryJobManager = Depends(get_library_job_manager),
    recommend_job_manager: RecommendJobManager = Depends(get_recommend_job_manager),
):
    progress = job_manager.status(session_id)
    if progress.state not in (LibraryPrepareState.DONE, LibraryPrepareState.PARTIAL):
        raise HTTPException(status_code=409, detail=NOT_PREPARED_DETAIL)
    if progress.profiles_total <= 0:
        raise HTTPException(status_code=409, detail=NOT_PREPARED_DETAIL)

    job_id = recommend_job_manager.start(session_id, request.prompt)
    return RecommendJobStartedResponse(job_id=job_id)


@router.get("/diagnostics", response_model=LibraryDiagnosticsResponse)
def library_diagnostics(
    session_id: str = Depends(get_session_id),
    diagnostics_use_case: LibraryDiagnosticsUseCase = Depends(get_library_diagnostics_use_case),
):
    report = diagnostics_use_case.run(session_id)
    return LibraryDiagnosticsResponse(
        moods=[
            MoodDiagnosticsResponse(
                mood_id=mood.mood_id,
                count=mood.count,
                mean_confidence=mood.mean_confidence,
                mean_positive_probability=mood.mean_positive_probability,
                mean_entropy=mood.mean_entropy,
                mean_probabilities=mood.mean_probabilities,
            )
            for mood in report.moods
        ],
        profiled_count=report.profiled_count,
        total_tracks=report.total_tracks,
        unprofiled_share=report.unprofiled_share,
        overall_mean_entropy=report.overall_mean_entropy,
    )
