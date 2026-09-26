"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel


class RecommendRequest(BaseModel):
    prompt: str


class TrackResponse(BaseModel):
    id: str
    name: str
    artist: str
    album: str
    energy: float
    valence: float
    tempo: float
    danceability: float
    acousticness: float
    instrumentalness: float
    cover_url: str | None
    external_url: str | None
    keep_probability: float | None


class ProfileResponse(BaseModel):
    energy: float
    valence: float
    tempo: float
    instrumentalness: float


class StageResponse(BaseModel):
    name: str
    profile: ProfileResponse
    tracks: list[TrackResponse]


class RecommendResponse(BaseModel):
    strategy: str
    strategy_probabilities: dict[str, float]
    stages: list[StageResponse]


class MeResponse(BaseModel):
    logged_in: bool
    display_name: str | None = None


class PlaylistSummaryResponse(BaseModel):
    id: str
    name: str
    image_url: str | None
    track_count: int
    snapshot_id: str


class PlaylistTrackResponse(BaseModel):
    id: str
    name: str
    artist: str
    album: str
    cover_url: str | None
    external_url: str | None
    keep_probability: float


class PlaylistStageResponse(BaseModel):
    name: str
    tracks: list[PlaylistTrackResponse]


class ExcludedResponse(BaseModel):
    no_lyrics: int
    instrumental: int
    no_profile: int = 0


class DetectedEmotionResponse(BaseModel):
    id: str
    label: str
    confidence: float


class DetectedSituationResponse(BaseModel):
    id: str
    label: str
    confidence: float


class DetectedTargetResponse(BaseModel):
    valence: float
    arousal: float


class DetectedResponse(BaseModel):
    emotion: DetectedEmotionResponse
    family_id: str
    situation: DetectedSituationResponse
    target: DetectedTargetResponse


class PlaylistRecommendResponse(BaseModel):
    strategy: str
    signals: dict[str, float]
    stages: list[PlaylistStageResponse]
    detected: DetectedResponse
    excluded: ExcludedResponse
    qualifying_count: int = 0
    threshold: float = 0.65


class LibraryPrepareStartedResponse(BaseModel):
    started: bool


class LibraryStatusResponse(BaseModel):
    state: str
    phase: str
    processed: int
    total: int
    cached: int
    playlists: int
    tracks: int
    with_lyrics: int
    instrumental: int
    missing: int
    playlists_processed: int = 0
    playlists_total: int = 0
    tracks_processed: int = 0
    tracks_total: int = 0
    profiles_processed: int = 0
    profiles_total: int = 0
    pending: int = 0
    failed_transient: int = 0
    error: str | None = None


class LibraryLyricsRowResponse(BaseModel):
    index: int
    track_id: str
    name: str
    artist: str
    status: str


class LibraryLyricsStatusResponse(BaseModel):
    state: str
    phase: str
    total: int
    processed: int
    with_lyrics: int
    instrumental: int
    missing: int
    pending: int
    playlists_processed: int = 0
    playlists_total: int = 0
    tracks_processed: int = 0
    tracks_total: int = 0
    failed_transient: int = 0
    rows: list[LibraryLyricsRowResponse]


class LibraryLyricsSummaryResponse(BaseModel):
    all_cached: bool


class LibraryRecommendRequest(BaseModel):
    prompt: str


class RecommendJobStartedResponse(BaseModel):
    job_id: str


class SavePlaylistRequest(BaseModel):
    name: str
    track_ids: list[str]


class SavePlaylistResponse(BaseModel):
    playlist_id: str


class RecommendJobStatusResponse(BaseModel):
    state: str
    phase: str
    processed: int
    total: int
    result: PlaylistRecommendResponse | None = None
    error: str | None = None
