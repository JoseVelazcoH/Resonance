"""Core domain models for the Mood DJ system.

These types carry no framework dependencies. They describe the vocabulary of the
domain: strategies, target audio profiles, tracks, stages and the final decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Strategy(str, Enum):
    """The high-level approach Laya picks for a listening session."""

    ACCOMPANY = "accompany"
    LIFT = "lift"
    ENERGIZE = "energize"
    CALM = "calm"


@dataclass(frozen=True)
class MoodProfile:
    """A target audio profile Laya wants the catalog to match.

    All fields are normalized to the 0.0-1.0 range except tempo, which is in BPM.
    """

    energy: float
    valence: float
    tempo: float
    instrumentalness: float


@dataclass(frozen=True)
class Track:
    """A single track candidate, enriched with catalog features and cover art."""

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
    cover_url: str | None = None
    external_url: str | None = None
    keep_probability: float | None = None


@dataclass(frozen=True)
class Stage:
    """One stage of the listening session: a target profile plus chosen tracks."""

    name: str
    profile: MoodProfile
    tracks: list[Track] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    """The full decision trace returned to the caller for transparency."""

    strategy: Strategy
    strategy_probabilities: dict[str, float]
    stages: list[Stage] = field(default_factory=list)


class LyricsStatus(str, Enum):
    """The outcome of looking up lyrics for a track."""

    LYRICS = "lyrics"
    INSTRUMENTAL = "instrumental"
    MISSING = "missing"


@dataclass(frozen=True)
class LyricsEntry:
    """A cached lyrics lookup result for one Spotify track."""

    track_id: str
    status: LyricsStatus
    text: str | None
    fetched_at: str


@dataclass(frozen=True)
class SpotifyTokens:
    """OAuth tokens for one authenticated Spotify user session."""

    access_token: str
    refresh_token: str
    expires_at: float


@dataclass(frozen=True)
class PlaylistSummary:
    """A user playlist as listed by `GET /v1/me/playlists`."""

    id: str
    name: str
    image_url: str | None
    track_count: int
    snapshot_id: str


@dataclass(frozen=True)
class PlaylistTrack:
    """A single track inside a playlist, mapped from the Spotify API response."""

    id: str
    name: str
    artist: str
    album: str
    duration_s: float
    cover_url: str | None
    external_url: str | None


class PrepareState(str, Enum):
    """The lifecycle state of a playlist preparation job."""

    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class PrepareProgress:
    """Progress counters for a playlist preparation job."""

    state: PrepareState = PrepareState.IDLE
    total: int = 0
    processed: int = 0
    with_lyrics: int = 0
    instrumental: int = 0
    missing: int = 0
    error: str | None = None


class LibraryPhase(str, Enum):
    """The stage a library preparation job is currently in.

    Preparation reads playlists, fetches lyrics, then computes a per-track mood
    profile (`READING_MOOD`) for every track that has lyrics. Prompt-dependent
    tone/fit judging still happens later, playlist by playlist, as part of the
    recommend job (see `RecommendFromLibraryUseCase`).
    """

    READING_PLAYLISTS = "reading playlists"
    FETCHING_LYRICS = "fetching lyrics"
    READING_MOOD = "reading mood"


class LibraryPrepareState(str, Enum):
    """The lifecycle state of a library preparation job."""

    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class LibraryPrepareProgress:
    """Progress counters for a whole-library preparation job.

    `processed`/`total` are scoped to the current `phase` (kept for backward
    compatibility): playlist count while reading playlists, track count while
    fetching lyrics. `playlists_processed`/`playlists_total` and
    `tracks_processed`/`tracks_total` are always populated regardless of the
    current phase, so a client can show an unambiguous, phase-aware label (and a
    combined progress bar) instead of guessing what `processed`/`total` currently
    means. `cached` is the number of tracks that already had lyrics cached when the
    fetch phase started; `processed`/`tracks_processed` include those from the very
    first poll, so the progress bar never appears to restart from zero.

    `pending` is the number of tracks still unresolved (no definitive lyrics
    status) after all fetch/retry passes finished; the job only reaches `DONE`
    when `pending` is 0. Otherwise it finishes as `PARTIAL`, with `pending` and
    `failed_transient` describing what is left, so a later `POST /library/prepare`
    can resume just those tracks.

    `profiles_processed`/`profiles_total` cover the `READING_MOOD` phase: the
    number of lyrics-bearing tracks already profiled (cached or freshly computed)
    out of the total that need one.
    """

    state: LibraryPrepareState = LibraryPrepareState.IDLE
    phase: str = LibraryPhase.READING_PLAYLISTS.value
    processed: int = 0
    total: int = 0
    cached: int = 0
    playlists: int = 0
    tracks: int = 0
    with_lyrics: int = 0
    instrumental: int = 0
    missing: int = 0
    playlists_processed: int = 0
    playlists_total: int = 0
    tracks_processed: int = 0
    tracks_total: int = 0
    profiles_processed: int = 0
    profiles_total: int = 0
    pending: int = 0
    failed_transient: int = 0
    error: str | None = None


class LibraryLyricsTrackStatus(str, Enum):
    """Per-track lyrics-download status, for the UI status table."""

    DOWNLOADED = "downloaded"
    DOWNLOADING = "downloading"
    PENDING = "pending"
    MISSING = "missing"
    INSTRUMENTAL = "instrumental"


@dataclass
class LibraryLyricsRow:
    """One row of the per-track lyrics status table."""

    index: int
    track_id: str
    name: str
    artist: str
    status: str


@dataclass(frozen=True)
class TrackMoodProfile:
    """A per-track mood profile computed from lyrics by `LayaTrackProfiler`.

    `valence`/`arousal` are normalized to [-1, 1]. `polarity_id`/`cluster_id`/
    `family_id` are the taxonomy node ids chosen by descending the emotion tree
    (see `mood_dj.domain.taxonomy`); `cluster_id` equals the polarity's implicit
    single cluster id when that level was skipped (only one child). Tree descent
    for tracks stops at the family level for performance, so `emotion_id` and
    `emotion_confidence` are `None` for profiles computed after that change;
    older cached profiles (a different `version`) may still carry a real
    `emotion_id` and `emotion_confidence`, kept here for backward compatibility.
    `situation_id`/`situation_confidence` are `None` for profiles computed since
    situations were dropped from track-level questions (a track's situation was
    replaced by a prompt-side situation relatedness bonus computed against the
    track's family/cluster, see `mood_dj.domain.track_ranking`); older cached
    profiles (a different `version`) may still carry real values here, kept for
    backward compatibility. `version` ties the profile to the exact taxonomy +
    question wording it was computed under, so a taxonomy change invalidates old
    rows automatically.
    """

    track_id: str
    valence: float
    arousal: float
    polarity_id: str
    cluster_id: str
    family_id: str
    version: str
    emotion_id: str | None = None
    emotion_confidence: float | None = None
    situation_id: str | None = None
    situation_confidence: float | None = None
