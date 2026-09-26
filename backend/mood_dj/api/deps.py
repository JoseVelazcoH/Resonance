"""FastAPI dependency wiring for auth and playlist endpoints."""

from __future__ import annotations

import time
from functools import lru_cache

from fastapi import Cookie, Depends, HTTPException

from mood_dj.adapters.in_memory_auth_state_store import InMemoryAuthStateStore
from mood_dj.adapters.laya_lyrics_judge import LayaLyricsJudge
from mood_dj.adapters.laya_prompt_profiler import LayaPromptProfiler
from mood_dj.adapters.laya_track_profiler import LayaTrackProfiler
from mood_dj.adapters.lrclib_lyrics import LrclibLyricsProvider
from mood_dj.adapters.sqlite_judgment_cache import SqliteJudgmentCache
from mood_dj.adapters.sqlite_lyrics_repository import SqliteLyricsRepository
from mood_dj.adapters.sqlite_mood_profile_repository import SqliteMoodProfileRepository
from mood_dj.adapters.sqlite_session_store import SqliteSessionStore
from mood_dj.adapters.spotify_auth import SpotifyAuthClient
from mood_dj.adapters.spotify_playlists import SpotifyPlaylistsClient
from mood_dj.application.library_lyrics_status import LibraryLyricsStatusStore
from mood_dj.application.library_store import LibraryStore
from mood_dj.application.prepare_library import PrepareLibraryUseCase
from mood_dj.application.prepare_library_job_manager import PrepareLibraryJobManager
from mood_dj.application.recommend_from_library import RecommendFromLibraryUseCase
from mood_dj.application.recommend_job_manager import RecommendJobManager
from mood_dj.config import Settings, load_settings
from mood_dj.domain.models import SpotifyTokens
from mood_dj.ports.auth_state_store import AuthStateStore
from mood_dj.ports.judgment_cache import JudgmentCache
from mood_dj.ports.lyrics_judge import LyricsJudge
from mood_dj.ports.lyrics_provider import LyricsProvider
from mood_dj.ports.lyrics_repository import LyricsRepository
from mood_dj.ports.mood_profile_repository import MoodProfileRepository
from mood_dj.ports.prompt_profiler import PromptProfiler
from mood_dj.ports.spotify_playlists import SpotifyPlaylistsClient as SpotifyPlaylistsClientPort
from mood_dj.ports.track_profiler import TrackProfiler
from mood_dj.ports.spotify_session_store import SpotifySessionStore

TOKEN_REFRESH_MARGIN_S = 30


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def get_auth_state_store() -> AuthStateStore:
    return InMemoryAuthStateStore()


@lru_cache(maxsize=1)
def get_session_store() -> SpotifySessionStore:
    return SqliteSessionStore(get_settings().app_db_path)


@lru_cache(maxsize=1)
def get_lyrics_repository() -> LyricsRepository:
    return SqliteLyricsRepository(get_settings().app_db_path)


@lru_cache(maxsize=1)
def get_auth_client() -> SpotifyAuthClient:
    return SpotifyAuthClient(client_id=get_settings().spotify_client_id)


@lru_cache(maxsize=1)
def get_playlists_client() -> SpotifyPlaylistsClientPort:
    return SpotifyPlaylistsClient()


@lru_cache(maxsize=1)
def get_lyrics_provider() -> LyricsProvider:
    return LrclibLyricsProvider()


@lru_cache(maxsize=1)
def get_judgment_cache() -> JudgmentCache:
    return SqliteJudgmentCache(get_settings().app_db_path)


@lru_cache(maxsize=1)
def get_lyrics_judge() -> LyricsJudge:
    # lru_cache ensures the Laya Router (and its loaded checkpoints) is built once
    # and shared across requests instead of being reloaded per call.
    return LayaLyricsJudge()


@lru_cache(maxsize=1)
def get_track_profiler() -> TrackProfiler:
    # lru_cache ensures the Laya Router (and its loaded checkpoints) is built once
    # and shared across requests instead of being reloaded per call.
    return LayaTrackProfiler(lyrics_char_budget=get_settings().track_lyrics_char_budget)


@lru_cache(maxsize=1)
def get_mood_profile_repository() -> MoodProfileRepository:
    return SqliteMoodProfileRepository(get_settings().app_db_path)


@lru_cache(maxsize=1)
def get_prompt_profiler() -> PromptProfiler:
    # lru_cache ensures the Laya Router (and its loaded checkpoints) is built once
    # and shared across requests instead of being reloaded per call.
    return LayaPromptProfiler()


@lru_cache(maxsize=1)
def get_library_store() -> LibraryStore:
    return LibraryStore()


@lru_cache(maxsize=1)
def get_library_lyrics_status_store() -> LibraryLyricsStatusStore:
    return LibraryLyricsStatusStore()


@lru_cache(maxsize=1)
def get_library_job_manager() -> PrepareLibraryJobManager:
    def factory() -> PrepareLibraryUseCase:
        return PrepareLibraryUseCase(
            playlists_client=get_playlists_client(),
            lyrics_repository=get_lyrics_repository(),
            lyrics_provider=get_lyrics_provider(),
            library_store=get_library_store(),
            lyrics_status_store=get_library_lyrics_status_store(),
            track_profiler=get_track_profiler(),
            mood_profile_repository=get_mood_profile_repository(),
            playlist_allowlist_file=get_settings().playlist_allowlist_file,
        )

    return PrepareLibraryJobManager(use_case_factory=factory, lyrics_status_store=get_library_lyrics_status_store())


@lru_cache(maxsize=1)
def get_recommend_from_library_use_case() -> RecommendFromLibraryUseCase:
    return RecommendFromLibraryUseCase(
        library_store=get_library_store(),
        lyrics_repository=get_lyrics_repository(),
        mood_profile_repository=get_mood_profile_repository(),
        prompt_profiler=get_prompt_profiler(),
        profile_version=get_track_profiler().version,
    )


@lru_cache(maxsize=1)
def get_recommend_job_manager() -> RecommendJobManager:
    return RecommendJobManager(use_case_factory=get_recommend_from_library_use_case)


def get_session_id(session_id: str | None = Cookie(default=None)) -> str:
    if session_id is None:
        raise HTTPException(status_code=401, detail="Not logged in")
    return session_id


def get_current_tokens(
    session_id: str = Depends(get_session_id),
    session_store: SpotifySessionStore = Depends(get_session_store),
    auth_client: SpotifyAuthClient = Depends(get_auth_client),
) -> SpotifyTokens:
    tokens = session_store.get(session_id)
    if tokens is None:
        raise HTTPException(status_code=401, detail="Not logged in")

    if tokens.expires_at <= time.time() + TOKEN_REFRESH_MARGIN_S:
        tokens = auth_client.refresh(tokens.refresh_token)
        session_store.save(session_id, tokens)

    return tokens
