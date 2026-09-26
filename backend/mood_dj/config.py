"""Application configuration, loaded from environment variables and `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DATASET_PATH = "data/tracks.parquet"
DEFAULT_APP_DB_PATH = "data/app.db"
DEFAULT_FRONTEND_URL = "http://127.0.0.1:5173"
DEFAULT_SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8000/auth/callback"
DEFAULT_PLAYLIST_ALLOWLIST_FILE = "playlists.local.json"
SESSION_COOKIE_NAME = "session_id"
DEFAULT_TRACK_LYRICS_CHAR_BUDGET = 500

# The backend package root (this file's directory's parent), used to resolve
# PLAYLIST_ALLOWLIST_FILE relative to `backend/` regardless of the process cwd.
BACKEND_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    spotify_client_id: str
    spotify_client_secret: str
    dataset_path: str
    app_db_path: str
    frontend_url: str
    spotify_redirect_uri: str
    playlist_allowlist_file: str = DEFAULT_PLAYLIST_ALLOWLIST_FILE
    track_lyrics_char_budget: int = DEFAULT_TRACK_LYRICS_CHAR_BUDGET


def load_settings() -> Settings:
    load_dotenv()
    allowlist_file = os.environ.get("PLAYLIST_ALLOWLIST_FILE", DEFAULT_PLAYLIST_ALLOWLIST_FILE)
    allowlist_path = allowlist_file if Path(allowlist_file).is_absolute() else str(BACKEND_ROOT / allowlist_file)
    return Settings(
        spotify_client_id=os.environ.get("SPOTIFY_CLIENT_ID", ""),
        spotify_client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET", ""),
        dataset_path=os.environ.get("DATASET_PATH", DEFAULT_DATASET_PATH),
        app_db_path=os.environ.get("APP_DB_PATH", DEFAULT_APP_DB_PATH),
        frontend_url=os.environ.get("FRONTEND_URL", DEFAULT_FRONTEND_URL),
        spotify_redirect_uri=os.environ.get("SPOTIFY_REDIRECT_URI", DEFAULT_SPOTIFY_REDIRECT_URI),
        playlist_allowlist_file=allowlist_path,
        track_lyrics_char_budget=int(
            os.environ.get("TRACK_LYRICS_CHAR_BUDGET", str(DEFAULT_TRACK_LYRICS_CHAR_BUDGET))
        ),
    )


def dataset_exists(settings: Settings) -> bool:
    return Path(settings.dataset_path).exists()
