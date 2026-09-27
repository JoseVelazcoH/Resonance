"""Spotify user auth adapter: Authorization Code flow with PKCE.

PKCE is used so the client secret is not required for the token exchange itself
(Spotify still accepts it, but it is not needed and this keeps the flow simpler and
consistent with public-client best practice). Scopes requested: playlist-read-private
and playlist-read-collaborative to read the user's playlists, and
playlist-modify-private to save a generated playlist back to their account.
Additional scopes for the Web Playback SDK (see
https://developer.spotify.com/documentation/web-playback-sdk): streaming (required to
instantiate an SDK player), user-read-email and user-read-private (required by the SDK
per the official quick start), and user-read-playback-state / user-modify-playback-state
so the backend can start/transfer playback on the SDK device via the Web API.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from typing import Protocol
from urllib.parse import urlencode

import httpx

from mood_dj.domain.models import SpotifyTokens

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPES = (
    "playlist-read-private playlist-read-collaborative playlist-modify-private "
    "streaming user-read-email user-read-private user-read-playback-state user-modify-playback-state"
)
REQUEST_TIMEOUT = 10.0


def generate_code_verifier() -> str:
    """Generate a PKCE code verifier: 43-128 chars from the unreserved URL charset."""
    return secrets.token_urlsafe(64)[:64]


def generate_code_challenge(code_verifier: str) -> str:
    """Derive the S256 PKCE code challenge from a code verifier."""
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def generate_state() -> str:
    """Generate a random opaque value to protect the OAuth redirect against CSRF."""
    return secrets.token_urlsafe(32)


def build_authorize_url(client_id: str, redirect_uri: str, state: str, code_verifier: str) -> str:
    """Build the Spotify `/authorize` URL the user is redirected to."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "scope": SCOPES,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": generate_code_challenge(code_verifier),
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


class SpotifyAuthHttpClient(Protocol):
    """Thin boundary around the Spotify token endpoint calls this adapter needs."""

    def exchange_code(self, code: str, redirect_uri: str, client_id: str, code_verifier: str) -> dict:
        """Exchange an authorization code for tokens."""
        ...

    def refresh_token(self, refresh_token: str, client_id: str) -> dict:
        """Exchange a refresh token for a new access token."""
        ...


class HttpxSpotifyAuthHttpClient:
    """Real Spotify token endpoint client, built on httpx."""

    def exchange_code(self, code: str, redirect_uri: str, client_id: str, code_verifier: str) -> dict:
        response = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "code_verifier": code_verifier,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def refresh_token(self, refresh_token: str, client_id: str) -> dict:
        response = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()


class SpotifyAuthClient:
    """Performs the PKCE token exchange and refresh against Spotify's token endpoint."""

    def __init__(self, client_id: str, http_client: SpotifyAuthHttpClient | None = None) -> None:
        self._client_id = client_id
        self._http_client = http_client or HttpxSpotifyAuthHttpClient()

    def exchange_code(self, code: str, redirect_uri: str, code_verifier: str) -> SpotifyTokens:
        payload = self._http_client.exchange_code(
            code=code, redirect_uri=redirect_uri, client_id=self._client_id, code_verifier=code_verifier
        )
        return self._tokens_from_payload(payload, fallback_refresh_token=None)

    def refresh(self, refresh_token: str) -> SpotifyTokens:
        payload = self._http_client.refresh_token(refresh_token=refresh_token, client_id=self._client_id)
        return self._tokens_from_payload(payload, fallback_refresh_token=refresh_token)

    def _tokens_from_payload(self, payload: dict, fallback_refresh_token: str | None) -> SpotifyTokens:
        expires_at = time.time() + payload.get("expires_in", 0)
        refresh_token = payload.get("refresh_token") or fallback_refresh_token
        return SpotifyTokens(
            access_token=payload["access_token"],
            refresh_token=refresh_token,
            expires_at=expires_at,
        )
