"""Spotify user auth endpoints: Authorization Code flow with PKCE."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from mood_dj.adapters.spotify_auth import build_authorize_url, generate_code_verifier, generate_state
from mood_dj.api.deps import (
    get_auth_client,
    get_auth_state_store,
    get_playlists_client,
    get_session_store,
    get_settings,
)
from mood_dj.api.schemas import MeResponse
from mood_dj.config import SESSION_COOKIE_NAME, Settings
from mood_dj.ports.auth_state_store import AuthStateStore
from mood_dj.ports.spotify_playlists import SpotifyPlaylistsClient
from mood_dj.ports.spotify_session_store import SpotifySessionStore

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/login")
def login(
    settings: Settings = Depends(get_settings),
    auth_state_store: AuthStateStore = Depends(get_auth_state_store),
):
    code_verifier = generate_code_verifier()
    state = generate_state()
    auth_state_store.save(state, code_verifier)

    url = build_authorize_url(
        client_id=settings.spotify_client_id,
        redirect_uri=settings.spotify_redirect_uri,
        state=state,
        code_verifier=code_verifier,
    )
    return RedirectResponse(url, status_code=307)


@router.get("/callback")
def callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    settings: Settings = Depends(get_settings),
    auth_state_store: AuthStateStore = Depends(get_auth_state_store),
    session_store: SpotifySessionStore = Depends(get_session_store),
    auth_client=Depends(get_auth_client),
):
    if error is not None or code is None or state is None:
        raise HTTPException(status_code=400, detail=f"Spotify auth failed: {error or 'missing code/state'}")

    code_verifier = auth_state_store.pop(state)
    if code_verifier is None:
        raise HTTPException(status_code=400, detail="Unknown or expired auth state")

    tokens = auth_client.exchange_code(
        code=code, redirect_uri=settings.spotify_redirect_uri, code_verifier=code_verifier
    )

    session_id = request.cookies.get(SESSION_COOKIE_NAME) or generate_state()
    session_store.save(session_id, tokens)

    response = RedirectResponse(settings.frontend_url, status_code=307)
    response.set_cookie(SESSION_COOKIE_NAME, session_id, httponly=True)
    return response


@router.get("/me", response_model=MeResponse)
def me(
    request: Request,
    session_store: SpotifySessionStore = Depends(get_session_store),
    playlists_client: SpotifyPlaylistsClient = Depends(get_playlists_client),
):
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    tokens = session_store.get(session_id) if session_id is not None else None
    if tokens is None:
        return MeResponse(logged_in=False, display_name=None)

    display_name = playlists_client.get_display_name(tokens.access_token)
    return MeResponse(logged_in=True, display_name=display_name)


@router.post("/logout")
def logout(
    request: Request,
    session_store: SpotifySessionStore = Depends(get_session_store),
):
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id is not None:
        session_store.delete(session_id)

    response = JSONResponse({"logged_out": True})
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response
