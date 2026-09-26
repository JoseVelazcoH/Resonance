"""FastAPI application exposing the Mood DJ recommendation use case."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from mood_dj.api import auth, library, playlists
from mood_dj.api.deps import get_settings
from mood_dj.api.schemas import (
    ProfileResponse,
    RecommendRequest,
    RecommendResponse,
    StageResponse,
    TrackResponse,
)
from mood_dj.application.recommend_playlist import RecommendPlaylistUseCase
from mood_dj.config import dataset_exists, load_settings

app = FastAPI(title="Laya Mood DJ", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(playlists.router)
app.include_router(playlists.recommend_jobs_router)
app.include_router(library.router)


@lru_cache(maxsize=1)
def get_use_case() -> RecommendPlaylistUseCase:
    from mood_dj.adapters.dataset_catalog import DatasetCatalog
    from mood_dj.adapters.laya_decision_engine import LayaDecisionEngine
    from mood_dj.adapters.spotify_covers import SpotifyCoverProvider

    settings = load_settings()
    return RecommendPlaylistUseCase(
        decision_engine=LayaDecisionEngine(),
        catalog=DatasetCatalog(settings.dataset_path),
        cover_provider=SpotifyCoverProvider(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
        ),
    )


@app.get("/health")
def health() -> dict:
    settings = load_settings()
    return {"status": "ok", "dataset_ready": dataset_exists(settings)}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    settings = load_settings()
    if not dataset_exists(settings):
        raise HTTPException(
            status_code=503,
            detail=(
                "Track dataset not found at "
                f"{settings.dataset_path}. Run scripts/build_dataset.py first."
            ),
        )

    use_case = get_use_case()
    decision = use_case.run(request.prompt)

    return RecommendResponse(
        strategy=decision.strategy.value,
        strategy_probabilities=decision.strategy_probabilities,
        stages=[
            StageResponse(
                name=stage.name,
                profile=ProfileResponse(
                    energy=stage.profile.energy,
                    valence=stage.profile.valence,
                    tempo=stage.profile.tempo,
                    instrumentalness=stage.profile.instrumentalness,
                ),
                tracks=[TrackResponse(**track.__dict__) for track in stage.tracks],
            )
            for stage in decision.stages
        ],
    )
