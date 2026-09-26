"""Integration test against the real multilingual Laya model.

Marked `laya` so it is skipped by default (it downloads real model weights and runs
actual inference). Run explicitly with: `uv run pytest -m laya`.
"""

from __future__ import annotations

import pytest

from mood_dj.adapters.laya_lyrics_judge import LayaLyricsJudge
from mood_dj.domain.models import PlaylistTrack, Strategy
from mood_dj.domain.playlist_strategy import resolve_playlist_strategy
from mood_dj.ports.lyrics_judge import TrackLyrics

pytestmark = pytest.mark.laya

SAD_SPANISH_LYRICS = (
    "Todo se derrumba y no se por que sigo aqui, llorando en silencio cada noche, "
    "extranando lo que ya no volvera, el dolor no se va, solo quiero desaparecer."
)

HAPPY_SPANISH_LYRICS = (
    "Hoy es un dia perfecto para bailar, el sol brilla y todo va a mejorar, "
    "estamos vivos y eso hay que celebrar, con amigos y risas hasta el amanecer."
)


def _track(track_id: str) -> PlaylistTrack:
    return PlaylistTrack(
        id=track_id, name="Song", artist="Artist", album="Album", duration_s=200.0,
        cover_url=None, external_url=None,
    )


def test_detect_signals_on_spanish_prompt_yields_lift_strategy() -> None:
    judge = LayaLyricsJudge()

    signals, probabilities = judge.detect_signals("Estoy triste pero quiero sentirme mejor")

    print(f"probabilities: {probabilities}")
    strategy = resolve_playlist_strategy(signals)
    print(f"resolved strategy: {strategy}")

    assert probabilities["feels_bad"] > 0.5, probabilities
    assert probabilities["wants_change"] > 0.5, probabilities
    assert strategy is Strategy.LIFT


def test_tone_ranks_sad_lyrics_below_happy_lyrics() -> None:
    judge = LayaLyricsJudge()
    tracks = [
        TrackLyrics(track=_track("sad"), text=SAD_SPANISH_LYRICS),
        TrackLyrics(track=_track("happy"), text=HAPPY_SPANISH_LYRICS),
    ]

    tones = judge.judge_tone(tracks)
    by_id = {t.track_id: t for t in tones}
    print(f"sad tone: {by_id['sad'].tone}, happy tone: {by_id['happy'].tone}")

    assert by_id["sad"].tone < by_id["happy"].tone
