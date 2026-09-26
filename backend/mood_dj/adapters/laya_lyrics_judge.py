"""LyricsJudge adapter backed by the real Laya Router, forced to the multilingual checkpoint.

The user's music is mostly Spanish, and the English checkpoint collapses off-English
(see `laya/router.py`'s module docstring: it scores near random-guessing on non-English
languages while reporting high confidence). The multilingual checkpoint (mmBERT-base,
322M params, 1024 tokens, 100+ languages per the model card) is selected explicitly with
`model="multilingual"` on every call, instead of relying on the Router's automatic
script-based routing, so behavior does not silently change if English lyrics show up in
an otherwise Spanish playlist.

Question types (confirmed by reading the installed `laya` package source, same schema
already used in `laya_decision_engine.py`):

- "noul" (yes/no): answer holds `{"noul": <probability the statement is true>}`.
- "score": answer holds `{"score": <expected index as float>}` over an ordered criteria list.

`predict_batch` shares a single forward pass across requests with an identical question
schema (see `Router._question_schema`), so `judge_tone` and `judge_fit` each build one
schema for their own question and reuse it across every track in the batch.
"""

from __future__ import annotations

from mood_dj.domain.models import Strategy
from mood_dj.domain.playlist_strategy import PlaylistSignals
from mood_dj.ports.lyrics_judge import JudgeProgressCallback, TrackFit, TrackLyrics, TrackTone

MODEL_NAME = "multilingual"

# Bump this when question wording changes, to invalidate cached judgments made under
# the old wording.
QUESTION_SET_VERSION = "v1"

DEFAULT_LYRICS_TRUNCATE_CHARS = 1500

# `judge_tone` and `judge_fit` split their work into chunks of this size, calling
# `predict_batch` once per chunk and reporting progress after each one, so a large
# batch reports incremental progress instead of blocking silently for the whole run.
DEFAULT_BATCH_SIZE = 8

# Score levels for lyrics tone, index 0..4, normalized to 0.0-1.0.
TONE_LEVELS = [
    "very sad",
    "sad",
    "neutral",
    "hopeful",
    "happy",
]

STRATEGY_FIT_INSTRUCTIONS = {
    Strategy.ACCOMPANY.value: (
        "The listener wants music that keeps them company in how they already feel, "
        "not a change of mood."
    ),
    Strategy.LIFT.value: (
        "The listener feels down and wants to be lifted toward feeling better, hopeful "
        "or positive."
    ),
    Strategy.ENERGIZE.value: "The listener wants energy and movement.",
    Strategy.CALM.value: "The listener wants to relax, wind down or sleep.",
}


def _signal_questions() -> dict:
    return {
        "feels_bad": {
            "type": "noul",
            "instructions": "Does the listener currently feel bad, sad or low?",
            "criteria": {"false": "The listener does not feel bad.", "true": "The listener feels bad."},
        },
        "wants_change": {
            "type": "noul",
            "instructions": "Does the listener want their mood to change, rather than stay the same?",
            "criteria": {
                "false": "The listener wants to stay in the same mood.",
                "true": "The listener wants their mood to change.",
            },
        },
        "wants_energy": {
            "type": "noul",
            "instructions": "Is the listener asking for energy, movement or something to power through an activity?",
            "criteria": {
                "false": "The listener is not asking for energy or movement.",
                "true": "The listener is asking for energy or movement.",
            },
        },
        "wants_rest": {
            "type": "noul",
            "instructions": "Is the listener asking to relax, wind down or sleep?",
            "criteria": {
                "false": "The listener is not asking to rest or relax.",
                "true": "The listener is asking to rest or relax.",
            },
        },
    }


class LayaLyricsJudge:
    """Adapts the Laya Router (multilingual checkpoint) to the LyricsJudge port."""

    def __init__(
        self,
        router=None,
        lyrics_truncate_chars: int = DEFAULT_LYRICS_TRUNCATE_CHARS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if router is None:
            from laya import Router

            router = Router()
        self._router = router
        self._lyrics_truncate_chars = lyrics_truncate_chars
        self._batch_size = batch_size

    def detect_signals(self, prompt: str) -> tuple[PlaylistSignals, dict[str, float]]:
        result = self._router.predict(prompt, _signal_questions(), model=MODEL_NAME)
        answers = result["answers"]
        probabilities = {question_id: float(answer["noul"]) for question_id, answer in answers.items()}
        signals = PlaylistSignals(
            feels_bad=probabilities["feels_bad"] >= 0.5,
            wants_change=probabilities["wants_change"] >= 0.5,
            wants_energy=probabilities["wants_energy"] >= 0.5,
            wants_rest=probabilities["wants_rest"] >= 0.5,
        )
        return signals, probabilities

    def judge_tone(
        self,
        tracks: list[TrackLyrics],
        on_progress: JudgeProgressCallback | None = None,
    ) -> list[TrackTone]:
        if not tracks:
            return []

        questions = {
            "tone": {
                "type": "score",
                "instructions": "What is the emotional tone of these song lyrics?",
                "criteria": TONE_LEVELS,
            },
        }
        max_index = len(TONE_LEVELS) - 1
        tones: list[TrackTone] = []

        for start in range(0, len(tracks), self._batch_size):
            chunk = tracks[start : start + self._batch_size]
            requests = [
                {
                    "state": {"lyrics": item.text[: self._lyrics_truncate_chars]},
                    "questions": questions,
                    "model": MODEL_NAME,
                }
                for item in chunk
            ]
            results = self._router.predict_batch(requests)

            for item, result in zip(chunk, results):
                tone = max(0.0, min(1.0, float(result["answers"]["tone"]["score"]) / max_index))
                tones.append(TrackTone(track_id=item.track.id, tone=tone))

            if on_progress is not None:
                on_progress(len(chunk))

        return tones

    def judge_fit(
        self,
        prompt: str,
        strategy: Strategy,
        tracks: list[TrackLyrics],
        on_progress: JudgeProgressCallback | None = None,
    ) -> list[TrackFit]:
        if not tracks:
            return []

        fit_instructions = STRATEGY_FIT_INSTRUCTIONS[strategy.value]
        questions = {
            "fit": {
                "type": "noul",
                "instructions": f"Do these lyrics fit what the listener asked for? {fit_instructions}",
                "criteria": {
                    "false": "These lyrics do not fit what the listener asked for.",
                    "true": "These lyrics fit what the listener asked for.",
                },
            },
        }
        fits: list[TrackFit] = []

        for start in range(0, len(tracks), self._batch_size):
            chunk = tracks[start : start + self._batch_size]
            requests = [
                {
                    "state": {
                        "prompt": prompt,
                        "lyrics": item.text[: self._lyrics_truncate_chars],
                    },
                    "questions": questions,
                    "model": MODEL_NAME,
                }
                for item in chunk
            ]
            results = self._router.predict_batch(requests)

            for item, result in zip(chunk, results):
                fits.append(TrackFit(track_id=item.track.id, fit=float(result["answers"]["fit"]["noul"])))

            if on_progress is not None:
                on_progress(len(chunk))

        return fits
