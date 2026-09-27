"""TrackProfiler adapter backed by the real Laya Router (multilingual checkpoint).

Computes a per-track mood profile from lyrics alone (no listener prompt involved).

v4 (flat mood): a single `predict_batch` pass answers two questions per track --
a flat `mood` choice over the 7 moods in `mood_dj/data/moods.json`, and a binary
`polarity` noul question (is the lyric emotionally positive overall). Both are
fed the PLAIN, FULL lyrics text as the request `state` (no dict wrapper, no
title/artist, no excerpting), with `max_len` set to the multilingual
checkpoint's own limit (1024 tokens). This replaces the earlier deep
emotion-tree descent (polarity -> cluster -> family) and the excerpt heuristic.

Both changes are backed by isolated-variable experiments on a 30-track
hand-labeled set (see the mood-profiling-accuracy change history and
`backend/eval/`): plain string state + full lyrics + the checkpoint's own
`max_len` measurably beat dict state and a 500-char excerpt, and a single flat
7-mood choice reached ~50% accuracy vs. ~25% for the 4-level family pick, while
a binary positive/negative polarity choice reached ~83%. Average latency in
that experiment was ~2.2s/track on CPU.

`choice` answers return `{"type": "choice", "choice": <key>, "probabilities": {...}}`;
confidence is the probability of the chosen key. `noul` answers return
`{"type": "noul", "noul": <probability_true>}` (see `laya_prompt_profiler.py`'s
established usage of the same question type for its playlist-strategy signals).
"""

from __future__ import annotations

import hashlib
from importlib import resources

from mood_dj.domain.models import TrackMoodProfile
from mood_dj.domain.moods import MoodCatalog, load_moods
from mood_dj.ports.track_profiler import ProfileProgressCallback, TrackForProfiling

MODEL_NAME = "multilingual"

# The multilingual checkpoint's own token limit (see the mood-profiling-accuracy
# experiment: passing the checkpoint's own max_len, instead of a smaller default,
# was one of the two biggest accuracy drivers alongside plain-string state).
MULTILINGUAL_MAX_LEN = 1024

# Bump this when question wording or the moods taxonomy changes what gets fed to
# or asked of the model. v4: flat 7-mood + binary polarity over full lyrics text
# (plain string state, no excerpting), replacing the v3 deep emotion-tree descent
# over a 500-char excerpt. v5: persist the full mood probability distribution
# (`TrackMoodProfile.mood_probabilities`), not just the argmax -- same questions
# asked of the model, but the cached row shape changed, so old rows must not be
# reused as if they carried a distribution.
QUESTION_WORDING_VERSION = "v5"

DEFAULT_BATCH_SIZE = 8

POSITIVE_POLARITY_KEY = "positive"
NEGATIVE_POLARITY_KEY = "negative"


def _read_taxonomy_bytes(filename: str) -> bytes:
    package = "mood_dj.data"
    with resources.files(package).joinpath(filename).open("rb") as fh:
        return fh.read()


def compute_version() -> str:
    """Hash `moods.json` plus question wording into one cache-busting id.

    Any change to `moods.json` (the mood set, labels, criteria, or family
    mapping) or the question wording below changes this string, which
    invalidates every cached `TrackMoodProfile` (they are keyed by
    `(track_id, version)`).
    """

    hasher = hashlib.sha256()
    hasher.update(_read_taxonomy_bytes("moods.json"))
    hasher.update(QUESTION_WORDING_VERSION.encode("utf-8"))
    return hasher.hexdigest()[:16]


def _mood_question(catalog: MoodCatalog) -> dict:
    return {
        "type": "choice",
        "instructions": "Identify the primary emotional response triggered by these song lyrics.",
        "criteria": {mood.id: mood.criterion for mood in catalog.moods},
    }


def _polarity_question() -> dict:
    return {
        "type": "noul",
        "instructions": "Are these song lyrics emotionally positive overall?",
        "criteria": {
            "false": "The lyrics read as emotionally negative overall.",
            "true": "The lyrics read as emotionally positive overall.",
        },
    }


class LayaTrackProfiler:
    """Adapts the Laya Router (multilingual checkpoint) to the TrackProfiler port."""

    def __init__(
        self,
        router=None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        moods: MoodCatalog | None = None,
        max_len: int = MULTILINGUAL_MAX_LEN,
    ) -> None:
        if router is None:
            from laya import Router

            router = Router()
        self._router = router
        self._batch_size = batch_size
        self._max_len = max_len
        self._moods = moods if moods is not None else load_moods()
        self.version = compute_version()

    def profile(
        self,
        tracks: list[TrackForProfiling],
        on_progress: ProfileProgressCallback | None = None,
    ) -> list[TrackMoodProfile]:
        if not tracks:
            return []

        questions = {
            "mood": _mood_question(self._moods),
            "polarity": _polarity_question(),
        }

        profiles: list[TrackMoodProfile] = []
        for start in range(0, len(tracks), self._batch_size):
            chunk = tracks[start : start + self._batch_size]
            requests = [self._build_request(track, questions) for track in chunk]
            results = self._router.predict_batch(requests)

            batch_profiles = []
            for track, result in zip(chunk, results):
                answers = result["answers"]
                mood_answer = answers["mood"]
                mood_id = mood_answer["choice"]
                mood_probabilities = {k: float(v) for k, v in mood_answer["probabilities"].items()}
                mood_confidence = mood_probabilities[mood_id]
                positive_probability = float(answers["polarity"]["noul"])

                batch_profiles.append(
                    TrackMoodProfile(
                        track_id=track.track_id,
                        mood_id=mood_id,
                        mood_confidence=mood_confidence,
                        positive_probability=positive_probability,
                        version=self.version,
                        mood_probabilities=mood_probabilities,
                    )
                )
            profiles.extend(batch_profiles)
            if on_progress is not None:
                on_progress(batch_profiles)

        return profiles

    def _build_request(self, track: TrackForProfiling, questions: dict) -> dict:
        # Plain string state (lyrics only, no title/artist/dict wrapper): the
        # isolated-variable experiment found this measurably beats a dict state
        # with title/artist/lyrics for track mood accuracy.
        return {
            "state": track.lyrics,
            "questions": questions,
            "model": MODEL_NAME,
            "max_len": self._max_len,
        }
