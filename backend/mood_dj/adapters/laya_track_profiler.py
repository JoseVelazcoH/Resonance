"""TrackProfiler adapter backed by the real Laya Router (multilingual checkpoint).

Computes a per-track mood profile from lyrics alone (no listener prompt involved):

1. One batched `predict_batch` call, shared across every track, answers three
   questions per track: valence (5-level score), arousal (5-level score) and
   polarity (3-way choice). Situations were dropped from the track-level
   question set (v3): situations describe the *listener's* context, not the
   song, and the two ten-option situation choices were the most expensive part
   of level 1. Prompt-side situation detection is unaffected; ranking now
   computes a prompt-situation relatedness bonus against the track's own
   family/cluster instead of a per-track situation pick (see
   `mood_dj.domain.track_ranking`).
2. The emotion tree is then descended per track down to the family level only
   (no per-track emotion-level call): cluster choice under the chosen polarity
   (skipped when the polarity has only one cluster), family choice under the
   chosen cluster. Tracks are grouped by their current tree node before each
   `predict_batch` call so every request in a call shares an identical question
   schema -- `predict_batch` only shares one forward pass across requests with
   the same schema (see `laya/router.py`). Stopping at family (instead of the
   finer-grained emotion leaf) removes one full sequential `predict_batch` pass
   per track, which is the dominant remaining cost after excerpting lyrics --
   see `extract_lyrics_excerpt` below. `TrackMoodProfile.emotion_id` /
   `emotion_confidence` are `None` for profiles computed this way; ranking
   (`mood_dj.domain.track_ranking`) treats family as the finest tier used for
   tracks.
3. The lyrics fed into every question above are not the full lyrics text but a
   compact excerpt from `extract_lyrics_excerpt` (chorus heuristic, capped at
   `TRACK_LYRICS_CHAR_BUDGET` characters, default 500), which is the single
   biggest driver of per-track inference cost on CPU.

`choice` answers return `{"type": "choice", "choice": <key>, "probabilities": {...}}`
(confirmed by reading the installed `laya` package's `agent.py`/`onnx_agent.py`);
confidence is the probability of the chosen key. `score` answers return
`{"score": <expected index as float>}`, taken from `laya_lyrics_judge.py`'s
established usage.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from importlib import resources

from mood_dj.domain.models import TrackMoodProfile
from mood_dj.domain.taxonomy import (
    EmotionCluster,
    EmotionFamily,
    EmotionPolarity,
    EmotionTree,
    list_children,
    load_emotion_tree,
)
from mood_dj.ports.track_profiler import ProfileProgressCallback, TrackForProfiling

MODEL_NAME = "multilingual"

# Bump this when question wording changes, or when the lyric-excerpt / tree-descent
# logic below changes what gets fed to the model (independent of QUESTION_SET_VERSION
# in laya_lyrics_judge.py -- these are a different, lyrics-only question set). v2:
# track descent stops at the family level (no per-track emotion predict_batch call)
# and the model input is a compact lyric excerpt instead of a flat truncation. v3:
# situation questions removed from level 1 (see module docstring); tracks no
# longer carry a `situation_id`/`situation_confidence` pick.
QUESTION_WORDING_VERSION = "v3"

DEFAULT_LYRICS_TRUNCATE_CHARS = 1500
DEFAULT_TRACK_LYRICS_CHAR_BUDGET = 500
DEFAULT_BATCH_SIZE = 8

_SECTION_TAG_RE = re.compile(r"^\s*\[[^\]]+\]\s*$", re.MULTILINE)
_BLANK_LINE_SPLIT_RE = re.compile(r"\n\s*\n+")


def extract_lyrics_excerpt(lyrics: str, char_budget: int = DEFAULT_TRACK_LYRICS_CHAR_BUDGET) -> str:
    """Return a compact excerpt of `lyrics` capped at `char_budget` characters.

    This feeds the model a much shorter input than the full lyrics, which is the
    dominant cost driver for track profiling. Strategy, in order:

    1. Strip section tags like "[Coro]" / "[Chorus]" (they carry no emotional
       content and would otherwise pollute the excerpt).
    2. Chorus heuristic: split the remaining text into blocks separated by blank
       lines. If any block appears 2+ times verbatim, it is very likely the
       chorus/hook -- the most emotionally representative part of the song --
       so return the most-repeated one (ties broken by first occurrence).
    3. Otherwise, fall back to a line-level repeat check (some lyric sources
       don't preserve blank-line stanza breaks): if any single non-empty line
       repeats 2+ times, return a window starting at its first occurrence.
    4. Otherwise, no repetition was found: return the first lines, filling as
       many whole blocks as fit under the budget.

    Always capped at `char_budget` characters. Empty/whitespace-only input
    returns an empty string.
    """

    if not lyrics or not lyrics.strip():
        return ""

    cleaned = _SECTION_TAG_RE.sub("", lyrics).strip()
    if not cleaned:
        return ""

    blocks = [b.strip() for b in _BLANK_LINE_SPLIT_RE.split(cleaned) if b.strip()]
    if not blocks:
        return cleaned[:char_budget]

    if len(blocks) > 1:
        block_counts = Counter(blocks)
        repeated_blocks = [b for b in block_counts if block_counts[b] >= 2]
        if repeated_blocks:
            best_block = max(repeated_blocks, key=lambda b: (block_counts[b], -blocks.index(b)))
            return best_block[:char_budget]

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) > 1:
        line_counts = Counter(lines)
        repeated_lines = [line for line in line_counts if line_counts[line] >= 2]
        if repeated_lines:
            chorus_line = max(repeated_lines, key=lambda line: (line_counts[line], -lines.index(line)))
            start = lines.index(chorus_line)
            excerpt_lines = lines[start:]
            excerpt = "\n".join(excerpt_lines)
            return excerpt[:char_budget]

    result_parts: list[str] = []
    total = 0
    for block in blocks:
        addition = block if not result_parts else "\n\n" + block
        if total + len(addition) > char_budget:
            remaining = char_budget - total
            if remaining > 0:
                result_parts.append(addition[:remaining])
            break
        result_parts.append(addition)
        total += len(addition)
    return "".join(result_parts)[:char_budget]

VALENCE_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]
AROUSAL_LEVELS = ["very calm", "calm", "moderate", "energetic", "very intense"]


def _read_taxonomy_bytes(filename: str) -> bytes:
    package = "mood_dj.data"
    with resources.files(package).joinpath(filename).open("rb") as fh:
        return fh.read()


def compute_version() -> str:
    """Hash the taxonomy data files plus question wording into one cache-busting id.

    Any change to `emotions.json` or the question wording below changes this
    string, which invalidates every cached `TrackMoodProfile` (they are keyed by
    `(track_id, version)`). `situations.json` no longer feeds track-level
    questions (see `QUESTION_WORDING_VERSION`), so it is intentionally excluded
    here; it still versions the prompt side independently.
    """

    hasher = hashlib.sha256()
    hasher.update(_read_taxonomy_bytes("emotions.json"))
    hasher.update(QUESTION_WORDING_VERSION.encode("utf-8"))
    return hasher.hexdigest()[:16]


def _score_to_unit_interval(score: float, level_count: int) -> float:
    """Map a 0..(level_count-1) expected index to [-1, 1]."""

    max_index = level_count - 1
    normalized = max(0.0, min(1.0, score / max_index))
    return normalized * 2.0 - 1.0


def _polarity_question(tree: EmotionTree) -> dict:
    return {
        "type": "choice",
        "instructions": "Which emotional polarity best matches these song lyrics?",
        "criteria": {
            polarity.id: f"The lyrics express a mostly {polarity.id} emotional polarity."
            for polarity in tree.polarities
        },
    }


def _choice_question(instructions: str, node_children) -> dict:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {child.id: getattr(child, "description", child.id) for child in node_children},
    }


class LayaTrackProfiler:
    """Adapts the Laya Router (multilingual checkpoint) to the TrackProfiler port."""

    def __init__(
        self,
        router=None,
        lyrics_truncate_chars: int = DEFAULT_LYRICS_TRUNCATE_CHARS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        tree: EmotionTree | None = None,
        lyrics_char_budget: int = DEFAULT_TRACK_LYRICS_CHAR_BUDGET,
    ) -> None:
        if router is None:
            from laya import Router

            router = Router()
        self._router = router
        self._lyrics_truncate_chars = lyrics_truncate_chars
        self._batch_size = batch_size
        self._lyrics_char_budget = lyrics_char_budget
        self._tree = tree if tree is not None else load_emotion_tree()
        self.version = compute_version()

    def profile(
        self,
        tracks: list[TrackForProfiling],
        on_progress: ProfileProgressCallback | None = None,
    ) -> list[TrackMoodProfile]:
        if not tracks:
            return []

        level1 = self._run_level1(tracks)

        profiles: list[TrackMoodProfile] = []
        for start in range(0, len(tracks), self._batch_size):
            chunk = tracks[start : start + self._batch_size]
            chunk_level1 = level1[start : start + self._batch_size]

            polarity_ids = [row["polarity_id"] for row in chunk_level1]
            cluster_ids, _cluster_confidences = self._descend_cluster(chunk, polarity_ids)
            family_ids, _family_confidences = self._descend_family(chunk, polarity_ids, cluster_ids)
            # Tree descent stops at the family level for tracks: the per-track
            # emotion-level predict_batch call is the most expensive remaining
            # sequential pass and ranking only needs family/cluster/polarity
            # granularity (see mood_dj.domain.track_ranking).

            batch_profiles = []
            for i, track in enumerate(chunk):
                row = chunk_level1[i]
                batch_profiles.append(
                    TrackMoodProfile(
                        track_id=track.track_id,
                        valence=row["valence"],
                        arousal=row["arousal"],
                        polarity_id=polarity_ids[i],
                        cluster_id=cluster_ids[i],
                        family_id=family_ids[i],
                        version=self.version,
                    )
                )
            profiles.extend(batch_profiles)
            if on_progress is not None:
                on_progress(batch_profiles)

        return profiles

    # -- level 1: one shared question schema for every track -----------------

    def _run_level1(self, tracks: list[TrackForProfiling]) -> list[dict]:
        questions = {
            "valence": {
                "type": "score",
                "instructions": "What is the emotional valence of these song lyrics?",
                "criteria": VALENCE_LEVELS,
            },
            "arousal": {
                "type": "score",
                "instructions": "What is the emotional arousal/energy of these song lyrics?",
                "criteria": AROUSAL_LEVELS,
            },
            "polarity": _polarity_question(self._tree),
        }

        results: list[dict] = []
        for start in range(0, len(tracks), self._batch_size):
            chunk = tracks[start : start + self._batch_size]
            requests = [self._build_request(track, questions) for track in chunk]
            batch_results = self._router.predict_batch(requests)
            for result in batch_results:
                answers = result["answers"]
                valence = _score_to_unit_interval(float(answers["valence"]["score"]), len(VALENCE_LEVELS))
                arousal = _score_to_unit_interval(float(answers["arousal"]["score"]), len(AROUSAL_LEVELS))
                polarity_id = answers["polarity"]["choice"]

                results.append(
                    {
                        "valence": valence,
                        "arousal": arousal,
                        "polarity_id": polarity_id,
                    }
                )
        return results

    def _build_request(self, track: TrackForProfiling, questions: dict) -> dict:
        excerpt = extract_lyrics_excerpt(track.lyrics, self._lyrics_char_budget)
        return {
            "state": {
                "title": track.title,
                "artist": track.artist,
                "lyrics": excerpt,
            },
            "questions": questions,
            "model": MODEL_NAME,
        }

    # -- tree descent: group tracks by current node so schemas match ---------

    def _find_polarity(self, polarity_id: str) -> EmotionPolarity:
        for polarity in self._tree.polarities:
            if polarity.id == polarity_id:
                return polarity
        raise KeyError(f"Unknown polarity id: {polarity_id!r}")

    def _find_cluster(self, polarity: EmotionPolarity, cluster_id: str) -> EmotionCluster:
        for cluster in polarity.clusters:
            if cluster.id == cluster_id:
                return cluster
        raise KeyError(f"Unknown cluster id: {cluster_id!r}")

    def _find_family(self, cluster: EmotionCluster, family_id: str) -> EmotionFamily:
        for family in cluster.families:
            if family.id == family_id:
                return family
        raise KeyError(f"Unknown family id: {family_id!r}")

    def _descend_cluster(
        self, tracks: list[TrackForProfiling], polarity_ids: list[str]
    ) -> tuple[list[str], list[float]]:
        cluster_ids: list[str | None] = [None] * len(tracks)
        confidences: list[float] = [1.0] * len(tracks)

        groups: dict[str, list[int]] = defaultdict(list)
        for i, polarity_id in enumerate(polarity_ids):
            groups[polarity_id].append(i)

        for polarity_id, indices in groups.items():
            polarity = self._find_polarity(polarity_id)
            clusters = list_children(polarity)
            if len(clusters) == 1:
                for i in indices:
                    cluster_ids[i] = clusters[0].id
                continue

            questions = {
                "cluster": _choice_question(
                    "Which emotional cluster within this polarity best matches these song lyrics?",
                    clusters,
                )
            }
            group_tracks = [tracks[i] for i in indices]
            for start in range(0, len(group_tracks), self._batch_size):
                sub = group_tracks[start : start + self._batch_size]
                sub_indices = indices[start : start + self._batch_size]
                requests = [self._build_request(track, questions) for track in sub]
                results = self._router.predict_batch(requests)
                for idx, result in zip(sub_indices, results):
                    answer = result["answers"]["cluster"]
                    cluster_ids[idx] = answer["choice"]
                    confidences[idx] = float(answer["probabilities"][answer["choice"]])

        return cluster_ids, confidences  # type: ignore[return-value]

    def _descend_family(
        self,
        tracks: list[TrackForProfiling],
        polarity_ids: list[str],
        cluster_ids: list[str],
    ) -> tuple[list[str], list[float]]:
        family_ids: list[str | None] = [None] * len(tracks)
        confidences: list[float] = [1.0] * len(tracks)

        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, (polarity_id, cluster_id) in enumerate(zip(polarity_ids, cluster_ids)):
            groups[(polarity_id, cluster_id)].append(i)

        for (polarity_id, cluster_id), indices in groups.items():
            polarity = self._find_polarity(polarity_id)
            cluster = self._find_cluster(polarity, cluster_id)
            families = list_children(cluster)
            if len(families) == 1:
                for i in indices:
                    family_ids[i] = families[0].id
                continue

            questions = {
                "family": _choice_question(
                    "Which emotion family within this cluster best matches these song lyrics?",
                    families,
                )
            }
            group_tracks = [tracks[i] for i in indices]
            for start in range(0, len(group_tracks), self._batch_size):
                sub = group_tracks[start : start + self._batch_size]
                sub_indices = indices[start : start + self._batch_size]
                requests = [self._build_request(track, questions) for track in sub]
                results = self._router.predict_batch(requests)
                for idx, result in zip(sub_indices, results):
                    answer = result["answers"]["family"]
                    family_ids[idx] = answer["choice"]
                    confidences[idx] = float(answer["probabilities"][answer["choice"]])

        return family_ids, confidences  # type: ignore[return-value]
