"""Cheap evaluation harness for Laya's mood taxonomy decisions.

Runs the REAL LayaTrackProfiler and LayaPromptProfiler (the Laya Router is
loaded once and shared) against a small hand-labeled eval set, WITHOUT
recomputing the user's cached track profiles and WITHOUT writing to
`backend/data/app.db`:

- Track lyrics are read from `backend/data/app.db` in read-only mode (SQLite
  URI `mode=ro`); the eval set file itself never stores lyrics, only track
  ids/names/artists and expected labels.
- The taxonomy fed to both profilers can be swapped via `--emotions` to
  evaluate a candidate taxonomy file (e.g. `eval/emotions.candidate.json`)
  without touching the packaged `mood_dj/data/emotions.json` (which would
  bump `compute_version()` and invalidate every cached profile).
- Nothing here calls `SqliteMoodProfileRepository.save` or otherwise persists
  anything; profiles are computed in memory and only printed/optionally
  dumped to `--json-out`.

Usage:
    uv run python scripts/eval_mood.py
    uv run python scripts/eval_mood.py --emotions eval/emotions.candidate.json
    uv run python scripts/eval_mood.py --beam-margin 0.2
    uv run python scripts/eval_mood.py --direct-score-weight 0.0
    uv run python scripts/eval_mood.py --json-out /tmp/eval_result.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from mood_dj.adapters.laya_prompt_profiler import LayaPromptProfiler  # noqa: E402
from mood_dj.adapters.laya_track_profiler import LayaTrackProfiler  # noqa: E402
from mood_dj.domain.taxonomy import (  # noqa: E402
    EmotionTree,
    load_emotion_tree,
    load_emotion_tree_from_path,
    load_situations,
)
from mood_dj.ports.track_profiler import TrackForProfiling  # noqa: E402

TOP_K_FOR_TOP2_ACCURACY = 2

DEFAULT_EVAL_SET_PATH = BACKEND_ROOT / "eval_set.local.json"
DEFAULT_APP_DB_PATH = BACKEND_ROOT / "data" / "app.db"


@dataclass
class TrackResult:
    track_id: str
    name: str
    artist: str
    expected_polarity: str
    expected_mood: list[str]
    predicted_mood: str
    predicted_polarity: str
    positive_probability: float
    top2_moods: list[str]
    mood_correct: bool
    top2_correct: bool
    polarity_correct: bool


@dataclass
class PromptResult:
    text: str
    expected_polarity: str | None
    expected_family: list[str]
    predicted_polarity: str
    predicted_family: str
    predicted_strategy: str
    target_valence: float
    target_arousal: float
    direct_valence: float
    polarity_correct: bool
    family_correct: bool


def load_eval_set(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_lyrics_readonly(db_path: Path, track_ids: list[str]) -> dict[str, str]:
    """Read lyrics text for the given track ids from `app.db`, read-only.

    Uses SQLite's URI `mode=ro` so this can never write to the database, even
    by accident (no table creation, no writes -- a plain read connection).
    """

    uri = f"file:{db_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        placeholders = ",".join("?" for _ in track_ids)
        rows = connection.execute(
            f"SELECT track_id, text FROM lyrics WHERE track_id IN ({placeholders}) AND status = 'lyrics'",
            track_ids,
        ).fetchall()
    finally:
        connection.close()
    return {track_id: text for track_id, text in rows}


def build_tree(emotions_path: str | None) -> EmotionTree:
    if emotions_path is None:
        return load_emotion_tree()
    return load_emotion_tree_from_path(emotions_path)


def run_tracks(
    eval_set: dict,
    tree: EmotionTree,
    db_path: Path,
    beam_margin: float | None,
) -> list[TrackResult]:
    track_items = eval_set.get("tracks", [])
    if not track_items:
        return []

    lyrics_by_id = read_lyrics_readonly(db_path, [t["track_id"] for t in track_items])

    tracks_for_profiling = []
    missing = []
    for item in track_items:
        lyrics = lyrics_by_id.get(item["track_id"])
        if not lyrics:
            missing.append(item["track_id"])
            continue
        tracks_for_profiling.append(
            TrackForProfiling(
                track_id=item["track_id"],
                artist=item["artist"],
                title=item["name"],
                lyrics=lyrics,
            )
        )
    if missing:
        print(f"WARNING: {len(missing)} eval tracks had no cached lyrics in app.db: {missing}", file=sys.stderr)

    # `beam_margin` only affects the prompt profiler below; LayaTrackProfiler's
    # flat mood choice has no tree to beam-search.
    profiler = LayaTrackProfiler()
    profiles = profiler.profile(tracks_for_profiling)
    profiles_by_id = {p.track_id: p for p in profiles}

    results = []
    for item in track_items:
        profile = profiles_by_id.get(item["track_id"])
        if profile is None:
            continue
        expected_mood = item.get("expected_mood", item.get("expected_family", []))
        predicted_polarity = "positive" if profile.positive_probability >= 0.5 else "negative"
        ranked_moods = sorted(profile.mood_probabilities.items(), key=lambda kv: kv[1], reverse=True)
        top2_moods = [mood_id for mood_id, _prob in ranked_moods[:TOP_K_FOR_TOP2_ACCURACY]]
        results.append(
            TrackResult(
                track_id=item["track_id"],
                name=item["name"],
                artist=item["artist"],
                expected_polarity=item["expected_polarity"],
                expected_mood=expected_mood,
                predicted_mood=profile.mood_id,
                predicted_polarity=predicted_polarity,
                positive_probability=profile.positive_probability,
                top2_moods=top2_moods,
                mood_correct=profile.mood_id in expected_mood,
                top2_correct=any(m in expected_mood for m in top2_moods),
                polarity_correct=predicted_polarity == item["expected_polarity"],
            )
        )
    return results


def run_prompts(
    eval_set: dict,
    tree: EmotionTree,
    beam_margin: float | None,
    direct_score_weight: float | None,
) -> list[PromptResult]:
    prompt_items = eval_set.get("prompts", [])
    if not prompt_items:
        return []

    situations = load_situations()
    kwargs = {}
    if direct_score_weight is not None:
        kwargs["direct_score_weight"] = direct_score_weight
    profiler = LayaPromptProfiler(tree=tree, situations=situations, beam_margin=beam_margin, **kwargs)

    results = []
    for item in prompt_items:
        profile = profiler.profile(item["text"])
        expected_polarity = item.get("expected_polarity")
        expected_family = item.get("expected_family", [])
        polarity_correct = expected_polarity is None or profile.emotion.polarity_id == expected_polarity
        family_correct = not expected_family or profile.emotion.family_id in expected_family
        results.append(
            PromptResult(
                text=item["text"],
                expected_polarity=expected_polarity,
                expected_family=expected_family,
                predicted_polarity=profile.emotion.polarity_id,
                predicted_family=profile.emotion.family_id,
                predicted_strategy=profile.strategy.value,
                target_valence=profile.target_valence,
                target_arousal=profile.target_arousal,
                direct_valence=profile.direct_valence,
                polarity_correct=polarity_correct,
                family_correct=family_correct,
            )
        )
    return results


def print_report(label: str, track_results: list[TrackResult], prompt_results: list[PromptResult], elapsed_s: float) -> None:
    print(f"\n=== {label} (runtime: {elapsed_s:.1f}s) ===")

    if track_results:
        polarity_acc = sum(r.polarity_correct for r in track_results) / len(track_results)
        mood_acc = sum(r.mood_correct for r in track_results) / len(track_results)
        top2_acc = sum(r.top2_correct for r in track_results) / len(track_results)
        print(f"\n-- Tracks (n={len(track_results)}) --")
        print(f"Polarity accuracy: {polarity_acc:.0%}   Mood (top-1) accuracy: {mood_acc:.0%}   Mood (top-2) accuracy: {top2_acc:.0%}")
        for r in track_results:
            mark_p = "OK" if r.polarity_correct else "XX"
            mark_m = "OK" if r.mood_correct else "XX"
            print(
                f"  [{mark_p}/{mark_m}] {r.artist} - {r.name}: "
                f"expected pol={r.expected_polarity} mood={r.expected_mood} | "
                f"got pol={r.predicted_polarity} ({r.positive_probability:.2f}) mood={r.predicted_mood} top2={r.top2_moods}"
            )

    if prompt_results:
        polarity_acc = sum(r.polarity_correct for r in prompt_results) / len(prompt_results)
        family_acc = sum(r.family_correct for r in prompt_results) / len(prompt_results)
        print(f"\n-- Prompts (n={len(prompt_results)}) --")
        print(f"Polarity accuracy: {polarity_acc:.0%}   Family accuracy: {family_acc:.0%}")
        for r in prompt_results:
            mark_p = "OK" if r.polarity_correct else "XX"
            mark_f = "OK" if r.family_correct else "XX"
            print(
                f"  [{mark_p}/{mark_f}] \"{r.text}\": "
                f"expected pol={r.expected_polarity} fam={r.expected_family} | "
                f"got pol={r.predicted_polarity} fam={r.predicted_family} strategy={r.predicted_strategy} "
                f"target_valence={r.target_valence:.2f} direct_valence={r.direct_valence:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET_PATH))
    parser.add_argument("--app-db", default=str(DEFAULT_APP_DB_PATH))
    parser.add_argument("--emotions", default=None, help="Path to a candidate emotions.json; default is packaged.")
    parser.add_argument("--beam-margin", type=float, default=None, help="Enable beam when top-2 margin < this.")
    parser.add_argument("--direct-score-weight", type=float, default=None, help="Override DIRECT_SCORE_WEIGHT.")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    eval_set = load_eval_set(Path(args.eval_set))
    tree = build_tree(args.emotions)

    start = time.monotonic()
    track_results = run_tracks(eval_set, tree, Path(args.app_db), args.beam_margin)
    prompt_results = run_prompts(eval_set, tree, args.beam_margin, args.direct_score_weight)
    elapsed = time.monotonic() - start

    label = args.label or (
        f"emotions={args.emotions or 'packaged'} beam_margin={args.beam_margin} "
        f"direct_score_weight={args.direct_score_weight}"
    )
    print_report(label, track_results, prompt_results, elapsed)

    if args.json_out:
        payload = {
            "label": label,
            "elapsed_s": elapsed,
            "tracks": [vars(r) for r in track_results],
            "prompts": [vars(r) for r in prompt_results],
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
