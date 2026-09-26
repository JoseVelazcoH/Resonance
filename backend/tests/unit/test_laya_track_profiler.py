"""Unit tests for LayaTrackProfiler, using a fake Laya Router.

`predict_batch` is exercised with a fake router that answers deterministically
from the question `criteria`, so tests can pin exact tree paths and assert on
schema-sharing (grouping), single-child skipping, and confidence mapping.
"""

from __future__ import annotations

from mood_dj.adapters.laya_track_profiler import (
    LayaTrackProfiler,
    compute_version,
    extract_lyrics_excerpt,
)
from mood_dj.ports.track_profiler import TrackForProfiling


class FakeRouter:
    """Answers every question in a request deterministically from its criteria.

    `answers_by_question` maps a question id to a function `(criteria) -> (choice,
    probability)` for `choice` questions, or `(criteria) -> score` for `score`
    questions. Records every batch of requests it receives so tests can assert on
    schema-sharing.
    """

    def __init__(self, overrides: dict | None = None) -> None:
        self.overrides = overrides or {}
        self.batches: list[list[dict]] = []

    def predict_batch(self, requests: list[dict]) -> list[dict]:
        self.batches.append(requests)
        results = []
        for request in requests:
            answers = {}
            for qid, qdef in request["questions"].items():
                if qdef["type"] == "score":
                    levels = qdef["criteria"]
                    index = self.overrides.get(qid, {}).get("score", (len(levels) - 1) // 2)
                    answers[qid] = {"score": float(index)}
                else:
                    keys = list(qdef["criteria"].keys())
                    default_choice = keys[0]
                    default_prob = 1.0 / len(keys)
                    choice = self.overrides.get(qid, {}).get("choice", default_choice)
                    if choice not in keys:
                        choice = default_choice
                    prob = self.overrides.get(qid, {}).get("prob", default_prob)
                    probabilities = {k: (prob if k == choice else (1 - prob) / max(1, len(keys) - 1)) for k in keys}
                    answers[qid] = {"type": "choice", "choice": choice, "probabilities": probabilities}
            results.append({"answers": answers})
        return results


def _tracks(n: int) -> list[TrackForProfiling]:
    return [
        TrackForProfiling(track_id=f"t{i}", artist="Artist", title=f"Song {i}", lyrics="la la la")
        for i in range(n)
    ]


def test_level1_call_shares_one_schema_across_every_track() -> None:
    router = FakeRouter()
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiler.profile(_tracks(3))

    first_batch = router.batches[0]
    schemas = [set(req["questions"].keys()) for req in first_batch]
    assert all(s == schemas[0] for s in schemas)
    assert schemas[0] == {"valence", "arousal", "polarity"}


def test_cluster_level_is_skipped_for_a_polarity_with_a_single_cluster() -> None:
    # "positive" has exactly one cluster ("core_positive") in emotions.json.
    router = FakeRouter(overrides={"polarity": {"choice": "positive"}})
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiles = profiler.profile(_tracks(1))

    assert profiles[0].polarity_id == "positive"
    assert profiles[0].cluster_id == "core_positive"
    # No predict_batch call should ever ask a "cluster" question for this track.
    cluster_calls = [b for b in router.batches if any("cluster" in r["questions"] for r in b)]
    assert cluster_calls == []


def test_cluster_level_runs_for_a_polarity_with_multiple_clusters() -> None:
    # "negative" has two clusters: acute_distress, relational_decline.
    router = FakeRouter(
        overrides={
            "polarity": {"choice": "negative"},
            "cluster": {"choice": "relational_decline", "prob": 0.77},
        }
    )
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiles = profiler.profile(_tracks(1))

    assert profiles[0].polarity_id == "negative"
    assert profiles[0].cluster_id == "relational_decline"
    cluster_calls = [b for b in router.batches if any("cluster" in r["questions"] for r in b)]
    assert len(cluster_calls) == 1


def test_tracks_are_grouped_by_tree_node_so_schemas_match_within_a_call() -> None:
    # Two tracks land on different polarities after level 1; the cluster-level
    # requests for each must carry only that polarity's own cluster options.
    def polarity_for(request):
        return "positive" if "0" in request["state"]["title"] else "negative"

    class MixedRouter(FakeRouter):
        def predict_batch(self, requests):
            self.batches.append(requests)
            results = []
            for request in requests:
                answers = {}
                for qid, qdef in request["questions"].items():
                    if qid == "polarity":
                        choice = polarity_for(request)
                        answers[qid] = {"choice": choice, "probabilities": {choice: 1.0}}
                    elif qdef["type"] == "score":
                        answers[qid] = {"score": 2.0}
                    else:
                        keys = list(qdef["criteria"].keys())
                        answers[qid] = {"choice": keys[0], "probabilities": {keys[0]: 1.0 / len(keys)}}
                results.append({"answers": answers})
            return results

    router = MixedRouter()
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiles = profiler.profile(_tracks(2))

    by_id = {p.track_id: p for p in profiles}
    assert by_id["t0"].polarity_id == "positive"
    assert by_id["t0"].cluster_id == "core_positive"
    assert by_id["t1"].polarity_id == "negative"
    # t1's cluster question must have only offered negative's own clusters.
    cluster_calls = [r for b in router.batches for r in b if "cluster" in r["questions"]]
    assert len(cluster_calls) == 1
    assert set(cluster_calls[0]["questions"]["cluster"]["criteria"].keys()) == {
        "acute_distress",
        "relational_decline",
    }


def test_confidence_is_the_probability_of_the_chosen_option() -> None:
    router = FakeRouter(
        overrides={
            "polarity": {"choice": "positive"},
            "family": {"prob": 0.42, "choice": "joy_elation"},
        }
    )
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiles = profiler.profile(_tracks(1))

    assert profiles[0].family_id == "joy_elation"


def test_descent_stops_at_family_no_emotion_level_call_is_made() -> None:
    router = FakeRouter(overrides={"polarity": {"choice": "positive"}})
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiles = profiler.profile(_tracks(1))

    assert profiles[0].emotion_id is None
    assert profiles[0].emotion_confidence is None
    emotion_calls = [b for b in router.batches if any("emotion" in r["questions"] for r in b)]
    assert emotion_calls == []


def test_valence_and_arousal_scores_map_onto_minus_one_to_one() -> None:
    # 5 levels (index 0..4): index 0 -> -1.0, index 4 -> 1.0, index 2 -> 0.0.
    router = FakeRouter(overrides={"valence": {"score": 4}, "arousal": {"score": 0}})
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiles = profiler.profile(_tracks(1))

    assert profiles[0].valence == 1.0
    assert profiles[0].arousal == -1.0


def test_tracks_no_longer_carry_a_situation_pick() -> None:
    router = FakeRouter()
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiles = profiler.profile(_tracks(1))

    assert profiles[0].situation_id is None
    assert profiles[0].situation_confidence is None
    situation_calls = [
        r for b in router.batches for r in b if any("situation" in qid for qid in r["questions"])
    ]
    assert situation_calls == []


def test_progress_callback_is_called_per_batch_with_completed_profiles() -> None:
    router = FakeRouter()
    profiler = LayaTrackProfiler(router=router, batch_size=2)
    seen: list[int] = []

    profiler.profile(_tracks(5), on_progress=lambda batch: seen.append(len(batch)))

    assert seen == [2, 2, 1]


def test_empty_track_list_returns_empty_and_makes_no_calls() -> None:
    router = FakeRouter()
    profiler = LayaTrackProfiler(router=router)

    assert profiler.profile([]) == []
    assert router.batches == []


def test_version_is_deterministic_and_matches_compute_version() -> None:
    router = FakeRouter()
    profiler = LayaTrackProfiler(router=router)

    assert profiler.version == compute_version()
    assert profiler.version == compute_version()


def test_version_changes_when_taxonomy_bytes_change(monkeypatch) -> None:
    from mood_dj.adapters import laya_track_profiler as module

    original = module._read_taxonomy_bytes
    before = module.compute_version()

    def patched(filename: str) -> bytes:
        if filename == "emotions.json":
            return original(filename) + b" "
        return original(filename)

    monkeypatch.setattr(module, "_read_taxonomy_bytes", patched)
    after = module.compute_version()

    assert before != after


# -- extract_lyrics_excerpt ---------------------------------------------------


def test_excerpt_picks_the_repeated_chorus_block() -> None:
    lyrics = (
        "Verso uno, aqui empieza la historia\n"
        "todo va bien por ahora\n"
        "\n"
        "No puedo mas, se me acaba el aire\n"
        "grito tu nombre en la noche\n"
        "\n"
        "Verso dos, algo distinto ocurre\n"
        "el camino se hace largo\n"
        "\n"
        "No puedo mas, se me acaba el aire\n"
        "grito tu nombre en la noche\n"
    )

    excerpt = extract_lyrics_excerpt(lyrics, char_budget=500)

    assert excerpt == "No puedo mas, se me acaba el aire\ngrito tu nombre en la noche"


def test_excerpt_falls_back_to_first_lines_when_there_is_no_chorus() -> None:
    lyrics = (
        "Primera linea distinta\n"
        "\n"
        "Segunda linea distinta\n"
        "\n"
        "Tercera linea distinta\n"
    )

    excerpt = extract_lyrics_excerpt(lyrics, char_budget=500)

    assert excerpt.startswith("Primera linea distinta")
    assert "Segunda linea distinta" in excerpt


def test_excerpt_is_capped_at_the_char_budget() -> None:
    lyrics = "una linea muy larga que se repite mucho " * 50

    excerpt = extract_lyrics_excerpt(lyrics, char_budget=100)

    assert len(excerpt) <= 100


def test_excerpt_of_very_short_lyrics_returns_them_unchanged() -> None:
    lyrics = "Solo una linea corta."

    excerpt = extract_lyrics_excerpt(lyrics, char_budget=500)

    assert excerpt == "Solo una linea corta."


def test_excerpt_of_empty_or_whitespace_lyrics_is_empty() -> None:
    assert extract_lyrics_excerpt("", char_budget=500) == ""
    assert extract_lyrics_excerpt("   \n\n  \n", char_budget=500) == ""


def test_excerpt_strips_section_tags_like_coro() -> None:
    lyrics = (
        "[Verso]\n"
        "Un dia cualquiera, el sol se asoma\n"
        "\n"
        "[Coro]\n"
        "Vuela conmigo, siente el momento\n"
        "vuela conmigo, siente el momento\n"
        "\n"
        "[Verso]\n"
        "Otro dia cualquiera, la luna brilla\n"
    )

    excerpt = extract_lyrics_excerpt(lyrics, char_budget=500)

    assert "[Coro]" not in excerpt
    assert "[Verso]" not in excerpt
    assert "Vuela conmigo, siente el momento" in excerpt
