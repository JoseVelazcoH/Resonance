"""Unit tests for LayaTrackProfiler, using a fake Laya Router.

v5 (flat mood + full distribution): one `predict_batch` pass answers a flat
`mood` choice (7 options, from `moods.json`) and a `noul` `polarity` question,
fed the plain full lyrics string as `state` with `max_len` set to the
multilingual checkpoint's limit.
"""

from __future__ import annotations

from mood_dj.adapters.laya_track_profiler import (
    MULTILINGUAL_MAX_LEN,
    LayaTrackProfiler,
    compute_version,
)
from mood_dj.ports.track_profiler import TrackForProfiling


class FakeRouter:
    """Answers every question in a request deterministically from its criteria.

    `overrides` maps a question id to `{"choice": <key>, "prob": <float>}` for a
    `choice` question, or `{"noul": <float>}` for a `noul` question. Records
    every batch of requests it receives so tests can assert on request shape.
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
                if qdef["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": self.overrides.get(qid, {}).get("noul", 0.5)}
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
        TrackForProfiling(track_id=f"t{i}", artist="Artist", title=f"Song {i}", lyrics=f"la la la {i}")
        for i in range(n)
    ]


def test_single_batch_call_shares_one_schema_across_every_track() -> None:
    router = FakeRouter()
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiler.profile(_tracks(3))

    assert len(router.batches) == 1
    first_batch = router.batches[0]
    schemas = [set(req["questions"].keys()) for req in first_batch]
    assert all(s == schemas[0] for s in schemas)
    assert schemas[0] == {"mood", "polarity"}


def test_request_state_is_the_plain_lyrics_string() -> None:
    router = FakeRouter()
    profiler = LayaTrackProfiler(router=router, batch_size=100)
    tracks = [TrackForProfiling(track_id="t1", artist="Artist", title="Song", lyrics="hola mundo")]

    profiler.profile(tracks)

    request = router.batches[0][0]
    assert request["state"] == "hola mundo"
    assert isinstance(request["state"], str)


def test_request_uses_the_multilingual_checkpoints_max_len_by_default() -> None:
    router = FakeRouter()
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiler.profile(_tracks(1))

    assert router.batches[0][0]["max_len"] == MULTILINGUAL_MAX_LEN


def test_custom_max_len_is_forwarded_to_the_request() -> None:
    router = FakeRouter()
    profiler = LayaTrackProfiler(router=router, batch_size=100, max_len=512)

    profiler.profile(_tracks(1))

    assert router.batches[0][0]["max_len"] == 512


def test_mood_choice_offers_all_seven_moods() -> None:
    router = FakeRouter()
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiler.profile(_tracks(1))

    mood_criteria = router.batches[0][0]["questions"]["mood"]["criteria"]
    assert set(mood_criteria.keys()) == {
        "love",
        "happiness",
        "comfort",
        "sadness",
        "loneliness",
        "anger",
        "fear",
    }


def test_profile_stores_mood_id_and_confidence_from_the_chosen_option() -> None:
    router = FakeRouter(overrides={"mood": {"choice": "anger", "prob": 0.73}})
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiles = profiler.profile(_tracks(1))

    assert profiles[0].mood_id == "anger"
    assert profiles[0].mood_confidence == 0.73


def test_profile_stores_the_full_mood_probability_distribution() -> None:
    router = FakeRouter(overrides={"mood": {"choice": "love", "prob": 0.4}})
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiles = profiler.profile(_tracks(1))

    distribution = profiles[0].mood_probabilities
    assert set(distribution.keys()) == {
        "love", "happiness", "comfort", "sadness", "loneliness", "anger", "fear",
    }
    assert distribution["love"] == 0.4
    assert sum(distribution.values()) == 1.0 or abs(sum(distribution.values()) - 1.0) < 1e-9


def test_profile_stores_positive_probability_from_the_polarity_noul() -> None:
    router = FakeRouter(overrides={"polarity": {"noul": 0.88}})
    profiler = LayaTrackProfiler(router=router, batch_size=100)

    profiles = profiler.profile(_tracks(1))

    assert profiles[0].positive_probability == 0.88


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


def test_version_changes_when_moods_json_bytes_change(monkeypatch) -> None:
    from mood_dj.adapters import laya_track_profiler as module

    original = module._read_taxonomy_bytes
    before = module.compute_version()

    def patched(filename: str) -> bytes:
        if filename == "moods.json":
            return original(filename) + b" "
        return original(filename)

    monkeypatch.setattr(module, "_read_taxonomy_bytes", patched)
    after = module.compute_version()

    assert before != after


def test_every_track_gets_a_profile_for_the_same_batch() -> None:
    router = FakeRouter()
    profiler = LayaTrackProfiler(router=router, batch_size=2)

    profiles = profiler.profile(_tracks(5))

    assert {p.track_id for p in profiles} == {f"t{i}" for i in range(5)}
    assert all(p.version == profiler.version for p in profiles)
