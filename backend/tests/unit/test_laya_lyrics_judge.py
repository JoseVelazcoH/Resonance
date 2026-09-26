"""Unit tests for the LayaLyricsJudge adapter, using a fake router (no model, no network)."""

from __future__ import annotations

from mood_dj.adapters.laya_lyrics_judge import MODEL_NAME, LayaLyricsJudge
from mood_dj.domain.models import PlaylistTrack, Strategy
from mood_dj.ports.lyrics_judge import TrackLyrics


def _track(track_id: str = "1") -> PlaylistTrack:
    return PlaylistTrack(
        id=track_id,
        name="Song",
        artist="Artist",
        album="Album",
        duration_s=200.0,
        cover_url=None,
        external_url=None,
    )


class FakeRouter:
    def __init__(self, predict_result=None, batch_results=None) -> None:
        self.predict_result = predict_result
        self.batch_results = batch_results or []
        self.predict_calls: list[tuple] = []
        self.predict_batch_calls: list[list[dict]] = []

    def predict(self, state, questions, model=None):
        self.predict_calls.append((state, questions, model))
        return self.predict_result

    def predict_batch(self, requests, batch_size=None):
        self.predict_batch_calls.append(list(requests))
        return self.batch_results


def test_detect_signals_maps_noul_probabilities_to_booleans() -> None:
    router = FakeRouter(
        predict_result={
            "answers": {
                "feels_bad": {"noul": 0.9},
                "wants_change": {"noul": 0.8},
                "wants_energy": {"noul": 0.1},
                "wants_rest": {"noul": 0.2},
            }
        }
    )
    judge = LayaLyricsJudge(router=router)

    signals, probabilities = judge.detect_signals("Estoy triste pero quiero sentirme mejor")

    assert signals.feels_bad is True
    assert signals.wants_change is True
    assert signals.wants_energy is False
    assert signals.wants_rest is False
    assert probabilities["feels_bad"] == 0.9


def test_detect_signals_forces_the_multilingual_model() -> None:
    router = FakeRouter(
        predict_result={
            "answers": {
                "feels_bad": {"noul": 0.1},
                "wants_change": {"noul": 0.1},
                "wants_energy": {"noul": 0.1},
                "wants_rest": {"noul": 0.1},
            }
        }
    )
    judge = LayaLyricsJudge(router=router)

    judge.detect_signals("any prompt")

    assert router.predict_calls[0][2] == MODEL_NAME
    assert MODEL_NAME == "multilingual"


def test_judge_tone_returns_tone_in_input_order() -> None:
    router = FakeRouter(
        batch_results=[
            {"answers": {"tone": {"score": 0.0}}},
            {"answers": {"tone": {"score": 4.0}}},
        ]
    )
    judge = LayaLyricsJudge(router=router)
    tracks = [
        TrackLyrics(track=_track("sad-song"), text="cry cry cry"),
        TrackLyrics(track=_track("happy-song"), text="joy joy joy"),
    ]

    tones = judge.judge_tone(tracks)

    assert tones[0].track_id == "sad-song"
    assert tones[0].tone == 0.0
    assert tones[1].track_id == "happy-song"
    assert tones[1].tone == 1.0


def test_judge_tone_uses_a_single_score_question_and_forces_multilingual() -> None:
    router = FakeRouter(batch_results=[{"answers": {"tone": {"score": 2.0}}}])
    judge = LayaLyricsJudge(router=router)

    judge.judge_tone([TrackLyrics(track=_track(), text="la la")])

    assert len(router.predict_batch_calls) == 1
    request = router.predict_batch_calls[0][0]
    assert request["model"] == MODEL_NAME
    assert request["questions"]["tone"]["type"] == "score"
    assert "fit" not in request["questions"]
    assert "prompt" not in request["state"]


def test_judge_tone_truncates_long_lyrics() -> None:
    router = FakeRouter(batch_results=[{"answers": {"tone": {"score": 2.0}}}])
    judge = LayaLyricsJudge(router=router, lyrics_truncate_chars=10)

    judge.judge_tone([TrackLyrics(track=_track(), text="x" * 100)])

    request = router.predict_batch_calls[0][0]
    assert len(request["state"]["lyrics"]) == 10


def test_judge_tone_returns_empty_list_for_no_tracks() -> None:
    router = FakeRouter()
    judge = LayaLyricsJudge(router=router)

    assert judge.judge_tone([]) == []
    assert router.predict_batch_calls == []


def test_judge_tone_processes_tracks_in_chunks_and_reports_progress() -> None:
    answer = {"answers": {"tone": {"score": 2.0}}}
    router = FakeRouter(batch_results=[answer, answer, answer, answer])
    judge = LayaLyricsJudge(router=router, batch_size=4)
    tracks = [TrackLyrics(track=_track(str(i)), text="la") for i in range(10)]
    progress_calls: list[int] = []

    tones = judge.judge_tone(tracks, on_progress=progress_calls.append)

    assert len(router.predict_batch_calls) == 3
    assert [len(call) for call in router.predict_batch_calls] == [4, 4, 2]
    assert progress_calls == [4, 4, 2]
    assert len(tones) == 10
    assert [t.track_id for t in tones] == [str(i) for i in range(10)]


def test_judge_fit_returns_fit_in_input_order() -> None:
    router = FakeRouter(
        batch_results=[
            {"answers": {"fit": {"noul": 0.9}}},
            {"answers": {"fit": {"noul": 0.1}}},
        ]
    )
    judge = LayaLyricsJudge(router=router)
    tracks = [
        TrackLyrics(track=_track("a"), text="cry cry cry"),
        TrackLyrics(track=_track("b"), text="joy joy joy"),
    ]

    fits = judge.judge_fit("I feel sad", Strategy.LIFT, tracks)

    assert fits[0].track_id == "a"
    assert fits[0].fit == 0.9
    assert fits[1].track_id == "b"
    assert fits[1].fit == 0.1


def test_judge_fit_uses_a_single_noul_question_and_forces_multilingual() -> None:
    router = FakeRouter(batch_results=[{"answers": {"fit": {"noul": 0.5}}}])
    judge = LayaLyricsJudge(router=router)

    judge.judge_fit("prompt", Strategy.ACCOMPANY, [TrackLyrics(track=_track(), text="la la")])

    assert len(router.predict_batch_calls) == 1
    request = router.predict_batch_calls[0][0]
    assert request["model"] == MODEL_NAME
    assert request["questions"]["fit"]["type"] == "noul"
    assert "tone" not in request["questions"]
    assert request["state"]["prompt"] == "prompt"


def test_judge_fit_truncates_long_lyrics() -> None:
    router = FakeRouter(batch_results=[{"answers": {"fit": {"noul": 0.5}}}])
    judge = LayaLyricsJudge(router=router, lyrics_truncate_chars=10)

    judge.judge_fit("prompt", Strategy.CALM, [TrackLyrics(track=_track(), text="x" * 100)])

    request = router.predict_batch_calls[0][0]
    assert len(request["state"]["lyrics"]) == 10


def test_judge_fit_returns_empty_list_for_no_tracks() -> None:
    router = FakeRouter()
    judge = LayaLyricsJudge(router=router)

    assert judge.judge_fit("prompt", Strategy.LIFT, []) == []
    assert router.predict_batch_calls == []


def test_judge_fit_processes_tracks_in_chunks_and_reports_progress() -> None:
    answer = {"answers": {"fit": {"noul": 0.5}}}
    router = FakeRouter(batch_results=[answer, answer, answer, answer])
    judge = LayaLyricsJudge(router=router, batch_size=4)
    tracks = [TrackLyrics(track=_track(str(i)), text="la") for i in range(10)]
    progress_calls: list[int] = []

    fits = judge.judge_fit("prompt", Strategy.LIFT, tracks, on_progress=progress_calls.append)

    assert len(router.predict_batch_calls) == 3
    assert [len(call) for call in router.predict_batch_calls] == [4, 4, 2]
    assert progress_calls == [4, 4, 2]
    assert len(fits) == 10
    assert [f.track_id for f in fits] == [str(i) for i in range(10)]


def test_judge_fit_without_progress_callback_still_works() -> None:
    router = FakeRouter(batch_results=[{"answers": {"fit": {"noul": 0.5}}}])
    judge = LayaLyricsJudge(router=router, batch_size=8)

    fits = judge.judge_fit("prompt", Strategy.LIFT, [TrackLyrics(track=_track(), text="la")])

    assert len(fits) == 1
