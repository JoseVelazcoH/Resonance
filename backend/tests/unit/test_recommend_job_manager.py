"""Unit tests for RecommendJobManager: one background job per session."""

from __future__ import annotations

import threading
import time

from mood_dj.application.recommend_job_manager import RecommendJobManager, RecommendJobState
from mood_dj.application.recommend_from_library import (
    LibraryTrackSummary,
    PlaylistRecommendation,
    RecommendPhase,
    RecommendRunProgress,
)


class BlockingUseCase:
    """A fake use case that blocks until released, so tests can inspect mid-run state."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.run_calls: list[tuple[str, str]] = []

    def run(self, session_id: str, prompt: str, on_progress=None, on_decisions=None, on_library_tracks=None):
        self.run_calls.append((session_id, prompt))
        self.started.set()
        self.release.wait(timeout=5)
        if on_progress is not None:
            on_progress(RecommendRunProgress(phase=RecommendPhase.RANKING_TRACKS.value, processed=1, total=1))
        if on_library_tracks is not None:
            on_library_tracks([LibraryTrackSummary(id="a", name="Song A", artist="Artist", cover_url=None)])
        return PlaylistRecommendation(
            strategy=None,
            signal_probabilities={},
            stages=[],
            detected=None,
            excluded_no_lyrics=0,
            excluded_instrumental=0,
            excluded_no_profile=0,
        )


def test_status_is_none_for_unknown_job() -> None:
    manager = RecommendJobManager(use_case_factory=lambda: BlockingUseCase())

    assert manager.status("nope", "session-1") is None


def test_start_runs_job_and_status_becomes_done() -> None:
    use_case = BlockingUseCase()
    use_case.release.set()
    manager = RecommendJobManager(use_case_factory=lambda: use_case)

    job_id = manager.start("session-1", "prompt")
    for _ in range(50):
        status = manager.status(job_id, "session-1")
        if status is not None and status.state is RecommendJobState.DONE:
            break
        time.sleep(0.02)

    status = manager.status(job_id, "session-1")
    assert status is not None
    assert status.state is RecommendJobState.DONE
    assert status.result is not None
    assert use_case.run_calls == [("session-1", "prompt")]


def test_start_returns_same_job_while_running_for_same_session() -> None:
    use_case = BlockingUseCase()
    manager = RecommendJobManager(use_case_factory=lambda: use_case)

    job_id = manager.start("session-1", "prompt")
    use_case.started.wait(timeout=5)
    second_job_id = manager.start("session-1", "other prompt")
    use_case.release.set()
    time.sleep(0.1)

    assert job_id == second_job_id
    assert len(use_case.run_calls) == 1


def test_different_sessions_get_independent_jobs() -> None:
    use_case = BlockingUseCase()
    manager = RecommendJobManager(use_case_factory=lambda: use_case)

    job_id_a = manager.start("session-a", "prompt")
    use_case.started.wait(timeout=5)
    use_case.release.set()
    time.sleep(0.1)

    use_case.started.clear()
    use_case.release.clear()
    job_id_b = manager.start("session-b", "prompt")

    assert job_id_a != job_id_b


def test_status_reports_error_when_use_case_raises() -> None:
    class FailingUseCase:
        def run(self, session_id, prompt, on_progress=None, on_decisions=None, on_library_tracks=None):
            raise RuntimeError("boom")

    manager = RecommendJobManager(use_case_factory=lambda: FailingUseCase())

    job_id = manager.start("session-1", "prompt")
    for _ in range(50):
        status = manager.status(job_id, "session-1")
        if status is not None and status.state is RecommendJobState.ERROR:
            break
        time.sleep(0.02)

    status = manager.status(job_id, "session-1")
    assert status is not None
    assert status.state is RecommendJobState.ERROR
    assert status.error == "boom"


def test_progress_callback_updates_phase_and_counts() -> None:
    manager = RecommendJobManager(use_case_factory=lambda: BlockingUseCase())
    use_case = manager._use_case_factory()
    manager._use_case_factory = lambda: use_case

    job_id = manager.start("session-1", "prompt")
    use_case.started.wait(timeout=5)
    use_case.release.set()
    for _ in range(50):
        status = manager.status(job_id, "session-1")
        if status is not None and status.state is RecommendJobState.DONE:
            break
        time.sleep(0.02)

    status = manager.status(job_id, "session-1")
    assert status is not None
    assert status.phase == RecommendPhase.RANKING_TRACKS.value
    assert status.processed == 1
    assert status.total == 1


def test_library_tracks_are_published_on_the_job() -> None:
    use_case = BlockingUseCase()
    use_case.release.set()
    manager = RecommendJobManager(use_case_factory=lambda: use_case)

    job_id = manager.start("session-1", "prompt")
    for _ in range(50):
        status = manager.status(job_id, "session-1")
        if status is not None and status.state is RecommendJobState.DONE:
            break
        time.sleep(0.02)

    status = manager.status(job_id, "session-1")
    assert status is not None
    assert status.library_tracks is not None
    assert status.library_tracks[0].id == "a"


def test_status_hides_jobs_owned_by_another_session() -> None:
    use_case = BlockingUseCase()
    use_case.release.set()
    manager = RecommendJobManager(use_case_factory=lambda: use_case)

    job_id = manager.start("session-1", "prompt")

    assert manager.status(job_id, "session-2") is None
    assert manager.status(job_id, "session-1") is not None
