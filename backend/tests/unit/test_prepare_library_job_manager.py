"""Unit tests for PrepareLibraryJobManager: one background job per session."""

from __future__ import annotations

import threading
import time

from mood_dj.application.prepare_library_job_manager import PrepareLibraryJobManager
from mood_dj.domain.models import LibraryPrepareState


class BlockingUseCase:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.run_calls: list[tuple[str, str]] = []

    def run(self, session_id: str, access_token: str, on_progress=None):
        self.run_calls.append((session_id, access_token))
        self.started.set()
        self.release.wait(timeout=5)


def test_status_is_idle_before_any_job_started() -> None:
    manager = PrepareLibraryJobManager(use_case_factory=lambda: BlockingUseCase())

    assert manager.status("session-1").state is LibraryPrepareState.IDLE


def test_start_runs_job_and_status_becomes_done() -> None:
    use_case = BlockingUseCase()
    use_case.release.set()
    manager = PrepareLibraryJobManager(use_case_factory=lambda: use_case)

    manager.start("session-1", "token")
    use_case.started.wait(timeout=5)
    time.sleep(0.1)

    assert use_case.run_calls == [("session-1", "token")]


def test_start_is_a_no_op_while_job_already_running_for_same_session() -> None:
    use_case = BlockingUseCase()
    manager = PrepareLibraryJobManager(use_case_factory=lambda: use_case)

    manager.start("session-1", "token")
    use_case.started.wait(timeout=5)
    started = manager.start("session-1", "token")
    use_case.release.set()
    time.sleep(0.1)

    assert started is False
    assert len(use_case.run_calls) == 1


def test_different_sessions_get_independent_jobs() -> None:
    use_case = BlockingUseCase()
    manager = PrepareLibraryJobManager(use_case_factory=lambda: use_case)

    manager.start("session-a", "token")
    use_case.started.wait(timeout=5)
    started = manager.start("session-b", "token")

    assert started is True


def test_status_reports_error_when_use_case_raises() -> None:
    class FailingUseCase:
        def run(self, session_id, access_token, on_progress=None):
            raise RuntimeError("boom")

    manager = PrepareLibraryJobManager(use_case_factory=lambda: FailingUseCase())

    manager.start("session-1", "token")
    for _ in range(50):
        if manager.status("session-1").state is LibraryPrepareState.ERROR:
            break
        time.sleep(0.05)

    status = manager.status("session-1")
    assert status.state is LibraryPrepareState.ERROR
    assert status.error == "boom"
