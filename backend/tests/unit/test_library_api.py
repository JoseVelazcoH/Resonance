"""Unit tests for the /library endpoints, with adapters replaced by fakes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mood_dj.api.deps import (
    get_current_tokens,
    get_library_diagnostics_use_case,
    get_library_job_manager,
    get_recommend_job_manager,
    get_session_id,
)
from mood_dj.api.main import app
from mood_dj.application.library_lyrics_status import LibraryLyricsSummary
from mood_dj.domain.mood_diagnostics import MoodDiagnostics, MoodDiagnosticsReport
from mood_dj.domain.models import LibraryLyricsRow, LibraryPrepareProgress, LibraryPrepareState, SpotifyTokens


class FakeLibraryJobManager:
    def __init__(
        self,
        status: LibraryPrepareProgress | None = None,
        rows: list[LibraryLyricsRow] | None = None,
        summary: LibraryLyricsSummary | None = None,
    ) -> None:
        self.started: list[tuple[str, str]] = []
        self._status = status or LibraryPrepareProgress(
            state=LibraryPrepareState.RUNNING,
            phase="fetching lyrics",
            processed=3,
            total=10,
            cached=1,
            playlists=5,
            tracks=100,
            with_lyrics=2,
            instrumental=1,
        )
        self._rows = rows if rows is not None else []
        self._summary = summary

    def start(self, session_id: str, access_token: str) -> bool:
        self.started.append((session_id, access_token))
        return True

    def status(self, session_id: str) -> LibraryPrepareProgress:
        return self._status

    def lyrics_rows(self, session_id: str, offset=None, limit=50):
        return self._rows, len(self._rows)

    def lyrics_summary(self, session_id: str):
        return self._summary


class FakeRecommendJobManager:
    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []

    def start(self, session_id: str, prompt: str) -> str:
        self.started.append((session_id, prompt))
        return "job-1"


def teardown_function() -> None:
    app.dependency_overrides = {}


def _with_tokens_and_session() -> None:
    app.dependency_overrides[get_current_tokens] = lambda: SpotifyTokens(
        access_token="tok", refresh_token="ref", expires_at=99999999999.0
    )
    app.dependency_overrides[get_session_id] = lambda: "session-1"


def test_prepare_requires_session() -> None:
    client = TestClient(app)

    response = client.post("/library/prepare")

    assert response.status_code == 401


def test_prepare_starts_job() -> None:
    _with_tokens_and_session()
    job_manager = FakeLibraryJobManager()
    app.dependency_overrides[get_library_job_manager] = lambda: job_manager
    client = TestClient(app)

    response = client.post("/library/prepare")

    assert response.status_code == 200
    assert response.json() == {"started": True}
    assert job_manager.started == [("session-1", "tok")]


def test_status_requires_session() -> None:
    client = TestClient(app)

    response = client.get("/library/status")

    assert response.status_code == 401


def test_status_returns_phase_and_progress_counts() -> None:
    app.dependency_overrides[get_session_id] = lambda: "session-1"
    app.dependency_overrides[get_library_job_manager] = lambda: FakeLibraryJobManager()
    client = TestClient(app)

    response = client.get("/library/status")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "running"
    assert body["phase"] == "fetching lyrics"
    assert body["processed"] == 3
    assert body["total"] == 10
    assert body["playlists"] == 5
    assert body["tracks"] == 100
    assert body["with_lyrics"] == 2
    assert body["instrumental"] == 1


def test_recommend_requires_session() -> None:
    client = TestClient(app)

    response = client.post("/library/recommend", json={"prompt": "sad"})

    assert response.status_code == 401


def test_recommend_returns_409_when_not_prepared() -> None:
    _with_tokens_and_session()
    app.dependency_overrides[get_library_job_manager] = lambda: FakeLibraryJobManager()  # state=running
    client = TestClient(app)

    response = client.post("/library/recommend", json={"prompt": "sad"})

    assert response.status_code == 409


def test_recommend_returns_409_when_prepared_but_zero_profiles() -> None:
    _with_tokens_and_session()
    done_status = LibraryPrepareProgress(
        state=LibraryPrepareState.DONE, phase="reading mood", processed=1, total=1, profiles_total=0
    )
    app.dependency_overrides[get_library_job_manager] = lambda: FakeLibraryJobManager(status=done_status)
    client = TestClient(app)

    response = client.post("/library/recommend", json={"prompt": "sad"})

    assert response.status_code == 409


def test_recommend_starts_job_and_returns_job_id_when_prepared() -> None:
    _with_tokens_and_session()
    done_status = LibraryPrepareProgress(
        state=LibraryPrepareState.DONE, phase="reading mood", processed=1, total=1, profiles_total=1
    )
    app.dependency_overrides[get_library_job_manager] = lambda: FakeLibraryJobManager(status=done_status)
    recommend_job_manager = FakeRecommendJobManager()
    app.dependency_overrides[get_recommend_job_manager] = lambda: recommend_job_manager
    client = TestClient(app)

    response = client.post("/library/recommend", json={"prompt": "I feel sad"})

    assert response.status_code == 202
    assert response.json() == {"job_id": "job-1"}
    assert recommend_job_manager.started == [("session-1", "I feel sad")]


def test_lyrics_status_requires_session() -> None:
    client = TestClient(app)

    response = client.get("/library/lyrics/status")

    assert response.status_code == 401


def test_lyrics_status_returns_rows_and_counts() -> None:
    app.dependency_overrides[get_session_id] = lambda: "session-1"
    rows = [LibraryLyricsRow(index=0, track_id="t1", name="Song", artist="Artist", status="downloaded")]
    summary = LibraryLyricsSummary(total=10, with_lyrics=7, instrumental=1, missing=1, pending=1, all_cached=False)
    app.dependency_overrides[get_library_job_manager] = lambda: FakeLibraryJobManager(rows=rows, summary=summary)
    client = TestClient(app)

    response = client.get("/library/lyrics/status")

    assert response.status_code == 200
    body = response.json()
    assert body["with_lyrics"] == 7
    assert body["instrumental"] == 1
    assert body["missing"] == 1
    assert body["pending"] == 1
    assert body["rows"] == [{"index": 0, "track_id": "t1", "name": "Song", "artist": "Artist", "status": "downloaded"}]


def test_lyrics_summary_reports_all_cached() -> None:
    app.dependency_overrides[get_session_id] = lambda: "session-1"
    summary = LibraryLyricsSummary(total=10, with_lyrics=10, instrumental=0, missing=0, pending=0, all_cached=True)
    app.dependency_overrides[get_library_job_manager] = lambda: FakeLibraryJobManager(summary=summary)
    client = TestClient(app)

    response = client.get("/library/lyrics/summary")

    assert response.status_code == 200
    assert response.json() == {"all_cached": True}


class FakeDiagnosticsUseCase:
    def __init__(self, report: MoodDiagnosticsReport) -> None:
        self._report = report

    def run(self, session_id: str) -> MoodDiagnosticsReport:
        return self._report


def test_diagnostics_requires_session() -> None:
    client = TestClient(app)

    response = client.get("/library/diagnostics")

    assert response.status_code == 401


def test_diagnostics_returns_per_mood_stats() -> None:
    app.dependency_overrides[get_session_id] = lambda: "session-1"
    report = MoodDiagnosticsReport(
        moods=(
            MoodDiagnostics(
                mood_id="happiness",
                count=2,
                mean_confidence=0.8,
                mean_positive_probability=0.75,
                mean_entropy=0.5,
                mean_probabilities={"happiness": 0.8, "love": 0.2},
            ),
        ),
        profiled_count=2,
        total_tracks=3,
        unprofiled_share=1 / 3,
        overall_mean_entropy=0.5,
    )
    app.dependency_overrides[get_library_diagnostics_use_case] = lambda: FakeDiagnosticsUseCase(report)
    client = TestClient(app)

    response = client.get("/library/diagnostics")

    assert response.status_code == 200
    body = response.json()
    assert body["profiled_count"] == 2
    assert body["total_tracks"] == 3
    assert body["moods"][0]["mood_id"] == "happiness"
    assert body["moods"][0]["mean_confidence"] == 0.8
    assert body["moods"][0]["mean_probabilities"]["love"] == 0.2
