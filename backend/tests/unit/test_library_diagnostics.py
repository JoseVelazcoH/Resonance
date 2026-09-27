"""Unit tests for LibraryDiagnosticsUseCase, using in-memory fakes."""

from __future__ import annotations

from mood_dj.application.library_diagnostics import LibraryDiagnosticsUseCase
from mood_dj.application.library_store import LibraryStore
from mood_dj.domain.models import TrackMoodProfile


def _profile(track_id: str, mood_id: str = "happiness") -> TrackMoodProfile:
    return TrackMoodProfile(
        track_id=track_id, mood_id=mood_id, mood_confidence=0.8, positive_probability=0.7, version="v1",
    )


class FakeMoodProfileRepository:
    def __init__(self, profiles: list[TrackMoodProfile]) -> None:
        self._profiles = profiles

    def get(self, track_id, version):
        for p in self._profiles:
            if p.track_id == track_id and p.version == version:
                return p
        return None

    def save(self, profile) -> None:
        self._profiles.append(profile)

    def get_all(self, version):
        return [p for p in self._profiles if p.version == version]


def test_no_prepared_library_yields_zero_total_tracks() -> None:
    use_case = LibraryDiagnosticsUseCase(LibraryStore(), FakeMoodProfileRepository([]), profile_version="v1")

    report = use_case.run("session-1")

    assert report.total_tracks == 0
    assert report.profiled_count == 0


def test_scopes_profiles_to_the_sessions_own_library_tracks() -> None:
    from mood_dj.domain.models import PlaylistTrack

    store = LibraryStore()
    store.save(
        "session-1",
        snapshot_map={"p1": "s1"},
        tracks=[PlaylistTrack(id="t1", name="A", artist="X", album="Y", duration_s=1.0, cover_url=None, external_url=None)],
        playlist_order=["p1"],
        tracks_by_playlist={"p1": ["t1"]},
    )
    repo = FakeMoodProfileRepository([_profile("t1"), _profile("t-from-another-session")])
    use_case = LibraryDiagnosticsUseCase(store, repo, profile_version="v1")

    report = use_case.run("session-1")

    assert report.total_tracks == 1
    assert report.profiled_count == 1
    assert report.moods[0].mood_id == "happiness"
