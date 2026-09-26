"""Unit tests for LibraryStore: caches a session's union track library by snapshot."""

from __future__ import annotations

from mood_dj.application.library_store import LibraryStore
from mood_dj.domain.models import PlaylistTrack


def _track(track_id: str) -> PlaylistTrack:
    return PlaylistTrack(
        id=track_id, name="Song", artist="Artist", album="Album", duration_s=200.0,
        cover_url=None, external_url=None,
    )


def test_get_returns_none_when_nothing_cached() -> None:
    store = LibraryStore()

    assert store.get("session-1") is None


def test_save_then_get_round_trips() -> None:
    store = LibraryStore()
    tracks = [_track("t1"), _track("t2")]

    store.save("session-1", snapshot_map={"p1": "snap1"}, tracks=tracks)
    snapshot = store.get("session-1")

    assert snapshot is not None
    assert snapshot.snapshot_map == {"p1": "snap1"}
    assert snapshot.tracks == tracks


def test_sessions_are_isolated() -> None:
    store = LibraryStore()
    store.save("session-1", snapshot_map={"p1": "snap1"}, tracks=[_track("t1")])

    assert store.get("session-2") is None
