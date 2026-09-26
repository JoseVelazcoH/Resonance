"""Unit tests for the local playlist allowlist filter."""

from __future__ import annotations

import json

from mood_dj.adapters.playlist_allowlist import filter_playlists_by_allowlist, load_playlist_allowlist
from mood_dj.domain.models import PlaylistSummary


def _playlist(name: str, playlist_id: str = "id") -> PlaylistSummary:
    return PlaylistSummary(id=playlist_id, name=name, image_url=None, track_count=1, snapshot_id="s1")


def test_load_playlist_allowlist_returns_none_when_file_missing(tmp_path):
    missing = tmp_path / "playlists.local.json"

    assert load_playlist_allowlist(str(missing)) is None


def test_load_playlist_allowlist_returns_none_when_list_empty(tmp_path):
    path = tmp_path / "playlists.local.json"
    path.write_text(json.dumps({"playlist_allowlist": []}))

    assert load_playlist_allowlist(str(path)) is None


def test_load_playlist_allowlist_reads_names(tmp_path):
    path = tmp_path / "playlists.local.json"
    path.write_text(json.dumps({"playlist_allowlist": ["Blues y Jazz", "Mars"]}))

    assert load_playlist_allowlist(str(path)) == ["Blues y Jazz", "Mars"]


def test_filter_matches_case_insensitively(tmp_path):
    playlists = [_playlist("blues Y jazz"), _playlist("Something Else")]

    filtered, not_found = filter_playlists_by_allowlist(playlists, ["Blues Y Jazz"])

    assert [p.name for p in filtered] == ["blues Y jazz"]
    assert not_found == []


def test_filter_matches_accent_insensitively(tmp_path):
    playlists = [_playlist("Particula")]

    filtered, not_found = filter_playlists_by_allowlist(playlists, ["Partícula"])

    assert [p.name for p in filtered] == ["Particula"]
    assert not_found == []


def test_filter_reports_names_not_found(tmp_path):
    playlists = [_playlist("Mars")]

    filtered, not_found = filter_playlists_by_allowlist(playlists, ["Mars", "Ghost Playlist"])

    assert [p.name for p in filtered] == ["Mars"]
    assert not_found == ["Ghost Playlist"]


def test_filter_with_none_allowlist_returns_all_playlists_unchanged():
    playlists = [_playlist("A"), _playlist("B")]

    filtered, not_found = filter_playlists_by_allowlist(playlists, None)

    assert filtered == playlists
    assert not_found == []
