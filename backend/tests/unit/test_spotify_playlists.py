"""Unit tests for the SpotifyPlaylistsClient adapter, using a fake HTTP client."""

from __future__ import annotations

from mood_dj.adapters.spotify_playlists import (
    CURRENT_USER_URL,
    HttpxSpotifyPlaylistsHttpClient,
    PLAYLISTS_URL,
    SpotifyPlaylistsClient,
    playlist_items_url,
    playlist_tracks_url,
    user_playlists_url,
)
from mood_dj.ports.spotify_playlists import SpotifyApiError


class FakeHttpClient:
    """Fakes GET/POST requests by returning a fixed page/response per exact URL."""

    def __init__(self, pages: dict[str, dict], posts: dict[str, tuple[int, dict]] | None = None) -> None:
        self.pages = pages
        self.posts = posts or {}
        self.calls: list[str] = []
        self.post_calls: list[tuple[str, dict]] = []

    def get(self, url: str, access_token: str) -> dict:
        self.calls.append(url)
        if url == CURRENT_USER_URL and url not in self.pages:
            return {"id": "me"}
        return self.pages[url]

    def post(self, url: str, access_token: str, payload: dict) -> tuple[int, dict]:
        self.post_calls.append((url, payload))
        return self.posts[url]


def test_list_playlists_maps_fields() -> None:
    http = FakeHttpClient(
        {
            PLAYLISTS_URL + "?limit=50": {
                "items": [
                    {
                        "id": "pl1",
                        "name": "My Playlist",
                        "images": [{"url": "https://img/1.jpg"}],
                        "tracks": {"total": 42},
                        "snapshot_id": "snap1",
                    }
                ],
                "next": None,
            }
        }
    )
    client = SpotifyPlaylistsClient(http_client=http)

    playlists = client.list_playlists("token")

    assert len(playlists) == 1
    assert playlists[0].id == "pl1"
    assert playlists[0].name == "My Playlist"
    assert playlists[0].image_url == "https://img/1.jpg"
    assert playlists[0].track_count == 42
    assert playlists[0].snapshot_id == "snap1"


def test_list_playlists_reads_track_count_from_items_key() -> None:
    # The current Spotify API reports the count under "items", not "tracks".
    http = FakeHttpClient(
        {
            PLAYLISTS_URL + "?limit=50": {
                "items": [{"id": "pl1", "name": "New Shape", "images": [], "items": {"total": 6}, "snapshot_id": "s"}],
                "next": None,
            }
        }
    )
    client = SpotifyPlaylistsClient(http_client=http)

    playlists = client.list_playlists("token")

    assert playlists[0].track_count == 6


def test_list_playlists_keeps_only_owned_or_collaborative_playlists() -> None:
    # Spotify answers 403 when reading tracks of playlists the user does not own.
    http = FakeHttpClient(
        {
            PLAYLISTS_URL + "?limit=50": {
                "items": [
                    {"id": "mine", "name": "Mine", "images": [], "owner": {"id": "me"}, "snapshot_id": "s"},
                    {"id": "followed", "name": "Followed", "images": [], "owner": {"id": "someone"}, "snapshot_id": "s"},
                    {
                        "id": "shared",
                        "name": "Shared",
                        "images": [],
                        "owner": {"id": "someone"},
                        "collaborative": True,
                        "snapshot_id": "s",
                    },
                ],
                "next": None,
            }
        }
    )
    client = SpotifyPlaylistsClient(http_client=http)

    playlists = client.list_playlists("token")

    assert [playlist.id for playlist in playlists] == ["mine", "shared"]


def test_list_playlists_handles_missing_images() -> None:
    http = FakeHttpClient(
        {
            PLAYLISTS_URL + "?limit=50": {
                "items": [{"id": "pl1", "name": "No Cover", "images": [], "tracks": {"total": 0}, "snapshot_id": "s"}],
                "next": None,
            }
        }
    )
    client = SpotifyPlaylistsClient(http_client=http)

    playlists = client.list_playlists("token")

    assert playlists[0].image_url is None


def test_list_playlists_follows_pagination() -> None:
    next_url = PLAYLISTS_URL + "?offset=1&limit=1"
    http = FakeHttpClient(
        {
            PLAYLISTS_URL + "?limit=50": {
                "items": [{"id": "pl1", "name": "A", "images": [], "tracks": {"total": 1}, "snapshot_id": "s"}],
                "next": next_url,
            },
            next_url: {
                "items": [{"id": "pl2", "name": "B", "images": [], "tracks": {"total": 1}, "snapshot_id": "s"}],
                "next": None,
            },
        }
    )
    client = SpotifyPlaylistsClient(http_client=http)

    playlists = client.list_playlists("token")

    assert [p.id for p in playlists] == ["pl1", "pl2"]


def _track_entry(track_id: str, key: str = "track") -> dict:
    return {
        key: {
            "id": track_id,
            "name": f"Song {track_id}",
            "artists": [{"name": "Artist One"}, {"name": "Artist Two"}],
            "album": {"name": "Album", "images": [{"url": "https://img/cover.jpg"}]},
            "duration_ms": 210000,
            "external_urls": {"spotify": f"https://open.spotify.com/track/{track_id}"},
        }
    }


def test_get_playlist_tracks_uses_item_key_with_track_fallback() -> None:
    url = playlist_items_url("pl1")
    http = FakeHttpClient(
        {url + "?limit=50": {"items": [_track_entry("t1", key="item"), _track_entry("t2", key="track")], "next": None}}
    )
    client = SpotifyPlaylistsClient(http_client=http)

    tracks = client.get_playlist_tracks("pl1", "token")

    assert [t.id for t in tracks] == ["t1", "t2"]
    assert tracks[0].artist == "Artist One"
    assert tracks[0].album == "Album"
    assert tracks[0].duration_s == 210.0
    assert tracks[0].cover_url == "https://img/cover.jpg"
    assert tracks[0].external_url == "https://open.spotify.com/track/t1"


def test_get_playlist_tracks_skips_null_local_and_episode_entries() -> None:
    local_track = _track_entry("local1", key="track")
    local_track["track"]["is_local"] = True
    episode = {"track": {"id": "ep1", "type": "episode"}}
    url = playlist_items_url("pl1")
    http = FakeHttpClient(
        {
            url + "?limit=50": {
                "items": [
                    {"track": None},
                    local_track,
                    episode,
                    _track_entry("t1", key="track"),
                ],
                "next": None,
            }
        }
    )
    client = SpotifyPlaylistsClient(http_client=http)

    tracks = client.get_playlist_tracks("pl1", "token")

    assert [t.id for t in tracks] == ["t1"]


def test_get_playlist_tracks_follows_pagination() -> None:
    url = playlist_items_url("pl1")
    next_url = url + "?offset=1&limit=1"
    http = FakeHttpClient(
        {
            url + "?limit=50": {"items": [_track_entry("t1", key="track")], "next": next_url},
            next_url: {"items": [_track_entry("t2", key="track")], "next": None},
        }
    )
    client = SpotifyPlaylistsClient(http_client=http)

    tracks = client.get_playlist_tracks("pl1", "token")

    assert [t.id for t in tracks] == ["t1", "t2"]


def test_httpx_client_can_be_constructed() -> None:
    HttpxSpotifyPlaylistsHttpClient()


def test_get_display_name_reads_profile_field() -> None:
    http = FakeHttpClient({CURRENT_USER_URL: {"id": "me", "display_name": "Jose"}})
    client = SpotifyPlaylistsClient(http_client=http)

    assert client.get_display_name("token") == "Jose"


def test_get_display_name_returns_none_when_missing() -> None:
    http = FakeHttpClient({CURRENT_USER_URL: {"id": "me"}})
    client = SpotifyPlaylistsClient(http_client=http)

    assert client.get_display_name("token") is None


def test_get_current_user_id() -> None:
    http = FakeHttpClient({CURRENT_USER_URL: {"id": "me"}})
    client = SpotifyPlaylistsClient(http_client=http)

    assert client.get_current_user_id("token") == "me"


def test_create_playlist_uses_me_playlists_endpoint() -> None:
    http = FakeHttpClient({}, posts={PLAYLISTS_URL: (201, {"id": "new-pl"})})
    client = SpotifyPlaylistsClient(http_client=http)

    playlist_id = client.create_playlist("me", "My Mood", "token")

    assert playlist_id == "new-pl"
    assert http.post_calls == [(PLAYLISTS_URL, {"name": "My Mood", "public": False})]


def test_create_playlist_falls_back_to_user_playlists_endpoint_on_404() -> None:
    http = FakeHttpClient(
        {},
        posts={
            PLAYLISTS_URL: (404, {}),
            user_playlists_url("me"): (201, {"id": "new-pl"}),
        },
    )
    client = SpotifyPlaylistsClient(http_client=http)

    playlist_id = client.create_playlist("me", "My Mood", "token")

    assert playlist_id == "new-pl"
    assert [url for url, _ in http.post_calls] == [PLAYLISTS_URL, user_playlists_url("me")]


def test_create_playlist_raises_spotify_api_error_on_failure() -> None:
    http = FakeHttpClient({}, posts={PLAYLISTS_URL: (403, {"error": {"message": "no permission"}})})
    client = SpotifyPlaylistsClient(http_client=http)

    try:
        client.create_playlist("me", "My Mood", "token")
        assert False, "expected SpotifyApiError"
    except SpotifyApiError as exc:
        assert exc.status_code == 403


def test_add_tracks_uses_items_endpoint() -> None:
    http = FakeHttpClient({}, posts={playlist_items_url("pl1"): (201, {})})
    client = SpotifyPlaylistsClient(http_client=http)

    client.add_tracks("pl1", ["t1", "t2"], "token")

    assert http.post_calls == [(playlist_items_url("pl1"), {"uris": ["spotify:track:t1", "spotify:track:t2"]})]


def test_add_tracks_falls_back_to_tracks_endpoint_on_404() -> None:
    http = FakeHttpClient(
        {},
        posts={
            playlist_items_url("pl1"): (404, {}),
            playlist_tracks_url("pl1"): (201, {}),
        },
    )
    client = SpotifyPlaylistsClient(http_client=http)

    client.add_tracks("pl1", ["t1"], "token")

    assert [url for url, _ in http.post_calls] == [playlist_items_url("pl1"), playlist_tracks_url("pl1")]


def test_add_tracks_does_nothing_for_empty_track_list() -> None:
    http = FakeHttpClient({})
    client = SpotifyPlaylistsClient(http_client=http)

    client.add_tracks("pl1", [], "token")

    assert http.post_calls == []


def test_add_tracks_raises_spotify_api_error_on_failure() -> None:
    http = FakeHttpClient({}, posts={playlist_items_url("pl1"): (500, {})})
    client = SpotifyPlaylistsClient(http_client=http)

    try:
        client.add_tracks("pl1", ["t1"], "token")
        assert False, "expected SpotifyApiError"
    except SpotifyApiError as exc:
        assert exc.status_code == 500
