"""Unit tests for the LRCLIB lyrics adapter, using a fake HTTP client."""

from __future__ import annotations

import httpx
import pytest

from mood_dj.adapters.lrclib_lyrics import LrclibLyricsProvider, clean_title
from mood_dj.domain.models import LyricsStatus


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Sesame Syrup - Remasterizado 2008", "Sesame Syrup"),
        ("Nudes - Remastered 2011", "Nudes"),
        ("Apocalypse - Live", "Apocalypse"),
        ("K. - Acoustic Version", "K."),
        ("Sunburnt (feat. Jane Doe)", "Sunburnt"),
        ("Track Name [Explicit]", "Track Name"),
        ("Clean Title", "Clean Title"),
    ],
)
def test_clean_title_strips_known_suffixes(raw: str, expected: str) -> None:
    assert clean_title(raw) == expected


class FakeHttpClient:
    """Fakes the two LRCLIB endpoints: `get` (exact match) and `search` (fallback)."""

    def __init__(self, get_response=None, get_error=None, search_response=None, search_error=None) -> None:
        self.get_response = get_response
        self.get_error = get_error
        self.search_response = search_response
        self.search_error = search_error
        self.get_calls: list[dict] = []
        self.search_calls: list[dict] = []

    def get(self, params: dict):
        self.get_calls.append(params)
        if self.get_error is not None:
            raise self.get_error
        return self.get_response

    def search(self, params: dict):
        self.search_calls.append(params)
        if self.search_error is not None:
            raise self.search_error
        return self.search_response


def _not_found() -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://lrclib.net/api/get")
    response = httpx.Response(404, request=request)
    return httpx.HTTPStatusError("not found", request=request, response=response)


def test_fetch_returns_lyrics_on_exact_match() -> None:
    http = FakeHttpClient(get_response={"plainLyrics": "la la la", "instrumental": False})
    provider = LrclibLyricsProvider(http_client=http)

    result = provider.fetch("Artist", "Title", "Album", 200.0)

    assert result.status is LyricsStatus.LYRICS
    assert result.text == "la la la"
    assert http.get_calls[0]["artist_name"] == "Artist"
    assert http.get_calls[0]["track_name"] == "Title"
    assert http.get_calls[0]["album_name"] == "Album"
    assert http.get_calls[0]["duration"] == 200


def test_fetch_returns_instrumental_on_exact_match() -> None:
    http = FakeHttpClient(get_response={"plainLyrics": None, "instrumental": True})
    provider = LrclibLyricsProvider(http_client=http)

    result = provider.fetch("Artist", "Title", "Album", 200.0)

    assert result.status is LyricsStatus.INSTRUMENTAL
    assert result.text is None


def test_fetch_falls_back_to_search_on_404() -> None:
    http = FakeHttpClient(
        get_error=_not_found(),
        search_response=[{"plainLyrics": "found via search", "instrumental": False}],
    )
    provider = LrclibLyricsProvider(http_client=http)

    result = provider.fetch("Artist", "Title - Remastered 2011", "Album", 200.0)

    assert result.status is LyricsStatus.LYRICS
    assert result.text == "found via search"
    assert http.search_calls[0]["track_name"] == "Title"
    assert http.search_calls[0]["artist_name"] == "Artist"


def test_fetch_handles_search_returning_a_non_list_error_object() -> None:
    http = FakeHttpClient(get_error=_not_found(), search_response={"code": 500, "message": "error"})
    provider = LrclibLyricsProvider(http_client=http)

    result = provider.fetch("Artist", "Title", "Album", 200.0)

    assert result.status is LyricsStatus.MISSING


def test_fetch_returns_missing_when_search_also_empty() -> None:
    http = FakeHttpClient(get_error=_not_found(), search_response=[])
    provider = LrclibLyricsProvider(http_client=http)

    result = provider.fetch("Artist", "Title", "Album", 200.0)

    assert result.status is LyricsStatus.MISSING


def test_fetch_returns_none_status_on_network_error() -> None:
    http = FakeHttpClient(get_error=httpx.ConnectError("boom"))
    provider = LrclibLyricsProvider(http_client=http)

    result = provider.fetch("Artist", "Title", "Album", 200.0)

    assert result.status is None


def test_fetch_returns_none_status_when_search_also_network_errors() -> None:
    http = FakeHttpClient(get_error=_not_found(), search_error=httpx.ConnectError("boom"))
    provider = LrclibLyricsProvider(http_client=http)

    result = provider.fetch("Artist", "Title", "Album", 200.0)

    assert result.status is None


def _status_error(status: int, retry_after: str | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://lrclib.net/api/get")
    headers = {"Retry-After": retry_after} if retry_after else {}
    response = httpx.Response(status, request=request, headers=headers)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


class SequencedHttpClient:
    """Fakes `get`, returning/raising each item in `get_sequence` in order."""

    def __init__(self, get_sequence: list) -> None:
        self.get_sequence = list(get_sequence)
        self.get_calls = 0

    def get(self, params: dict):
        item = self.get_sequence[self.get_calls]
        self.get_calls += 1
        if isinstance(item, Exception):
            raise item
        return item

    def search(self, params: dict):
        raise AssertionError("search should not be called in these tests")


class FakeSleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def test_fetch_retries_transient_503_then_succeeds() -> None:
    http = SequencedHttpClient(
        [_status_error(503), _status_error(503), {"plainLyrics": "la la la", "instrumental": False}]
    )
    sleeper = FakeSleeper()
    provider = LrclibLyricsProvider(http_client=http, sleep_fn=sleeper, random_fn=lambda: 0.0)

    result = provider.fetch("Artist", "Title", "Album", 200.0)

    assert result.status is LyricsStatus.LYRICS
    assert http.get_calls == 3
    assert len(sleeper.calls) == 2


def test_fetch_honors_retry_after_header() -> None:
    http = SequencedHttpClient([_status_error(429, retry_after="7"), {"plainLyrics": "x", "instrumental": False}])
    sleeper = FakeSleeper()
    provider = LrclibLyricsProvider(http_client=http, sleep_fn=sleeper, random_fn=lambda: 0.0)

    provider.fetch("Artist", "Title", "Album", 200.0)

    assert sleeper.calls == [7.0]


def test_fetch_caps_backoff_delay_at_max_delay() -> None:
    http = SequencedHttpClient([_status_error(503)] * 4 + [{"plainLyrics": "x", "instrumental": False}])
    sleeper = FakeSleeper()
    provider = LrclibLyricsProvider(
        http_client=http, max_attempts=5, base_delay_s=1.0, max_delay_s=30.0, sleep_fn=sleeper, random_fn=lambda: 1.0
    )

    provider.fetch("Artist", "Title", "Album", 200.0)

    assert all(delay <= 30.0 for delay in sleeper.calls)


def test_fetch_gives_up_after_max_attempts_and_logs_one_warning(caplog) -> None:
    http = SequencedHttpClient([_status_error(503)] * 10)
    sleeper = FakeSleeper()
    provider = LrclibLyricsProvider(http_client=http, max_attempts=3, sleep_fn=sleeper, random_fn=lambda: 0.0)

    with caplog.at_level("WARNING"):
        result = provider.fetch("Artist", "Title", "Album", 200.0)

    assert result.status is None
    assert http.get_calls == 3
    assert len(sleeper.calls) == 2
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "giving up after 3 attempt(s)" in warnings[0].getMessage()


def test_fetch_does_not_retry_on_404_goes_straight_to_search_fallback() -> None:
    http = FakeHttpClient(get_error=_not_found(), search_response=[{"plainLyrics": "found", "instrumental": False}])
    sleeper = FakeSleeper()
    provider = LrclibLyricsProvider(http_client=http, sleep_fn=sleeper)

    result = provider.fetch("Artist", "Title", "Album", 200.0)

    assert result.status is LyricsStatus.LYRICS
    assert sleeper.calls == []


def test_fetch_retries_network_errors_like_timeouts() -> None:
    http = SequencedHttpClient(
        [httpx.TimeoutException("slow"), {"plainLyrics": "la", "instrumental": False}]
    )
    sleeper = FakeSleeper()
    provider = LrclibLyricsProvider(http_client=http, sleep_fn=sleeper, random_fn=lambda: 0.0)

    result = provider.fetch("Artist", "Title", "Album", 200.0)

    assert result.status is LyricsStatus.LYRICS
    assert len(sleeper.calls) == 1
