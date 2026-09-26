"""LyricsProvider adapter backed by the LRCLIB API (https://lrclib.net).

Tries an exact match first via `GET /api/get` (artist, track, album, duration).
On a 404, falls back to `GET /api/search` with a cleaned track title, since LRCLIB's
exact match is strict about title suffixes like "- Remastered 2011" that Spotify
commonly appends but the underlying recording's lyrics entry does not carry.

Transient failures (HTTP 429/500/502/503/504, timeouts, connection errors) are
retried with exponential backoff and jitter, honoring `Retry-After` when the server
sends one. A 404 is a definitive "no exact match" and goes straight to the search
fallback without retrying. Network errors and exhausted retries are reported as
`LyricsLookupResult(status=None)` so the caller does not cache them: the track is
retried on a later preparation run. Each final failure logs one concise warning line
(no traceback spam); individual retry attempts log at debug level.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Callable, Protocol, TypeVar

import httpx

from mood_dj.domain.models import LyricsStatus
from mood_dj.ports.lyrics_provider import LyricsLookupResult

logger = logging.getLogger(__name__)

GET_URL = "https://lrclib.net/api/get"
SEARCH_URL = "https://lrclib.net/api/search"
USER_AGENT = "laya-spotify/0.1.0 (https://github.com/laya-spotify; playlist lyrics lookup)"
REQUEST_TIMEOUT = 10.0

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY_S = 1.0
DEFAULT_MAX_DELAY_S = 30.0

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

_TITLE_SUFFIX_PATTERNS = [
    r"\s*\(feat\.[^)]*\)",
    r"\s*\[[^\]]*\]",
    r"\s*-\s*Remaster(?:ed|izado)?(?:\s+\d{4})?",
    r"\s*-\s*Live(?:\s+.*)?",
    r"\s*-\s*Acoustic(?:\s+Version)?",
]
_TITLE_SUFFIX_RE = re.compile("|".join(_TITLE_SUFFIX_PATTERNS), re.IGNORECASE)

T = TypeVar("T")


def clean_title(title: str) -> str:
    """Strip common Spotify title suffixes that break LRCLIB's exact match."""
    cleaned = _TITLE_SUFFIX_RE.sub("", title)
    return cleaned.strip()


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a `Retry-After` header value (seconds form only) into a float."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(0.0, seconds)


class RetryExhausted(Exception):
    """Raised internally when a transient failure survives all retry attempts."""


class LrclibHttpClient(Protocol):
    """Thin boundary around the two LRCLIB HTTP calls this adapter needs."""

    def get(self, params: dict) -> dict:
        """Call `/api/get`. Raises `httpx.HTTPStatusError` on 404."""
        ...

    def search(self, params: dict):
        """Call `/api/search`. Returns a list of matches, or an error object."""
        ...


class HttpxLrclibHttpClient:
    """Real LRCLIB HTTP client, built on httpx."""

    def get(self, params: dict) -> dict:
        response = httpx.get(
            GET_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()

    def search(self, params: dict):
        response = httpx.get(
            SEARCH_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()


class LrclibLyricsProvider:
    """Looks up lyrics on LRCLIB, with a cleaned-title search fallback.

    Transient errors are retried in-place (blocking the calling thread) with
    exponential backoff and jitter. `sleep_fn`/`random_fn` are injectable so tests
    never actually sleep or depend on real randomness.
    """

    def __init__(
        self,
        http_client: LrclibHttpClient | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        base_delay_s: float = DEFAULT_BASE_DELAY_S,
        max_delay_s: float = DEFAULT_MAX_DELAY_S,
        sleep_fn: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self._http_client = http_client or HttpxLrclibHttpClient()
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._sleep_fn = sleep_fn
        self._random_fn = random_fn

    def fetch(self, artist: str, title: str, album: str, duration_s: float) -> LyricsLookupResult:
        try:
            payload = self._call_with_retry(
                lambda: self._http_client.get(
                    {
                        "artist_name": artist,
                        "track_name": title,
                        "album_name": album,
                        "duration": int(duration_s),
                    }
                ),
                "GET /api/get",
                artist,
                title,
            )
        except RetryExhausted:
            return LyricsLookupResult(status=None)
        except httpx.HTTPStatusError as error:
            if error.response.status_code != httpx.codes.NOT_FOUND:
                raise
            return self._search_fallback(artist, title)

        return self._result_from_payload(payload)

    def _search_fallback(self, artist: str, title: str) -> LyricsLookupResult:
        try:
            matches = self._call_with_retry(
                lambda: self._http_client.search({"track_name": clean_title(title), "artist_name": artist}),
                "GET /api/search",
                artist,
                title,
            )
        except RetryExhausted:
            return LyricsLookupResult(status=None)

        if not isinstance(matches, list) or not matches:
            return LyricsLookupResult(status=LyricsStatus.MISSING)

        return self._result_from_payload(matches[0])

    def _call_with_retry(self, call: Callable[[], T], op_name: str, artist: str, title: str) -> T:
        """Run `call`, retrying transient failures with backoff+jitter.

        A 404 `HTTPStatusError` is re-raised immediately (it is definitive, not
        transient). Any other transient failure that survives `max_attempts`
        raises `RetryExhausted` after logging one warning line.
        """
        attempt = 0
        while True:
            try:
                return call()
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if status == httpx.codes.NOT_FOUND:
                    raise
                if status not in _TRANSIENT_STATUS_CODES or attempt >= self._max_attempts - 1:
                    logger.warning(
                        "LRCLIB %s failed for %r - %r: HTTP %d (giving up after %d attempt(s))",
                        op_name, artist, title, status, attempt + 1,
                    )
                    raise RetryExhausted from error
                retry_after = _parse_retry_after(error.response.headers.get("Retry-After"))
                delay = self._retry_delay(attempt, retry_after)
                logger.debug(
                    "LRCLIB %s transient HTTP %d for %r - %r, retrying in %.1fs (attempt %d/%d)",
                    op_name, status, artist, title, delay, attempt + 1, self._max_attempts,
                )
                self._sleep_fn(delay)
                attempt += 1
            except httpx.HTTPError as error:
                if attempt >= self._max_attempts - 1:
                    logger.warning(
                        "LRCLIB %s network error for %r - %r: %s (giving up after %d attempt(s))",
                        op_name, artist, title, error, attempt + 1,
                    )
                    raise RetryExhausted from error
                delay = self._retry_delay(attempt)
                logger.debug(
                    "LRCLIB %s network error for %r - %r, retrying in %.1fs (attempt %d/%d)",
                    op_name, artist, title, delay, attempt + 1, self._max_attempts,
                )
                self._sleep_fn(delay)
                attempt += 1

    def _retry_delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(retry_after, self._max_delay_s)
        base = min(self._base_delay_s * (2**attempt), self._max_delay_s)
        jittered = base * (0.5 + self._random_fn() * 0.5)
        return min(jittered, self._max_delay_s)

    def _result_from_payload(self, payload: dict) -> LyricsLookupResult:
        if payload.get("instrumental"):
            return LyricsLookupResult(status=LyricsStatus.INSTRUMENTAL, text=None)
        text = payload.get("plainLyrics")
        if not text:
            return LyricsLookupResult(status=LyricsStatus.MISSING)
        return LyricsLookupResult(status=LyricsStatus.LYRICS, text=text)
