"""Port for judging a listener's mood and per-track lyrics tone/fit with Laya."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from mood_dj.domain.models import PlaylistTrack, Strategy
from mood_dj.domain.playlist_strategy import PlaylistSignals

# Called with the number of tracks judged in the batch that just completed, so
# callers can accumulate a running total across chunks.
JudgeProgressCallback = Callable[[int], None]


@dataclass(frozen=True)
class TrackLyrics:
    """A playlist track paired with its (already truncated) lyrics text."""

    track: PlaylistTrack
    text: str


@dataclass(frozen=True)
class TrackTone:
    """The prompt-independent emotional tone of a track's lyrics.

    0.0 is very sad, 1.0 is very happy. Tone depends only on the lyrics, so it is
    computed once per track and reused across every future prompt.
    """

    track_id: str
    tone: float


@dataclass(frozen=True)
class TrackFit:
    """The prompt-dependent probability that a track's lyrics fit the listener's ask."""

    track_id: str
    fit: float


class LyricsJudge(Protocol):
    """Judges mood signals, lyrics tone and lyrics fit using the multilingual Laya checkpoint."""

    def detect_signals(self, prompt: str) -> tuple[PlaylistSignals, dict[str, float]]:
        """Return the boolean mood signals plus their raw probabilities."""
        ...

    def judge_tone(
        self,
        tracks: list[TrackLyrics],
        on_progress: JudgeProgressCallback | None = None,
    ) -> list[TrackTone]:
        """Judge the prompt-independent tone of each track, in the same order as `tracks`.

        Implementations may process `tracks` in chunks and call `on_progress` with
        the chunk size after each chunk completes, so callers can report incremental
        progress on long-running batches.
        """
        ...

    def judge_fit(
        self,
        prompt: str,
        strategy: Strategy,
        tracks: list[TrackLyrics],
        on_progress: JudgeProgressCallback | None = None,
    ) -> list[TrackFit]:
        """Judge how well each track's lyrics fit the prompt, in the same order as `tracks`."""
        ...
