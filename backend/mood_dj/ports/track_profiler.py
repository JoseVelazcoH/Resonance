"""Port for computing per-track mood profiles from lyrics with Laya."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from mood_dj.domain.models import TrackMoodProfile

# Called with the number of tracks profiled in the batch that just completed, so
# callers can accumulate a running total across chunks and persist incrementally.
ProfileProgressCallback = Callable[[list[TrackMoodProfile]], None]


@dataclass(frozen=True)
class TrackForProfiling:
    """A track's identity plus its (already truncated) lyrics text."""

    track_id: str
    artist: str
    title: str
    lyrics: str


class TrackProfiler(Protocol):
    """Computes a `TrackMoodProfile` for every given track from its lyrics."""

    version: str

    def profile(
        self,
        tracks: list[TrackForProfiling],
        on_progress: ProfileProgressCallback | None = None,
    ) -> list[TrackMoodProfile]:
        """Return one `TrackMoodProfile` per track, in the same order as `tracks`.

        Implementations should call `on_progress` with each completed batch of
        profiles (not just a count) so callers can persist them immediately and
        resume cleanly if interrupted.
        """
        ...
