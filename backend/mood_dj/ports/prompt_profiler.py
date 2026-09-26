"""Port for analyzing a listener's prompt (no track lyrics) into a `PromptProfile`."""

from __future__ import annotations

from typing import Protocol

from mood_dj.domain.prompt_profile import PromptProfile


class PromptProfiler(Protocol):
    """Computes a `PromptProfile` from the listener's raw prompt text."""

    def profile(self, prompt: str) -> PromptProfile:
        """Always returns a profile: never raises on an unclear or short prompt."""
        ...
