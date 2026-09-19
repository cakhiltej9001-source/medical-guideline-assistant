"""Bounded retrieval assistance; synthetic text is never citation evidence."""

from typing import Protocol


class AugmentationError(RuntimeError):
    """An optional retrieval aid was unavailable or invalid."""


class RetrievalAssistant(Protocol):
    def hypothesize(self, query: str, maximum_chars: int) -> str: ...

    def rewrite(self, query: str) -> str: ...
