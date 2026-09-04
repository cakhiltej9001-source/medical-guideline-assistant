"""Typed configuration for indexing and hybrid retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class RetrievalConfigError(ValueError):
    """Raised when retrieval configuration is missing or unsafe."""


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    dimensions: int
    batch_size: int
    timeout_seconds: int
    maximum_attempts: int
    minimum_batch_interval_seconds: float = 0.0


@dataclass(frozen=True)
class SearchConfig:
    bm25_candidates: int
    dense_candidates: int
    final_results: int
    rrf_constant: int


@dataclass(frozen=True)
class RetrievalConfig:
    retrieval_version: int
    database_path: str
    embedding: EmbeddingConfig
    search: SearchConfig

    @classmethod
    def from_path(cls, path: Path) -> "RetrievalConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            embedding = EmbeddingConfig(**raw["embedding"])
            search = SearchConfig(**raw["search"])
            config = cls(
                retrieval_version=int(raw["retrieval_version"]),
                database_path=str(raw["database_path"]),
                embedding=embedding,
                search=search,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RetrievalConfigError(f"Could not load retrieval config: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if self.embedding.provider != "gemini":
            raise RetrievalConfigError("Only the gemini embedding provider is configured.")
        if not self.embedding.model.startswith("gemini-embedding-"):
            raise RetrievalConfigError("Embedding model must be a Gemini embedding model.")
        if not 128 <= self.embedding.dimensions <= 3072:
            raise RetrievalConfigError("Embedding dimensions must be between 128 and 3072.")
        if not 1 <= self.embedding.batch_size <= 100:
            raise RetrievalConfigError("Embedding batch size must be between 1 and 100.")
        if not 1 <= self.embedding.maximum_attempts <= 5:
            raise RetrievalConfigError("Maximum attempts must be between 1 and 5.")
        if self.embedding.timeout_seconds <= 0:
            raise RetrievalConfigError("Embedding timeout must be positive.")
        if not 0 <= self.embedding.minimum_batch_interval_seconds <= 60:
            raise RetrievalConfigError(
                "Minimum embedding batch interval must be between 0 and 60 seconds."
            )
        values = (
            self.search.bm25_candidates,
            self.search.dense_candidates,
            self.search.final_results,
            self.search.rrf_constant,
        )
        if any(value <= 0 for value in values):
            raise RetrievalConfigError("All search limits must be positive.")
        if self.search.final_results > (
            self.search.bm25_candidates + self.search.dense_candidates
        ):
            raise RetrievalConfigError("Final result count exceeds the candidate pool.")
