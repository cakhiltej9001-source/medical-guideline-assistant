"""Cross-encoder reranking and confidence scoring for retrieved evidence."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Protocol

from .config import RerankingConfig
from .index import SearchResult


class RerankingError(RuntimeError):
    """Raised when relevance scores cannot be produced safely."""


class RerankingProvider(Protocol):
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]: ...

    def score_pair(self, left: str, right: str) -> float: ...


class CrossEncoderReranker:
    """Jointly score query/passage pairs with a local FastEmbed ONNX model."""

    def __init__(self, config: RerankingConfig, model: Any | None = None) -> None:
        self.config = config
        self._model = model

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(model_name=self.config.model)
            except Exception as exc:
                raise RerankingError(
                    f"Could not load approved cross-encoder: {type(exc).__name__}"
                ) from exc
        return self._model

    @staticmethod
    def _probability(raw_score: float) -> float:
        if not math.isfinite(raw_score):
            raise RerankingError("Cross-encoder returned a non-finite score.")
        if raw_score >= 0:
            return 1.0 / (1.0 + math.exp(-raw_score))
        exponential = math.exp(raw_score)
        return exponential / (1.0 + exponential)

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        try:
            model = self._load_model()
            raw_scores = [next(model.rerank(left, [right])) for left, right in pairs]
            scores = [self._probability(float(score)) for score in raw_scores]
        except RerankingError:
            raise
        except Exception as exc:
            raise RerankingError(
                f"Cross-encoder scoring failed: {type(exc).__name__}"
            ) from exc
        if len(scores) != len(pairs):
            raise RerankingError("Cross-encoder returned an unexpected score count.")
        return scores

    def score_pair(self, left: str, right: str) -> float:
        """Return a normalized semantic support/relevance score for one text pair."""
        return self._score_pairs([(left, right)])[0]

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        candidates = results[: self.config.candidate_count]
        if not candidates:
            return []
        try:
            raw_scores = list(
                self._load_model().rerank(query, [result.text for result in candidates])
            )
            scores = [self._probability(float(score)) for score in raw_scores]
        except RerankingError:
            raise
        except Exception as exc:
            raise RerankingError(
                f"Cross-encoder scoring failed: {type(exc).__name__}"
            ) from exc
        if len(scores) != len(candidates):
            raise RerankingError("Cross-encoder returned an unexpected score count.")
        rescored = [
            replace(result, rerank_score=score)
            for result, score in zip(candidates, scores)
        ]
        return sorted(
            rescored,
            key=lambda result: (
                -(result.rerank_score or 0.0),
                -result.rrf_score,
                result.chunk_id,
            ),
        )
