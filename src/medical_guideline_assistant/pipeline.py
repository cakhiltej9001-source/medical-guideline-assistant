"""Safe orchestration boundary between input checks and evidence retrieval."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .retrieval.config import RetrievalConfig
from .retrieval.embeddings import EmbeddingError, EmbeddingProvider
from .retrieval.index import SearchResult, search_index
from .retrieval.reranker import CrossEncoderReranker, RerankingError, RerankingProvider
from .safety.guardrails import OUT_OF_SCOPE_REFUSAL, SafetyDecision, evaluate_input


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalOutcome:
    status: str
    safety: SafetyDecision
    results: list[SearchResult]
    message: str | None
    retrieval_mode: str
    reranking_status: str = "not_run"
    top_confidence: float | None = None
    confidence_threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "safety": self.safety.to_dict(),
            "results": [asdict(result) for result in self.results],
            "message": self.message,
            "retrieval_mode": self.retrieval_mode,
            "reranking_status": self.reranking_status,
            "top_confidence": self.top_confidence,
            "confidence_threshold": self.confidence_threshold,
        }


def retrieve_safely(
    query: str,
    database_path: Path,
    config: RetrievalConfig,
    embedding_provider: EmbeddingProvider | None = None,
    reranker: RerankingProvider | None = None,
    source_ids: tuple[str, ...] | None = None,
) -> RetrievalOutcome:
    """Block unsafe inputs, then retrieve; never generate an answer here."""
    decision = evaluate_input(query)
    if not decision.allowed:
        return RetrievalOutcome(
            status="refused",
            safety=decision,
            results=[],
            message=decision.refusal_message,
            retrieval_mode="not_run",
        )

    retrieval_query = decision.retrieval_query or decision.normalized_query
    retrieval_mode = "hybrid" if embedding_provider is not None else "lexical"
    try:
        results = search_index(
            database_path=database_path,
            query=retrieval_query,
            config=config,
            embedding_provider=embedding_provider,
            source_ids=source_ids,
        )
    except EmbeddingError as exc:
        LOGGER.warning(
            "Dense retrieval failed; using lexical fallback; error_type=%s",
            type(exc).__name__,
        )
        results = search_index(
            database_path=database_path,
            query=retrieval_query,
            config=config,
            embedding_provider=None,
            source_ids=source_ids,
        )
        retrieval_mode = "lexical_fallback"
    if not results:
        return RetrievalOutcome(
            status="insufficient_evidence",
            safety=decision,
            results=[],
            message=OUT_OF_SCOPE_REFUSAL,
            retrieval_mode=retrieval_mode,
        )
    top_confidence: float | None = None
    reranking_status = "disabled"
    if config.reranking.enabled:
        try:
            active_reranker = reranker or CrossEncoderReranker(config.reranking)
            results = active_reranker.rerank(retrieval_query, results)
        except RerankingError as exc:
            LOGGER.warning("Reranking failed closed; error_type=%s", type(exc).__name__)
            return RetrievalOutcome(
                status="insufficient_evidence",
                safety=decision,
                results=[],
                message=OUT_OF_SCOPE_REFUSAL,
                retrieval_mode=retrieval_mode,
                reranking_status="unavailable",
                confidence_threshold=config.reranking.minimum_top_score,
            )
        reranking_status = "applied"
        top_confidence = results[0].rerank_score if results else None
        if top_confidence is None or top_confidence < config.reranking.minimum_top_score:
            return RetrievalOutcome(
                status="insufficient_evidence",
                safety=decision,
                results=[],
                message=OUT_OF_SCOPE_REFUSAL,
                retrieval_mode=retrieval_mode,
                reranking_status=reranking_status,
                top_confidence=top_confidence,
                confidence_threshold=config.reranking.minimum_top_score,
            )
    return RetrievalOutcome(
        status="evidence_retrieved",
        safety=decision,
        results=results,
        message=None,
        retrieval_mode=retrieval_mode,
        reranking_status=reranking_status,
        top_confidence=top_confidence,
        confidence_threshold=(
            config.reranking.minimum_top_score if config.reranking.enabled else None
        ),
    )
