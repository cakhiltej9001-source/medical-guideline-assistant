"""Safe orchestration boundary between input checks and evidence retrieval."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .retrieval.config import RetrievalConfig
from .retrieval.augmentation import AugmentationError, RetrievalAssistant
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
    hyde_status: str = "disabled"
    correction_status: str = "disabled"
    initial_top_score: float | None = None
    removed_chunks: int = 0

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
            "hyde_status": self.hyde_status,
            "correction_status": self.correction_status,
            "initial_top_score": self.initial_top_score,
            "removed_chunks": self.removed_chunks,
        }


def retrieve_safely(
    query: str,
    database_path: Path,
    config: RetrievalConfig,
    embedding_provider: EmbeddingProvider | None = None,
    reranker: RerankingProvider | None = None,
    source_ids: tuple[str, ...] | None = None,
    retrieval_assistant: RetrievalAssistant | None = None,
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
    hypothetical_vector = None
    hyde_status = "disabled"
    if config.augmentation.hyde_enabled:
        hyde_status = "unavailable"
        if retrieval_assistant is not None and embedding_provider is not None:
            try:
                hypothesis = retrieval_assistant.hypothesize(
                    decision.normalized_query, config.augmentation.maximum_hypothesis_chars
                )
                if not isinstance(hypothesis, str) or not 1 <= len(hypothesis.strip()) <= config.augmentation.maximum_hypothesis_chars:
                    raise AugmentationError("Invalid hypothetical passage length.")
                vectors = embedding_provider.embed_documents([hypothesis], ["Hypothetical search passage"])
                if len(vectors) != 1 or len(vectors[0]) != config.embedding.dimensions:
                    raise AugmentationError("Invalid hypothetical embedding dimensions.")
                hypothetical_vector = vectors[0]
                hyde_status = "applied"
            except (AugmentationError, EmbeddingError):
                hyde_status = "failed_fallback"
    retrieval_mode = "hybrid" if embedding_provider is not None else "lexical"
    try:
        results = search_index(
            database_path=database_path,
            query=retrieval_query,
            config=config,
            embedding_provider=embedding_provider,
            source_ids=source_ids,
            hypothetical_vector=hypothetical_vector,
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
        hyde_status = "discarded_embedding_failure" if hypothetical_vector is not None else hyde_status
    top_confidence: float | None = None
    reranking_status = "disabled"
    correction_status = "not_needed" if config.augmentation.crag_enabled else "disabled"
    initial_top_score = None
    removed_chunks = 0
    if config.reranking.enabled:
        try:
            active_reranker = reranker or CrossEncoderReranker(config.reranking)
            results = active_reranker.rerank(retrieval_query, results)
            initial_top_score = results[0].rerank_score if results else None
            if config.augmentation.crag_enabled and (
                initial_top_score is None or initial_top_score < config.augmentation.correction_threshold
            ):
                # One correction only; keep the original intent for all relevance scoring.
                correction_query = retrieval_query
                correction_status = "broadened"
                if retrieval_assistant is not None:
                    try:
                        rewritten = retrieval_assistant.rewrite(decision.normalized_query)
                        if not isinstance(rewritten, str) or not 1 <= len(rewritten.strip()) <= 500:
                            raise AugmentationError("Invalid correction query.")
                        rewrite_decision = evaluate_input(rewritten)
                        if not rewrite_decision.allowed:
                            raise AugmentationError("Correction query failed safety checks.")
                        correction_query = rewrite_decision.retrieval_query or rewritten
                        correction_status = "rewritten"
                    except AugmentationError:
                        correction_status = "rewrite_failed_broadened"
                expanded = replace(config, search=replace(
                    config.search,
                    bm25_candidates=min(100, config.search.bm25_candidates * 2),
                    dense_candidates=min(100, config.search.dense_candidates * 2),
                    final_results=min(20, config.search.final_results * 2),
                ))
                try:
                    corrected = search_index(
                        database_path, correction_query, expanded,
                        embedding_provider if retrieval_mode != "lexical_fallback" else None,
                        source_ids=source_ids,
                        hypothetical_vector=hypothetical_vector if retrieval_mode != "lexical_fallback" else None,
                    )
                except EmbeddingError:
                    corrected = search_index(database_path, correction_query, expanded, source_ids=source_ids)
                    correction_status += "_lexical_fallback"
                # Score corrected candidates in bounded batches so the reranker's top-N
                # cap cannot silently discard candidates from the expanded pool.
                rescored = []
                batch_size = config.reranking.candidate_count
                for start in range(0, len(corrected), batch_size):
                    rescored.extend(active_reranker.rerank(retrieval_query, corrected[start:start + batch_size]))
                unique = {result.chunk_id: result for result in results}
                unique.update({result.chunk_id: result for result in rescored})
                results = sorted(unique.values(), key=lambda item: (-(item.rerank_score or 0), item.chunk_id))[:config.search.final_results]
            if any(result.rerank_score is None or not math.isfinite(result.rerank_score)
                   or not 0 <= result.rerank_score <= 1 for result in results):
                raise RerankingError("Invalid relevance score.")
            if config.augmentation.crag_enabled:
                before = len(results)
                results = [result for result in results if result.rerank_score >= config.reranking.minimum_top_score]
                removed_chunks = before - len(results)
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
                hyde_status=hyde_status,
                correction_status="unavailable",
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
                hyde_status=hyde_status,
                correction_status=correction_status,
                initial_top_score=initial_top_score,
                removed_chunks=removed_chunks,
            )
    return RetrievalOutcome(
        status="evidence_retrieved" if results else "insufficient_evidence",
        safety=decision,
        results=results,
        message=None if results else OUT_OF_SCOPE_REFUSAL,
        retrieval_mode=retrieval_mode,
        reranking_status=reranking_status,
        top_confidence=top_confidence,
        confidence_threshold=(
            config.reranking.minimum_top_score if config.reranking.enabled else None
        ),
        hyde_status=hyde_status,
        correction_status=correction_status,
        initial_top_score=initial_top_score,
        removed_chunks=removed_chunks,
    )
