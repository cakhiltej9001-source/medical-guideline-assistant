"""End-to-end safe retrieval and grounded answer orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .generation.config import GenerationConfig
from .generation.gemini import GenerationError, GroundedGenerator
from .generation.grounding import (
    GroundedAnswer,
    GroundingValidationError,
    validate_grounded_payload,
)
from .pipeline import RetrievalOutcome, retrieve_safely
from .retrieval.config import RetrievalConfig
from .retrieval.embeddings import EmbeddingProvider
from .retrieval.reranker import RerankingProvider
from .safety.guardrails import OUT_OF_SCOPE_REFUSAL, evaluate_input


@dataclass(frozen=True)
class AnswerOutcome:
    status: str
    message: str | None
    answer: GroundedAnswer | None
    retrieval: RetrievalOutcome
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "answer": asdict(self.answer) if self.answer else None,
            "retrieval": self.retrieval.to_dict(),
            "diagnostic": self.diagnostic,
        }


def preflight_query(query: str) -> AnswerOutcome | None:
    """Return an immediate refusal before configs, clients, retrieval, or API access."""
    safety = evaluate_input(query)
    if safety.allowed:
        return None
    retrieval = RetrievalOutcome(
        status="refused",
        safety=safety,
        results=[],
        message=safety.refusal_message,
        retrieval_mode="not_run",
    )
    return AnswerOutcome(
        status="refused",
        message=safety.refusal_message,
        answer=None,
        retrieval=retrieval,
    )


def answer_query(
    query: str,
    database_path: Path,
    retrieval_config: RetrievalConfig,
    generation_config: GenerationConfig,
    generator: GroundedGenerator,
    embedding_provider: EmbeddingProvider | None = None,
    reranker: RerankingProvider | None = None,
) -> AnswerOutcome:
    """Refuse early, retrieve evidence, generate claims, then validate citations."""
    preflight = preflight_query(query)
    if preflight is not None:
        return preflight
    retrieval = retrieve_safely(
        query=query,
        database_path=database_path,
        config=retrieval_config,
        embedding_provider=embedding_provider,
        reranker=reranker,
    )
    if retrieval.status != "evidence_retrieved":
        return AnswerOutcome(
            status=retrieval.status,
            message=retrieval.message,
            answer=None,
            retrieval=retrieval,
        )

    evidence = retrieval.results[: generation_config.maximum_context_chunks]
    try:
        validation_error: GroundingValidationError | None = None
        for validation_attempt in range(generation_config.maximum_validation_attempts):
            payload = generator.generate(
                retrieval.safety.normalized_query,
                evidence,
            )
            try:
                answer = validate_grounded_payload(
                    payload=payload,
                    results=evidence,
                    disclaimer=generation_config.standard_disclaimer,
                    minimum_claim_token_overlap=(
                        generation_config.minimum_claim_token_overlap
                    ),
                    support_scorer=(
                        reranker
                        if reranker is not None and hasattr(reranker, "score_pair")
                        else None
                    ),
                    minimum_claim_support_score=(
                        generation_config.minimum_claim_support_score
                    ),
                )
                break
            except GroundingValidationError as exc:
                validation_error = exc
                if validation_attempt + 1 == generation_config.maximum_validation_attempts:
                    raise
        else:
            raise validation_error or GroundingValidationError(
                "No generated response passed validation."
            )
    except (GenerationError, GroundingValidationError) as exc:
        return AnswerOutcome(
            status="output_blocked",
            message=OUT_OF_SCOPE_REFUSAL,
            answer=None,
            retrieval=retrieval,
            diagnostic=f"{type(exc).__name__}: {exc}",
        )

    if answer.status == "insufficient_evidence":
        return AnswerOutcome(
            status="insufficient_evidence",
            message=OUT_OF_SCOPE_REFUSAL,
            answer=answer,
            retrieval=retrieval,
        )
    return AnswerOutcome(
        status="answered",
        message=None,
        answer=answer,
        retrieval=retrieval,
    )
