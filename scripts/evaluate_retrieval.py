"""Measure hybrid retrieval, cross-encoder reranking, and page-level recall."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.retrieval.config import (  # noqa: E402
    RetrievalConfig,
    RetrievalConfigError,
)
from medical_guideline_assistant.retrieval.embeddings import (  # noqa: E402
    EmbeddingError,
    GeminiEmbeddingProvider,
)
from medical_guideline_assistant.retrieval.index import (  # noqa: E402
    RetrievalIndexError,
    search_index,
)
from medical_guideline_assistant.retrieval.reranker import (  # noqa: E402
    CrossEncoderReranker,
    RerankingError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--k", type=int)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    provider = None
    try:
        dataset = json.loads(
            (PROJECT_ROOT / "eval" / "retrieval_cases.json").read_text(encoding="utf-8")
        )
        config = RetrievalConfig.from_path(PROJECT_ROOT / "configs" / "retrieval.json")
        k = args.k or int(dataset["default_k"])
        if not 1 <= k <= config.search.final_results:
            raise RetrievalIndexError(
                f"Evaluation k must be between 1 and {config.search.final_results}."
            )
        if args.hybrid:
            provider = GeminiEmbeddingProvider(config.embedding)

        reranker = CrossEncoderReranker(config.reranking) if config.reranking.enabled else None
        case_reports = []
        reciprocal_rank_total = 0.0
        hits = 0
        page_recall_total = 0.0
        confident_cases = 0
        for case in dataset["cases"]:
            results = search_index(
                PROJECT_ROOT / config.database_path,
                case["query"],
                config,
                embedding_provider=provider,
            )
            if reranker is not None:
                results = reranker.rerank(case["query"], results)
            relevant_pages = set(case["relevant_pages"])
            retrieved_pages = {
                page
                for result in results[:k]
                if result.source_id == case["expected_source_id"]
                for page in result.pages
            }
            page_recall = (
                len(relevant_pages.intersection(retrieved_pages)) / len(relevant_pages)
                if relevant_pages
                else 0.0
            )
            page_recall_total += page_recall
            rank = next(
                (
                    position
                    for position, result in enumerate(results[:k], start=1)
                    if result.source_id == case["expected_source_id"]
                    and relevant_pages.intersection(result.pages)
                ),
                None,
            )
            if rank is not None:
                hits += 1
                reciprocal_rank_total += 1 / rank
            top_confidence = results[0].rerank_score if results else None
            confidence_passed = (
                top_confidence is None
                or top_confidence >= config.reranking.minimum_top_score
            )
            confident_cases += int(confidence_passed)
            case_reports.append(
                {
                    "case_id": case["case_id"],
                    "hit": rank is not None,
                    "rank": rank,
                    "page_recall": page_recall,
                    "top_confidence": top_confidence,
                    "confidence_passed": confidence_passed,
                    "top_chunk_id": results[0].chunk_id if results else None,
                }
            )

        case_count = len(case_reports)
        hit_rate = hits / case_count if case_count else 0.0
        page_coverage = page_recall_total / case_count if case_count else 0.0
        report = {
            "mode": "hybrid_reranked" if provider else "lexical_reranked",
            "cases": case_count,
            f"hit_rate_at_{k}": hit_rate,
            f"answerable_recall_at_{k}": hit_rate,
            f"gold_page_coverage_at_{k}": page_coverage,
            f"mrr_at_{k}": reciprocal_rank_total / case_count if case_count else 0.0,
            "confidence_acceptance_rate": confident_cases / case_count if case_count else 0.0,
            "minimum_required_answerable_recall": dataset[
                "minimum_answerable_recall_at_k"
            ],
            "passed": (
                hit_rate >= float(dataset["minimum_answerable_recall_at_k"])
                and confident_cases == case_count
            ),
            "case_reports": case_reports,
        }
    except (
        EmbeddingError,
        RetrievalConfigError,
        RetrievalIndexError,
        RerankingError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Retrieval evaluation failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if provider is not None:
            provider.close()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
