"""Measure page-level retrieval recall before adding answer generation."""

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

        case_reports = []
        reciprocal_rank_total = 0.0
        hits = 0
        for case in dataset["cases"]:
            results = search_index(
                PROJECT_ROOT / config.database_path,
                case["query"],
                config,
                embedding_provider=provider,
            )
            relevant_pages = set(case["relevant_pages"])
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
            case_reports.append(
                {
                    "case_id": case["case_id"],
                    "hit": rank is not None,
                    "rank": rank,
                    "top_chunk_id": results[0].chunk_id if results else None,
                }
            )

        case_count = len(case_reports)
        recall = hits / case_count if case_count else 0.0
        report = {
            "mode": "hybrid" if provider else "lexical_only",
            "cases": case_count,
            f"recall_at_{k}": recall,
            f"mrr_at_{k}": reciprocal_rank_total / case_count if case_count else 0.0,
            "minimum_required_recall": dataset["minimum_recall_at_k"],
            "passed": recall >= float(dataset["minimum_recall_at_k"]),
            "case_reports": case_reports,
        }
    except (
        EmbeddingError,
        RetrievalConfigError,
        RetrievalIndexError,
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
