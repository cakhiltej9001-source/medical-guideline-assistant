"""Evaluate the calibrated top-1 cross-encoder confidence refusal gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.retrieval.config import RetrievalConfig  # noqa: E402
from medical_guideline_assistant.retrieval.index import search_index  # noqa: E402
from medical_guideline_assistant.retrieval.reranker import CrossEncoderReranker  # noqa: E402


def main() -> int:
    dataset = json.loads(
        (PROJECT_ROOT / "eval" / "confidence_cases.json").read_text(encoding="utf-8")
    )
    config = RetrievalConfig.from_path(PROJECT_ROOT / "configs" / "retrieval.json")
    reranker = CrossEncoderReranker(config.reranking)
    reports = []
    for case in dataset["cases"]:
        results = search_index(
            PROJECT_ROOT / config.database_path, case["query"], config,
            embedding_provider=None,
        )
        ranked = reranker.rerank(case["query"], results)
        score = ranked[0].rerank_score if ranked else None
        accepted = score is not None and score >= config.reranking.minimum_top_score
        reports.append({
            "case_id": case["case_id"], "top_confidence": score,
            "threshold": config.reranking.minimum_top_score,
            "expected_accept": case["expected_accept"], "actual_accept": accepted,
            "passed": accepted == case["expected_accept"],
        })
    accuracy = sum(report["passed"] for report in reports) / len(reports)
    output = {
        "cases": len(reports), "confidence_gate_accuracy": accuracy,
        "minimum_required_accuracy": dataset["minimum_accuracy"],
        "passed": accuracy >= float(dataset["minimum_accuracy"]), "case_reports": reports,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
