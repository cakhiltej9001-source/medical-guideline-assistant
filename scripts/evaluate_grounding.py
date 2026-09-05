"""Evaluate citation integrity, support, and generated-output validation offline."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.generation.grounding import (  # noqa: E402
    GroundingValidationError,
    validate_grounded_payload,
)
from medical_guideline_assistant.retrieval.index import SearchResult  # noqa: E402


class FixedSupportScorer:
    def __init__(self, score: float) -> None:
        self.score = score

    def score_pair(self, left: str, right: str) -> float:
        return self.score


def main() -> int:
    dataset = json.loads(
        (PROJECT_ROOT / "eval" / "grounding_cases.json").read_text(encoding="utf-8")
    )
    reports = []
    for case in dataset["cases"]:
        evidence = SearchResult(
            chunk_id="official:1", source_id="official", title="Official Guideline",
            source_url="https://example.gov/guideline.pdf", pages=[1], sections=["Test"],
            safety_tags=[], text=case["evidence"], rrf_score=0.1, lexical_rank=1,
            dense_rank=1, lexical_score=1.0, dense_score=0.8,
        )
        try:
            validate_grounded_payload(
                {"status": "answered", "claims": [{"text": case["claim"], "chunk_ids": case["chunk_ids"]}]},
                [evidence], "Informational only.", 0.2,
                support_scorer=FixedSupportScorer(case["support_score"]),
                minimum_claim_support_score=0.2,
            )
            actual_valid = True
        except GroundingValidationError:
            actual_valid = False
        reports.append({
            "case_id": case["case_id"], "expected_valid": case["expected_valid"],
            "actual_valid": actual_valid, "passed": actual_valid == case["expected_valid"],
        })
    accuracy = sum(report["passed"] for report in reports) / len(reports)
    output = {
        "cases": len(reports), "grounding_validation_accuracy": accuracy,
        "minimum_required_accuracy": dataset["minimum_accuracy"],
        "passed": accuracy >= float(dataset["minimum_accuracy"]), "case_reports": reports,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
