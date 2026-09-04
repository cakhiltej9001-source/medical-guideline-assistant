"""Evaluate deterministic pre-retrieval safety classification."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.safety.guardrails import evaluate_input  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        dataset = json.loads(
            (PROJECT_ROOT / "eval" / "safety_cases.json").read_text(encoding="utf-8")
        )
        reports = []
        correct = 0
        for case in dataset["cases"]:
            decision = evaluate_input(case["query"])
            passed = decision.category == case["expected_category"]
            correct += int(passed)
            reports.append(
                {
                    "case_id": case["case_id"],
                    "expected": case["expected_category"],
                    "actual": decision.category,
                    "passed": passed,
                    "retrieval_blocked": decision.retrieval_query is None,
                }
            )
        accuracy = correct / len(reports) if reports else 0.0
        output = {
            "cases": len(reports),
            "accuracy": accuracy,
            "minimum_required_accuracy": dataset["minimum_accuracy"],
            "passed": accuracy >= float(dataset["minimum_accuracy"]),
            "case_reports": reports,
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Safety evaluation failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
