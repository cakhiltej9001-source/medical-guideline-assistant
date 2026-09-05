"""Evaluate the deterministic post-generation safety classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.generation.grounding import classify_output  # noqa: E402


def main() -> int:
    try:
        dataset = json.loads(
            (PROJECT_ROOT / "eval" / "output_safety_cases.json").read_text(encoding="utf-8")
        )
        reports = []
        for case in dataset["cases"]:
            decision = classify_output(case["text"])
            passed = decision.category == case["expected_category"]
            reports.append({
                "case_id": case["case_id"],
                "expected": case["expected_category"],
                "actual": decision.category,
                "passed": passed,
            })
        accuracy = sum(report["passed"] for report in reports) / len(reports)
        output = {
            "cases": len(reports),
            "output_safety_accuracy": accuracy,
            "minimum_required_accuracy": dataset["minimum_accuracy"],
            "passed": accuracy >= float(dataset["minimum_accuracy"]),
            "case_reports": reports,
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Output-safety evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
