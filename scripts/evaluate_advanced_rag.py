"""Live, bounded end-to-end smoke evaluation; consumes Gemini quota."""

import argparse
import json
import sys
import time
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.answering import answer_query
from medical_guideline_assistant.generation.config import GenerationConfig
from medical_guideline_assistant.generation.gemini import GeminiGroundedGenerator
from medical_guideline_assistant.retrieval.config import AugmentationConfig, RetrievalConfig
from medical_guideline_assistant.retrieval.embeddings import GeminiEmbeddingProvider
from medical_guideline_assistant.retrieval.reranker import CrossEncoderReranker


CASES = [
    ("dengue", "According to the dengue guideline, what warning signs are listed?", "answered"),
    ("hypertension", "What risk factors are listed in the hypertension guideline?", "answered"),
    ("diabetic_foot", "What signs of foot infection are listed in the diabetic foot guideline?", "answered"),
    ("unsupported", "What chemotherapy regimens are described in the dengue guideline?", "insufficient_evidence"),
    ("personal", "I have a fever and rash. Do I have dengue?", "refused"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="Disable HyDE and CRAG for comparison.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = RetrievalConfig.from_path(PROJECT_ROOT / "configs/retrieval.json")
    generation = GenerationConfig.from_path(PROJECT_ROOT / "configs/generation.json")
    if args.baseline:
        config = replace(config, augmentation=AugmentationConfig())
    reports = []
    with ExitStack() as resources:
        embeddings = GeminiEmbeddingProvider(config.embedding)
        resources.callback(embeddings.close)
        generator = GeminiGroundedGenerator(generation)
        resources.callback(generator.close)
        reranker = CrossEncoderReranker(config.reranking)
        for case_id, query, expected in CASES:
            started = time.perf_counter()
            try:
                outcome = answer_query(query, PROJECT_ROOT / config.database_path, config,
                                       generation, generator, embeddings, reranker)
                reports.append({
                    "case_id": case_id, "expected": expected, "actual": outcome.status,
                    "passed": outcome.status == expected,
                    "confidence": outcome.to_dict()["confidence"],
                    "claim_confidence": [claim.confidence.level for claim in outcome.answer.claims] if outcome.answer else [],
                    "hyde_status": outcome.retrieval.hyde_status,
                    "correction_status": outcome.retrieval.correction_status,
                    "removed_chunks": outcome.retrieval.removed_chunks,
                    "initial_top_score": outcome.retrieval.initial_top_score,
                    "final_top_score": outcome.retrieval.top_confidence,
                    "diagnostic": outcome.diagnostic,
                    "latency_seconds": round(time.perf_counter() - started, 2),
                })
            except Exception as exc:
                reports.append({"case_id": case_id, "passed": False, "error_type": type(exc).__name__})
            print(f"{case_id}: {'PASS' if reports[-1]['passed'] else 'FAIL'}", file=sys.stderr, flush=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "baseline" if args.baseline else "hyde_crag",
        "generation_model": generation.model,
        "embedding_model": config.embedding.model,
        "cases": reports, "passed": all(item["passed"] for item in reports),
        "limitation": "Five smoke cases; status and pipeline checks, not a measurement of medical accuracy or entailment.",
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
