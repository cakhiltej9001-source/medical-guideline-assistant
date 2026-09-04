"""Run input safety checks and retrieve evidence without answer generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.pipeline import retrieve_safely  # noqa: E402
from medical_guideline_assistant.retrieval.config import RetrievalConfig  # noqa: E402
from medical_guideline_assistant.retrieval.embeddings import (  # noqa: E402
    GeminiEmbeddingProvider,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--hybrid", action="store_true")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = RetrievalConfig.from_path(PROJECT_ROOT / "configs" / "retrieval.json")
    provider = GeminiEmbeddingProvider(config.embedding) if args.hybrid else None
    try:
        outcome = retrieve_safely(
            query=args.query,
            database_path=PROJECT_ROOT / config.database_path,
            config=config,
            embedding_provider=provider,
        )
    finally:
        if provider is not None:
            provider.close()

    output = outcome.to_dict()
    for result in output["results"]:
        text = result["text"]
        result["text"] = text[:500] + ("…" if len(text) > 500 else "")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
