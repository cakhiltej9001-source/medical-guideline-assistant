"""Run the current end-to-end RAG pipeline from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.answering import answer_query  # noqa: E402
from medical_guideline_assistant.generation.config import (  # noqa: E402
    GenerationConfig,
    GenerationConfigError,
)
from medical_guideline_assistant.generation.gemini import (  # noqa: E402
    GenerationError,
    GeminiGroundedGenerator,
)
from medical_guideline_assistant.retrieval.config import (  # noqa: E402
    RetrievalConfig,
    RetrievalConfigError,
)
from medical_guideline_assistant.retrieval.embeddings import (  # noqa: E402
    EmbeddingError,
    GeminiEmbeddingProvider,
)
from medical_guideline_assistant.retrieval.index import RetrievalIndexError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    embedding_provider = None
    generator = None
    try:
        retrieval_config = RetrievalConfig.from_path(
            PROJECT_ROOT / "configs" / "retrieval.json"
        )
        generation_config = GenerationConfig.from_path(
            PROJECT_ROOT / "configs" / "generation.json"
        )
        embedding_provider = GeminiEmbeddingProvider(retrieval_config.embedding)
        generator = GeminiGroundedGenerator(generation_config)
        outcome = answer_query(
            query=args.query,
            database_path=PROJECT_ROOT / retrieval_config.database_path,
            retrieval_config=retrieval_config,
            generation_config=generation_config,
            generator=generator,
            embedding_provider=embedding_provider,
        )
    except (
        EmbeddingError,
        GenerationConfigError,
        GenerationError,
        RetrievalConfigError,
        RetrievalIndexError,
        OSError,
    ) as exc:
        print(f"Request failed safely: {exc}", file=sys.stderr)
        return 1
    finally:
        if generator is not None:
            generator.close()
        if embedding_provider is not None:
            embedding_provider.close()

    output = outcome.to_dict()
    for result in output["retrieval"]["results"]:
        text = result["text"]
        result["text"] = text[:300] + ("…" if len(text) > 300 else "")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
