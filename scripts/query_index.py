"""Inspect retrieval results without invoking answer generation."""

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
    parser.add_argument("query")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Embed the query with Gemini and combine dense and BM25 rankings.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        help="Optional exact source filter; may be supplied more than once.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    provider = None
    try:
        config = RetrievalConfig.from_path(PROJECT_ROOT / "configs" / "retrieval.json")
        if args.hybrid:
            provider = GeminiEmbeddingProvider(config.embedding)
        results = search_index(
            database_path=PROJECT_ROOT / config.database_path,
            query=args.query,
            config=config,
            embedding_provider=provider,
            source_ids=tuple(args.source_id) if args.source_id else None,
        )
        if config.reranking.enabled:
            results = CrossEncoderReranker(config.reranking).rerank(args.query, results)
    except (EmbeddingError, RetrievalConfigError, RetrievalIndexError, RerankingError) as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if provider is not None:
            provider.close()

    output = []
    for result in results:
        item = result.to_dict()
        item["text"] = item["text"][:500] + ("…" if len(item["text"]) > 500 else "")
        output.append(item)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
