"""Build the local SQLite retrieval index from audited guideline chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.ingestion.audit import (  # noqa: E402
    CorpusAuditError,
    read_json,
    read_jsonl,
)
from medical_guideline_assistant.ingestion.manifest import (  # noqa: E402
    ManifestError,
    load_manifest,
)
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
    build_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Call Gemini for missing vectors. Without this flag, build BM25 only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    provider = None
    try:
        retrieval_config = RetrievalConfig.from_path(
            PROJECT_ROOT / "configs" / "retrieval.json"
        )
        manifest = load_manifest(PROJECT_ROOT / "configs" / "sources.json")
        audit = read_json(PROJECT_ROOT / "data" / "processed" / "corpus.audit.json")
        if audit.get("status") != "passed":
            raise RetrievalIndexError(
                "Corpus audit must pass before indexing; run scripts/audit_corpus.py."
            )

        records = []
        for document in manifest["documents"]:
            records.extend(
                read_jsonl(
                    PROJECT_ROOT
                    / "data"
                    / "processed"
                    / f"{document['source_id']}.chunks.jsonl"
                )
            )
        if len(records) != audit.get("chunks"):
            raise RetrievalIndexError(
                "Chunk count changed after the audit; run scripts/audit_corpus.py again."
            )

        if args.with_embeddings:
            provider = GeminiEmbeddingProvider(retrieval_config.embedding)
        database_path = PROJECT_ROOT / retrieval_config.database_path
        report = build_index(
            database_path=database_path,
            records=records,
            config=retrieval_config,
            embedding_provider=provider,
        )
    except (
        CorpusAuditError,
        EmbeddingError,
        ManifestError,
        RetrievalConfigError,
        RetrievalIndexError,
        OSError,
    ) as exc:
        print(f"Index build failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if provider is not None:
            provider.close()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
