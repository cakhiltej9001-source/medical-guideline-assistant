"""Audit all configured chunks before creating retrieval indexes."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.ingestion.audit import (  # noqa: E402
    CorpusAuditError,
    audit_corpus,
    read_json,
    read_jsonl,
    write_audit_report,
)
from medical_guideline_assistant.ingestion.chunker import (  # noqa: E402
    ChunkingConfig,
    ChunkingError,
)
from medical_guideline_assistant.ingestion.manifest import (  # noqa: E402
    ManifestError,
    load_manifest,
)


def main() -> int:
    try:
        manifest = load_manifest(PROJECT_ROOT / "configs" / "sources.json")
        config = ChunkingConfig.from_path(PROJECT_ROOT / "configs" / "chunking.json")
        records_by_source = {}
        metadata_by_source = {}
        for document in manifest["documents"]:
            source_id = document["source_id"]
            records_by_source[source_id] = read_jsonl(
                PROJECT_ROOT / "data" / "processed" / f"{source_id}.chunks.jsonl"
            )
            metadata_by_source[source_id] = read_json(
                PROJECT_ROOT
                / "data"
                / "raw"
                / Path(document["filename"]).with_suffix(".metadata.json")
            )
        report = audit_corpus(
            documents=manifest["documents"],
            records_by_source=records_by_source,
            metadata_by_source=metadata_by_source,
            config=config,
        )
        write_audit_report(
            report, PROJECT_ROOT / "data" / "processed" / "corpus.audit.json"
        )
    except (CorpusAuditError, ChunkingError, ManifestError, OSError) as exc:
        print(f"Corpus audit failed to run: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
