"""Create section-aware chunks for one extracted guideline document."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.ingestion.chunker import (  # noqa: E402
    ChunkingConfig,
    ChunkingError,
    build_chunk_records,
    read_jsonl,
    write_chunk_outputs,
)
from medical_guideline_assistant.ingestion.manifest import (  # noqa: E402
    ManifestError,
    load_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(PROJECT_ROOT / "configs" / "sources.json")
        config = ChunkingConfig.from_path(PROJECT_ROOT / "configs" / "chunking.json")
        documents = {
            document["source_id"]: document for document in manifest["documents"]
        }
        document = documents.get(args.source_id)
        if document is None:
            print(f"Unknown source_id: {args.source_id}", file=sys.stderr)
            return 2

        input_path = (
            PROJECT_ROOT / "data" / "processed" / f"{args.source_id}.pages.jsonl"
        )
        page_records = read_jsonl(input_path)
        metadata_path = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / Path(document["filename"]).with_suffix(".metadata.json")
        )
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records, report = build_chunk_records(
            page_records=page_records,
            document=document,
            source_metadata=source_metadata,
            config=config,
        )
        output_path = (
            PROJECT_ROOT / "data" / "processed" / f"{args.source_id}.chunks.jsonl"
        )
        write_chunk_outputs(records, report, output_path)
    except (ChunkingError, ManifestError, OSError, json.JSONDecodeError) as exc:
        print(f"Chunking failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
