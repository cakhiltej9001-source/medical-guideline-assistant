"""Extract verified, page-level text for one downloaded source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.ingestion.manifest import (  # noqa: E402
    ManifestError,
    load_manifest,
)
from medical_guideline_assistant.ingestion.extractor import (  # noqa: E402
    ExtractionError,
    extract_pdf_pages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-id",
        required=True,
        help="Exact source_id from configs/sources.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(PROJECT_ROOT / "configs" / "sources.json")
    except ManifestError as exc:
        print(f"Manifest error: {exc}", file=sys.stderr)
        return 1

    documents = {
        document["source_id"]: document for document in manifest["documents"]
    }
    document = documents.get(args.source_id)
    if document is None:
        print(f"Unknown source_id: {args.source_id}", file=sys.stderr)
        return 2

    pdf_path = PROJECT_ROOT / "data" / "raw" / document["filename"]
    metadata_path = pdf_path.with_suffix(".metadata.json")
    output_path = (
        PROJECT_ROOT / "data" / "processed" / f"{args.source_id}.pages.jsonl"
    )

    try:
        report = extract_pdf_pages(
            pdf_path=pdf_path,
            metadata_path=metadata_path,
            source_id=args.source_id,
            output_path=output_path,
        )
    except ExtractionError as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
