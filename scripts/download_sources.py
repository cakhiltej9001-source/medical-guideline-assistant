"""Command-line entry point for downloading one approved source document."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.ingestion.downloader import (  # noqa: E402
    SourceDownloadError,
    download_pdf,
)
from medical_guideline_assistant.ingestion.manifest import (  # noqa: E402
    ManifestError,
    load_manifest,
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
    matches = [
        document
        for document in manifest["documents"]
        if document["source_id"] == args.source_id
    ]
    if not matches:
        available = ", ".join(document["source_id"] for document in manifest["documents"])
        print(f"Unknown source_id. Available values: {available}", file=sys.stderr)
        return 2

    try:
        metadata = download_pdf(
            document=matches[0],
            source_page=matches[0].get("source_page", manifest["source_page"]),
            output_dir=PROJECT_ROOT / "data" / "raw",
        )
    except SourceDownloadError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
