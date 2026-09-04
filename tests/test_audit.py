"""Unit tests for corpus-wide pre-index integrity checks."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.ingestion.audit import audit_corpus  # noqa: E402
from medical_guideline_assistant.ingestion.chunker import (  # noqa: E402
    ChunkingConfig,
    estimate_tokens,
)


CONFIG = ChunkingConfig(
    chunker_version=1,
    minimum_estimated_tokens=10,
    target_estimated_tokens=20,
    maximum_estimated_tokens=30,
    overlap_estimated_tokens=2,
    skip_extraction_statuses=frozenset({"empty", "image_review_required"}),
)
DOCUMENT = {
    "source_id": "test_source",
    "title": "Test Guideline",
    "publisher": "Test Publisher",
    "filename": "test.pdf",
    "include_page_ranges": [[3, 5]],
    "exclude_pages_from_index": [],
}
METADATA = {
    "final_url": "https://clinicalestablishments.mohfw.gov.in/test.pdf",
    "sha256": "a" * 64,
}


def make_record(pages: list[int]) -> dict[str, object]:
    text = "Supported guideline evidence for retrieval and citation."
    return {
        "chunk_id": "test_source:p0003-p0003:abc123",
        "source_id": "test_source",
        "title": "Test Guideline",
        "source_url": METADATA["final_url"],
        "source_sha256": METADATA["sha256"],
        "page_start": pages[0],
        "page_end": pages[-1],
        "pages": pages,
        "estimated_tokens": estimate_tokens(text),
        "safety_tags": [],
        "text": text,
    }


class CorpusAuditTests(unittest.TestCase):
    def test_valid_corpus_passes(self) -> None:
        report = audit_corpus(
            documents=[DOCUMENT],
            records_by_source={"test_source": [make_record([3])]},
            metadata_by_source={"test_source": METADATA},
            config=CONFIG,
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["chunks"], 1)
        self.assertEqual(report["unique_chunk_ids"], 1)

    def test_page_outside_included_range_fails(self) -> None:
        report = audit_corpus(
            documents=[DOCUMENT],
            records_by_source={"test_source": [make_record([6])]},
            metadata_by_source={"test_source": METADATA},
            config=CONFIG,
        )
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("outside include ranges" in error for error in report["errors"])
        )


if __name__ == "__main__":
    unittest.main()
