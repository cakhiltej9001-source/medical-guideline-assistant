"""Unit tests for section-aware chunking and safety metadata."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.ingestion.chunker import (  # noqa: E402
    ChunkingConfig,
    build_chunk_records,
    detect_safety_tags,
    estimate_tokens,
    is_hard_section_heading,
    is_section_heading,
)


CONFIG = ChunkingConfig(
    chunker_version=1,
    minimum_estimated_tokens=40,
    target_estimated_tokens=60,
    maximum_estimated_tokens=80,
    overlap_estimated_tokens=10,
    skip_extraction_statuses=frozenset({"empty", "image_review_required"}),
)
DOCUMENT = {
    "source_id": "test_source",
    "title": "Test Guideline",
    "publisher": "Test Publisher",
    "category": "Test",
    "document_date": "2026",
    "language": "en",
}
METADATA = {
    "final_url": "https://clinicalestablishments.mohfw.gov.in/test.pdf",
    "sha256": "a" * 64,
}


class HeadingAndSafetyTests(unittest.TestCase):
    def test_recognizes_section_heading(self) -> None:
        self.assertTrue(is_section_heading("KEY RECOMMENDATION: SCREENING"))
        self.assertFalse(is_section_heading("This is ordinary explanatory text."))
        self.assertFalse(is_section_heading("(USA)."))
        self.assertTrue(is_hard_section_heading("ANNEXURE 3: FORMULARY"))
        self.assertFalse(is_hard_section_heading("KEY RECOMMENDATION: SCREENING"))

    def test_detects_dosage_and_emergency_content(self) -> None:
        tags = detect_safety_tags("Initial dosages include 5 mg in an emergency.")
        self.assertEqual(tags, ["dosage_content", "emergency_content"])

    def test_token_estimate_is_conservative(self) -> None:
        self.assertGreater(estimate_tokens("A short medical sentence."), 4)


class ChunkRecordTests(unittest.TestCase):
    def _page_records(self) -> list[dict[str, object]]:
        text = " ".join(
            f"Recommendation {number} provides general screening information."
            for number in range(1, 45)
        )
        return [
            {
                "page_number": 1,
                "text": f"SECTION 1: SCREENING\n{text}",
                "extraction_status": "ok",
            },
            {
                "page_number": 2,
                "text": "Flowchart image",
                "extraction_status": "image_review_required",
            },
        ]

    def test_chunks_respect_hard_limit_and_skip_unverified_image_page(self) -> None:
        records, report = build_chunk_records(
            self._page_records(), DOCUMENT, METADATA, CONFIG
        )
        self.assertTrue(records)
        self.assertTrue(all(record["estimated_tokens"] <= 80 for record in records))
        self.assertEqual(report["skipped_pages"], [2])

    def test_chunk_ids_are_stable(self) -> None:
        first, _ = build_chunk_records(
            self._page_records(), DOCUMENT, METADATA, CONFIG
        )
        second, _ = build_chunk_records(
            self._page_records(), DOCUMENT, METADATA, CONFIG
        )
        self.assertEqual(
            [record["chunk_id"] for record in first],
            [record["chunk_id"] for record in second],
        )

    def test_consecutive_headings_do_not_create_heading_only_chunks(self) -> None:
        pages = [
            {
                "page_number": 1,
                "text": (
                    "SECTION 1: APPENDICES\n"
                    "ANNEXURE 1: DEFINITIONS\n"
                    + " ".join(["General explanatory content."] * 15)
                ),
                "extraction_status": "ok",
            }
        ]
        records, _ = build_chunk_records(pages, DOCUMENT, METADATA, CONFIG)
        self.assertEqual(len(records), 1)
        self.assertIn("General explanatory content", records[0]["text"])

    def test_manual_page_exclusion_is_recorded(self) -> None:
        pages = [
            {
                "page_number": 1,
                "text": "TABLE OF CONTENTS\nDuplicate navigation text.",
                "extraction_status": "ok",
            },
            {
                "page_number": 2,
                "text": " ".join(["Supported guideline content."] * 15),
                "extraction_status": "ok",
            },
        ]
        document = {**DOCUMENT, "exclude_pages_from_index": [1]}
        records, report = build_chunk_records(pages, document, METADATA, CONFIG)
        self.assertTrue(records)
        self.assertTrue(all(record["page_start"] == 2 for record in records))
        self.assertEqual(report["manually_excluded_pages"], [1])

    def test_chunks_do_not_cross_a_skipped_page_gap(self) -> None:
        pages = [
            {
                "page_number": 1,
                "text": " ".join(["First page evidence."] * 12),
                "extraction_status": "ok",
            },
            {
                "page_number": 2,
                "text": "Unverified flowchart",
                "extraction_status": "image_review_required",
            },
            {
                "page_number": 3,
                "text": " ".join(["Third page evidence."] * 12),
                "extraction_status": "ok",
            },
        ]
        records, _ = build_chunk_records(pages, DOCUMENT, METADATA, CONFIG)
        self.assertFalse(any(1 in record["pages"] and 3 in record["pages"] for record in records))
        self.assertTrue(all(2 not in record["pages"] for record in records))

    def test_include_page_ranges_limit_the_corpus(self) -> None:
        pages = [
            {
                "page_number": number,
                "text": " ".join([f"Evidence from page {number}."] * 12),
                "extraction_status": "ok",
            }
            for number in range(1, 5)
        ]
        document = {**DOCUMENT, "include_page_ranges": [[2, 3]]}
        records, report = build_chunk_records(pages, document, METADATA, CONFIG)
        indexed_pages = {
            page for record in records for page in record["pages"]
        }
        self.assertEqual(indexed_pages, {2, 3})
        self.assertEqual(report["manually_excluded_pages"], [1, 4])


if __name__ == "__main__":
    unittest.main()
