"""Unit tests for deterministic text-cleaning helpers."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.ingestion.extractor import (  # noqa: E402
    classify_page_extraction,
    canonical_boundary_line,
    clean_page_lines,
    detect_repeated_boundary_lines,
    normalize_line,
)


class NormalizeLineTests(unittest.TestCase):
    def test_normalizes_spacing_and_unicode_hyphens(self) -> None:
        self.assertEqual(normalize_line("high\u2011risk   group"), "high-risk group")

    def test_canonicalizes_footer_page_number(self) -> None:
        self.assertEqual(
            canonical_boundary_line("National Standard Treatment Guidelines 17"),
            "National Standard Treatment Guidelines <PAGE_NUMBER>",
        )


class BoundaryCleaningTests(unittest.TestCase):
    def test_detects_repeated_footer(self) -> None:
        pages = [
            [
                "Heading",
                f"Body topic number-{number}",
                f"Different content number-{number}",
                str(number),
                "Repeated document footer",
            ]
            for number in range(1, 6)
        ]
        self.assertEqual(
            detect_repeated_boundary_lines(pages),
            {"Heading", "Repeated document footer"},
        )

    def test_detects_footer_with_varying_page_number(self) -> None:
        pages = [
            [
                f"Unique body number-{number}",
                f"Additional unique number-{number}",
                f"National Standard Treatment Guidelines {number}",
            ]
            for number in range(1, 6)
        ]
        self.assertEqual(
            detect_repeated_boundary_lines(pages),
            {"National Standard Treatment Guidelines <PAGE_NUMBER>"},
        )

    def test_removes_edge_page_number_but_keeps_body_number(self) -> None:
        lines = ["12", "Recommendation", "2", "Supporting text", "Repeated footer"]
        cleaned = clean_page_lines(lines, {"Repeated footer"})
        self.assertEqual(cleaned, "Recommendation\n2\nSupporting text")


class ExtractionStatusTests(unittest.TestCase):
    def test_flags_image_dominant_low_text_page(self) -> None:
        self.assertEqual(
            classify_page_extraction(character_count=9, image_count=2),
            "image_review_required",
        )

    def test_accepts_normal_text_page(self) -> None:
        self.assertEqual(
            classify_page_extraction(character_count=2000, image_count=0), "ok"
        )


if __name__ == "__main__":
    unittest.main()
