"""Unit tests for source validation that do not require internet access."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.ingestion.downloader import (  # noqa: E402
    SourceDownloadError,
    validate_source_url,
)


class ValidateSourceUrlTests(unittest.TestCase):
    def test_accepts_approved_mohfw_https_url(self) -> None:
        validate_source_url(
            "https://clinicalestablishments.mohfw.gov.in/sites/default/files/example.pdf"
        )

    def test_accepts_approved_ncvbdc_https_url(self) -> None:
        validate_source_url("https://ncvbdc.mohfw.gov.in/Doc/dengue-guideline.pdf")

    def test_rejects_http(self) -> None:
        with self.assertRaises(SourceDownloadError):
            validate_source_url("http://clinicalestablishments.mohfw.gov.in/example.pdf")

    def test_rejects_unapproved_hostname(self) -> None:
        with self.assertRaises(SourceDownloadError):
            validate_source_url("https://example.com/guideline.pdf")

    def test_rejects_hostname_suffix_attack(self) -> None:
        with self.assertRaises(SourceDownloadError):
            validate_source_url(
                "https://clinicalestablishments.mohfw.gov.in.example.com/guideline.pdf"
            )


if __name__ == "__main__":
    unittest.main()
