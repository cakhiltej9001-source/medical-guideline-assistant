"""Tests for source-version and licensing governance metadata."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.ingestion.manifest import (  # noqa: E402
    ManifestError,
    load_manifest,
)


class ManifestTests(unittest.TestCase):
    def test_curated_manifest_has_required_governance_metadata(self) -> None:
        manifest = load_manifest(PROJECT_ROOT / "configs" / "sources.json")
        self.assertEqual(len(manifest["documents"]), 3)
        self.assertTrue(all(item["status"] == "active" for item in manifest["documents"]))

    def test_manifest_rejects_missing_license_status(self) -> None:
        document = {
            "source_id": "official:1",
            "title": "Guideline",
            "publisher": "Government",
            "document_date": "2026",
            "version": "2026 edition",
            "status": "active",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(json.dumps({"documents": [document]}), encoding="utf-8")
            with self.assertRaises(ManifestError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
