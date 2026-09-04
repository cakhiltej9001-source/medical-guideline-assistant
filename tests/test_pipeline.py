"""Tests proving unsafe queries cannot reach retrieval or embeddings."""

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.pipeline import retrieve_safely  # noqa: E402
from medical_guideline_assistant.retrieval.config import (  # noqa: E402
    EmbeddingConfig,
    RetrievalConfig,
    SearchConfig,
)
from medical_guideline_assistant.retrieval.embeddings import EmbeddingError  # noqa: E402
from medical_guideline_assistant.retrieval.index import build_index  # noqa: E402


CONFIG = RetrievalConfig(
    retrieval_version=1,
    database_path="unused.sqlite3",
    embedding=EmbeddingConfig(
        provider="gemini",
        model="gemini-embedding-001",
        dimensions=3,
        batch_size=2,
        timeout_seconds=10,
        maximum_attempts=2,
    ),
    search=SearchConfig(5, 5, 5, 60),
)
RECORD = {
    "chunk_id": "dengue:1",
    "source_id": "dengue",
    "title": "Dengue Guideline",
    "category": "Dengue",
    "document_date": "2023",
    "source_url": "https://example.gov/dengue.pdf",
    "source_sha256": "a" * 64,
    "pages": [40],
    "sections": ["Warning signs"],
    "safety_tags": [],
    "text": "The guideline lists warning signs associated with dengue.",
}


class MustNotBeCalledProvider:
    dimensions = 3

    def embed_documents(self, texts, titles):
        raise AssertionError("embedding must not be called")

    def embed_query(self, query):
        raise AssertionError("embedding must not be called")

    def close(self):
        return None


class FailingQueryEmbeddingProvider:
    dimensions = 3

    def embed_query(self, query):
        raise EmbeddingError("simulated temporary failure")


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "index.sqlite3"
        build_index(self.database, [RECORD], CONFIG)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_emergency_is_blocked_before_embedding(self) -> None:
        outcome = retrieve_safely(
            "My child is struggling to breathe.",
            self.database,
            CONFIG,
            MustNotBeCalledProvider(),
        )
        self.assertEqual(outcome.status, "refused")
        self.assertEqual(outcome.safety.category, "emergency")
        self.assertEqual(outcome.results, [])

    def test_allowed_question_retrieves_cited_evidence(self) -> None:
        outcome = retrieve_safely(
            "What dengue warning signs are listed?", self.database, CONFIG
        )
        self.assertEqual(outcome.status, "evidence_retrieved")
        self.assertEqual(outcome.results[0].pages, [40])
        self.assertEqual(outcome.retrieval_mode, "lexical")

    def test_embedding_failure_falls_back_to_lexical_retrieval(self) -> None:
        outcome = retrieve_safely(
            "What dengue warning signs are listed?",
            self.database,
            CONFIG,
            FailingQueryEmbeddingProvider(),
        )
        self.assertEqual(outcome.status, "evidence_retrieved")
        self.assertEqual(outcome.retrieval_mode, "lexical_fallback")
        self.assertEqual(outcome.results[0].pages, [40])

    def test_no_lexical_evidence_returns_insufficient(self) -> None:
        outcome = retrieve_safely("Explain asthma inhalers.", self.database, CONFIG)
        self.assertEqual(outcome.status, "insufficient_evidence")
        self.assertEqual(outcome.results, [])


if __name__ == "__main__":
    unittest.main()
