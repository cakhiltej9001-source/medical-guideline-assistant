"""Tests for safe end-to-end answer orchestration without external APIs."""

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.answering import answer_query  # noqa: E402
from medical_guideline_assistant.generation.config import GenerationConfig  # noqa: E402
from medical_guideline_assistant.retrieval.config import (  # noqa: E402
    EmbeddingConfig,
    RetrievalConfig,
    SearchConfig,
)
from medical_guideline_assistant.retrieval.index import build_index  # noqa: E402


RETRIEVAL_CONFIG = RetrievalConfig(
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
GENERATION_CONFIG = GenerationConfig(
    generation_version=1,
    provider="gemini",
    model="gemini-3.5-flash-lite",
    temperature=0.1,
    maximum_output_tokens=512,
    timeout_seconds=30,
    maximum_attempts=2,
    maximum_validation_attempts=2,
    maximum_context_chunks=3,
    minimum_claim_token_overlap=0.2,
    standard_disclaimer="Informational only.",
)
RECORD = {
    "chunk_id": "dengue:37",
    "source_id": "dengue",
    "title": "Dengue Guideline",
    "category": "Dengue",
    "document_date": "2023",
    "source_url": "https://example.gov/dengue.pdf",
    "source_sha256": "a" * 64,
    "pages": [37],
    "sections": ["Warning signs"],
    "safety_tags": [],
    "text": "Warning signs include persistent vomiting and abdominal pain.",
}


class FakeGenerator:
    def __init__(self, chunk_id: str = "dengue:37") -> None:
        self.calls = 0
        self.chunk_id = chunk_id

    def generate(self, query, results):
        self.calls += 1
        return {
            "status": "answered",
            "claims": [
                {
                    "text": "Warning signs include persistent vomiting and abdominal pain.",
                    "chunk_ids": [self.chunk_id],
                }
            ],
        }

    def close(self):
        return None


class RepairableGenerator(FakeGenerator):
    def generate(self, query, results):
        self.calls += 1
        chunk_id = "invented:chunk" if self.calls == 1 else self.chunk_id
        return {
            "status": "answered",
            "claims": [
                {
                    "text": "Warning signs include persistent vomiting and abdominal pain.",
                    "chunk_ids": [chunk_id],
                }
            ],
        }


class AnsweringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "index.sqlite3"
        build_index(self.database, [RECORD], RETRIEVAL_CONFIG)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_refusal_never_calls_generator(self) -> None:
        generator = FakeGenerator()
        outcome = answer_query(
            "I have a fever and rash. Do I have dengue?",
            self.database,
            RETRIEVAL_CONFIG,
            GENERATION_CONFIG,
            generator,
        )
        self.assertEqual(outcome.status, "refused")
        self.assertEqual(generator.calls, 0)

    def test_allowed_question_returns_validated_citation(self) -> None:
        generator = FakeGenerator()
        outcome = answer_query(
            "What dengue warning signs are listed?",
            self.database,
            RETRIEVAL_CONFIG,
            GENERATION_CONFIG,
            generator,
        )
        self.assertEqual(outcome.status, "answered")
        self.assertEqual(outcome.answer.claims[0].citations[0].pages, [37])

    def test_invented_model_citation_blocks_entire_output(self) -> None:
        generator = FakeGenerator("invented:chunk")
        outcome = answer_query(
            "What dengue warning signs are listed?",
            self.database,
            RETRIEVAL_CONFIG,
            GENERATION_CONFIG,
            generator,
        )
        self.assertEqual(outcome.status, "output_blocked")
        self.assertIsNone(outcome.answer)
        self.assertEqual(generator.calls, 2)

    def test_second_valid_generation_recovers_from_first_validation_failure(self) -> None:
        generator = RepairableGenerator()
        outcome = answer_query(
            "What dengue warning signs are listed?",
            self.database,
            RETRIEVAL_CONFIG,
            GENERATION_CONFIG,
            generator,
        )
        self.assertEqual(outcome.status, "answered")
        self.assertEqual(generator.calls, 2)
        self.assertEqual(outcome.answer.claims[0].citations[0].pages, [37])


if __name__ == "__main__":
    unittest.main()
