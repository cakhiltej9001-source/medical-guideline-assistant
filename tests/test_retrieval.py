"""Tests for SQLite BM25, dense retrieval, caching, and filters."""

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.retrieval.config import (  # noqa: E402
    EmbeddingConfig,
    RetrievalConfig,
    SearchConfig,
)
from medical_guideline_assistant.retrieval.index import (  # noqa: E402
    RetrievalIndexError,
    build_index,
    search_index,
)


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
    search=SearchConfig(
        bm25_candidates=3,
        dense_candidates=3,
        final_results=3,
        rrf_constant=60,
    ),
)


def record(
    chunk_id: str, source_id: str, title: str, text: str, page: int
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "title": title,
        "category": title,
        "document_date": "2026",
        "source_url": f"https://example.gov/{source_id}.pdf",
        "source_sha256": source_id.ljust(64, "0")[:64],
        "pages": [page],
        "sections": ["Clinical guidance"],
        "safety_tags": [],
        "text": text,
    }


RECORDS = [
    record(
        "dengue:1",
        "dengue",
        "Dengue",
        "Dengue warning signs include abdominal pain and persistent vomiting.",
        24,
    ),
    record(
        "hypertension:1",
        "hypertension",
        "Hypertension",
        "Blood pressure measurement supports hypertension screening.",
        10,
    ),
    record(
        "foot:1",
        "diabetic_foot",
        "Diabetic Foot",
        "Foot ulcer prevention includes regular inspection of the feet.",
        8,
    ),
]


class FakeEmbeddingProvider:
    dimensions = 3

    def __init__(self) -> None:
        self.document_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        if "dengue" in lowered:
            return [1.0, 0.0, 0.0]
        if "blood pressure" in lowered or "hypertension" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts, titles):
        self.document_calls += 1
        return [self._vector(f"{title} {text}") for title, text in zip(titles, texts)]

    def embed_query(self, query):
        return self._vector(query)

    def close(self):
        return None


class RetrievalIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "index.sqlite3"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_lexical_index_searches_and_filters_sources(self) -> None:
        report = build_index(self.database, RECORDS, CONFIG)
        self.assertEqual(report["mode"], "lexical_only")
        results = search_index(self.database, "blood pressure", CONFIG)
        self.assertEqual(results[0].source_id, "hypertension")
        filtered = search_index(
            self.database,
            "screening",
            CONFIG,
            source_ids=("hypertension",),
        )
        self.assertTrue(filtered)
        self.assertTrue(all(item.source_id == "hypertension" for item in filtered))

    def test_hybrid_search_and_vector_cache(self) -> None:
        first_provider = FakeEmbeddingProvider()
        report = build_index(self.database, RECORDS, CONFIG, first_provider)
        self.assertEqual(report["embedded_chunks"], 3)
        self.assertEqual(first_provider.document_calls, 2)
        self.assertEqual(report["completed_api_batches"], 2)

        results = search_index(
            self.database, "dengue danger signs", CONFIG, first_provider
        )
        self.assertEqual(results[0].source_id, "dengue")
        self.assertIsNotNone(results[0].dense_rank)

        second_provider = FakeEmbeddingProvider()
        rebuilt = build_index(self.database, RECORDS, CONFIG, second_provider)
        self.assertEqual(rebuilt["embedding_cache_hits"], 3)
        self.assertEqual(second_provider.document_calls, 0)

    def test_query_length_guardrail(self) -> None:
        build_index(self.database, RECORDS, CONFIG)
        with self.assertRaises(RetrievalIndexError):
            search_index(self.database, "x" * 501, CONFIG)


if __name__ == "__main__":
    unittest.main()
