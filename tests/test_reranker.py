"""Unit tests for cross-encoder ordering and normalized confidence scores."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.retrieval.config import RerankingConfig  # noqa: E402
from medical_guideline_assistant.retrieval.index import SearchResult  # noqa: E402
from medical_guideline_assistant.retrieval.reranker import (  # noqa: E402
    CrossEncoderReranker,
)


def result(chunk_id: str, text: str, rrf_score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        source_id="official",
        title="Guideline",
        source_url="https://example.gov/guideline.pdf",
        pages=[1],
        sections=["Section"],
        safety_tags=[],
        text=text,
        rrf_score=rrf_score,
        lexical_rank=1,
        dense_rank=1,
        lexical_score=1.0,
        dense_score=0.5,
    )


class FakeCrossEncoder:
    def rerank(self, query, documents):
        self.pairs = getattr(self, "pairs", []) + [(query, document) for document in documents]
        return iter([-2.0, 2.0][: len(documents)])


class RerankerTests(unittest.TestCase):
    def test_cross_encoder_reorders_rrf_candidates(self) -> None:
        model = FakeCrossEncoder()
        reranker = CrossEncoderReranker(
            RerankingConfig(
                enabled=True,
                model="Xenova/ms-marco-MiniLM-L-6-v2",
                candidate_count=2,
            ),
            model=model,
        )
        ranked = reranker.rerank(
            "dengue warning signs",
            [result("a", "unrelated", 0.04), result("b", "warning signs", 0.03)],
        )
        self.assertEqual([item.chunk_id for item in ranked], ["b", "a"])
        self.assertGreater(ranked[0].rerank_score, ranked[1].rerank_score)
        self.assertEqual(len(model.pairs), 2)

    def test_support_pair_score_is_normalized(self) -> None:
        reranker = CrossEncoderReranker(
            RerankingConfig(enabled=True), model=FakeCrossEncoder()
        )
        score = reranker.score_pair("claim", "evidence")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
