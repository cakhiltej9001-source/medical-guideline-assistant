"""Deterministic behavior checks for HyDE, corpus-only CRAG and confidence."""

import math
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from tests.test_answering import GENERATION_CONFIG, FakeGenerator, RECORD
from tests.test_pipeline import CONFIG
from medical_guideline_assistant.answering import answer_query, preflight_query
from medical_guideline_assistant.confidence import confidence_from_scores
from medical_guideline_assistant.generation.gemini import GeminiGroundedGenerator, GenerationError
from medical_guideline_assistant.generation.grounding import validate_grounded_payload, GroundingValidationError
from medical_guideline_assistant.pipeline import retrieve_safely
from medical_guideline_assistant.retrieval.augmentation import AugmentationError
from medical_guideline_assistant.retrieval.config import AugmentationConfig, RerankingConfig, RetrievalConfig, RetrievalConfigError
from medical_guideline_assistant.retrieval.embeddings import EmbeddingError
from medical_guideline_assistant.retrieval.index import SearchResult
from medical_guideline_assistant.retrieval.reranker import RerankingError


QUERY = "What dengue warning signs are listed?"
CONFIG_ADVANCED = replace(CONFIG, reranking=RerankingConfig(enabled=True, candidate_count=5),
                          augmentation=AugmentationConfig(True, True))
RESULT = SearchResult(chunk_id=RECORD["chunk_id"], source_id="dengue", title="Dengue",
                      source_url=RECORD["source_url"], pages=[37], sections=[], safety_tags=[],
                      text=RECORD["text"], rrf_score=.03, lexical_rank=1, dense_rank=1,
                      lexical_score=1., dense_score=.9, rerank_score=.9)


class AdvancedRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.helper = Mock()
        self.helper.hypothesize.return_value = "Synthetic dengue warning-sign passage."
        self.helper.rewrite.return_value = "What warning signs does the dengue guideline describe?"
        self.embeddings = Mock()
        self.embeddings.embed_documents.return_value = [[1., 0., 0.]]
        self.reranker = Mock()
        self.reranker.rerank.side_effect = lambda query, results: results

    def run_retrieval(self, **kwargs):
        return retrieve_safely(QUERY, Path("unused"), CONFIG_ADVANCED,
                               self.embeddings, self.reranker, retrieval_assistant=self.helper, **kwargs)

    @patch("medical_guideline_assistant.pipeline.search_index", return_value=[RESULT])
    def test_hyde_vector_used_but_never_returned_as_evidence(self, search):
        outcome = self.run_retrieval(source_ids=("dengue",))
        self.assertEqual(outcome.hyde_status, "applied")
        self.assertEqual(search.call_args.kwargs["hypothetical_vector"], [1., 0., 0.])
        self.assertEqual(search.call_args.kwargs["source_ids"], ("dengue",))
        self.assertEqual(outcome.results[0].text, RECORD["text"])
        self.assertNotIn("Synthetic", str(outcome.to_dict()))
        self.helper.rewrite.assert_not_called()

    @patch("medical_guideline_assistant.pipeline.search_index", return_value=[RESULT])
    def test_hyde_failure_preserves_standard_retrieval(self, search):
        self.helper.hypothesize.side_effect = AugmentationError("unavailable")
        outcome = self.run_retrieval()
        self.assertEqual(outcome.status, "evidence_retrieved")
        self.assertEqual(outcome.hyde_status, "failed_fallback")
        self.assertIsNone(search.call_args.kwargs["hypothetical_vector"])

    @patch("medical_guideline_assistant.pipeline.search_index", return_value=[RESULT])
    def test_oversize_hypothesis_is_not_embedded(self, search):
        self.helper.hypothesize.return_value = "x" * 1201
        self.run_retrieval()
        self.embeddings.embed_documents.assert_not_called()

    @patch("medical_guideline_assistant.pipeline.search_index")
    def test_unsafe_request_never_calls_hyde_or_search(self, search):
        result = retrieve_safely("I have a fever, help me", Path("unused"), CONFIG_ADVANCED,
                                 self.embeddings, self.reranker, retrieval_assistant=self.helper)
        self.assertEqual(result.status, "refused")
        self.helper.hypothesize.assert_not_called()
        search.assert_not_called()

    @patch("medical_guideline_assistant.pipeline.search_index")
    def test_crag_recovers_empty_retrieval_once_and_scores_original_intent(self, search):
        search.side_effect = [[], [RESULT]]
        outcome = self.run_retrieval(source_ids=("dengue",))
        self.assertEqual(outcome.status, "evidence_retrieved")
        self.assertEqual(outcome.correction_status, "rewritten")
        self.assertEqual(search.call_count, 2)
        self.helper.rewrite.assert_called_once()
        self.assertEqual(search.call_args.kwargs["source_ids"], ("dengue",))
        for call in self.reranker.rerank.call_args_list:
            self.assertEqual(call.args[0], QUERY)

    @patch("medical_guideline_assistant.pipeline.search_index")
    def test_persistent_low_scores_refuse_and_prune_after_one_correction(self, search):
        weak = replace(RESULT, rerank_score=.05)
        search.side_effect = [[weak], [weak]]
        outcome = self.run_retrieval()
        self.assertEqual(outcome.status, "insufficient_evidence")
        self.assertEqual(outcome.results, [])
        self.assertEqual(outcome.removed_chunks, 1)
        self.assertEqual(search.call_count, 2)

    @patch("medical_guideline_assistant.pipeline.search_index")
    def test_unsafe_rewrite_falls_back_to_original_query(self, search):
        self.helper.rewrite.return_value = "I have fever, diagnose me"
        search.side_effect = [[replace(RESULT, rerank_score=.4)], [RESULT]]
        outcome = self.run_retrieval()
        self.assertEqual(outcome.correction_status, "rewrite_failed_broadened")
        self.assertEqual(search.call_args.args[1], QUERY)

    @patch("medical_guideline_assistant.pipeline.search_index")
    def test_correction_embedding_failure_uses_local_search(self, search):
        search.side_effect = [[replace(RESULT, rerank_score=.4)], EmbeddingError("offline"), [RESULT]]
        outcome = self.run_retrieval()
        self.assertEqual(outcome.status, "evidence_retrieved")
        self.assertIn("lexical_fallback", outcome.correction_status)

    @patch("medical_guideline_assistant.pipeline.search_index", return_value=[RESULT])
    def test_reranker_unavailable_still_fails_closed(self, search):
        self.reranker.rerank.side_effect = RerankingError("offline")
        outcome = self.run_retrieval()
        self.assertEqual(outcome.status, "insufficient_evidence")
        self.assertEqual(outcome.reranking_status, "unavailable")

    @patch("medical_guideline_assistant.pipeline.search_index", return_value=[RESULT])
    def test_disabled_features_make_no_helper_calls(self, search):
        result = retrieve_safely(QUERY, Path("unused"), replace(CONFIG_ADVANCED, augmentation=AugmentationConfig()),
                                self.embeddings, self.reranker, retrieval_assistant=self.helper)
        self.assertEqual(result.hyde_status, "disabled")
        self.helper.hypothesize.assert_not_called()
        self.helper.rewrite.assert_not_called()


class ConfidenceAndPromptTests(unittest.TestCase):
    def test_confidence_uses_weakest_score_and_handles_unknown(self):
        self.assertEqual(confidence_from_scores([.95, .6]).level, "moderate")
        self.assertEqual(confidence_from_scores([.8, .95]).level, "high")
        self.assertEqual(confidence_from_scores([.2, .95]).level, "low")
        for scores in ([None], [math.nan], [math.inf], [-1], [2], []):
            self.assertEqual(confidence_from_scores(scores).level, "not_assessed")

    def test_refusal_confidence_is_not_assessed(self):
        outcome = preflight_query("I have a fever, help me")
        self.assertEqual(outcome.to_dict()["confidence"]["level"], "not_assessed")

    def test_claim_confidence_is_computed_and_nonfinite_support_is_blocked(self):
        payload = FakeGenerator().generate(QUERY, [RESULT])
        scorer = Mock()
        scorer.score_pair.return_value = .6
        answer = validate_grounded_payload(payload, [RESULT], "info", .2, scorer, .2)
        self.assertEqual(answer.claims[0].confidence.level, "moderate")
        for score in (math.nan, math.inf, -1, 2):
            scorer.score_pair.return_value = score
            with self.assertRaises(GroundingValidationError):
                validate_grounded_payload(payload, [RESULT], "info", .2, scorer, .2)

    @patch("medical_guideline_assistant.answering.retrieve_safely")
    def test_validation_retry_uses_repair_prompt(self, retrieve):
        from medical_guideline_assistant.pipeline import RetrievalOutcome
        from medical_guideline_assistant.safety.guardrails import evaluate_input
        retrieve.return_value = RetrievalOutcome("evidence_retrieved", evaluate_input(QUERY), [RESULT], None, "hybrid")
        generator = Mock()
        generator.generate.return_value = FakeGenerator("invented").generate(QUERY, [RESULT])
        generator.repair.return_value = FakeGenerator().generate(QUERY, [RESULT])
        scorer = Mock()
        scorer.score_pair.return_value = .7
        outcome = answer_query(QUERY, Path("unused"), CONFIG_ADVANCED, GENERATION_CONFIG, generator, reranker=scorer)
        self.assertEqual(outcome.status, "answered")
        self.assertEqual(outcome.confidence.level, "moderate")
        generator.repair.assert_called_once()

    def test_provider_repair_and_assistance_are_bounded(self):
        generator = object.__new__(GeminiGroundedGenerator)
        generator._request = Mock(return_value={"text": "Dengue guideline terminology."})
        generator.hypothesize(QUERY, 1200)
        self.assertEqual(generator._request.call_args.kwargs["maximum_attempts"], 1)
        generator.repair(QUERY, [RESULT])
        self.assertIn("REPAIR PASS", generator._request.call_args.args[0])
        generator._request.side_effect = GenerationError("unavailable")
        with self.assertRaises(AugmentationError):
            generator.rewrite(QUERY)

    def test_config_enables_features_and_rejects_invalid_threshold(self):
        config = RetrievalConfig.from_path(Path(__file__).resolve().parents[1] / "configs/retrieval.json")
        self.assertTrue(config.augmentation.hyde_enabled)
        self.assertTrue(config.augmentation.crag_enabled)
        with self.assertRaises(RetrievalConfigError):
            replace(config, augmentation=replace(config.augmentation, correction_threshold=.01)).validate()


if __name__ == "__main__":
    unittest.main()
