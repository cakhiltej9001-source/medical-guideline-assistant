"""Tests for grounded claim validation and output safety."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.generation.grounding import (  # noqa: E402
    GroundingValidationError,
    build_grounded_prompt,
    classify_output,
    validate_grounded_payload,
)
from medical_guideline_assistant.retrieval.index import SearchResult  # noqa: E402


RESULT = SearchResult(
    chunk_id="dengue:p0037:abc",
    source_id="dengue",
    title="Dengue Guideline",
    source_url="https://example.gov/dengue.pdf",
    pages=[37],
    sections=["Warning signs"],
    safety_tags=[],
    text="Warning signs include persistent vomiting, abdominal pain, lethargy, and bleeding manifestations.",
    rrf_score=0.03,
    lexical_rank=2,
    dense_rank=1,
    lexical_score=8.0,
    dense_score=0.8,
)


class LowSupportScorer:
    def score_pair(self, left, right):
        return 0.05


class GroundingTests(unittest.TestCase):
    def test_output_classifier_blocks_personalized_diagnosis(self) -> None:
        decision = classify_output("You likely have dengue and should take this medicine.")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "personalized_advice")

    def test_output_classifier_blocks_emergency_instruction(self) -> None:
        decision = classify_output("Seek immediate medical attention now.")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "emergency_guidance")

    def test_output_classifier_allows_neutral_guideline_summary(self) -> None:
        decision = classify_output("The guideline lists persistent vomiting as a warning sign.")
        self.assertTrue(decision.allowed)

    def test_prompt_marks_chunk_and_treats_it_as_evidence(self) -> None:
        prompt = build_grounded_prompt("What warning signs are listed?", [RESULT])
        self.assertIn("<chunk id=", prompt)
        self.assertIn(RESULT.chunk_id, prompt)

    def test_valid_claim_resolves_trusted_citation_metadata(self) -> None:
        answer = validate_grounded_payload(
            {
                "status": "answered",
                "claims": [
                    {
                        "text": "Warning signs include persistent vomiting and abdominal pain.",
                        "chunk_ids": [RESULT.chunk_id],
                    }
                ],
            },
            [RESULT],
            "Disclaimer",
            0.2,
        )
        self.assertEqual(answer.claims[0].citations[0].pages, [37])

    def test_allows_bounded_multi_item_guideline_list(self) -> None:
        claims = [
            {
                "text": "Warning signs include persistent vomiting and abdominal pain.",
                "chunk_ids": [RESULT.chunk_id],
            }
            for _ in range(10)
        ]
        answer = validate_grounded_payload(
            {"status": "answered", "claims": claims},
            [RESULT],
            "Disclaimer",
            0.2,
        )
        self.assertEqual(len(answer.claims), 10)

    def test_rejects_invented_citation_id(self) -> None:
        with self.assertRaises(GroundingValidationError):
            validate_grounded_payload(
                {
                    "status": "answered",
                    "claims": [{"text": "Persistent vomiting is listed.", "chunk_ids": ["fake"]}],
                },
                [RESULT],
                "Disclaimer",
                0.2,
            )

    def test_rejects_dosage_in_generated_claim(self) -> None:
        with self.assertRaises(GroundingValidationError):
            validate_grounded_payload(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Take 500 mg of medicine.",
                            "chunk_ids": [RESULT.chunk_id],
                        }
                    ],
                },
                [RESULT],
                "Disclaimer",
                0.0,
            )

    def test_allows_lifestyle_quantity_without_medication_context(self) -> None:
        result = replace(
            RESULT,
            text="The guideline describes a heart-healthy diet with salt below 5 g/day.",
        )
        answer = validate_grounded_payload(
            {
                "status": "answered",
                "claims": [
                    {
                        "text": "The guideline describes limiting salt to below 5 g/day.",
                        "chunk_ids": [result.chunk_id],
                    }
                ],
            },
            [result],
            "Disclaimer",
            0.1,
        )
        self.assertEqual(answer.status, "answered")

    def test_rejects_weakly_related_claim(self) -> None:
        with self.assertRaises(GroundingValidationError):
            validate_grounded_payload(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Astronomers discovered a distant galaxy.",
                            "chunk_ids": [RESULT.chunk_id],
                        }
                    ],
                },
                [RESULT],
                "Disclaimer",
                0.2,
            )

    def test_rejects_claim_below_semantic_support_threshold(self) -> None:
        with self.assertRaises(GroundingValidationError):
            validate_grounded_payload(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Warning signs include persistent vomiting.",
                            "chunk_ids": [RESULT.chunk_id],
                        }
                    ],
                },
                [RESULT],
                "Disclaimer",
                0.2,
                support_scorer=LowSupportScorer(),
                minimum_claim_support_score=0.2,
            )


if __name__ == "__main__":
    unittest.main()
