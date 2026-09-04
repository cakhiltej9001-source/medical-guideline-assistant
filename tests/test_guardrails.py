"""Tests for pre-retrieval medical safety decisions."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.safety.guardrails import (  # noqa: E402
    evaluate_input,
    rewrite_query,
)


class RequiredSafetyExamplesTests(unittest.TestCase):
    def test_refuses_personalized_dose_request(self) -> None:
        decision = evaluate_input(
            "My blood pressure is 160/100. Which medicine and dose should I take?"
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "dosage")
        self.assertIsNone(decision.retrieval_query)

    def test_refuses_personalized_diagnosis(self) -> None:
        decision = evaluate_input(
            "I have fever, body pain, and a rash. Do I have dengue?"
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "diagnosis")

    def test_refuses_emergency_before_retrieval(self) -> None:
        decision = evaluate_input(
            "My child is struggling to breathe. Tell me what treatment to do at home."
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "emergency")
        self.assertIn("immediate", decision.refusal_message.casefold())


class GuardrailPrecisionTests(unittest.TestCase):
    def test_allows_general_guideline_warning_sign_question(self) -> None:
        decision = evaluate_input(
            "According to the dengue guideline, what warning signs are listed?"
        )
        self.assertTrue(decision.allowed)

    def test_allows_general_blood_pressure_measurement_question(self) -> None:
        decision = evaluate_input(
            "How should blood pressure be measured for hypertension screening?"
        )
        self.assertTrue(decision.allowed)

    def test_refuses_general_dosage_request(self) -> None:
        decision = evaluate_input("Summarize all medication doses in the guideline.")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "dosage")

    def test_refuses_first_person_symptom_help_request(self) -> None:
        decision = evaluate_input("i am having fever, help me")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "diagnosis")
        self.assertIsNone(decision.retrieval_query)

    def test_rejects_oversized_input(self) -> None:
        decision = evaluate_input("a" * 501)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "invalid_input")

    def test_expands_only_known_abbreviations(self) -> None:
        rewritten = rewrite_query("Compare SBP, DBP, and HCT.")
        self.assertEqual(
            rewritten,
            "Compare systolic blood pressure, diastolic blood pressure, and hematocrit.",
        )


if __name__ == "__main__":
    unittest.main()
