"""Render-level smoke test for the Streamlit page."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTests(unittest.TestCase):
    def test_initial_page_renders_without_exceptions(self) -> None:
        app = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py").run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Medical Guideline Assistant")
        self.assertIn("Educational demonstration only", app.warning[0].value)
        self.assertEqual(app.selectbox[0].label, "Common guideline questions")
        self.assertEqual(len(app.selectbox[0].options), 10)
        self.assertEqual(app.text_input[0].max_chars, 500)
        self.assertEqual(app.button[0].label, "Search official guidelines")

    def test_sample_question_populates_editable_search_box(self) -> None:
        app = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py").run(timeout=30)
        app.selectbox[0].select("Dengue — warning signs")
        app.run(timeout=30)

        self.assertEqual(
            app.text_input[0].value,
            "According to the dengue guideline, what warning signs are listed?",
        )

    def test_personal_symptom_request_renders_refusal(self) -> None:
        app = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py").run(timeout=30)
        app.text_input[0].set_value("i am having fever, help me")
        app.button[0].click()
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertTrue(
            any("cannot diagnose" in warning.value for warning in app.warning)
        )


if __name__ == "__main__":
    unittest.main()
