"""Tests for the UI's session rate limiter."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_guideline_assistant.ui.rate_limit import consume_request


class RateLimitTests(unittest.TestCase):
    def test_accepts_requests_below_limit(self) -> None:
        decision = consume_request([90.0, 95.0], now=100.0, maximum_requests=3)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.timestamps, [90.0, 95.0, 100.0])

    def test_rejects_request_at_limit_with_retry_time(self) -> None:
        decision = consume_request([50.0, 70.0], now=100.0, maximum_requests=2)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.retry_after_seconds, 10)

    def test_expired_and_future_timestamps_are_removed(self) -> None:
        decision = consume_request([-1.0, 39.9, 40.0, 101.0], now=100.0)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.timestamps, [100.0])

    def test_rejects_invalid_settings(self) -> None:
        with self.assertRaises(ValueError):
            consume_request([], now=1.0, maximum_requests=0)


if __name__ == "__main__":
    unittest.main()
