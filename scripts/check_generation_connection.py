"""Safely diagnose one minimal Gemini text-generation request."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sanitize(message: str) -> str:
    message = re.sub(r"https?://\S+", "[url-redacted]", message)
    message = re.sub(r"(?:AIza|AQ\.)[0-9A-Za-z._-]+", "[key-redacted]", message)
    return message[:800]


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print(json.dumps({"connected": False, "reason": "missing_key"}))
        return 1
    client = genai.Client(api_key=key)
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Reply with the single word OK.",
        )
        print(json.dumps({"connected": True, "has_text": bool(response.text)}))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "connected": False,
                    "error_type": type(exc).__name__,
                    "detail": _sanitize(str(exc)),
                }
            )
        )
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
