"""List generation-capable Gemini models visible to the configured project."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print(json.dumps({"error": "missing_key"}))
        return 1
    client = genai.Client(api_key=key)
    try:
        names = []
        for model in client.models.list():
            actions = set(model.supported_actions or [])
            if "generateContent" in actions and "gemini" in model.name:
                names.append(model.name.removeprefix("models/"))
    finally:
        client.close()
    print(json.dumps(sorted(names), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
