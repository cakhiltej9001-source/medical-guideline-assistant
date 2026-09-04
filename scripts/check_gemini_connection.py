"""Safely test Gemini embeddings without printing credentials or raw responses."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_status(error: Exception) -> object:
    for owner in (error, getattr(error, "response", None)):
        if owner is None:
            continue
        for name in ("status_code", "code", "status"):
            value = getattr(owner, name, None)
            if isinstance(value, (int, str)):
                return value
    return None


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print(json.dumps({"connected": False, "reason": "missing_key"}))
        return 1
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=30_000),
        )
        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents="connection test",
                config=types.EmbedContentConfig(output_dimensionality=768),
            )
        finally:
            client.close()
        dimensions = len(response.embeddings[0].values)
        print(json.dumps({"connected": True, "dimensions": dimensions}))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "connected": False,
                    "error_type": type(exc).__name__,
                    "status": _safe_status(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
