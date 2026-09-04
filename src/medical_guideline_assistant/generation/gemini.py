"""Gemini Flash-Lite provider for structured grounded claims."""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Protocol

from ..retrieval.index import SearchResult
from .config import GenerationConfig
from .grounding import ANSWER_SCHEMA, SYSTEM_INSTRUCTION, build_grounded_prompt


class GenerationError(RuntimeError):
    """Raised when a bounded generation request cannot return usable JSON."""


class GroundedGenerator(Protocol):
    def generate(self, query: str, results: list[SearchResult]) -> dict[str, Any]: ...

    def close(self) -> None: ...


class GeminiGroundedGenerator:
    RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(self, config: GenerationConfig, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key or key == "replace_with_your_google_ai_studio_key":
            raise GenerationError("GEMINI_API_KEY is missing from the local environment.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GenerationError(
                "google-genai is not installed; run pip install -r requirements.txt"
            ) from exc
        self._config = config
        self._types = types
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=config.timeout_seconds * 1000),
        )

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        for owner in (error, getattr(error, "response", None)):
            if owner is None:
                continue
            for attribute in ("status_code", "code", "status"):
                value = getattr(owner, attribute, None)
                if isinstance(value, int):
                    return value
        return None

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            value = headers.get("retry-after") or headers.get("Retry-After")
            try:
                return min(60.0, max(0.0, float(value)))
            except (TypeError, ValueError):
                pass
        match = re.search(
            r"retryDelay[^0-9]{0,30}(\d+(?:\.\d+)?)s", str(error), flags=re.I
        )
        return min(60.0, float(match.group(1))) if match else None

    def generate(self, query: str, results: list[SearchResult]) -> dict[str, Any]:
        if not results:
            raise GenerationError("Generation requires at least one evidence chunk.")
        prompt = build_grounded_prompt(query, results)
        last_error: Exception | None = None
        for attempt in range(1, self._config.maximum_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._config.model,
                    contents=prompt,
                    config=self._types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=self._config.temperature,
                        max_output_tokens=self._config.maximum_output_tokens,
                        response_mime_type="application/json",
                        response_json_schema=ANSWER_SCHEMA,
                    ),
                )
                if not response.text:
                    raise GenerationError("Gemini returned an empty structured response.")
                payload = json.loads(response.text)
                if not isinstance(payload, dict):
                    raise GenerationError("Gemini response must be a JSON object.")
                return payload
            except (GenerationError, json.JSONDecodeError):
                raise
            except Exception as exc:
                last_error = exc
                status = self._status_code(exc)
                if status not in self.RETRYABLE_STATUS_CODES or attempt == self._config.maximum_attempts:
                    break
                server_delay = self._retry_after_seconds(exc) or 0.0
                delay = max(server_delay, min(30.0, 2.0 ** attempt))
                time.sleep(delay + random.uniform(0, 0.25))

        status = self._status_code(last_error) if last_error else None
        detail = f" HTTP {status}" if status else ""
        error_type = type(last_error).__name__ if last_error else "unknown error"
        raise GenerationError(
            f"Gemini generation failed after bounded retries:{detail} {error_type}."
        ) from last_error

    def close(self) -> None:
        self._client.close()
