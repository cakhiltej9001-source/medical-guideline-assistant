"""Embedding provider boundary with a rate-limit-aware Gemini implementation."""

from __future__ import annotations

import math
import os
import random
import re
import time
from collections.abc import Sequence
from typing import Protocol

from .config import EmbeddingConfig


class EmbeddingError(RuntimeError):
    """Raised when embedding configuration or API calls fail safely."""


class EmbeddingProvider(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed_documents(
        self, texts: Sequence[str], titles: Sequence[str]
    ) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...

    def close(self) -> None: ...


def normalize_vector(vector: Sequence[float]) -> list[float]:
    magnitude = math.sqrt(sum(float(value) ** 2 for value in vector))
    if not math.isfinite(magnitude) or magnitude == 0:
        raise EmbeddingError("Embedding vector has an invalid magnitude.")
    return [float(value) / magnitude for value in vector]


class GeminiEmbeddingProvider:
    """Generate retrieval-specific embeddings without exposing the API key."""

    RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(self, config: EmbeddingConfig, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key or key == "replace_with_your_google_ai_studio_key":
            raise EmbeddingError(
                "GEMINI_API_KEY is missing. Create a Google AI Studio key and set "
                "it as an environment variable; never put it in source code."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise EmbeddingError(
                "google-genai is not installed; run pip install -r requirements.txt"
            ) from exc

        self._config = config
        self._types = types
        self._last_request_started = 0.0
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=config.timeout_seconds * 1000),
        )

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

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

    def _pace_request(self, minimum_delay: float | None = None) -> None:
        interval = max(
            self._config.minimum_batch_interval_seconds,
            minimum_delay or 0.0,
        )
        remaining = interval - (time.monotonic() - self._last_request_started)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_started = time.monotonic()

    def _embed(self, contents: Sequence[str], task_type: str) -> list[list[float]]:
        if not contents or any(not content.strip() for content in contents):
            raise EmbeddingError("Embedding input must contain non-empty text.")

        last_error: Exception | None = None
        retry_delay: float | None = None
        for attempt in range(1, self._config.maximum_attempts + 1):
            try:
                self._pace_request(retry_delay)
                response = self._client.models.embed_content(
                    model=self._config.model,
                    contents=list(contents),
                    config=self._types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._config.dimensions,
                    ),
                )
                vectors = [embedding.values for embedding in response.embeddings or []]
                if len(vectors) != len(contents):
                    raise EmbeddingError(
                        "Gemini returned a different number of vectors than inputs."
                    )
                normalized = [normalize_vector(vector) for vector in vectors]
                if any(len(vector) != self.dimensions for vector in normalized):
                    raise EmbeddingError("Gemini returned an unexpected vector dimension.")
                return normalized
            except EmbeddingError:
                raise
            except Exception as exc:  # SDK errors vary across supported versions.
                last_error = exc
                retryable = self._status_code(exc) in self.RETRYABLE_STATUS_CODES
                if not retryable or attempt == self._config.maximum_attempts:
                    break
                server_delay = self._retry_after_seconds(exc)
                exponential = min(30.0, 2.0 * (2 ** (attempt - 1)))
                retry_delay = max(server_delay or 0.0, exponential) + random.uniform(0, 0.25)

        status = self._status_code(last_error) if last_error else None
        detail = f" (HTTP {status})" if status else ""
        error_type = type(last_error).__name__ if last_error else "unknown error"
        raise EmbeddingError(
            f"Gemini embedding request failed after bounded retries{detail}; "
            f"SDK error type: {error_type}."
        ) from last_error

    def embed_documents(
        self, texts: Sequence[str], titles: Sequence[str]
    ) -> list[list[float]]:
        if len(texts) != len(titles):
            raise EmbeddingError("Each document text must have one title.")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._config.batch_size):
            batch_texts = texts[start : start + self._config.batch_size]
            batch_titles = titles[start : start + self._config.batch_size]
            titled_texts = [
                f"Title: {title}\n\n{text}" for title, text in zip(batch_titles, batch_texts)
            ]
            vectors.extend(self._embed(titled_texts, "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query], "RETRIEVAL_QUERY")[0]

    def close(self) -> None:
        self._client.close()
