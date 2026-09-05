"""Configuration for grounded answer generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class GenerationConfigError(ValueError):
    """Raised when generation configuration is invalid."""


@dataclass(frozen=True)
class GenerationConfig:
    generation_version: int
    provider: str
    model: str
    temperature: float
    maximum_output_tokens: int
    timeout_seconds: int
    maximum_attempts: int
    maximum_validation_attempts: int
    maximum_context_chunks: int
    minimum_claim_token_overlap: float
    standard_disclaimer: str
    minimum_claim_support_score: float = 0.20

    @classmethod
    def from_path(cls, path: Path) -> "GenerationConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            config = cls(**raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GenerationConfigError(f"Could not load generation config: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if self.provider != "gemini":
            raise GenerationConfigError("Only Gemini generation is configured.")
        if not self.model.startswith("gemini-"):
            raise GenerationConfigError("Generation model must be a Gemini model.")
        if not 0 <= self.temperature <= 1:
            raise GenerationConfigError("Temperature must be between 0 and 1.")
        if not 128 <= self.maximum_output_tokens <= 4096:
            raise GenerationConfigError("Maximum output tokens must be 128 to 4096.")
        if self.timeout_seconds <= 0 or not 1 <= self.maximum_attempts <= 5:
            raise GenerationConfigError("Timeout or maximum attempts is invalid.")
        if not 1 <= self.maximum_validation_attempts <= 3:
            raise GenerationConfigError("Maximum validation attempts must be 1 to 3.")
        if not 1 <= self.maximum_context_chunks <= 10:
            raise GenerationConfigError("Maximum context chunks must be 1 to 10.")
        if not 0 <= self.minimum_claim_token_overlap <= 1:
            raise GenerationConfigError("Claim overlap threshold must be between 0 and 1.")
        if not 0 <= self.minimum_claim_support_score <= 1:
            raise GenerationConfigError("Claim support threshold must be between 0 and 1.")
        if not self.standard_disclaimer.strip():
            raise GenerationConfigError("A standard disclaimer is required.")
