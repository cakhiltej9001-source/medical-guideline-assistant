"""Evidence-strength labels, deliberately not probabilities of medical accuracy."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EvidenceConfidence:
    level: str
    score: float | None
    reason: str


def confidence_from_scores(scores: list[float | None]) -> EvidenceConfidence:
    if not scores or any(score is None or not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        return EvidenceConfidence("not_assessed", None, "Complete relevance and support scores are unavailable.")
    weakest = min(scores)
    level = "high" if weakest >= 0.8 else "moderate" if weakest >= 0.5 else "low"
    return EvidenceConfidence(level, weakest, "Minimum cited-evidence relevance and claim-support score; an uncalibrated heuristic, not medical certainty.")
