"""Small, deterministic session rate limiter for the demonstration UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    timestamps: list[float]
    retry_after_seconds: int


def consume_request(
    timestamps: list[float],
    now: float,
    maximum_requests: int = 5,
    window_seconds: int = 60,
) -> RateLimitDecision:
    """Apply a sliding-window limit and return the pruned timestamp history."""
    if maximum_requests < 1 or window_seconds < 1:
        raise ValueError("Rate-limit settings must be positive.")
    recent = sorted(
        timestamp
        for timestamp in timestamps
        if 0 <= timestamp <= now and now - timestamp < window_seconds
    )
    if len(recent) >= maximum_requests:
        retry_after = max(1, int(window_seconds - (now - recent[0]) + 0.999))
        return RateLimitDecision(False, recent, retry_after)
    recent.append(now)
    return RateLimitDecision(True, recent, 0)
