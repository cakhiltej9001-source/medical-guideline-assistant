"""Validate a complete chunk corpus before building retrieval indexes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .chunker import ChunkingConfig, estimate_tokens


REQUIRED_FIELDS = frozenset(
    {
        "chunk_id",
        "source_id",
        "title",
        "source_url",
        "source_sha256",
        "page_start",
        "page_end",
        "pages",
        "estimated_tokens",
        "safety_tags",
        "text",
    }
)
ALLOWED_SAFETY_TAGS = frozenset({"dosage_content", "emergency_content"})


class CorpusAuditError(RuntimeError):
    """Raised when corpus inputs cannot be read or validated."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusAuditError(f"Could not read {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            return [json.loads(line) for line in file_handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusAuditError(f"Could not read {path}: {exc}") from exc


def _allowed_pages(document: dict[str, Any]) -> set[int] | None:
    ranges = document.get("include_page_ranges", [])
    if not ranges:
        return None
    pages: set[int] = set()
    for start, end in ranges:
        pages.update(range(int(start), int(end) + 1))
    return pages


def audit_corpus(
    documents: list[dict[str, Any]],
    records_by_source: dict[str, list[dict[str, Any]]],
    metadata_by_source: dict[str, dict[str, Any]],
    config: ChunkingConfig,
) -> dict[str, Any]:
    """Return a serializable integrity report for all configured sources."""
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    aggregate_tags: Counter[str] = Counter()
    source_reports: list[dict[str, Any]] = []
    total_chunks = 0
    below_minimum = 0

    configured_ids = {str(document["source_id"]) for document in documents}
    unexpected = sorted(set(records_by_source).difference(configured_ids))
    if unexpected:
        errors.append(f"Unexpected chunk sources: {unexpected}")

    for document in documents:
        source_id = str(document["source_id"])
        records = records_by_source.get(source_id)
        metadata = metadata_by_source.get(source_id)
        source_errors_before = len(errors)
        source_warnings_before = len(warnings)

        if records is None:
            errors.append(f"{source_id}: chunk file is missing")
            records = []
        if metadata is None:
            errors.append(f"{source_id}: source metadata is missing")
            metadata = {}
        if not records:
            errors.append(f"{source_id}: no chunks are available")

        excluded = {int(page) for page in document.get("exclude_pages_from_index", [])}
        allowed = _allowed_pages(document)
        source_tags: Counter[str] = Counter()
        source_below_minimum = 0

        for index, record in enumerate(records, start=1):
            label = f"{source_id} chunk {index}"
            missing = sorted(REQUIRED_FIELDS.difference(record))
            if missing:
                errors.append(f"{label}: missing fields {missing}")
                continue

            chunk_id = str(record["chunk_id"])
            if chunk_id in seen_ids:
                errors.append(f"{label}: duplicate chunk_id {chunk_id}")
            seen_ids.add(chunk_id)

            if record["source_id"] != source_id:
                errors.append(f"{label}: source_id does not match manifest")
            if record["title"] != document["title"]:
                errors.append(f"{label}: title does not match manifest")
            if record["source_url"] != metadata.get("final_url"):
                errors.append(f"{label}: source URL does not match download metadata")
            if record["source_sha256"] != metadata.get("sha256"):
                errors.append(f"{label}: source hash does not match download metadata")

            text = str(record["text"]).strip()
            if not text:
                errors.append(f"{label}: text is empty")
            actual_tokens = estimate_tokens(text)
            if record["estimated_tokens"] != actual_tokens:
                errors.append(f"{label}: estimated token count is stale")
            if actual_tokens > config.maximum_estimated_tokens:
                errors.append(f"{label}: exceeds maximum token limit")
            if actual_tokens < config.minimum_estimated_tokens:
                source_below_minimum += 1

            try:
                pages = [int(page) for page in record["pages"]]
                page_start = int(record["page_start"])
                page_end = int(record["page_end"])
            except (TypeError, ValueError):
                errors.append(f"{label}: page citation metadata is invalid")
                continue
            if not pages or pages != sorted(set(pages)):
                errors.append(f"{label}: pages must be a non-empty sorted unique list")
            elif page_start != pages[0] or page_end != pages[-1]:
                errors.append(f"{label}: page_start/page_end do not match pages")
            leaked_exclusions = sorted(set(pages).intersection(excluded))
            if leaked_exclusions:
                errors.append(f"{label}: contains excluded pages {leaked_exclusions}")
            if allowed is not None:
                leaked_range = sorted(set(pages).difference(allowed))
                if leaked_range:
                    errors.append(f"{label}: pages outside include ranges {leaked_range}")

            tags = set(record["safety_tags"])
            unknown_tags = sorted(tags.difference(ALLOWED_SAFETY_TAGS))
            if unknown_tags:
                errors.append(f"{label}: unknown safety tags {unknown_tags}")
            source_tags.update(tags)

        if source_below_minimum:
            warnings.append(
                f"{source_id}: {source_below_minimum} chunks are below the soft minimum"
            )
        aggregate_tags.update(source_tags)
        total_chunks += len(records)
        below_minimum += source_below_minimum
        source_reports.append(
            {
                "source_id": source_id,
                "chunks": len(records),
                "unique_chunk_ids": len({r.get("chunk_id") for r in records}),
                "below_soft_minimum": source_below_minimum,
                "safety_tag_counts": dict(sorted(source_tags.items())),
                "errors": len(errors) - source_errors_before,
                "warnings": len(warnings) - source_warnings_before,
            }
        )

    return {
        "status": "passed" if not errors else "failed",
        "sources": len(documents),
        "chunks": total_chunks,
        "unique_chunk_ids": len(seen_ids),
        "below_soft_minimum": below_minimum,
        "safety_tag_counts": dict(sorted(aggregate_tags.items())),
        "source_reports": source_reports,
        "errors": errors,
        "warnings": warnings,
    }


def write_audit_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
