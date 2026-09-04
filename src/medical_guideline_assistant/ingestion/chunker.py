"""Create section-aware, citation-ready chunks from extracted PDF pages."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
NUMBERED_PARAGRAPH_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)+\.?|\([a-z0-9ivx]+\)|[•●▪])\s+",
    flags=re.IGNORECASE,
)
NUMBERED_HEADING_PATTERN = re.compile(r"^\d+(?:\.\d+)*\s*:\s+\S")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
DOSAGE_AMOUNT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml|units?|iu)\b",
    flags=re.IGNORECASE,
)
DOSAGE_LANGUAGE_PATTERN = re.compile(
    r"\b(?:doses?|dosages?|dosing|titration|once daily|twice daily)\b",
    flags=re.IGNORECASE,
)
EMERGENCY_LANGUAGE_PATTERN = re.compile(
    r"\b(?:emergency|emergencies|life-threatening|urgent referral)\b",
    flags=re.IGNORECASE,
)


class ChunkingError(RuntimeError):
    """Raised when page records or chunking configuration are invalid."""


@dataclass(frozen=True)
class ChunkingConfig:
    chunker_version: int
    minimum_estimated_tokens: int
    target_estimated_tokens: int
    maximum_estimated_tokens: int
    overlap_estimated_tokens: int
    skip_extraction_statuses: frozenset[str]

    @classmethod
    def from_path(cls, path: Path) -> "ChunkingConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            config = cls(
                chunker_version=int(raw["chunker_version"]),
                minimum_estimated_tokens=int(raw["minimum_estimated_tokens"]),
                target_estimated_tokens=int(raw["target_estimated_tokens"]),
                maximum_estimated_tokens=int(raw["maximum_estimated_tokens"]),
                overlap_estimated_tokens=int(raw["overlap_estimated_tokens"]),
                skip_extraction_statuses=frozenset(raw["skip_extraction_statuses"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ChunkingError(f"Could not load chunking configuration: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not (
            0 < self.minimum_estimated_tokens
            <= self.target_estimated_tokens
            <= self.maximum_estimated_tokens
        ):
            raise ChunkingError("Chunk token limits must satisfy 0 < min <= target <= max.")
        if not 0 <= self.overlap_estimated_tokens < self.minimum_estimated_tokens:
            raise ChunkingError("Chunk overlap must be non-negative and smaller than min.")


@dataclass(frozen=True)
class TextUnit:
    text: str
    page_start: int
    page_end: int
    section: str
    estimated_tokens: int
    break_before: bool = False
    is_heading: bool = False


def estimate_tokens(text: str) -> int:
    """Return a conservative tokenizer-independent estimate for configuration."""
    lexical_tokens = len(TOKEN_PATTERN.findall(text))
    return math.ceil(lexical_tokens * 1.15)


def is_section_heading(line: str) -> bool:
    stripped = line.strip()
    if not 3 <= len(stripped) <= 140:
        return False
    lowered = stripped.lower()
    if lowered.startswith(
        (
            "section ",
            "annexure",
            "key recommendation",
            "full guideline",
            "background document",
            "table of contents",
        )
    ):
        return True
    if NUMBERED_HEADING_PATTERN.match(stripped):
        return True
    letters = [character for character in stripped if character.isalpha()]
    return (
        len(letters) >= 8
        and sum(character.isupper() for character in letters) / len(letters) >= 0.85
    )


def is_hard_section_heading(line: str) -> bool:
    """Return true for major document boundaries that must not share a chunk."""
    return line.strip().lower().startswith(("section ", "annexure", "full guideline"))


def _split_long_text(
    text: str,
    page_start: int,
    page_end: int,
    section: str,
    maximum_tokens: int,
    break_before: bool,
) -> list[TextUnit]:
    """Split unusually long sentences using words while respecting the hard limit."""
    words = text.split()
    units: list[TextUnit] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word])
        if current and estimate_tokens(candidate) > maximum_tokens:
            unit_text = " ".join(current)
            units.append(
                TextUnit(
                    text=unit_text,
                    page_start=page_start,
                    page_end=page_end,
                    section=section,
                    estimated_tokens=estimate_tokens(unit_text),
                    break_before=break_before if not units else False,
                )
            )
            current = [word]
        else:
            current.append(word)

    if current:
        unit_text = " ".join(current)
        units.append(
            TextUnit(
                text=unit_text,
                page_start=page_start,
                page_end=page_end,
                section=section,
                estimated_tokens=estimate_tokens(unit_text),
                break_before=break_before if not units else False,
            )
        )
    return units


def _paragraph_to_units(
    lines: list[str],
    page_start: int,
    page_end: int,
    section: str,
    maximum_tokens: int,
) -> list[TextUnit]:
    text = " ".join(lines).strip()
    if not text:
        return []

    sentences = [part.strip() for part in SENTENCE_BOUNDARY_PATTERN.split(text) if part.strip()]
    units: list[TextUnit] = []
    for sentence_index, sentence in enumerate(sentences):
        units.extend(
            _split_long_text(
                text=sentence,
                page_start=page_start,
                page_end=page_end,
                section=section,
                maximum_tokens=maximum_tokens,
                break_before=sentence_index == 0,
            )
        )
    return units


def page_records_to_units(
    page_records: Iterable[dict[str, Any]],
    config: ChunkingConfig,
    excluded_pages: frozenset[int] = frozenset(),
) -> tuple[list[TextUnit], list[int]]:
    """Convert physical lines into heading-aware sentence units."""
    units: list[TextUnit] = []
    skipped_pages: list[int] = []
    current_section = "Document overview"
    paragraph_lines: list[str] = []
    paragraph_page_start = 0
    paragraph_page_end = 0

    def flush_paragraph() -> None:
        nonlocal paragraph_lines, paragraph_page_start, paragraph_page_end
        units.extend(
            _paragraph_to_units(
                lines=paragraph_lines,
                page_start=paragraph_page_start,
                page_end=paragraph_page_end,
                section=current_section,
                maximum_tokens=config.maximum_estimated_tokens,
            )
        )
        paragraph_lines = []
        paragraph_page_start = 0
        paragraph_page_end = 0

    for record in page_records:
        page_number = int(record["page_number"])
        status = str(record.get("extraction_status", "ok"))
        if page_number in excluded_pages or status in config.skip_extraction_statuses:
            flush_paragraph()
            skipped_pages.append(page_number)
            continue

        for raw_line in str(record.get("text", "")).splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if is_section_heading(line):
                flush_paragraph()
                current_section = line
                units.append(
                    TextUnit(
                        text=line,
                        page_start=page_number,
                        page_end=page_number,
                        section=current_section,
                        estimated_tokens=estimate_tokens(line),
                        break_before=True,
                        is_heading=True,
                    )
                )
                continue

            if paragraph_lines and NUMBERED_PARAGRAPH_PATTERN.match(line):
                flush_paragraph()

            if not paragraph_lines:
                paragraph_page_start = page_number
            paragraph_page_end = page_number
            paragraph_lines.append(line)

            if line.endswith((".", "?", "!")) and sum(map(len, paragraph_lines)) >= 120:
                flush_paragraph()

    flush_paragraph()
    return units, skipped_pages


def _unit_token_total(units: list[TextUnit]) -> int:
    return sum(unit.estimated_tokens for unit in units)


def _overlap_tail(units: list[TextUnit], overlap_tokens: int) -> list[TextUnit]:
    if overlap_tokens == 0:
        return []
    tail: list[TextUnit] = []
    token_total = 0
    for unit in reversed(units):
        tail.insert(0, unit)
        token_total += unit.estimated_tokens
        if token_total >= overlap_tokens:
            break
    return tail


def pack_units(units: list[TextUnit], config: ChunkingConfig) -> list[list[TextUnit]]:
    """Pack units into bounded chunks with a trailing-unit overlap."""
    packed: list[list[TextUnit]] = []
    current: list[TextUnit] = []

    for unit in units:
        current_tokens = _unit_token_total(current)
        projected_tokens = current_tokens + unit.estimated_tokens
        crosses_page_gap = bool(current) and unit.page_start > max(
            current_unit.page_end for current_unit in current
        ) + 1
        current_has_body = any(not current_unit.is_heading for current_unit in current)
        starts_new_section = unit.is_heading and (
            current_tokens >= config.minimum_estimated_tokens
            or (is_hard_section_heading(unit.text) and current_has_body)
        )

        if crosses_page_gap:
            packed.append(current)
            current = []
        elif current and starts_new_section:
            packed.append(current)
            current = []
        elif current and (
                projected_tokens > config.target_estimated_tokens
                and current_tokens >= config.minimum_estimated_tokens
        ):
            packed.append(current)
            current = _overlap_tail(current, config.overlap_estimated_tokens)

        while current and (
            _unit_token_total(current) + unit.estimated_tokens
            > config.maximum_estimated_tokens
        ):
            current.pop(0)

        if unit.estimated_tokens > config.maximum_estimated_tokens:
            raise ChunkingError("A text unit exceeds the configured hard token limit.")
        current.append(unit)

    if current:
        if packed and _unit_token_total(current) < config.minimum_estimated_tokens:
            previous = packed[-1]
            overlap_length = 0
            maximum_overlap = min(len(previous), len(current))
            for candidate_length in range(maximum_overlap, 0, -1):
                if previous[-candidate_length:] == current[:candidate_length]:
                    overlap_length = candidate_length
                    break
            merged = previous + current[overlap_length:]
            if _unit_token_total(merged) <= config.maximum_estimated_tokens:
                packed[-1] = merged
            else:
                packed.append(current)
        else:
            packed.append(current)
    return packed


def _render_chunk_text(units: list[TextUnit]) -> str:
    rendered = ""
    for unit in units:
        separator = "\n\n" if unit.break_before and rendered else " " if rendered else ""
        rendered += separator + unit.text
    return rendered.strip()


def detect_safety_tags(text: str) -> list[str]:
    tags: list[str] = []
    if DOSAGE_AMOUNT_PATTERN.search(text) or DOSAGE_LANGUAGE_PATTERN.search(text):
        tags.append("dosage_content")
    if EMERGENCY_LANGUAGE_PATTERN.search(text):
        tags.append("emergency_content")
    return tags


def _stable_chunk_id(source_id: str, page_start: int, page_end: int, text: str) -> str:
    identity = f"{source_id}|{page_start}|{page_end}|{text}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:16]
    return f"{source_id}:p{page_start:04d}-p{page_end:04d}:{digest}"


def build_chunk_records(
    page_records: list[dict[str, Any]],
    document: dict[str, Any],
    source_metadata: dict[str, Any],
    config: ChunkingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build citation-ready chunk records plus an audit report."""
    all_page_numbers = {int(record["page_number"]) for record in page_records}
    manually_excluded = {
        int(page_number) for page_number in document.get("exclude_pages_from_index", [])
    }
    include_page_ranges = document.get("include_page_ranges", [])
    if include_page_ranges:
        included_pages: set[int] = set()
        try:
            for page_start, page_end in include_page_ranges:
                page_start = int(page_start)
                page_end = int(page_end)
                if page_start <= 0 or page_end < page_start:
                    raise ValueError("invalid page range")
                included_pages.update(range(page_start, page_end + 1))
        except (TypeError, ValueError) as exc:
            raise ChunkingError(f"Invalid include_page_ranges: {include_page_ranges}") from exc
        manually_excluded.update(all_page_numbers.difference(included_pages))
    manually_excluded_pages = frozenset(manually_excluded)
    units, skipped_pages = page_records_to_units(
        page_records, config, excluded_pages=manually_excluded_pages
    )
    packed_chunks = pack_units(units, config)
    records: list[dict[str, Any]] = []

    for chunk_units in packed_chunks:
        text = _render_chunk_text(chunk_units)
        page_start = min(unit.page_start for unit in chunk_units)
        page_end = max(unit.page_end for unit in chunk_units)
        pages = sorted(
            {
                page_number
                for unit in chunk_units
                for page_number in range(unit.page_start, unit.page_end + 1)
            }
        )
        sections = list(dict.fromkeys(unit.section for unit in chunk_units))
        estimated_token_count = estimate_tokens(text)
        if estimated_token_count > config.maximum_estimated_tokens:
            raise ChunkingError("Rendered chunk exceeds the configured hard token limit.")

        records.append(
            {
                "chunk_id": _stable_chunk_id(
                    document["source_id"], page_start, page_end, text
                ),
                "source_id": document["source_id"],
                "title": document["title"],
                "publisher": document["publisher"],
                "category": document.get("category"),
                "document_date": document.get("document_date"),
                "language": document.get("language"),
                "source_url": source_metadata["final_url"],
                "source_sha256": source_metadata["sha256"],
                "page_start": page_start,
                "page_end": page_end,
                "pages": pages,
                "sections": sections,
                "estimated_tokens": estimated_token_count,
                "safety_tags": detect_safety_tags(text),
                "text": text,
            }
        )

    token_counts = [record["estimated_tokens"] for record in records]
    tag_counts: dict[str, int] = {}
    for record in records:
        for tag in record["safety_tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    report = {
        "source_id": document["source_id"],
        "chunker_version": config.chunker_version,
        "source_sha256": source_metadata["sha256"],
        "chunks_total": len(records),
        "estimated_tokens": {
            "minimum": min(token_counts, default=0),
            "median": median(token_counts) if token_counts else 0,
            "maximum": max(token_counts, default=0),
            "below_configured_minimum": sum(
                count < config.minimum_estimated_tokens for count in token_counts
            ),
            "above_configured_maximum": sum(
                count > config.maximum_estimated_tokens for count in token_counts
            ),
        },
        "overlap_estimated_tokens": config.overlap_estimated_tokens,
        "skipped_pages": skipped_pages,
        "manually_excluded_pages": sorted(manually_excluded_pages),
        "include_page_ranges": include_page_ranges,
        "safety_tag_counts": tag_counts,
        "corpus_version": (
            f"{document['source_id']}-{source_metadata['sha256'][:12]}-"
            f"chunker-v{config.chunker_version}"
        ),
    }
    return records, report


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            return [json.loads(line) for line in file_handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ChunkingError(f"Could not read page records: {exc}") from exc


def write_chunk_outputs(
    records: list[dict[str, Any]], report: dict[str, Any], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file_handle:
            for record in records:
                file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
