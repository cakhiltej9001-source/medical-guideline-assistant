"""Extract page-level text from a verified PDF and report extraction quality."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import pdfplumber


BOUNDARY_LINE_COUNT = 3
PAGE_NUMBER_EDGE_LINE_COUNT = 2
REPEATED_BOUNDARY_RATIO = 0.20
LOW_TEXT_CHARACTER_LIMIT = 150
PAGE_NUMBER_PATTERN = re.compile(r"^\d{1,4}$")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
TRAILING_PAGE_NUMBER_PATTERN = re.compile(r"\s+\d{1,4}$")
UNICODE_HYPHENS = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


class ExtractionError(RuntimeError):
    """Raised when a PDF cannot be verified or extracted safely."""


def normalize_line(line: str) -> str:
    """Normalize Unicode and horizontal spacing without joining layout lines."""
    normalized = unicodedata.normalize("NFKC", line).translate(UNICODE_HYPHENS)
    normalized = normalized.replace("\x00", "")
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def canonical_boundary_line(line: str) -> str:
    """Normalize variable trailing page numbers for footer frequency checks."""
    normalized = normalize_line(line)
    return TRAILING_PAGE_NUMBER_PATTERN.sub(" <PAGE_NUMBER>", normalized)


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_checksum(pdf_path: Path, metadata_path: Path) -> str:
    """Refuse extraction when the PDF differs from its download metadata."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = str(metadata["sha256"]).lower()
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"Could not read checksum metadata: {exc}") from exc

    actual = calculate_sha256(pdf_path)
    if actual != expected:
        raise ExtractionError(
            "PDF checksum does not match its provenance metadata; re-download the source."
        )
    return actual


def detect_repeated_boundary_lines(page_lines: Iterable[list[str]]) -> set[str]:
    """Find normalized lines repeatedly occurring near page edges."""
    pages = list(page_lines)
    nonempty_pages = [lines for lines in pages if any(line.strip() for line in lines)]
    if not nonempty_pages:
        return set()

    counts: Counter[str] = Counter()
    for lines in nonempty_pages:
        normalized = [canonical_boundary_line(line) for line in lines]
        boundary = normalized[:BOUNDARY_LINE_COUNT] + normalized[-BOUNDARY_LINE_COUNT:]
        counts.update(
            {
                line
                for line in boundary
                if line and not PAGE_NUMBER_PATTERN.fullmatch(line)
            }
        )

    threshold = max(3, math.ceil(len(nonempty_pages) * REPEATED_BOUNDARY_RATIO))
    return {line for line, count in counts.items() if count >= threshold}


def clean_page_lines(lines: list[str], repeated_boundary_lines: set[str]) -> str:
    """Remove page furniture while preserving semantic line boundaries."""
    normalized = [normalize_line(line) for line in lines]
    last_index = len(normalized) - 1
    kept: list[str] = []

    for index, line in enumerate(normalized):
        if not line:
            continue
        near_boundary = (
            index < BOUNDARY_LINE_COUNT
            or index > last_index - BOUNDARY_LINE_COUNT
        )
        if near_boundary and canonical_boundary_line(line) in repeated_boundary_lines:
            continue
        near_page_number_edge = (
            index < PAGE_NUMBER_EDGE_LINE_COUNT
            or index > last_index - PAGE_NUMBER_EDGE_LINE_COUNT
        )
        if near_page_number_edge and PAGE_NUMBER_PATTERN.fullmatch(line):
            continue
        kept.append(line)

    return "\n".join(kept).strip()


def classify_page_extraction(character_count: int, image_count: int) -> str:
    """Flag pages that need OCR or human review instead of silent acceptance."""
    if character_count < LOW_TEXT_CHARACTER_LIMIT and image_count > 0:
        return "image_review_required"
    if character_count == 0:
        return "empty"
    if character_count < LOW_TEXT_CHARACTER_LIMIT:
        return "low_text"
    return "ok"


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file_handle:
            for record in records:
                file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_pdf_pages(
    pdf_path: Path,
    metadata_path: Path,
    source_id: str,
    output_path: Path,
) -> dict[str, Any]:
    """Extract cleaned page records and return a quality report."""
    if not pdf_path.is_file():
        raise ExtractionError(f"PDF does not exist: {pdf_path}")

    checksum = verify_source_checksum(pdf_path, metadata_path)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            raw_pages = [(page.extract_text() or "").splitlines() for page in pdf.pages]
            image_counts = [len(page.images) for page in pdf.pages]
    except Exception as exc:
        raise ExtractionError(f"PDF text extraction failed: {exc}") from exc

    repeated_lines = detect_repeated_boundary_lines(raw_pages)
    records: list[dict[str, Any]] = []
    character_counts: list[int] = []
    empty_pages: list[int] = []
    low_text_pages: list[int] = []
    image_review_required_pages: list[int] = []

    for page_number, (lines, image_count) in enumerate(
        zip(raw_pages, image_counts, strict=True), start=1
    ):
        cleaned_text = clean_page_lines(lines, repeated_lines)
        character_count = len(cleaned_text)
        extraction_status = classify_page_extraction(character_count, image_count)
        character_counts.append(character_count)
        if extraction_status == "empty":
            empty_pages.append(page_number)
        elif extraction_status == "low_text":
            low_text_pages.append(page_number)
        elif extraction_status == "image_review_required":
            image_review_required_pages.append(page_number)
        records.append(
            {
                "source_id": source_id,
                "page_number": page_number,
                "text": cleaned_text,
                "character_count": character_count,
                "image_count": image_count,
                "extraction_status": extraction_status,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(records, output_path)

    nonempty_counts = [count for count in character_counts if count > 0]
    report = {
        "source_id": source_id,
        "source_sha256": checksum,
        "extractor": "pdfplumber",
        "extractor_version": pdfplumber.__version__,
        "pages_total": len(records),
        "pages_with_text": len(records) - len(empty_pages),
        "empty_pages": empty_pages,
        "low_text_pages": low_text_pages,
        "image_review_required_pages": image_review_required_pages,
        "characters_total": sum(character_counts),
        "characters_per_nonempty_page": {
            "minimum": min(nonempty_counts, default=0),
            "median": median(nonempty_counts) if nonempty_counts else 0,
            "maximum": max(nonempty_counts, default=0),
        },
        "removed_repeated_boundary_lines": sorted(repeated_lines),
        "output_path": str(output_path),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
