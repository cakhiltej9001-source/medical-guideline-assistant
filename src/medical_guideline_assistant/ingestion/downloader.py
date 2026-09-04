"""Download explicitly approved MOHFW PDF sources with provenance checks."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ALLOWED_HOSTS = frozenset(
    {
        "clinicalestablishments.mohfw.gov.in",
        "www.clinicalestablishments.mohfw.gov.in",
        "ncvbdc.mohfw.gov.in",
    }
)
MAX_PDF_BYTES = 25 * 1024 * 1024
CHUNK_BYTES = 64 * 1024


class SourceDownloadError(RuntimeError):
    """Raised when a source fails validation or cannot be downloaded safely."""


def validate_source_url(url: str) -> None:
    """Require HTTPS and an exact, allowlisted MOHFW hostname."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    if parsed.scheme != "https":
        raise SourceDownloadError("Source URL must use HTTPS.")
    if hostname not in ALLOWED_HOSTS:
        raise SourceDownloadError(f"Source hostname is not allowlisted: {hostname!r}")
    if parsed.username or parsed.password:
        raise SourceDownloadError("Source URL must not contain credentials.")
    if parsed.port not in (None, 443):
        raise SourceDownloadError("Source URL must use the standard HTTPS port.")


def build_http_session() -> requests.Session:
    """Create a session with bounded retries for temporary server failures."""
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "MedicalGuidelineAssistant-CourseProject/0.1"}
    )
    session.mount("https://", adapter)
    return session


def _validate_filename(filename: str) -> None:
    path = Path(filename)
    if path.name != filename or path.suffix.lower() != ".pdf":
        raise SourceDownloadError("Manifest filename must be a plain .pdf filename.")


def download_pdf(
    document: dict[str, Any],
    source_page: str,
    output_dir: Path,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Download one manifest document and write a provenance sidecar."""
    required = {"source_id", "title", "publisher", "pdf_url", "filename"}
    missing = sorted(required.difference(document))
    if missing:
        raise SourceDownloadError(f"Manifest entry is missing: {', '.join(missing)}")

    url = str(document["pdf_url"])
    filename = str(document["filename"])
    validate_source_url(url)
    _validate_filename(filename)

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    temporary = output_dir / f"{filename}.part"
    client = session or build_http_session()

    try:
        with client.get(url, stream=True, timeout=(5, 60), allow_redirects=True) as response:
            response.raise_for_status()

            for redirect in response.history:
                validate_source_url(redirect.url)
            validate_source_url(response.url)

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/pdf":
                raise SourceDownloadError(
                    f"Expected application/pdf, received {content_type or 'no content type'}.")

            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > MAX_PDF_BYTES:
                raise SourceDownloadError("PDF exceeds the configured 25 MB limit.")

            digest = hashlib.sha256()
            total_bytes = 0
            first_bytes = b""

            with temporary.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    if not chunk:
                        continue
                    if not first_bytes:
                        first_bytes = chunk[:4]
                    total_bytes += len(chunk)
                    if total_bytes > MAX_PDF_BYTES:
                        raise SourceDownloadError("PDF exceeded the 25 MB limit while downloading.")
                    digest.update(chunk)
                    file_handle.write(chunk)

            if first_bytes != b"%PDF":
                raise SourceDownloadError("Downloaded content does not have a PDF signature.")

            os.replace(temporary, destination)
            metadata = {
                "source_id": document["source_id"],
                "title": document["title"],
                "publisher": document["publisher"],
                "category": document.get("category"),
                "document_date": document.get("document_date"),
                "language": document.get("language"),
                "source_page": source_page,
                "original_url": url,
                "final_url": response.url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "content_type": content_type,
                "size_bytes": total_bytes,
                "sha256": digest.hexdigest(),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
            }

        metadata_path = destination.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return metadata
    except requests.RequestException as exc:
        raise SourceDownloadError(f"Network request failed: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
