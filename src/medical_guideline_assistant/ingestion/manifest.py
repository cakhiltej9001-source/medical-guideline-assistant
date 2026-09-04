"""Load and validate the curated source manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ManifestError(RuntimeError):
    """Raised when a curated source manifest is invalid or unreadable."""


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a source manifest and reject duplicate source identifiers."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read source manifest: {exc}") from exc

    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ManifestError("Manifest must contain a non-empty documents list.")

    source_ids = [document.get("source_id") for document in documents]
    if None in source_ids or len(source_ids) != len(set(source_ids)):
        raise ManifestError("Every manifest document needs a unique source_id.")

    return manifest
