"""SQLite FTS5/BM25 index, dense vectors, and hybrid rank fusion."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import RetrievalConfig
from .embeddings import EmbeddingProvider, normalize_vector


QUERY_TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)
QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "can",
        "for",
        "how",
        "in",
        "is",
        "of",
        "should",
        "the",
        "to",
        "what",
        "when",
        "which",
        "with",
    }
)


class RetrievalIndexError(RuntimeError):
    """Raised when an index cannot be built or queried safely."""


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    source_id: str
    title: str
    source_url: str
    pages: list[int]
    sections: list[str]
    safety_tags: list[str]
    text: str
    rrf_score: float
    lexical_rank: int | None
    dense_rank: int | None
    lexical_score: float | None
    dense_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vector_to_blob(vector: list[float]) -> bytes:
    return array("f", normalize_vector(vector)).tobytes()


def _blob_to_vector(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA foreign_keys=ON;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT,
            document_date TEXT,
            source_url TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            pages_json TEXT NOT NULL,
            sections_json TEXT NOT NULL,
            safety_tags_json TEXT NOT NULL,
            text TEXT NOT NULL,
            vector BLOB,
            vector_dim INTEGER
        );
        CREATE INDEX chunks_source_id_idx ON chunks(source_id);
        CREATE INDEX chunks_category_idx ON chunks(category);
        CREATE VIRTUAL TABLE chunk_fts USING fts5(
            chunk_id UNINDEXED,
            title,
            text,
            tokenize='porter unicode61 remove_diacritics 2'
        );
        """
    )


def _load_vector_cache(
    database_path: Path, model: str, dimensions: int
) -> dict[tuple[str, str], bytes]:
    cached: dict[tuple[str, str], bytes] = {}
    if database_path.exists():
        try:
            connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if (
                metadata.get("embedding_model") == model
                and int(metadata.get("embedding_dimensions", 0)) == dimensions
            ):
                cached.update(
                    {
                        (chunk_id, source_sha256): vector
                        for chunk_id, source_sha256, vector in connection.execute(
                            "SELECT chunk_id, source_sha256, vector FROM chunks "
                            "WHERE vector IS NOT NULL"
                        )
                    }
                )
        except (sqlite3.Error, ValueError):
            pass
        finally:
            if "connection" in locals():
                connection.close()

    cache_path = database_path.with_name("embedding_cache.sqlite3")
    if cache_path.exists():
        try:
            cache_connection = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
            cached.update(
                {
                    (chunk_id, source_sha256): vector
                    for chunk_id, source_sha256, vector in cache_connection.execute(
                        "SELECT chunk_id, source_sha256, vector FROM vectors "
                        "WHERE model = ? AND dimensions = ?",
                        (model, dimensions),
                    )
                }
            )
        except sqlite3.Error:
            pass
        finally:
            if "cache_connection" in locals():
                cache_connection.close()
    return cached


def _save_vector_cache_batch(
    database_path: Path,
    model: str,
    dimensions: int,
    rows: list[tuple[str, str, bytes]],
) -> None:
    """Checkpoint successful API batches so a later 429 can resume safely."""
    cache_path = database_path.with_name("embedding_cache.sqlite3")
    try:
        connection = sqlite3.connect(cache_path)
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY(model, dimensions, chunk_id, source_sha256)
                )
                """
            )
            connection.executemany(
                "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?, ?)",
                [
                    (model, dimensions, chunk_id, source_sha256, vector)
                    for chunk_id, source_sha256, vector in rows
                ],
            )
    except sqlite3.Error as exc:
        raise RetrievalIndexError(f"Could not checkpoint embedding cache: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()


def build_index(
    database_path: Path,
    records: list[dict[str, Any]],
    config: RetrievalConfig,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Atomically build an index, reusing unchanged vectors when possible."""
    if not records:
        raise RetrievalIndexError("Cannot build an index from an empty corpus.")
    if len({record["chunk_id"] for record in records}) != len(records):
        raise RetrievalIndexError("Chunk IDs must be globally unique.")
    if embedding_provider and embedding_provider.dimensions != config.embedding.dimensions:
        raise RetrievalIndexError("Embedding provider dimensions do not match config.")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_suffix(database_path.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    cache = _load_vector_cache(
        database_path, config.embedding.model, config.embedding.dimensions
    )
    vectors: dict[str, bytes] = {}
    cache_hits = 0
    missing: list[dict[str, Any]] = []
    for record in records:
        cached = cache.get((record["chunk_id"], record["source_sha256"]))
        if cached is None:
            missing.append(record)
        else:
            vectors[record["chunk_id"]] = cached
            cache_hits += 1

    completed_api_batches = 0
    if embedding_provider and missing:
        for start in range(0, len(missing), config.embedding.batch_size):
            batch = missing[start : start + config.embedding.batch_size]
            generated = embedding_provider.embed_documents(
                [record["text"] for record in batch],
                [record["title"] for record in batch],
            )
            if len(generated) != len(batch):
                raise RetrievalIndexError("Embedding provider returned an incomplete batch.")
            cache_rows: list[tuple[str, str, bytes]] = []
            for record, vector in zip(batch, generated):
                if len(vector) != config.embedding.dimensions:
                    raise RetrievalIndexError(
                        "Embedding vector dimension does not match config."
                    )
                blob = _vector_to_blob(vector)
                vectors[record["chunk_id"]] = blob
                cache_rows.append(
                    (record["chunk_id"], record["source_sha256"], blob)
                )
            _save_vector_cache_batch(
                database_path,
                config.embedding.model,
                config.embedding.dimensions,
                cache_rows,
            )
            completed_api_batches += 1

    try:
        connection = sqlite3.connect(temporary_path)
        _create_schema(connection)
        with connection:
            for record in records:
                vector = vectors.get(record["chunk_id"])
                connection.execute(
                    """
                    INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["chunk_id"],
                        record["source_id"],
                        record["title"],
                        record.get("category"),
                        record.get("document_date"),
                        record["source_url"],
                        record["source_sha256"],
                        json.dumps(record["pages"]),
                        json.dumps(record.get("sections", []), ensure_ascii=False),
                        json.dumps(record.get("safety_tags", [])),
                        record["text"],
                        vector,
                        config.embedding.dimensions if vector is not None else None,
                    ),
                )
                connection.execute(
                    "INSERT INTO chunk_fts(chunk_id, title, text) VALUES (?, ?, ?)",
                    (record["chunk_id"], record["title"], record["text"]),
                )
            metadata = {
                "retrieval_version": str(config.retrieval_version),
                "embedding_model": config.embedding.model,
                "embedding_dimensions": str(config.embedding.dimensions),
                "chunk_count": str(len(records)),
                "embedded_chunk_count": str(len(vectors)),
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RetrievalIndexError(f"SQLite integrity check failed: {integrity}")
        connection.close()
        database_path.with_name(database_path.name + "-wal").unlink(missing_ok=True)
        database_path.with_name(database_path.name + "-shm").unlink(missing_ok=True)
        temporary_path.replace(database_path)
    except (sqlite3.Error, OSError) as exc:
        raise RetrievalIndexError(f"Could not build retrieval index: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
        temporary_path.unlink(missing_ok=True)

    return {
        "database_path": str(database_path),
        "chunks": len(records),
        "embedded_chunks": len(vectors),
        "embedding_cache_hits": cache_hits,
        "embedding_cache_misses": len(missing),
        "estimated_api_batches": (
            math.ceil(len(missing) / config.embedding.batch_size)
            if embedding_provider
            else 0
        ),
        "completed_api_batches": completed_api_batches,
        "mode": "hybrid" if len(vectors) == len(records) else "lexical_only",
    }


def _validate_query(query: str) -> str:
    normalized = " ".join(query.split())
    if not normalized:
        raise RetrievalIndexError("Search query cannot be empty.")
    if len(normalized) > 500:
        raise RetrievalIndexError("Search query exceeds the 500-character guardrail.")
    return normalized


def _fts_query(query: str) -> str:
    all_tokens = QUERY_TOKEN_PATTERN.findall(query.casefold())
    tokens = [token for token in all_tokens if token not in QUERY_STOPWORDS]
    if not tokens:
        tokens = all_tokens
    if not tokens:
        raise RetrievalIndexError("Search query contains no searchable words.")
    return " OR ".join(f'"{token}"' for token in tokens[:30])


def _source_filter(source_ids: tuple[str, ...] | None) -> tuple[str, list[str]]:
    if not source_ids:
        return "", []
    placeholders = ",".join("?" for _ in source_ids)
    return f" AND c.source_id IN ({placeholders})", list(source_ids)


def _lexical_candidates(
    connection: sqlite3.Connection,
    query: str,
    limit: int,
    source_ids: tuple[str, ...] | None,
) -> list[tuple[str, float]]:
    filter_sql, filter_values = _source_filter(source_ids)
    rows = connection.execute(
        f"""
        SELECT c.chunk_id, bm25(chunk_fts, 0.0, 1.0, 1.0) AS score
        FROM chunk_fts JOIN chunks AS c ON c.chunk_id = chunk_fts.chunk_id
        WHERE chunk_fts MATCH ? {filter_sql}
        ORDER BY score ASC LIMIT ?
        """,
        [_fts_query(query), *filter_values, limit],
    ).fetchall()
    return [(row["chunk_id"], -float(row["score"])) for row in rows]


def _dense_candidates(
    connection: sqlite3.Connection,
    query_vector: list[float],
    limit: int,
    source_ids: tuple[str, ...] | None,
) -> list[tuple[str, float]]:
    vector = normalize_vector(query_vector)
    filter_sql, filter_values = _source_filter(source_ids)
    rows = connection.execute(
        f"""
        SELECT c.chunk_id, c.vector, c.vector_dim FROM chunks AS c
        WHERE c.vector IS NOT NULL {filter_sql}
        """,
        filter_values,
    ).fetchall()
    scored: list[tuple[str, float]] = []
    for row in rows:
        candidate = _blob_to_vector(row["vector"])
        if len(candidate) != len(vector) or row["vector_dim"] != len(vector):
            raise RetrievalIndexError("Stored vector dimensions are inconsistent.")
        score = sum(left * right for left, right in zip(vector, candidate))
        if math.isfinite(score):
            scored.append((row["chunk_id"], score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


def search_index(
    database_path: Path,
    query: str,
    config: RetrievalConfig,
    embedding_provider: EmbeddingProvider | None = None,
    source_ids: tuple[str, ...] | None = None,
) -> list[SearchResult]:
    """Retrieve BM25 and dense candidates, then fuse their ranks."""
    normalized_query = _validate_query(query)
    if not database_path.exists():
        raise RetrievalIndexError(f"Index does not exist: {database_path}")
    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        lexical = _lexical_candidates(
            connection,
            normalized_query,
            config.search.bm25_candidates,
            source_ids,
        )
        dense: list[tuple[str, float]] = []
        if embedding_provider is not None:
            dense = _dense_candidates(
                connection,
                embedding_provider.embed_query(normalized_query),
                config.search.dense_candidates,
                source_ids,
            )

        lexical_map = {
            chunk_id: (rank, score)
            for rank, (chunk_id, score) in enumerate(lexical, start=1)
        }
        dense_map = {
            chunk_id: (rank, score)
            for rank, (chunk_id, score) in enumerate(dense, start=1)
        }
        candidate_ids = set(lexical_map).union(dense_map)
        fused = []
        for chunk_id in candidate_ids:
            score = 0.0
            if chunk_id in lexical_map:
                score += 1 / (config.search.rrf_constant + lexical_map[chunk_id][0])
            if chunk_id in dense_map:
                score += 1 / (config.search.rrf_constant + dense_map[chunk_id][0])
            fused.append((chunk_id, score))
        fused.sort(key=lambda item: (-item[1], item[0]))

        results: list[SearchResult] = []
        for chunk_id, rrf_score in fused[: config.search.final_results]:
            row = connection.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            lexical_rank, lexical_score = lexical_map.get(chunk_id, (None, None))
            dense_rank, dense_score = dense_map.get(chunk_id, (None, None))
            results.append(
                SearchResult(
                    chunk_id=row["chunk_id"],
                    source_id=row["source_id"],
                    title=row["title"],
                    source_url=row["source_url"],
                    pages=json.loads(row["pages_json"]),
                    sections=json.loads(row["sections_json"]),
                    safety_tags=json.loads(row["safety_tags_json"]),
                    text=row["text"],
                    rrf_score=rrf_score,
                    lexical_rank=lexical_rank,
                    dense_rank=dense_rank,
                    lexical_score=lexical_score,
                    dense_score=dense_score,
                )
            )
        return results
    except sqlite3.Error as exc:
        raise RetrievalIndexError(f"Could not query retrieval index: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
