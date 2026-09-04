# Milestone 3: Hybrid Retrieval Design

## Goal

Retrieve guideline passages before asking a language model to write an answer. The
retriever returns chunk IDs, source URLs, and exact PDF pages so later generation
can cite evidence and refuse when evidence is weak.

## Why SQLite

SQLite is free, local, reproducible, and sufficient for the current 443-chunk
corpus. FTS5 provides Porter-stemmed BM25 keyword search. Embedding vectors are
stored in the same database, and an exact cosine scan is acceptable at this size.
If the corpus grows to tens of thousands of chunks, the vector layer can be
replaced without changing the retrieval interface.

## Hybrid ranking

1. BM25 retrieves up to 20 lexical candidates.
2. Gemini retrieves up to 20 semantically similar candidates.
3. Reciprocal-rank fusion combines ranks using `1 / (60 + rank)`.
4. The top 10 fused chunks continue to later confidence scoring and reranking.

If the query-embedding request fails after bounded retries, retrieval degrades to
the local BM25 index. The response is marked `lexical_fallback` so the interface
can disclose the reduced retrieval mode while still applying normal citation and
output validation.

Rank fusion is used because BM25 and cosine scores are on different scales.

## Gemini embedding choice

The current text-only index uses `gemini-embedding-001`, 768 dimensions, and the
`RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY` task types. Chunk inputs are below the
model's text limit. Titles are prefixed to document text to improve retrieval.

The provider batches 16 chunks, spaces calls by at least 15 seconds, makes at most
three attempts for retryable API errors, uses a 30-second timeout, and never prints
the API key. Each successful batch is checkpointed by chunk ID, source hash, model,
and dimension, so rate-limit retries resume without resending unchanged chunks.

## Local commands

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Build and test BM25 without an API key:

```powershell
python scripts/build_index.py
python scripts/evaluate_retrieval.py
```

After creating a Google AI Studio key, copy the ignored template and edit `.env`
locally. Do not paste the key into chat or source code:

```powershell
Copy-Item .env.example .env
notepad .env
python scripts/build_index.py --with-embeddings
python scripts/evaluate_retrieval.py --hybrid
```

Inspect results without generating a medical answer:

```powershell
python scripts/query_index.py "What are the warning signs of dengue?" --hybrid
```

The process environment is also supported. If it is used, remove it when finished:

```powershell
Remove-Item Env:GEMINI_API_KEY
```

## Guardrails and current metrics

- Only the three audited public guidelines are embedded; no patient data is
  accepted or stored.
- Queries longer than 500 characters are rejected before retrieval.
- Metadata filters can restrict results to exact approved source IDs.
- The lexical baseline reaches Recall@5 = 1.00 and MRR@5 = 0.575 on six manually
  curated smoke questions. Hybrid retrieval preserves Recall@5 = 1.00 and raises
  MRR@5 to 0.833. This set is too small for a production claim.
- Before deployment, expand the evaluation set, measure latency and API failures,
  and choose a confidence/refusal threshold on held-out questions.
