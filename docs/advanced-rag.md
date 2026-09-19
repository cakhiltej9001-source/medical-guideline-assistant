# HyDE, corrective retrieval, refined prompts, and confidence

## What changed

The answer path now uses optional HyDE and corrective retrieval before generation.
Both are enabled in `configs/retrieval.json`. The source corpus, embedding model,
and index format are unchanged, so the index does not need to be rebuilt.

## HyDE: hypothetical document embeddings

After the input safety check, Gemini drafts one short hypothetical guideline
passage. It is limited to 1,200 characters and embedded as a document. Search
combines three ranked lists using reciprocal-rank fusion: BM25 for the original
query, dense retrieval for the original query, and dense retrieval for the
hypothetical passage.

Only real indexed chunks are returned. The synthetic passage is never displayed,
logged, indexed, cited, or passed to answer generation as evidence. This adapts
[Gao et al.'s HyDE](https://aclanthology.org/2023.acl-long.99/) to Gemini and the
existing hybrid search. A generated passage can be wrong or off-topic, so the
original question is still used for reranking and validation. If HyDE generation
or embedding fails, the pipeline falls back to the original retrieval path.

## Corpus-only CRAG adaptation

The cross-encoder evaluates retrieval against the original question. When the
top score is below `0.65`, or no result is found, the pipeline makes one correction:

1. Gemini creates a conservative query rewrite; the application rechecks its
   length and safety.
2. Candidate pools expand from 20 to 40 per retriever and fused results from 10
   to 20, using the same corpus and source filters. If rewriting fails, the
   original query is broadened instead. If embeddings fail, local BM25 is used.
3. New candidates are reranked against the original question in bounded batches,
   deduplicated with the initial results, and reduced to the best ten.
4. Passages scoring below `0.20` are removed. If none remain, the application
   returns its insufficient-evidence refusal.

Strong initial matches skip the corrective search but still undergo weak-passage
filtering. A reranker failure fails closed, and there is no correction loop.

This follows the evaluate/correct/refine idea from
[Yan et al.'s CRAG](https://arxiv.org/abs/2401.15884), adapted to the assignment's
approved-source restriction. It does not reproduce the paper's external web
search or sentence-strip decomposition. Filtering complete chunks preserves page
citations and surrounding context.

## Prompt refinement

The generation prompt requests concise atomic claims, preserves question
qualifiers, avoids unrelated material, and abstains on unsupported or conflicting
evidence. It identifies chunks with source IDs, titles, pages, and chunk IDs. A
failed validation invokes a distinct repair prompt that requests fewer qualitative
claims and removes numeric clinical content that could violate output guardrails.
The existing maximum of two drafts and all validators remain in place. Gemini
does not generate the confidence labels.

## Evidence confidence on every response and claim

For each accepted claim, application code takes the minimum of its cited chunks'
query-relevance scores and its claim-to-evidence support score. Overall answer
confidence uses the weakest claim score, so a strong claim cannot hide a weak one.

| Label | Rule |
| --- | --- |
| High | Minimum score is at least 0.80 |
| Moderate | Minimum score is at least 0.50 and below 0.80 |
| Low | Minimum score is below 0.50, or evidence is insufficient without a numeric answer score |
| Not assessed | Safety refusal, blocked output, operational failure, or missing scores |

These are uncalibrated evidence-strength bands. They are not probabilities of
correctness or medical certainty. The cross-encoder measures relevance rather
than logical entailment. The thresholds need a larger labeled validation set
before any reliability claim. NaN, infinite, and out-of-range scores are rejected.

The UI shows overall and per-claim labels. An expandable panel reports HyDE and
correction status. CLI JSON includes labels, scores, reasons, and retrieval
diagnostics without including the hypothetical passage.

## Cost, configuration, and scope

HyDE adds at most one model request and one document-embedding operation for each
allowed question. Weak retrieval can add one rewrite request, one search, and
extra local reranking. Auxiliary generation uses one attempt and a 400-token cap;
embedding and final-answer calls keep their configured bounded retries. These
steps increase latency and consume free-tier quota. Safety refusals make no model
calls.

Set `augmentation.hyde_enabled` or `augmentation.crag_enabled` to `false` to
disable either feature independently. CRAG requires the reranker. The existing
index supports all modes. The UI and `scripts/ask.py` use the advanced settings;
the low-level `query_index.py` and `evaluate_retrieval.py` remain baseline tools.

## Verification

```powershell
python -m unittest discover -v
python scripts/evaluate_advanced_rag.py --output docs/advanced-rag-smoke.json
python scripts/evaluate_advanced_rag.py --baseline --output docs/baseline-rag-smoke.json
```

The live evaluator covers three allowed topics, one unsupported question, and one
personal diagnosis refusal. It records status, confidence, retrieval path, and
latency. Five smoke cases cannot measure medical accuracy, semantic entailment,
or prove that HyDE/CRAG improves retrieval. A fair comparison needs a larger held-
out set measuring recall, claim support, refusal errors, cost, and latency.

The recorded September 19, 2026 runs produced these observed outcomes:

| Mode | Expected outcomes | Observation |
| --- | ---: | --- |
| Baseline (HyDE/CRAG disabled) | 4/5 | The dengue answer was blocked because one generated claim failed semantic-support validation. |
| HyDE + corpus-only CRAG | 5/5 | Three answers were validated, the unsupported question was corrected then refused, and the personal query was refused before model access. |

The JSON reports preserve per-case latency and diagnostics. The advanced run was
slower because it made additional generation and embedding calls. Model output
can vary, and a single five-case run is not a statistically valid improvement
claim. The comparison demonstrates that both execution modes and their failure
handling are measurable.
