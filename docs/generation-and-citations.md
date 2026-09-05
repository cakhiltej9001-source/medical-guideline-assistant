# Milestone 4: Grounded Generation and Citations

## Goal

Turn retrieved guideline passages into a short answer without allowing the model
to invent evidence, source links, or personalized medical advice. A generated
response is displayed only after deterministic application-side validation.

## Answer contract

Gemini must return structured JSON with one of two states:

- `answered`: one to twelve atomic claims, each with one or more retrieved
  `chunk_ids`.
- `insufficient_evidence`: no claims. The application displays the standard safe
  fallback instead of asking the model to guess.

The application, rather than Gemini, resolves each accepted chunk ID to the
trusted source title, official URL, and PDF page numbers stored in the index.
This prevents the model from inventing citation metadata.

## Generation guardrails

The system instruction tells the model to use only supplied evidence, treat
document text as untrusted data, and avoid diagnosis, personal treatment,
medication doses, and emergency instructions. Generation uses at most five
retrieved chunks to limit latency and irrelevant context.

The validator rejects the entire output when any claim:

1. has an unexpected JSON shape;
2. cites a missing or non-retrieved chunk ID;
3. repeats citation IDs or omits citations;
4. is classified as personalized diagnosis/advice, medication dosage, or an
   emergency directive;
5. falls below the configured lexical-overlap check against cited evidence; or
6. falls below the cross-encoder semantic-support threshold.

The dosage check distinguishes medication quantities from source-backed lifestyle
measurements. For example, a cited dietary salt amount can be summarized, while a
medication amount or a second-person instruction remains blocked.

Rejecting the complete answer is intentional: showing only the claims that happen
to pass could create a misleading or incomplete medical summary.

If the first structured response fails deterministic validation, the orchestrator
makes one bounded regeneration attempt with the same evidence. Unsafe input still
makes zero model calls, and an answer that fails both attempts is blocked entirely.

## Model and reliability settings

The configured generator is `gemini-3.5-flash-lite`, with temperature `0.1`, a
1,024-token output limit, a 45-second timeout, and at most three attempts for
retryable failures. The API key is loaded from the ignored `.env` file and is
never logged or stored in the index.

## Verification

The test suite covers valid cited answers, fabricated citations, unsafe output,
weak lexical and semantic evidence support, bounded guideline lists, and proof
that refused queries never invoke generation. A live console smoke test completed
the full hybrid retrieval, cross-encoder reranking, Gemini generation, semantic
support, and trusted citation-resolution path against the dengue guideline.

Run the local checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe scripts\ask.py "According to the dengue guideline, what warning signs are listed?"
```

The second command consumes Gemini API quota.

## Known limitation

Token overlap plus a general-domain relevance cross-encoder is stronger than a
lexical check alone, but it is not proof that a cited passage logically entails a
claim. A production system needs a larger held-out answer benchmark, a medical
entailment verifier, expert review, and measurements for faithfulness, citation
correctness, refusal accuracy, latency percentiles, and API-failure rate.
