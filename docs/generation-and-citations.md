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
4. contains dosage or personalized-directive patterns; or
5. falls below the configured lexical-overlap check against cited evidence.

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

The test suite covers a valid cited answer, a fabricated citation, unsafe dosage
language, weak evidence overlap, bounded guideline lists, and proof that refused
queries never invoke generation. A live smoke test answered the dengue warning
signs question using real indexed evidence from the official guideline and
resolved every displayed citation to page 37.

Run the local checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe scripts\ask.py "According to the dengue guideline, what warning signs are listed?"
```

The second command consumes Gemini API quota.

## Known limitation

Token overlap is a useful low-cost tripwire, not proof that a cited passage
semantically entails a claim. Before deployment, add a held-out answer benchmark
and measure claim faithfulness, citation correctness, refusal accuracy, latency,
and API-failure rate. A stronger verifier or human review is appropriate for any
medical use beyond this educational demonstration.
