# Medical Guideline Assistant

An educational retrieval-augmented generation (RAG) application that answers
general questions using a curated collection of official Indian government
medical guidelines. The application retrieves evidence before generation,
validates every citation, and refuses personalized diagnosis, treatment, dosage,
patient-data interpretation, and emergency requests.

> **Safety notice:** This is a course project, not a medical device. It provides
> information from indexed official guidelines only and does not provide
> personalized medical advice. Do not submit names, medical records, symptoms, or
> other personal health information.

## Project goal

The project demonstrates how to:

- build a reproducible corpus from allowlisted government PDF URLs;
- combine BM25 keyword retrieval with Gemini semantic embeddings;
- constrain Gemini to retrieved evidence and structured JSON;
- resolve citations from trusted application metadata rather than model text;
- refuse unsafe or insufficiently supported requests; and
- evaluate retrieval quality and safety behavior.

## Indexed topics

The current corpus contains 443 audited chunks from three public guidelines:

1. **National Guidelines for Clinical Management of Dengue Fever 2023** —
   NCVBDC/MOHFW.
2. **Screening, Diagnosis, Assessment, and Management of Primary Hypertension in
   Adults in India** — MOHFW.
3. **The Diabetic Foot: Prevention and Management in India (Draft)** — MOHFW.

Source URLs, document dates, page inclusion rules, and filenames are maintained in
[`configs/sources.json`](configs/sources.json). The downloader accepts HTTPS URLs
only from explicitly approved MOHFW/NCVBDC hostnames, validates that each response
is a PDF, computes its SHA-256 digest, and saves provenance metadata. This is how
external MOHFW links are retrieved without accepting arbitrary internet content.

## Architecture

```text
User question
    |
    v
Pre-retrieval safety gate ---- unsafe/personal request ---> Safe refusal
    |
    v
Query normalization and conservative abbreviation expansion
    |
    v
BM25 retrieval + Gemini dense retrieval
    |                 |
    |                 +--- API failure ---> Local BM25 fallback
    v
Reciprocal-rank fusion and top evidence chunks
    |
    v
Gemini structured claim generation
    |
    v
Schema, safety, evidence-overlap, and citation-ID validation
    |
    +--- validation failure ---> Safe refusal
    v
Answer claims + official URL + exact PDF page(s) + disclaimer
```

## Technology

- Python 3.12+
- Streamlit
- SQLite with FTS5/BM25
- Google Gemini API through `google-genai`
- `gemini-embedding-001` embeddings at 768 dimensions
- `gemini-3.5-flash-lite` structured generation
- `pdfplumber` for PDF text extraction
- Python `unittest` for automated tests

## Safety design

The safety gate runs before retrieval or API access. It blocks:

- first-person symptoms and requests for diagnosis;
- personalized treatment or medication changes;
- medication doses and dosing summaries;
- uploaded or pasted personal medical records;
- emergency/crisis requests; and
- obvious questions outside the indexed corpus.

Generated output is accepted only when it has the exact expected JSON structure,
contains one to twelve bounded claims, cites only retrieved chunk IDs, passes a
minimum evidence-overlap check, and contains no medication dose or personalized
directive. Source title, URL, and page numbers are attached from the trusted local
index after validation. If any claim fails, the entire answer is blocked.

These deterministic checks are guardrails, not proof of clinical correctness.
See [`docs/scope-and-safety.md`](docs/scope-and-safety.md) and
[`docs/generation-and-citations.md`](docs/generation-and-citations.md).

## Local setup

Create a Google AI Studio API key, then clone the repository and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder locally:

```dotenv
GEMINI_API_KEY=your_key_here
```

Never commit `.env` or `.streamlit/secrets.toml`.

Start the application:

```powershell
python -m streamlit run streamlit_app.py
```

Then open <http://localhost:8501>.

## Example questions

- According to the dengue guideline, what warning signs are listed?
- What phases of dengue illness are described in the guideline?
- What risk factors are listed in the hypertension guideline?
- How should blood pressure be measured for hypertension screening?
- What general lifestyle changes are described in the hypertension guideline?
- What signs of foot infection are listed in the diabetic foot guideline?

The same questions are available from the application's dropdown.

## Rebuild the corpus and index

The committed runtime index makes hosted startup deterministic. To reproduce it
from the allowlisted sources:

```powershell
python scripts/download_sources.py
python scripts/extract_pdf.py
python scripts/chunk_pages.py
python scripts/audit_corpus.py
python scripts/build_index.py --with-embeddings
```

Embedding creation is checkpointed after each successful batch and paced for
free-tier limits. Downloaded PDFs, extracted intermediate files, and the embedding
cache are intentionally excluded from Git.

## Command-line usage

Inspect retrieval without answer generation:

```powershell
python scripts/query_index.py "What are the warning signs of dengue?" --hybrid
```

Run the complete guarded pipeline:

```powershell
python scripts/ask.py "According to the dengue guideline, what warning signs are listed?"
```

## Tests and evaluation

Run all tests:

```powershell
python -m unittest discover -v
```

Run the safety and retrieval evaluations:

```powershell
python scripts/evaluate_safety.py
python scripts/evaluate_retrieval.py
python scripts/evaluate_retrieval.py --hybrid
```

Current small smoke benchmarks:

| Evaluation | Result |
| --- | ---: |
| Automated tests | 57 passing |
| Safety classification | 13/13 (100%) |
| BM25 Recall@5 | 1.00 |
| BM25 MRR@5 | 0.575 |
| Hybrid Recall@5 | 1.00 |
| Hybrid MRR@5 | 0.833 |

The evaluation sets are deliberately small and do not justify production or
clinical-performance claims. A real deployment needs substantially larger held-out
sets for retrieval recall, refusal accuracy, claim faithfulness, citation
correctness, latency percentiles, and API failure rates.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at <https://share.streamlit.io> and connect the GitHub account.
3. Create an app from the repository, using `streamlit_app.py` as the entrypoint.
4. Open **Advanced settings** and add the secret:

   ```toml
   GEMINI_API_KEY = "your_rotated_key_here"
   ```

5. Deploy, then test an allowed question, a refusal, and an unsupported question.

Use a newly rotated key for deployment. The application's per-session limiter is
only a demonstration control; a public production service needs a shared
server-side limiter, quotas, monitoring, abuse controls, and privacy-safe logs.

## Project structure

```text
configs/                  Source, chunking, retrieval, and generation settings
data/index/               Verified runtime SQLite index
docs/                     Design decisions and source-integrity notes
eval/                     Safety and retrieval smoke datasets
scripts/                  Ingestion, indexing, query, and evaluation commands
src/medical_guideline_assistant/
  ingestion/              Download, extraction, chunking, and auditing
  retrieval/              Embeddings, SQLite/BM25, dense search, rank fusion
  safety/                 Pre-retrieval guardrails
  generation/             Prompt, Gemini provider, and grounding validator
  ui/                     UI helpers such as the session rate limiter
tests/                    Automated unit and render-level tests
streamlit_app.py          Streamlit entrypoint
```

## Known limitations

- The corpus covers only three guidelines and includes one document marked draft.
- Lexical overlap is a low-cost grounding tripwire, not semantic entailment.
- Free Gemini quotas can introduce throttling and latency.
- Session rate limiting is not sufficient for an anonymous public application.
- This project must not be used for diagnosis, treatment, emergencies, or clinical
  decision-making.

## Documentation

- [`docs/source-integrity-log.md`](docs/source-integrity-log.md)
- [`docs/retrieval-design.md`](docs/retrieval-design.md)
- [`docs/generation-and-citations.md`](docs/generation-and-citations.md)
- [`docs/streamlit-interface.md`](docs/streamlit-interface.md)
