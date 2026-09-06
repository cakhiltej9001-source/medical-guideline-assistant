# Medical Guideline Assistant

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://medical-guideline-assistant-luowhfprn5urrtnhgrgzq7.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/Google-Gemini%20API-8E75B2?logo=googlegemini&logoColor=white)
![RAG](https://img.shields.io/badge/Architecture-Hybrid%20RAG-0A7EA4)
![Tests](https://img.shields.io/badge/Tests-66%20passing-2EA44F)
![Safety](https://img.shields.io/badge/Safety%20evals-19%2F19-2EA44F?logo=checkmarx&logoColor=white)

> 🩺 **Trusted guidelines in. Grounded answers out.**

An educational retrieval-augmented generation (RAG) application that answers
general questions using a curated collection of official Indian government
medical guidelines. The application retrieves evidence before generation,
validates every citation, and refuses personalized diagnosis, treatment, dosage,
patient-data interpretation, and emergency requests.

> **Safety notice:** This is a course project, not a medical device. It provides
> information from indexed official guidelines only and does not provide
> personalized medical advice. Do not submit names, medical records, symptoms, or
> other personal health information.

## 🚀 Live demo

[Open the Medical Guideline Assistant on Streamlit Community Cloud](https://medical-guideline-assistant-luowhfprn5urrtnhgrgzq7.streamlit.app/)

## 👋 Start here if you are new to AI projects

Imagine a reader looking for one explanation inside several long government
guideline PDFs. This project helps locate the relevant passages and turn them
into a readable summary with source references. It is designed for general
document questions, such as “What warning signs are listed in the dengue
guideline?” Its purpose is to explain the documents, rather than assess a person's
health.

The main idea is **retrieve first, then generate**. The application searches a
prepared collection of documents, selects useful passages, and gives those
passages to Gemini to draft an answer. Application code checks the result before
displaying it. This workflow is called **retrieval-augmented generation**, or RAG.

Think of it as an open-book exercise: the search system finds the pages, the
language model writes the explanation, and the validator checks its references
and applies the project's output rules. Having the book available helps, but
does not guarantee that every explanation is correct.

**Suggested reading path:** try the demo, read the concepts below, follow the
end-to-end walkthrough, and then explore the code using the file guide.

### 🧭 Guide to this README

- [Try the app and understand the screenshots](#-working-application)
- [Learn the core concepts](#-core-concepts-explained)
- [Follow the complete workflow](#-end-to-end-walkthrough)
- [Understand the technology choices](#-technology)
- [Run it locally](#-local-setup)
- [Understand the evaluation results](#-tests-and-evaluation)
- [Troubleshoot common issues](#-common-issues-and-resolutions)
- [Prepare for interviews](#-common-interview-questions-and-sample-answers)

## 📸 Working application

The deployed interface includes ready-made questions, a free-text guideline
question field, and an educational-use safety notice.

![Medical Guideline Assistant Streamlit interface](docs/assets/streamlit-working-app.png)

Answers are grounded in retrieved guideline passages and include the official
source, exact PDF page, and a direct link to the government document.

![Grounded Streamlit answer with an official citation](docs/assets/streamlit-grounded-answer.png)

### How to try it

1. Open the live demo and choose a question from **Common guideline questions**.
2. The selection fills the **Guideline question** box. You can edit the wording.
3. Click **Search official guidelines** and wait for the result.
4. Read the answer and its source title and PDF page references. Follow the
   **Open official source** link to inspect the original document.
5. Notice the request latency below the result: it measures how long that request
   took, not the quality or certainty of the answer.

In the first screenshot, the dropdown provides a starting point for users who do
not know what the corpus covers. In the second, the answer and citation show how
an explanation can be traced back to a document. The screenshots illustrate the
interface; exact wording and retrieved pages can differ between requests.

An insufficient-evidence message means the system could not find or validate
enough support. A safety refusal means the question asks for something outside
the assistant's permitted role. An operational error can indicate a connection,
model, index, or API quota problem. These outcomes have different causes even
though none displays a generated answer.

## 🎯 Project goal

The project demonstrates how to:

- build a reproducible corpus from allowlisted government PDF URLs;
- combine BM25 keyword retrieval with Gemini embeddings and local cross-encoder reranking;
- constrain Gemini to retrieved evidence and structured JSON;
- resolve citations from trusted application metadata rather than model text;
- refuse unsafe or insufficiently supported requests; and
- evaluate retrieval quality and safety behavior.

## 📚 Indexed topics

The current corpus contains 443 audited chunks from three public guidelines:

1. **National Guidelines for Clinical Management of Dengue Fever 2023** —
   NCVBDC/MOHFW.
2. **Screening, Diagnosis, Assessment, and Management of Primary Hypertension in
   Adults in India** — MOHFW.
3. **The Diabetic Foot: Prevention and Management in India (Draft)** — MOHFW.

Source URLs, editions, document dates, license-review status, active-version status,
page inclusion rules, and filenames are maintained in
[`configs/sources.json`](configs/sources.json). The downloader accepts HTTPS URLs
only from explicitly approved MOHFW/NCVBDC hostnames, validates that each response
is a PDF, computes its SHA-256 digest, and saves provenance metadata. This is how
external MOHFW links are retrieved without accepting arbitrary internet content.

## 🧠 Architecture

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
Local ONNX cross-encoder reranking
    |
    +--- top score < 0.20 / model unavailable ---> Insufficient-evidence refusal
    v
Gemini structured claim generation
    |
    v
Schema, output safety, lexical + semantic support, and citation-ID validation
    |
    +--- validation failure ---> Safe refusal
    v
Answer claims + official URL + exact PDF page(s) + disclaimer
```

## 🧩 Core concepts explained

| Concept | Plain-language meaning | How this project uses it |
| --- | --- | --- |
| Large language model (LLM) | A model that generates text from the input it receives. Fluent text can still contain mistakes. | Gemini drafts explanations from the selected guideline passages. |
| Corpus / knowledge base | The collection of documents the application is allowed to search. | Three curated MOHFW/NCVBDC guidelines, represented by 443 text chunks. |
| Ingestion | Preparing source documents so software can search them. | Download approved PDFs, extract text, clean it, split it, and record source metadata. |
| Token | A unit of text processed by a model; it can be a word, part of a word, or punctuation. | The chunker estimates token counts to keep passages and prompts bounded. |
| Chunk and overlap | A chunk is a small passage; overlap repeats some text at a boundary to preserve context. | Target about 300 estimated tokens per chunk, with a 400-token maximum and roughly 50-token overlap. |
| Metadata | Information describing a passage and where it came from. | Source ID, title, URL, pages, sections, date, and content hash help trace evidence. |
| Embedding / vector | A numerical representation that helps compare text by meaning. | Gemini represents each chunk as 768 numbers and represents each query in the same space. |
| Cosine similarity | A comparison of the directions of two vectors. | Helps rank passages whose embeddings are similar to the question's embedding. |
| Index | A prepared data structure used for searching. | SQLite stores searchable text, source metadata, and embedding vectors. |
| BM25 / lexical search | A word-based ranking method that considers term frequency, rarity, and document length. | Finds passages containing the user's terms or related word forms. |
| Dense / semantic search | Search using embeddings rather than only matching words. | Helps find a passage when the question uses different wording. |
| Hybrid search | Combining lexical and semantic search. | Retrieves candidates from both BM25 and Gemini embeddings. |
| Reciprocal-rank fusion (RRF) | Combining result lists according to positions rather than raw scores. | Merges the two rankings before reranking. |
| Cross-encoder reranker | A model that reads the question and a candidate passage together to score relevance. | Reorders the ten fused candidates using a local ONNX model. |
| Grounding and citation | Grounding ties a claim to evidence; a citation tells the reader where to inspect that evidence. | Each generated claim must cite retrieved chunk IDs, which resolve to official sources and pages. |
| Guardrail | A rule or check that limits what the system accepts or displays. | Input rules, relevance thresholds, output safety checks, and citation validation. |
| Hallucination | A generated statement that is invented or unsupported by the available evidence. | The pipeline reduces this risk through retrieval and validation; it cannot eliminate it. |

### Why use both search methods?

Suppose a question uses “BP measurement” while a guideline says “blood pressure
measurement.” Conservative abbreviation expansion helps normalize the query.
BM25 can find matching terms, while embeddings can help with paraphrases. Neither
method guarantees that a passage actually answers the question, so the fused
candidates pass through a reranker before generation.

The fusion rule is `1 / (60 + rank)` for each list in which a passage appears.
For example, a passage ranked first by BM25 and third by dense search receives
`1/61 + 1/63`, approximately `0.0323`. This is a ranking score, not a probability
that the passage is correct.

### Why retrieve before reranking?

Chunk embeddings can be prepared in advance and reused. A cross-encoder must
process each question–passage pair together, so it does more work at query time.
This project first finds a small candidate set, then spends that extra computation
on ten passages instead of the whole corpus.

The reranker's raw scores are transformed into values between 0 and 1. The
configured top-result threshold is `0.20`: below it, the system refuses for
insufficient evidence. **A score of 0.80 does not mean an answer is 80% medically
correct.** These are relevance scores, and the threshold has only been checked
against a small project dataset.

## 🔎 End-to-end walkthrough

The project has two workflows: preparation runs when the corpus is built or
updated; answering runs each time a user submits a question.

### A. Prepare the searchable documents

1. **Select official sources.** The source manifest lists approved PDF URLs,
   editions, dates, and page inclusion rules. The app does not search the whole
   internet for every question.
2. **Download and verify.** The downloader checks the hostname and PDF response,
   then records a SHA-256 hash. A hash is a fingerprint for detecting changed
   bytes; it does not prove that the document is correct or current.
3. **Extract and clean text.** `pdfplumber` reads text page by page. The pipeline
   removes repeated headers/footers and flags pages needing image review. It does
   not assume every scanned page can be read successfully.
4. **Create chunks.** The chunker splits text into passages while preserving page
   references. Some short chunks remain at natural boundaries; the corpus audit
   reports these rather than silently hiding them.
5. **Audit the corpus.** Checks cover chunk IDs, allowed page ranges, size limits,
   and source consistency before indexing.
6. **Embed and store.** Gemini creates chunk embeddings. SQLite stores these
   vectors alongside text and metadata, and FTS5 provides the keyword index.
   Cached vectors allow unchanged chunks to be reused during a rebuild.

This is document indexing; it does not train or fine-tune Gemini. Updating a
guideline requires reviewing the source and rebuilding the affected data.

### B. Answer a guideline question

Using “According to the dengue guideline, what warning signs are listed?” as a
workflow example:

1. **Check the input.** The safety gate checks whether the question is allowed.
   Personal diagnosis, dosing, and emergency requests are refused before retrieval.
2. **Normalize wording.** Text cleanup and known abbreviation expansion prepare
   the search query without broadly inventing additional medical meaning.
3. **Retrieve candidates.** BM25 returns up to 20 passages and dense search
   returns up to 20. Source filters are available for exact approved source IDs.
4. **Fuse and rerank.** RRF selects ten candidates, and the cross-encoder orders
   them by relevance to the question. The confidence gate checks the top score.
5. **Generate from context.** Up to five selected passages are supplied to Gemini
   with instructions to use only that evidence and return structured claims.
6. **Validate the answer.** Code checks the JSON structure, cited IDs, output
   safety, lexical overlap, and—on the Streamlit and configured CLI paths—the
   cross-encoder score between each claim and its cited text. Relevance is a
   support heuristic, not a test of logical entailment.
7. **Display or refuse.** Accepted claims receive source metadata from the index.
   If validation fails, one bounded regeneration attempt is allowed; an answer
   that still fails is blocked.

### What does structured output mean?

Instead of requesting one unrestricted paragraph, the application requests JSON,
a format with named fields that Python can inspect. This **illustrative example**
shows the shape only; its placeholder ID is not a real citation:

```json
{
  "status": "answered",
  "claims": [
    {
      "text": "A short statement supported by the retrieved passage.",
      "chunk_ids": ["example-source:example-chunk"]
    }
  ]
}
```

The model supplies a claim and an ID. The application looks up the ID in the
retrieved evidence and obtains the official title, URL, and PDF pages itself.
Unknown IDs are rejected. This protects citation metadata, but a real citation
alone does not prove that the associated statement is supported.

## 🛠️ Technology

| Tool | Responsibility | Why it is used here |
| --- | --- | --- |
| Python 3.12+ | Connects ingestion, retrieval, generation, validation, and UI code. | One language keeps a student's first end-to-end project easier to follow. |
| Streamlit | Builds the dropdown, text input, answer display, and citation links. | Provides a Python interface without requiring a separate frontend framework. |
| SQLite + FTS5 | Stores text, vectors, metadata, and the BM25 index in one local database. | Fits a small corpus and avoids running a separate database service. Dense similarity is calculated in Python, not by a hosted vector database. |
| `pdfplumber` | Extracts page-level PDF text. | Preserves page references needed for citations and corpus review. |
| `requests` | Downloads the allowlisted guideline PDFs. | Supports the controlled HTTP download stage. |
| `google-genai` | Calls Google's embedding and generation APIs. | Uses one SDK for both Gemini roles. An API key identifies and authorizes the application to the service. |
| `gemini-embedding-001` | Generates 768-dimensional document and query vectors. | Enables semantic retrieval; it does not write the final answer. |
| `gemini-3.5-flash-lite` | Generates structured answer claims from the selected passages. | The configured text-generation model supplies the explanation after retrieval. |
| FastEmbed + ONNX Runtime | Runs `Xenova/ms-marco-MiniLM-L-6-v2` locally. | Adds cross-encoder reranking without an additional paid reranking API or a PyTorch runtime dependency. ONNX is a model format; ONNX Runtime executes the model. |
| `python-dotenv` / Streamlit Secrets | Loads the Gemini key for local / hosted runs. | Keeps configuration separate from committed application code. |
| Python `unittest` + Streamlit AppTest | Checks components and UI behavior. | Allows regressions to be detected with repeatable tests. |

The model names above describe the repository's configuration. API availability
and quotas depend on the Google project. The application is designed around
free-tier usage, but a free-tier limit is not a guarantee of unlimited requests.

### Where does computation happen?

SQLite search, vector comparison, reranking, and validation run in the Python
application—on your computer for a local run, or on the Streamlit host for the
public app. Gemini API calls run remotely. Document text is sent for embeddings
during index preparation; allowed query text is sent for query embeddings, and
the question plus selected passages are sent for answer generation.

The public repository includes the prepared index, so a visitor can try the app
without downloading and embedding all PDFs again. The reranker model is fetched
separately on first use and cached. The repository is organized into ordinary
Python modules; no agent framework is required for this fixed sequence of steps.

## 🛡️ Safety design

The safety gate runs before retrieval or API access. It blocks:

- first-person symptoms and requests for diagnosis;
- personalized treatment or medication changes;
- medication doses and dosing summaries;
- questions containing personal medical records (the UI does not offer uploads);
- emergency/crisis requests; and
- obvious questions outside the indexed corpus.

Generated output is accepted only when it has the exact expected JSON structure,
contains one to twelve bounded claims, cites only retrieved chunk IDs, passes
lexical and cross-encoder semantic-support checks, and passes the output classifier
for personalized advice/diagnosis, medication dosage, and emergency directives.
Source title, URL, and page numbers are attached from the trusted local index after
validation. If any claim fails, the entire answer is blocked.

These deterministic checks are guardrails, not proof of clinical correctness.
See [`docs/scope-and-safety.md`](docs/scope-and-safety.md) and
[`docs/generation-and-citations.md`](docs/generation-and-citations.md).

## 💻 Local setup

Create a Google AI Studio API key, then clone the repository and run:

```powershell
git clone https://github.com/cakhiltej9001-source/medical-guideline-assistant.git
cd medical-guideline-assistant
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

These commands are for Windows PowerShell. They assume Git and Python are
installed. A **virtual environment** (`.venv`) isolates this project's packages
from your other Python projects; activation makes `python` use that environment.
`requirements.txt` lists the dependency versions to install. If you already cloned
the repository, start from its folder and skip the first two commands.

`localhost` means your own computer. It is useful for development, but sharing
that address does not give someone else access to your local app. The public
Streamlit URL runs a hosted copy. An internet connection is needed for Gemini
requests and for the first reranker download.

## 💬 Example questions

- According to the dengue guideline, what warning signs are listed?
- What phases of dengue illness are described in the guideline?
- What risk factors are listed in the hypertension guideline?
- How should blood pressure be measured for hypertension screening?
- What general lifestyle changes are described in the hypertension guideline?
- What signs of foot infection are listed in the diabetic foot guideline?

The same questions are available from the application's dropdown.

## 🔄 Rebuild the corpus and index

The committed runtime index avoids rebuilding the corpus on hosted startup. To
reproduce it from the allowlisted sources:

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

## ⌨️ Command-line usage

Inspect retrieval without answer generation:

```powershell
python scripts/query_index.py "What are the warning signs of dengue?" --hybrid
```

Run the complete guarded pipeline:

```powershell
python scripts/ask.py "According to the dengue guideline, what warning signs are listed?"
```

## ✅ Tests and evaluation

Run all tests:

```powershell
python -m unittest discover -v
```

Run the complete offline and retrieval evaluations:

```powershell
python scripts/evaluate_safety.py
python scripts/evaluate_output_safety.py
python scripts/evaluate_grounding.py
python scripts/evaluate_confidence.py
python scripts/evaluate_retrieval.py
python scripts/evaluate_retrieval.py --hybrid
```

Recorded small smoke benchmarks (these are project checks, not clinical validation):

| Evaluation | Result |
| --- | ---: |
| Automated tests | 66 passing |
| Input safety classification | 13/13 (100%) |
| Output safety classification | 6/6 (100%) |
| Citation/grounding validation | 5/5 (100%) |
| Confidence gate | 5/5 (100%), threshold 0.20 |
| Lexical + reranker Answerable Recall@5 | 1.00 |
| Lexical + reranker MRR@5 | 0.917 |
| Lexical gold-page coverage@5 | 0.722 |
| Hybrid + reranker Answerable Recall@5 | 1.00 |
| Hybrid + reranker MRR@5 | 0.917 |
| Hybrid gold-page coverage@5 | 0.806 |

### What do these numbers mean?

- **Automated tests** check specific expected behaviors, including validation,
  error handling, and UI rendering. Passing tests does not cover every possible input.
- **Safety accuracy** is the fraction of labeled examples classified as expected.
  A 13/13 result describes those 13 examples, not all possible unsafe requests.
- **Answerable Recall@5** is the evaluation script's name for **Hit Rate@5**:
  the fraction of questions with at least one annotated relevant page represented
  in the first five results. It does not mean all relevant evidence was retrieved.
- **MRR@5** averages the reciprocal rank of the first relevant result: rank 1
  contributes 1, rank 2 contributes 0.5, and no relevant result in the top five
  contributes 0. Higher values mean useful evidence appears earlier.
- **Gold-page coverage@5** measures how many annotated relevant pages appear in
  the top five results, divided by the annotated pages for each question, then
  averaged across questions. This captures missing evidence that hit rate can hide.
- **Confidence-gate accuracy** checks whether the relevance threshold accepts or
  rejects five labeled answerable/unanswerable questions as expected.
- **Citation/grounding validation** checks five constructed cases using fixed
  support scores. It tests validator acceptance/rejection behavior, not the real
  model's semantic faithfulness or medical accuracy.

For example, retrieving one of four relevant pages counts as a hit, but only
25% page coverage for that question. This is why more than one metric is needed.
The hybrid retrieval evaluation calls Gemini; local reranker evaluations can
also need internet access on the first run to download the model.

The evaluation sets are deliberately small and do not justify production or
clinical-performance claims. A real deployment needs substantially larger held-out
sets for retrieval recall, refusal accuracy, claim faithfulness, citation
correctness, latency percentiles, and API failure rates.

## 🔧 Common issues and resolutions

During development, the interface displayed generic request failures, semantic
search fallback notices, and answers blocked by validation. Those messages identify
an outcome, not necessarily its root cause. The table below explains these and
other common troubleshooting scenarios; it is not a claim that every issue occurred
or that one fix resolves every failure.

| Issue / symptom | Likely cause | Resolution and verification |
| --- | --- | --- |
| **“The request failed safely”** | A broad error handler caught a failure in the index, API, model, or another pipeline stage. The screenshot alone cannot identify which. | Check the developer console/host logs, then isolate retrieval, generation, and validation using the commands below. Record the failing stage and sanitized error type rather than guessing that the API key is wrong. |
| **Missing key or authentication failure** | `GEMINI_API_KEY` is absent, invalid, or configured in a different environment. | Set the key in local `.env` or hosted Streamlit Secrets, as appropriate, and restart/reload the app. Run the relevant connection check. Local `.env` does not configure the hosted deployment. Never put keys into questions or GitHub. |
| **Quota or rate-limit failure** | Provider limits or the app's session limiter were reached. | Wait before retrying and inspect the project's quota. Avoid repeated clicks or unbounded retries. The app permits five requests per session per 60 seconds; that is separate from provider limits and is not global abuse protection. |
| **Semantic search unavailable; keyword fallback used** | Query embedding retrieval failed, for example due to connectivity or API access. | Test embeddings separately. BM25 fallback lets local retrieval continue, but does not replace Gemini answer generation. Reranking and validation still apply. Verify hybrid retrieval again after the underlying failure is resolved. |
| **Missing or unreadable local index** | The SQLite file is absent, corrupt, or the configured path is wrong. | Check `database_path` in `configs/retrieval.json` and the corresponding file in `data/index/`. Use the committed index or follow the rebuild steps above. Confirm a local retrieval command succeeds before testing generation. |
| **Model unavailable or embedding mismatch** | Model access/configuration changed, or query vectors no longer match the indexed embedding model/dimensions. | Check the configured model and access. When changing embedding model or dimensionality, regenerate the document embeddings and index too; do not mix incompatible vectors. Test both retrieval and generation after a configuration change. |
| **Insufficient-evidence refusal** | The topic is absent from the corpus, no useful passage was retrieved, or the top reranker score is below the threshold. | Try a general question about one of the three indexed topics. For an expected answerable question, inspect retrieved passages and scores, then evaluate any retrieval change. Do not lower the threshold merely to force an answer. |
| **Answer blocked by validation** | Generated JSON, citation IDs, claim support, or output safety failed a check. | Inspect the failed check with a non-sensitive example. Compare each claim with its cited passage, and improve retrieval or generation if needed. Add a regression test. Keep validation enabled; a plausible-looking answer is not enough. |
| **Personal symptom question is refused** | The safety gate intentionally blocks personalized medical requests. | This is expected behavior, not an API error. Use a general document question such as “What phases of dengue illness are described in the guideline?” Do not reword a personal request to evade the gate. |
| **First request is slow / reranker fails to load** | Cold startup or the first ONNX model download; failed downloads or resource limits can prevent loading. | Allow the initial load to finish and inspect download/resource errors if it fails. Reuse the cached model on subsequent requests. If reranking is unavailable, the app must refuse rather than bypass its confidence gate. |
| **PDF extraction produces little or broken text** | Scanned pages, tables, or layout interfere with extraction. | Review flagged pages against the original PDF, verify page references, and rerun the corpus audit. OCR would require an additional reviewed ingestion step; do not assume the current extractor reads every image. |
| **Missing Python module or app will not start** | Packages were installed into another Python environment or dependencies are incomplete. | Use the project virtual environment and run `python -m pip install -r requirements.txt`, followed by `python -m streamlit run streamlit_app.py`. If activation is blocked, call `.\.venv\Scripts\python.exe` directly rather than weakening system security settings. |
| **Works locally but not on the public app** | Hosted secrets, dependencies, files, or startup state differ from local development. | Check the deployed branch and `streamlit_app.py` entrypoint, hosted secrets, index presence, and deployment logs. After redeployment, test one allowed question, one safety refusal, and one unsupported question. |
| **Someone else cannot open `localhost:8501`** | `localhost` refers to the viewer's own computer. | Share the public Streamlit demo link near the top of this README, not the local development address. |

### Debug one stage at a time

Run these from the repository root with the project environment active:

```powershell
# 1. Check local retrieval and reranking without Gemini answer generation.
python scripts/query_index.py "What are the warning signs of dengue?"

# 2. Test embedding access independently.
python scripts/check_gemini_connection.py

# 3. Test text-generation access independently.
python scripts/check_generation_connection.py

# 4. Check semantic retrieval plus the local ranking stages.
python scripts/query_index.py "What are the warning signs of dengue?" --hybrid

# 5. Exercise the complete guarded answering path.
python scripts/ask.py "According to the dengue guideline, what warning signs are listed?"
```

The first command can need internet for the initial reranker download. The API
checks make real requests and may consume quota; they load local `.env`, not hosted
Streamlit Secrets. They currently use hard-coded model names, so keep them aligned
with the application configuration when changing models. A successful connection
test proves access to that API operation, not end-to-end answer correctness.

After a code fix, run `python -m unittest discover -v` and the evaluation relevant
to the changed stage. For example, retrieval changes need retrieval/confidence
checks, while validator changes need grounding/output-safety checks. Reproduce
the original failure and verify it is resolved without weakening refusals.

### Keep debugging safe

- Record the stage, error category, and reproduction steps using synthetic or
  general guideline questions. Review logs before sharing them.
- Never publish API keys, `.env`, Streamlit Secrets, or personal medical data.
  If a key has been exposed, revoke/rotate it and update the affected environments.
- Do not disable citation checks, safety filters, or confidence gates to make a
  failed request appear successful.

Latency is the time spent handling a request. This project limits context size,
caches reusable resources, and bounds retries to control latency and API use.
Repeated retries still consume time and can consume quota. A production system
should track slow requests, failures, and usage without logging personal health
information or secrets.

## ☁️ Deploy on Streamlit Community Cloud

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

## 🗂️ Project structure

```text
configs/                  Source, chunking, retrieval, and generation settings
data/index/               Verified runtime SQLite index
docs/                     Design decisions and source-integrity notes
eval/                     Retrieval, confidence, grounding, and safety datasets
scripts/                  Ingestion, indexing, query, and evaluation commands
src/medical_guideline_assistant/
  ingestion/              Download, extraction, chunking, and auditing
  retrieval/              Embeddings, BM25/dense search, RRF, cross-encoder
  safety/                 Pre-retrieval guardrails
  generation/             Prompt, Gemini provider, and grounding validator
  ui/                     UI helpers such as the session rate limiter
tests/                    Automated unit and render-level tests
streamlit_app.py          Streamlit entrypoint
```

### A beginner-friendly code reading order

1. Start with [`streamlit_app.py`](streamlit_app.py) to connect the visible
   dropdown, form, and result display to Python code.
2. Read [`answering.py`](src/medical_guideline_assistant/answering.py) and
   [`pipeline.py`](src/medical_guideline_assistant/pipeline.py) to follow how the
   stages are connected and how failures stop an answer.
3. Explore [`index.py`](src/medical_guideline_assistant/retrieval/index.py) and
   [`reranker.py`](src/medical_guideline_assistant/retrieval/reranker.py) to see
   candidate retrieval and pairwise relevance scoring.
4. Read [`grounding.py`](src/medical_guideline_assistant/generation/grounding.py)
   to understand why model output is checked before display.
5. Compare the settings in [`configs/`](configs/) with the cases in
   [`eval/`](eval/) and the checks in [`tests/`](tests/).

This separation is **modular design**: UI code handles interaction, retrieval
code finds evidence, generation code drafts and validates claims, and ingestion
code prepares documents. Smaller responsibilities make changes easier to test
and bugs easier to locate. Configuration files hold tunable settings separately
from the logic that uses them.

## 🎤 Common interview questions and sample answers

These answers are intentionally concise. In an interview, explain each idea in
your own words and connect it to a design choice or measurement from this project.

**1. What problem does this project solve?**

It gives educational answers from a small, curated collection of official
MOHFW/NCVBDC guidelines. Unlike a general chatbot, it retrieves evidence first,
cites supporting PDF pages, and refuses personalized diagnosis, treatment,
dosage, emergency, and unsupported requests.

**2. What is retrieval-augmented generation (RAG)?**

RAG retrieves relevant passages from an external knowledge base and supplies
them to an LLM as context before generation. The model therefore answers from
inspectable evidence instead of relying only on knowledge stored in its
parameters.

**3. Why use RAG instead of fine-tuning?**

Medical guidelines can change and each factual claim should be traceable to a
source. RAG lets us update documents without retraining and supports page-level
citations. Fine-tuning may improve behavior or style, but it does not replace
source retrieval.

**4. How are external MOHFW documents ingested safely?**

The downloader reads URLs only from `configs/sources.json`, requires HTTPS,
accepts only allowlisted MOHFW/NCVBDC hostnames, validates the PDF response, and
records metadata plus a SHA-256 digest. It never crawls an arbitrary URL supplied
by a user.

**5. How are PDFs prepared for retrieval?**

`pdfplumber` extracts text page by page. The pipeline cleans the text, creates
bounded logical chunks, and attaches metadata such as document ID, title, source
URL, page number, section, version, and content hash before indexing.

**6. Why does chunk size matter?**

Very small chunks can lose context, while very large chunks dilute relevant
information and consume more prompt tokens. This project uses configurable,
overlapping chunks and evaluates retrieval instead of assuming one size is
always best.

**7. Why combine BM25 and dense retrieval?**

BM25 is strong for exact terms, abbreviations, and guideline wording. Dense
embeddings help when the user's wording differs from the document. Combining
them improves recall and makes retrieval more robust than either method alone.

**8. How are BM25 and dense results combined?**

Each retriever returns a ranked list. Reciprocal-rank fusion rewards chunks near
the top of either list without requiring their raw scores to use the same scale.
The fused candidates are then jointly scored as query/passage pairs by a local
cross-encoder. This improves precision and produces the top-1 confidence used by
the refusal gate.

**9. How does the project reduce hallucinations?**

The prompt tells Gemini to use only retrieved context and return structured JSON.
After generation, deterministic code checks the schema, claim count, lexical and
semantic evidence support, output-safety category, and every cited chunk ID. A
failed answer is retried within a small limit or replaced with a safe refusal.

**10. How are citations kept trustworthy?**

The model cites internal chunk IDs, but it does not control the displayed title,
URL, or page number. After validation, the application resolves those fields
from trusted index metadata, so a fabricated URL or page cannot be displayed as
a citation.

**11. Where are safety guardrails applied?**

A deterministic gate runs before retrieval and API calls to block personal
symptoms, diagnosis, treatment, dosage, records, emergencies, and out-of-scope
questions. A second validation layer scans generated claims for unsafe language.
This defense-in-depth approach is stronger than relying on the prompt alone.

**12. What happens when Gemini is unavailable?**

If Gemini embedding retrieval fails, the system falls back to the local BM25
index and reports that mode. The local cross-encoder still reranks and applies
the confidence gate. If reranking, generation, or evidence validation fails, the
system fails closed with a controlled insufficient-evidence response.

**13. Why use SQLite with FTS5?**

SQLite is lightweight, reproducible, inexpensive, and easy to deploy with a
small curated corpus. FTS5 provides local BM25 search without another service.
At larger scale, a managed vector database or search engine would improve
concurrency, filtering, observability, and index operations.

**14. How was the system evaluated?**

The project tracks automated tests, Answerable Recall@5, MRR@5, gold-page
coverage, confidence-gate accuracy, input/output safety, grounding validation,
and live latency. The current six-question hybrid-reranked smoke set achieves
Answerable Recall@5 of 1.00, MRR@5 of 0.917, and gold-page coverage@5 of 0.806,
but the dataset is too small for clinical or production claims.

**15. How do you control cost, quotas, and latency?**

The app performs the inexpensive safety gate first, limits retrieved context and
output size, caches local data, uses a lexical fallback, bounds validation
retries, and applies a per-session request limiter. Production would also need
shared rate limiting, queues, monitoring, budgets, and latency-percentile alerts.

**16. How would you improve this project for production?**

I would add a larger versioned corpus, scheduled re-ingestion, a medical-domain
reranker and entailment model, held-out evaluations reviewed by medical experts, shared
abuse protection, privacy-safe observability, key rotation, audit logs, and a
formal clinical safety and regulatory review.

**17. What is the biggest limitation?**

The corpus contains only three guidelines, one of which is marked as a draft,
and neither lexical overlap nor cross-encoder relevance proves semantic entailment
(that the evidence actually implies the claim). The system
therefore refuses aggressively and must not be used for diagnosis or clinical
decision-making.

## ⚠️ Known limitations

- The corpus covers only three guidelines and includes one document marked draft.
- Cross-encoder relevance plus lexical overlap improves support checking but is
  not a formal entailment proof.
- The first hosted query may be slower while the 80 MB ONNX reranker is cached.
- Free Gemini quotas can introduce throttling and latency.
- Session rate limiting is not sufficient for an anonymous public application.
- This project must not be used for diagnosis, treatment, emergencies, or clinical
  decision-making.

## 📖 Documentation

- [`docs/source-integrity-log.md`](docs/source-integrity-log.md)
- [`docs/retrieval-design.md`](docs/retrieval-design.md)
- [`docs/generation-and-citations.md`](docs/generation-and-citations.md)
- [`docs/streamlit-interface.md`](docs/streamlit-interface.md)
