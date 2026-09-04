# Milestone 5: Streamlit Interface

Streamlit was selected because it provides a usable Python-only interface with
very little front-end code. The UI calls the same tested orchestration layer as
the command-line program; it does not duplicate retrieval or safety logic.

## Safety and production-minded behavior

- The page clearly states the corpus boundary and asks users not to submit
  personal medical data.
- A 500-character UI limit matches the backend input guardrail.
- A dropdown provides nine curated, non-personal sample questions across all
  three indexed guideline topics and copies the selection into the editable
  search field.
- Five submissions per rolling minute are allowed per browser session to reduce
  accidental quota exhaustion. This is a demonstration control, not a secure
  distributed rate limiter.
- Refused questions are stopped before retrieval and generation.
- Only application-validated claims and citation metadata are shown.
- Internal exceptions and diagnostics are not displayed to users.
- Each provider is closed after a request, and latency is shown for observation.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The app reads `GEMINI_API_KEY` from the ignored local `.env` file. On a hosting
platform, configure it as a secret rather than committing it.

## Before public deployment

Replace the session-only limiter with a shared server-side limiter, add request
logging without query or patient data, define concurrency limits and timeouts,
and monitor refusal accuracy, grounded-claim rate, p50/p95 latency, API failures,
and cost/quota consumption.
