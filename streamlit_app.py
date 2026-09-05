"""Streamlit interface for the Medical Guideline Assistant."""

from __future__ import annotations

import os
import sys
import time
import logging
from contextlib import ExitStack
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SAMPLE_QUESTIONS = {
    "Select a sample question": "",
    "Dengue — warning signs": (
        "According to the dengue guideline, what warning signs are listed?"
    ),
    "Dengue — phases of illness": (
        "What phases of dengue illness are described in the guideline?"
    ),
    "Dengue — patient groups": (
        "How does the dengue guideline classify patients into Groups A, B, and C?"
    ),
    "Hypertension — risk factors": (
        "What risk factors are listed in the hypertension guideline?"
    ),
    "Hypertension — blood pressure measurement": (
        "How should blood pressure be measured for hypertension screening?"
    ),
    "Hypertension — lifestyle changes": (
        "What general lifestyle changes are described in the hypertension guideline?"
    ),
    "Diabetic foot — prevention": (
        "Summarize general diabetic foot prevention from the indexed guideline."
    ),
    "Diabetic foot — infection signs": (
        "What signs of foot infection are listed in the diabetic foot guideline?"
    ),
    "Diabetic foot — risk assessment": (
        "How does the guideline describe diabetic foot risk assessment?"
    ),
}

from medical_guideline_assistant.answering import (  # noqa: E402
    AnswerOutcome,
    answer_query,
    preflight_query,
)
from medical_guideline_assistant.generation.config import GenerationConfig  # noqa: E402
from medical_guideline_assistant.generation.gemini import (  # noqa: E402
    GeminiGroundedGenerator,
)
from medical_guideline_assistant.retrieval.config import RetrievalConfig  # noqa: E402
from medical_guideline_assistant.retrieval.embeddings import (  # noqa: E402
    GeminiEmbeddingProvider,
)
from medical_guideline_assistant.retrieval.reranker import (  # noqa: E402
    CrossEncoderReranker,
)
from medical_guideline_assistant.ui.rate_limit import consume_request  # noqa: E402


st.set_page_config(
    page_title="Medical Guideline Assistant",
    page_icon="📘",
    layout="centered",
)
load_dotenv(PROJECT_ROOT / ".env", override=False)
LOGGER = logging.getLogger(__name__)


@st.cache_data
def load_configs() -> tuple[RetrievalConfig, GenerationConfig]:
    return (
        RetrievalConfig.from_path(PROJECT_ROOT / "configs" / "retrieval.json"),
        GenerationConfig.from_path(PROJECT_ROOT / "configs" / "generation.json"),
    )


def get_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        secret = st.secrets.get("GEMINI_API_KEY")
    except FileNotFoundError:
        return None
    return str(secret) if secret else None


@st.cache_resource
def get_reranker(model: str, candidate_count: int, threshold: float):
    from medical_guideline_assistant.retrieval.config import RerankingConfig

    return CrossEncoderReranker(
        RerankingConfig(
            enabled=True,
            model=model,
            candidate_count=candidate_count,
            minimum_top_score=threshold,
        )
    )


def execute_query(query: str) -> AnswerOutcome:
    preflight = preflight_query(query)
    if preflight is not None:
        return preflight
    retrieval_config, generation_config = load_configs()
    api_key = get_api_key()
    reranker = None
    if retrieval_config.reranking.enabled:
        reranker = get_reranker(
            retrieval_config.reranking.model,
            retrieval_config.reranking.candidate_count,
            retrieval_config.reranking.minimum_top_score,
        )
    with ExitStack() as resources:
        embedding_provider = GeminiEmbeddingProvider(retrieval_config.embedding, api_key)
        resources.callback(embedding_provider.close)
        generator = GeminiGroundedGenerator(generation_config, api_key)
        resources.callback(generator.close)
        return answer_query(
            query=query,
            database_path=PROJECT_ROOT / retrieval_config.database_path,
            retrieval_config=retrieval_config,
            generation_config=generation_config,
            generator=generator,
            embedding_provider=embedding_provider,
            reranker=reranker,
        )


def render_answer(outcome: AnswerOutcome, latency_seconds: float) -> None:
    if outcome.retrieval.retrieval_mode == "lexical_fallback":
        st.info(
            "Semantic search was temporarily unavailable, so this response used "
            "the local keyword index. Citations were still validated."
        )
    if outcome.status == "answered" and outcome.answer:
        st.subheader("Guideline answer")
        seen_citations: set[tuple[str, str, tuple[int, ...]]] = set()
        for number, claim in enumerate(outcome.answer.claims, start=1):
            st.write(f"{number}. {claim.text}")
            for citation in claim.citations:
                key = (citation.title, citation.source_url, tuple(citation.pages))
                if key in seen_citations:
                    continue
                seen_citations.add(key)
                pages = ", ".join(str(page) for page in citation.pages)
                st.caption(f"Source: {citation.title} — PDF page(s) {pages}")
                st.link_button(
                    f"Open official source (page {pages})",
                    citation.source_url,
                )
        st.warning(outcome.answer.disclaimer)
    elif outcome.status == "refused":
        st.warning(outcome.message)
    elif outcome.status == "insufficient_evidence":
        st.info(outcome.message)
    else:
        st.error(
            "The answer was blocked because it could not be validated safely. "
            "Try a narrower question about the indexed guidelines."
        )
    st.caption(f"Request latency: {latency_seconds:.2f} seconds")


def apply_sample_question() -> None:
    selected_label = st.session_state.get("sample_question", "")
    selected_query = SAMPLE_QUESTIONS.get(selected_label, "")
    if selected_query:
        st.session_state["guideline_query"] = selected_query


st.title("Medical Guideline Assistant")
st.write(
    "Ask for general information from the indexed official MOHFW/NCVBDC "
    "guidelines on dengue, hypertension, and diabetic foot care."
)
st.warning(
    "Educational demonstration only. Do not enter names, reports, symptoms, or "
    "other personal medical data. This app does not diagnose, prescribe, give "
    "dosages, or handle emergencies."
)
st.caption(
    "Try: “According to the dengue guideline, what warning signs are listed?”"
)
st.selectbox(
    "Common guideline questions",
    options=list(SAMPLE_QUESTIONS),
    key="sample_question",
    on_change=apply_sample_question,
)

with st.form("guideline-question"):
    query = st.text_input(
        "Guideline question",
        key="guideline_query",
        max_chars=500,
        placeholder="Ask a general question about an indexed guideline",
    )
    submitted = st.form_submit_button("Search official guidelines", type="primary")

if submitted:
    cleaned_query = query.strip()
    if not cleaned_query:
        st.warning("Enter a guideline question first.")
    else:
        previous = list(st.session_state.get("request_timestamps", []))
        decision = consume_request(previous, time.monotonic())
        st.session_state["request_timestamps"] = decision.timestamps
        if not decision.allowed:
            st.warning(
                "Session rate limit reached. Try again in about "
                f"{decision.retry_after_seconds} seconds."
            )
        else:
            started = time.perf_counter()
            try:
                with st.spinner("Retrieving and validating guideline evidence…"):
                    outcome = execute_query(cleaned_query)
                if outcome.status == "output_blocked" and outcome.diagnostic:
                    LOGGER.warning(
                        "Generated output was blocked; diagnostic=%s",
                        outcome.diagnostic,
                    )
                render_answer(outcome, time.perf_counter() - started)
            except Exception as exc:
                LOGGER.error("UI request failed safely; error_type=%s", type(exc).__name__)
                st.error(
                    "The request failed safely. Check the local index, API key, "
                    "internet connection, and API quota, then try again."
                )
