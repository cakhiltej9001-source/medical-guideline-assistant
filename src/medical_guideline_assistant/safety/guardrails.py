"""Deterministic pre-retrieval safety checks and conservative query rewriting."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass


PERSONALIZED_REFUSAL = (
    "I can explain information contained in the indexed MOHFW guidelines, but I "
    "cannot diagnose a condition or recommend personalized treatment, medication, "
    "or dosage. Please consult a qualified healthcare professional."
)
EMERGENCY_REFUSAL = (
    "I cannot assess or manage medical emergencies. Please contact your local "
    "emergency service or seek immediate help from a qualified healthcare professional."
)
OUT_OF_SCOPE_REFUSAL = (
    "I don't have enough information in the indexed MOHFW guidelines to answer that question."
)
INVALID_INPUT_REFUSAL = "Please enter a medical-guideline question of 500 characters or fewer."


PERSONAL_CONTEXT = re.compile(
    r"\b(?:i|i'm|i've|me|my|mine|we|our|my child|my baby|my mother|my father|"
    r"my husband|my wife|my patient)\b",
    flags=re.IGNORECASE,
)
IMMEDIATE_PERSON_CONTEXT = re.compile(
    r"\b(?:someone|this person|the patient)\s+(?:is|has|cannot|can't)\b",
    flags=re.IGNORECASE,
)
EMERGENCY_SIGNAL = re.compile(
    r"\b(?:cannot breathe|can't breathe|struggling to breathe|not breathing|"
    r"unconscious|collapsed|seizure|severe bleeding|bleeding heavily|overdose|"
    r"suicidal|chest pain|medical emergency)\b",
    flags=re.IGNORECASE,
)
DIAGNOSIS_REQUEST = re.compile(
    r"\b(?:do i have|have i got|diagnose me|what (?:disease|condition) do i have|"
    r"is this (?:dengue|hypertension|an infection)|am i (?:infected|diabetic))\b",
    flags=re.IGNORECASE,
)
SYMPTOM_OR_RESULT = re.compile(
    r"\b(?:fever|rash|pain|vomiting|bleeding|dizzy|dizziness|swelling|ulcer|"
    r"blood pressure|bp|lab result|test result|platelet|glucose|symptoms?)\b",
    flags=re.IGNORECASE,
)
DOSAGE_REQUEST = re.compile(
    r"\b(?:doses?|dosages?|dosing|how many tablets?|how much (?:medicine|medication)|"
    r"what strength|which strength)\b",
    flags=re.IGNORECASE,
)
PERSONAL_TREATMENT_REQUEST = re.compile(
    r"\b(?:should|can|may|must)\s+i\s+(?:take|use|start|stop|change|increase|decrease)\b|"
    r"\b(?:which|what)\s+(?:medicine|medication|drug|treatment)\s+should\s+i\b",
    flags=re.IGNORECASE,
)
PERSONAL_RECORD = re.compile(
    r"\b(?:my|our)\s+(?:prescription|medical record|lab report|test report|scan|"
    r"blood report|doctor's note)\b",
    flags=re.IGNORECASE,
)
OBVIOUS_NONMEDICAL_TOPIC = re.compile(
    r"\b(?:weather forecast|stock price|share price|cricket score|football score|"
    r"movie review|write (?:python|javascript) code|cooking recipe)\b",
    flags=re.IGNORECASE,
)

ABBREVIATIONS = {
    "BP": "blood pressure",
    "CBC": "complete blood count",
    "DBP": "diastolic blood pressure",
    "HCT": "hematocrit",
    "HTN": "hypertension",
    "PAD": "peripheral artery disease",
    "SBP": "systolic blood pressure",
}


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    category: str
    normalized_query: str
    retrieval_query: str | None
    refusal_message: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_query(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query)
    normalized = "".join(
        character
        for character in normalized
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
    )
    return " ".join(normalized.split())


def rewrite_query(query: str) -> str:
    """Expand only an audited set of unambiguous abbreviations."""
    rewritten = query
    for abbreviation, expansion in ABBREVIATIONS.items():
        rewritten = re.sub(
            rf"\b{re.escape(abbreviation)}\b",
            expansion,
            rewritten,
            flags=re.IGNORECASE,
        )
    return rewritten


def evaluate_input(query: str) -> SafetyDecision:
    """Return an allow/refuse decision before retrieval or API access."""
    normalized = normalize_query(query)
    if not normalized or len(normalized) > 500:
        return SafetyDecision(
            False, "invalid_input", normalized, None, INVALID_INPUT_REFUSAL
        )

    personal = bool(PERSONAL_CONTEXT.search(normalized))
    immediate_person = personal or bool(IMMEDIATE_PERSON_CONTEXT.search(normalized))
    if immediate_person and EMERGENCY_SIGNAL.search(normalized):
        return SafetyDecision(False, "emergency", normalized, None, EMERGENCY_REFUSAL)
    if PERSONAL_RECORD.search(normalized):
        return SafetyDecision(
            False, "personal_medical_data", normalized, None, PERSONALIZED_REFUSAL
        )
    if DOSAGE_REQUEST.search(normalized):
        return SafetyDecision(False, "dosage", normalized, None, PERSONALIZED_REFUSAL)
    if PERSONAL_TREATMENT_REQUEST.search(normalized):
        return SafetyDecision(
            False, "personalized_treatment", normalized, None, PERSONALIZED_REFUSAL
        )
    if DIAGNOSIS_REQUEST.search(normalized) or (
        personal and SYMPTOM_OR_RESULT.search(normalized)
    ):
        return SafetyDecision(False, "diagnosis", normalized, None, PERSONALIZED_REFUSAL)
    if OBVIOUS_NONMEDICAL_TOPIC.search(normalized):
        return SafetyDecision(
            False, "outside_corpus", normalized, None, OUT_OF_SCOPE_REFUSAL
        )

    return SafetyDecision(
        True,
        "allowed_guideline_query",
        normalized,
        rewrite_query(normalized),
        None,
    )
