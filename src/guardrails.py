"""Deterministic safety and reliability controls for the RAG workflow."""

from dataclasses import dataclass
import re
from typing import Sequence

from src.rag_pipeline import GroundedAnswer, NO_EVIDENCE_ANSWER
from src.retriever import RetrievedPassage


MAX_QUESTION_CHARACTERS = 1_000
MAX_ANSWER_CHARACTERS = 8_000

CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|system)\s+instructions?", re.I),
    re.compile(r"reveal\s+(the\s+)?(system|developer)\s+prompt", re.I),
    re.compile(r"show\s+(me\s+)?(your\s+)?hidden\s+instructions?", re.I),
    re.compile(r"bypass\s+(the\s+)?(safety|guardrails?|rules?)", re.I),
)


@dataclass(frozen=True)
class SafetyCheck:
    """One inspectable guardrail decision."""

    name: str
    passed: bool
    detail: str


def validate_question(question: str) -> tuple[str, tuple[SafetyCheck, ...]]:
    """Normalize a question and reject malformed or adversarial instructions."""
    normalized = " ".join(question.split())
    checks: list[SafetyCheck] = []

    length_ok = 5 <= len(normalized) <= MAX_QUESTION_CHARACTERS
    checks.append(
        SafetyCheck(
            "Question length",
            length_ok,
            f"Question contains {len(normalized)} characters.",
        )
    )
    if not length_ok:
        raise ValueError(
            f"Questions must contain 5 to {MAX_QUESTION_CHARACTERS:,} characters."
        )

    characters_ok = CONTROL_CHARACTER_PATTERN.search(normalized) is None
    checks.append(
        SafetyCheck(
            "Control characters",
            characters_ok,
            "No unsafe control characters were detected.",
        )
    )
    if not characters_ok:
        raise ValueError("The question contains unsupported control characters.")

    injection_match = next(
        (pattern for pattern in INJECTION_PATTERNS if pattern.search(normalized)), None
    )
    injection_ok = injection_match is None
    checks.append(
        SafetyCheck(
            "Prompt injection",
            injection_ok,
            "No instruction-bypass pattern was detected.",
        )
    )
    if not injection_ok:
        raise ValueError(
            "This question appears to request bypassing the assistant's instructions. "
            "Ask a direct question about the uploaded documents instead."
        )
    return normalized, tuple(checks)


def validate_final_answer(
    answer: GroundedAnswer,
    passages: Sequence[RetrievedPassage],
) -> tuple[SafetyCheck, ...]:
    """Check answer size, evidence labels, and refusal behavior before display."""
    text = answer.text.strip()
    if not text:
        raise ValueError("The final answer is empty.")
    if len(text) > MAX_ANSWER_CHARACTERS:
        raise ValueError("The final answer exceeds the configured safety limit.")

    allowed = {f"S{passage.rank}" for passage in passages}
    invalid = set(answer.citation_ids) - allowed
    if invalid:
        raise ValueError(
            "The final answer contains invalid citations: "
            + ", ".join(sorted(invalid))
        )
    if text != NO_EVIDENCE_ANSWER and not answer.citation_ids:
        raise ValueError("The final answer must cite at least one evidence passage.")

    return (
        SafetyCheck(
            "Answer length",
            True,
            f"Answer contains {len(text)} characters.",
        ),
        SafetyCheck(
            "Evidence citations",
            True,
            "Every source label points to a retrieved passage.",
        ),
        SafetyCheck(
            "Human review",
            True,
            "The interface keeps supporting passages available for verification.",
        ),
    )
