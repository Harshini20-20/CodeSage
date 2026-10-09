"""Validate source labels in generated CodeSage answers (Phase 2E).

This lightweight check validates citation *references*, not the semantic truth of
an answer. It ensures that explicit [SOURCE N] markers point to sources supplied
to the model and reports invalid markers rather than silently trusting them.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

_SOURCE_REFERENCE = re.compile(r"\[\s*SOURCE\s+(\d+)\s*\]", re.IGNORECASE)


@dataclass(frozen=True)
class CitationValidationResult:
    """Generated answer with invalid source references identified."""

    answer: str
    invalid_source_references: tuple[int, ...] = ()


def validate_source_citations(answer: str, source_count: int) -> CitationValidationResult:
    """Replace invalid [SOURCE N] markers and report the referenced indices.

    Valid indices are one-based because prompts label context blocks starting at
    SOURCE 1. This function deliberately does not invent citations for uncited
    claims or attempt to prove semantic entailment.
    """
    if not isinstance(answer, str):
        answer = str(answer)
    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 0:
        raise ValueError("source_count must be a non-negative integer")

    invalid: list[int] = []

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 1 <= index <= source_count:
            return match.group(0)
        invalid.append(index)
        return "[invalid source reference removed]"

    cleaned = _SOURCE_REFERENCE.sub(replace, answer)
    unique_invalid = tuple(dict.fromkeys(invalid))
    if unique_invalid:
        refs = ", ".join(f"SOURCE {index}" for index in unique_invalid)
        cleaned += (
            "\n\nEvidence note: the answer included source references not present "
            f"in the retrieved context ({refs}); those reference markers were removed. "
            "This check validates source labels, not the truth of every claim."
        )
    return CitationValidationResult(cleaned, unique_invalid)


__all__ = ["CitationValidationResult", "validate_source_citations"]
