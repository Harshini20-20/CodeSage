"""Prompt templates (placeholders - final wording comes with the LLM step)."""

from __future__ import annotations

EXPLANATION_PROMPT_TEMPLATE: str = ""  # TODO: define in the LLM implementation step.


def build_explanation_prompt(question: str, context_chunks: list[str]) -> str:
    """Build the LLM prompt from a question and retrieved context."""
    raise NotImplementedError("Prompt building is not implemented yet.")
