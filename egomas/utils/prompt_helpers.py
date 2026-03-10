"""
Prompt building helpers for EgoMAS inference.
"""
from egomas.utils.prompts import ANSWER_HEADER, PLANNER_SYSTEM


def build_question_prompt(question: str, options: list) -> str:
    """Build question + options prompt (A–E)."""
    lines = [question, "Options:"]
    for label, opt in zip("ABCDE", options):
        lines.append(f"{label}) {opt}")
    return "\n".join(lines)


def get_context_text(contexts: list[dict]) -> str:
    """Join context dicts into a single string (caption field)."""
    return "\n\n".join(c["caption"] for c in contexts)


def build_planner_prompt(shared_context: str, question_prompt: str) -> str:
    """Build full planner prompt from shared context and question."""
    return PLANNER_SYSTEM.format(
        context=shared_context,
        question=question_prompt,
    )


def build_answer_prompt(
    shared_context: str,
    retrieved_contexts: list[str],
    question_prompt: str,
) -> str:
    """Build full answer prompt from shared context, retrieved contexts, and question."""
    return ANSWER_HEADER.format(
        shared_context=shared_context,
        retrieved_contexts="\n\n".join(retrieved_contexts),
        question=question_prompt,
    )
