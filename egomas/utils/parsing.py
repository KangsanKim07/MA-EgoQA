"""
Parsing utilities for planner responses and model predictions.
"""
import ast
import json
import random

from egomas.utils.constants import CODEBLOCK_PATTERN, PERSON_NAMES, VALID_OPTIONS


def extract_codeblock_text(text: str) -> str:
    """Extract content from optional ```json/``` code block."""
    match = CODEBLOCK_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_planner_response(planner_text: str, fallback_question: str) -> list[dict]:
    """
    Parse planner output into list of {name, query}. On failure, return
    a single random person with the original question as query.
    """
    raw = extract_codeblock_text(planner_text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass
    return [
        {"name": random.choice(PERSON_NAMES), "query": fallback_question}
    ]


def normalize_prediction(pred: str) -> str:
    """Normalize model prediction to single letter a–e."""
    if not pred:
        return ""
    s = (
        pred.lower()
        .strip("*")
        .strip("option ")
        .replace("the correct answer is", "")
        .replace("<answer>", "")
        .strip(":")
        .strip()
    )
    return s[0] if s else ""


def get_prediction_index(pred: str) -> int:
    """Return index of predicted option (0–4) or -1 if invalid."""
    normalized = normalize_prediction(pred)
    if normalized in VALID_OPTIONS:
        return VALID_OPTIONS.index(normalized)
    return -1
