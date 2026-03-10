"""
EgoMAS utilities: constants, I/O, parsing, prompts, retrieval formatting, evaluation.
"""
from egomas.utils.constants import (
    CODEBLOCK_PATTERN,
    PERSON_NAMES,
    VALID_OPTIONS,
)
from egomas.utils.eval import compute_accuracy
from egomas.utils.io import (
    load_benchmark,
    load_bm25_data,
    load_json,
    load_min_captions,
    save_json,
)
from egomas.utils.parsing import (
    extract_codeblock_text,
    get_prediction_index,
    normalize_prediction,
    parse_planner_response,
)
from egomas.utils.prompt_helpers import (
    build_answer_prompt,
    build_planner_prompt,
    build_question_prompt,
    get_context_text,
)
from egomas.utils.retrieval_format import (
    format_retrieved_context,
    format_retrieved_item,
)

__all__ = [
    "CODEBLOCK_PATTERN",
    "PERSON_NAMES",
    "VALID_OPTIONS",
    "compute_accuracy",
    "load_benchmark",
    "load_bm25_data",
    "load_json",
    "load_min_captions",
    "save_json",
    "extract_codeblock_text",
    "get_prediction_index",
    "normalize_prediction",
    "parse_planner_response",
    "build_answer_prompt",
    "build_planner_prompt",
    "build_question_prompt",
    "get_context_text",
    "format_retrieved_context",
    "format_retrieved_item",
]
