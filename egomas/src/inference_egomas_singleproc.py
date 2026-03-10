"""
Single-process inference for MA-EgoQA using planner + retriever + answer pipeline.
"""
import os
import random
from typing import Any

random.seed(42)

from google import genai
from google.genai import types
from tqdm import tqdm

from egomas.src.index_bm25 import BM25TextRetriever
from egomas.utils.io import load_benchmark, save_json
from egomas.utils.parsing import get_prediction_index, parse_planner_response
from egomas.utils.prompt_helpers import (
    build_answer_prompt,
    build_planner_prompt,
    build_question_prompt,
    get_context_text,
)
from egomas.src.retrieval_helpers import (
    DEFAULT_AGENT_MEMORY_TOP_K,
    DEFAULT_RETRIEVE_TOP_K,
    DEFAULT_RETRIEVE_TOP_K_FALLBACK,
    DEFAULT_SCORE_THRESHOLD,
    retrieve_for_person,
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BENCHMARK_PATH = "data/MA-EgoQA_bm25.json"
OUTPUT_PATH = "egomas_result.json"
BM25_PKL_PATH = "data/30sec_bm25.pkl"
MODEL_NAME = "models/gemini-2.5-flash"

SHARED_MEMORY_TOP_K = 20
AGENT_MEMORY_TOP_K = DEFAULT_AGENT_MEMORY_TOP_K
PLANNER_MAX_TOKENS = 10000
ANSWER_MAX_TOKENS = 4096


# -----------------------------------------------------------------------------
# Planner
# -----------------------------------------------------------------------------
def run_planner(
    client: genai.Client,
    shared_context: str,
    question_prompt: str,
    fallback_question: str,
    verbose: bool = True,
) -> list[dict]:
    """Call planner model and return parsed selection list."""
    prompt = build_planner_prompt(shared_context, question_prompt)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=PLANNER_MAX_TOKENS),
    )
    raw = (response.candidates[0].content.parts[0].text or "").strip()
    if verbose:
        print("Planner text:", raw)
    selection = parse_planner_response(raw, fallback_question)
    if verbose:
        print("Selection:", selection)
    return selection


# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------
def retrieve_agent_contexts(
    retriever: BM25TextRetriever,
    selection: list[dict],
) -> list[str]:
    """For each planner selection, retrieve top-k agent memories and return formatted strings."""
    retrieved_list: list[str] = []
    for s in selection:
        try:
            name, query = s["name"], s["query"]
        except (KeyError, TypeError):
            continue
        ctx = retrieve_for_person(
            retriever,
            name,
            query,
            top_k=AGENT_MEMORY_TOP_K,
            top_k_retrieve=DEFAULT_RETRIEVE_TOP_K,
            top_k_fallback=DEFAULT_RETRIEVE_TOP_K_FALLBACK,
            score_threshold=DEFAULT_SCORE_THRESHOLD,
        )
        if ctx:
            retrieved_list.append(ctx)
    return retrieved_list


# -----------------------------------------------------------------------------
# Answer model
# -----------------------------------------------------------------------------
def run_answer(
    client: genai.Client,
    shared_context: str,
    retrieved_contexts: list[str],
    question_prompt: str,
    verbose: bool = True,
) -> str:
    """Build answer prompt, call model, return raw prediction string."""
    answer_prompt = build_answer_prompt(
        shared_context, retrieved_contexts, question_prompt
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=answer_prompt,
        config=types.GenerateContentConfig(max_output_tokens=ANSWER_MAX_TOKENS),
    )
    pred = (response.candidates[0].content.parts[0].text or "").strip()
    if verbose:
        print("Final pred:", pred)
    return pred


# -----------------------------------------------------------------------------
# Per-item pipeline & main
# -----------------------------------------------------------------------------
def process_item(
    elem: dict[str, Any],
    client: genai.Client,
    retriever: BM25TextRetriever,
    verbose: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Run planner -> retrieval -> answer for one benchmark item. Returns (elem with 'pred', is_correct)."""
    question_prompt = build_question_prompt(elem["question"], elem["options"])
    contexts = elem["bm25"][:SHARED_MEMORY_TOP_K]
    shared_context = get_context_text(contexts)

    selection = run_planner(
        client, shared_context, question_prompt, elem["question"], verbose=verbose
    )
    retrieved = retrieve_agent_contexts(retriever, selection)
    pred = run_answer(
        client, shared_context, retrieved, question_prompt, verbose=verbose
    )

    pred_idx = get_prediction_index(pred)
    gt_idx = elem["options"].index(elem["answer"])
    correct = pred_idx == gt_idx

    return {**elem, "pred": pred}, correct


def main(
    benchmark_path: str = BENCHMARK_PATH,
    output_path: str = OUTPUT_PATH,
    bm25_path: str = BM25_PKL_PATH,
    verbose: bool = True,
) -> None:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    retriever = BM25TextRetriever.load_vectorized_format(bm25_path)
    benchmark_data = load_benchmark(benchmark_path)

    results = []
    correct_count = 0

    for elem in tqdm(benchmark_data, total=len(benchmark_data)):
        elem_out, is_correct = process_item(
            elem, client, retriever, verbose=verbose
        )
        results.append(elem_out)
        if is_correct:
            correct_count += 1
        if verbose:
            print(f"Accuracy: {correct_count / len(results):.2%}")

    save_json(results, output_path)
    print(f"Saved {len(results)} results to {output_path}")
    print(f"Final accuracy: {correct_count / len(results):.2%}")


if __name__ == "__main__":
    main()
