"""
EgoMAS inference: multi-agent memory retrieval + Gemini for MA-EgoQA.
"""
import os
import random
from multiprocessing import Pool, cpu_count, Manager

from google import genai
from google.genai import types
from tqdm import tqdm

from egomas.src.index_bm25 import BM25TextRetriever
from egomas.src.retrieval_helpers import (
    DEFAULT_AGENT_MEMORY_TOP_K,
    DEFAULT_RETRIEVE_TOP_K,
    DEFAULT_RETRIEVE_TOP_K_FALLBACK,
    DEFAULT_SCORE_THRESHOLD,
    retrieve_for_person,
)
from egomas.utils.eval import compute_accuracy
from egomas.utils.io import load_bm25_data
from egomas.utils.parsing import parse_planner_response
from egomas.utils.prompt_helpers import (
    build_answer_prompt,
    build_planner_prompt,
    build_question_prompt,
    get_context_text,
)

random.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CAPTION_DIR = "data/caption/10min"
BM25_DATA_PATH = "data/MA-EgoQA_bm25.json"
BM25_INDEX_PATH = "data/30sec_bm25.pkl"
MODEL_NAME = "models/gemini-2.5-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

SHARED_MEMORY_TOP_K = 20
AGENT_MEMORY_TOP_K = DEFAULT_AGENT_MEMORY_TOP_K
RETRIEVE_TOP_K = DEFAULT_RETRIEVE_TOP_K
RETRIEVE_TOP_K_FALLBACK = DEFAULT_RETRIEVE_TOP_K_FALLBACK
SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD


# ---------------------------------------------------------------------------
# Single-sample pipeline
# ---------------------------------------------------------------------------
def run_planner(client, contexts: list, question_prompt: str) -> list[dict]:
    """Call Gemini to get (name, query) list for agent memory retrieval."""
    context_text = get_context_text(contexts)
    planner_prompt = build_planner_prompt(context_text, question_prompt)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=planner_prompt,
        config=types.GenerateContentConfig(max_output_tokens=20000),
    )
    text = response.candidates[0].content.parts[0].text or ""
    return parse_planner_response(text, question_prompt)


def run_answer(
    client, shared_contexts: list, retrieved_contexts: list[str], question_prompt: str
) -> str:
    """Call Gemini to get final answer (single letter A–E)."""
    shared_text = get_context_text(shared_contexts)
    answer_prompt = build_answer_prompt(
        shared_text, retrieved_contexts, question_prompt
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=answer_prompt,
        config=types.GenerateContentConfig(max_output_tokens=20000),
    )
    return (response.candidates[0].content.parts[0].text or "").strip()


def process_one_elem(
    elem: dict, client, retriever: BM25TextRetriever
) -> dict:
    """Run planner -> retrieval -> answer for one QA item; attach elem['pred']."""
    question_prompt = build_question_prompt(elem["question"], elem["options"])
    shared_contexts = elem["bm25"][:SHARED_MEMORY_TOP_K]

    selection = run_planner(client, shared_contexts, question_prompt)

    retrieved = []
    for s in selection:
        try:
            name, query = s["name"], s["query"]
        except (KeyError, TypeError):
            continue
        ctx = retrieve_for_person(
        retriever, name, query,
        top_k=AGENT_MEMORY_TOP_K,
        top_k_retrieve=RETRIEVE_TOP_K,
        top_k_fallback=RETRIEVE_TOP_K_FALLBACK,
        score_threshold=SCORE_THRESHOLD,
    )
        retrieved.append(ctx)

    pred = run_answer(client, shared_contexts, retrieved, question_prompt)
    elem["pred"] = pred
    return elem


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------
def worker(chunk: list, result_queue) -> None:
    """Load retriever and client, process chunk, put each result on queue."""
    retriever = BM25TextRetriever.load_vectorized_format(BM25_INDEX_PATH)
    client = genai.Client(api_key=os.getenv(GEMINI_API_KEY_ENV, ""))
    for elem in chunk:
        try:
            out_elem = process_one_elem(elem, client, retriever)
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            elem["pred"] = ""
            out_elem = elem
        result_queue.put(out_elem)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    bm25_data = load_bm25_data(BM25_DATA_PATH)
    data_to_process = bm25_data
    total = len(data_to_process)
    n_workers = min(32, max(1, cpu_count() - 1))
    chunk_size = max(1, (total + n_workers - 1) // n_workers)
    chunks = [
        data_to_process[i : i + chunk_size]
        for i in range(0, total, chunk_size)
    ]

    with Manager() as manager:
        result_queue = manager.Queue()
        with Pool(n_workers) as pool:
            async_handles = [
                pool.apply_async(worker, (c, result_queue)) for c in chunks
            ]
            result_list = []
            pbar = tqdm(total=total, unit="sample")
            for _ in range(total):
                elem = result_queue.get()
                result_list.append(elem)
                acc = compute_accuracy(result_list)
                pbar.update(1)
                pbar.set_postfix_str(f"Acc: {acc:.2%}")
            pbar.close()
            for h in async_handles:
                h.get()

    acc = compute_accuracy(result_list)
    print(f"Accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()
