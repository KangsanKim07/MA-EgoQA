"""
Shared retrieval logic for EgoMAS: BM25 retrieve + filter by person + format.
"""
from egomas.src.index_bm25 import BM25TextRetriever
from egomas.utils.retrieval_format import format_retrieved_context

# Defaults (can be overridden by callers)
DEFAULT_AGENT_MEMORY_TOP_K = 5
DEFAULT_RETRIEVE_TOP_K = 100
DEFAULT_RETRIEVE_TOP_K_FALLBACK = 1000
DEFAULT_SCORE_THRESHOLD = 10.0


def retrieve_for_person(
    retriever: BM25TextRetriever,
    name: str,
    query: str,
    top_k: int = DEFAULT_AGENT_MEMORY_TOP_K,
    top_k_retrieve: int = DEFAULT_RETRIEVE_TOP_K,
    top_k_fallback: int = DEFAULT_RETRIEVE_TOP_K_FALLBACK,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> str:
    """Retrieve captions for one person and format as context string."""
    name_upper = name.upper()
    results, captions, scores = retriever.retrieve(
        query, top_k=top_k_retrieve, return_scores=True
    )
    named_results = [
        {"id": rid, "caption": cap, "score": sc}
        for rid, cap, sc in zip(results, captions, scores)
        if name_upper in rid
    ]
    if len(named_results) < top_k:
        results, captions, scores = retriever.retrieve(
            query, top_k=top_k_fallback, return_scores=True
        )
        named_results = [
            {"id": rid, "caption": cap, "score": sc}
            for rid, cap, sc in zip(results, captions, scores)
            if name_upper in rid
        ]
    return format_retrieved_context(
        named_results, top_k=top_k, score_threshold=score_threshold
    )
