"""
Formatting utilities for BM25 retrieval results.
"""


def format_retrieved_item(item_id: str, caption: str) -> str:
    """Format a single retrieved item (id, caption) into a readable line."""
    parts = item_id.split("_")
    if len(parts) >= 4:
        day, _, name, ctime = parts[0], parts[1], parts[2], parts[3]
        time_str = f"{ctime[:2]}:{ctime[2:4]}:{ctime[4:6]}"
        return f"[{day} {time_str}] {name.capitalize()}: {caption}"
    return f"{item_id}: {caption}"


def format_retrieved_context(
    named_results: list[dict],
    top_k: int = 5,
    score_threshold: float = 10.0,
) -> str:
    """Format retrieved (id, caption, score) list into a single context string."""
    lines = []
    for r in named_results[:top_k]:
        id_, caption, score = r["id"], r["caption"], r["score"]
        if score < score_threshold:
            continue
        lines.append(format_retrieved_item(id_, caption))
    return "\n".join(lines)
