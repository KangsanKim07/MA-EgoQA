"""
Data loading utilities for EgoMAS.
"""
import json
import os


def load_json(path: str) -> dict | list:
    """Load a single JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict | list, path: str, indent: int = 4) -> None:
    """Save data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_bm25_data(path: str) -> list:
    """Load MA-EgoQA dataset with BM25 pre-retrieved contexts."""
    return load_json(path)


def load_min_captions(caption_dir: str) -> dict:
    """Load and merge all caption JSON files from a directory."""
    merged = {}
    for filename in os.listdir(caption_dir):
        filepath = os.path.join(caption_dir, filename)
        if not filename.endswith(".json"):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            merged.update(json.load(f))
    return merged


def load_benchmark(path: str) -> list:
    """Load benchmark JSON (e.g. MA-EgoQA_bm25.json)."""
    return load_json(path)
