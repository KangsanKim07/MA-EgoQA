"""
Evaluation utilities for MA-EgoQA.
"""
from egomas.utils.parsing import get_prediction_index


def compute_accuracy(result_list: list[dict]) -> float:
    """Accuracy: fraction of samples where pred matches ground-truth option index."""
    if not result_list:
        return 0.0
    correct = sum(
        1
        for e in result_list
        if get_prediction_index(e.get("pred") or "") == e["options"].index(e["answer"])
    )
    return correct / len(result_list)
