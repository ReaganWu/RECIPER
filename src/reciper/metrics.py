from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked_ids[:k] if item in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    relevant = set(relevant_ids)
    score = 0.0
    for rank, item in enumerate(ranked_ids[:k], start=1):
        if item in relevant:
            score += 1.0 / math.log2(rank + 1)
    return score


def ndcg_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    ideal = dcg_at_k(list(relevant_ids), relevant_ids, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevant_ids, k) / ideal


def mrr(ranked_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, item in enumerate(ranked_ids, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
