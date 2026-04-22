import math
from collections.abc import Iterable

from tsann.filters import passes_filters
from tsann.types import Query, Record, SearchResult


def recall_at_k(result: SearchResult, truth: SearchResult, k: int) -> float:
    if k <= 0:
        return 1.0
    truth_ids = truth.ids[:k]
    if not truth_ids:
        return 1.0 if not result.ids else 0.0
    return len(set(result.ids[:k]) & set(truth_ids)) / len(truth_ids)


def recall_at_1(result: SearchResult, truth: SearchResult) -> float:
    return recall_at_k(result, truth, 1)


def valid_result_rate(result: SearchResult, query: Query, records: dict[int, Record]) -> float:
    if not result.ids:
        return 1.0
    valid = sum(1 for rid in result.ids if rid in records and passes_filters(records[rid], query))
    return valid / len(result.ids)


def ndcg_at_k(result: SearchResult, truth: SearchResult, k: int) -> float:
    if k <= 0 or not truth.ids:
        return 1.0
    relevance = {rid: max(len(truth.ids) - idx, 0) for idx, rid in enumerate(truth.ids[:k])}

    def dcg(ids: Iterable[int]) -> float:
        score = 0.0
        for idx, rid in enumerate(list(ids)[:k]):
            rel = relevance.get(rid, 0)
            score += rel / math.log2(idx + 2)
        return score

    ideal = dcg(truth.ids[:k])
    return 1.0 if ideal == 0 else dcg(result.ids[:k]) / ideal


def merge_topk(items: Iterable[tuple[int, float]], k: int) -> tuple[list[int], list[float]]:
    best_by_id: dict[int, float] = {}
    for rid, dist in items:
        if rid not in best_by_id or dist < best_by_id[rid]:
            best_by_id[rid] = dist
    ranked = sorted(best_by_id.items(), key=lambda item: (item[1], item[0]))[:k]
    return [rid for rid, _ in ranked], [dist for _, dist in ranked]
