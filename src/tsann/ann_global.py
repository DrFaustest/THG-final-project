import time

import numpy as np

from tsann.config import IndexConfig
from tsann.distance import l2
from tsann.filters import passes_filters, validate_query
from tsann.indexes.base import BaseTemporalSubsetIndex
from tsann.indexes.hnsw_wrapper import HnswVectorIndex
from tsann.metrics import merge_topk
from tsann.types import Query, Record, SearchResult


class GlobalAnnThenFilterIndex(BaseTemporalSubsetIndex):
    def __init__(self, config: IndexConfig | None = None) -> None:
        self.config = config or IndexConfig()
        self.records: list[Record] = []
        self.id_to_record: dict[int, Record] = {}
        self.index: HnswVectorIndex | None = None
        self.dim: int | None = None

    def build(self, records: list[Record]) -> None:
        self.records = []
        self.id_to_record = {}
        for record in records:
            if record.id in self.id_to_record:
                raise ValueError(f"Duplicate record id {record.id}")
            self.records.append(record)
            self.id_to_record[record.id] = record
        self._rebuild_vector_index()

    def insert(self, record: Record) -> None:
        if record.id in self.id_to_record:
            raise ValueError(f"Duplicate record id {record.id}")
        self.records.append(record)
        self.id_to_record[record.id] = record
        self._rebuild_vector_index()

    def _rebuild_vector_index(self) -> None:
        if not self.records:
            self.index = None
            self.dim = None
            return
        self.dim = int(self.records[0].vector.shape[0])
        ids = [record.id for record in self.records]
        vectors = np.stack([record.vector.astype(np.float32) for record in self.records])
        self.index = HnswVectorIndex(
            self.dim,
            ef_construction=self.config.hnsw_ef_construction,
            m=self.config.hnsw_m,
            ef_search=self.config.hnsw_ef_search,
        )
        self.index.build(ids, vectors)

    def search(self, query: Query) -> SearchResult:
        validate_query(query)
        start = time.perf_counter()
        if self.index is None or query.k == 0:
            return SearchResult([], [], self._metadata(start, 0, 0, 0, 0, 0))

        max_budget = self.config.global_max_budget or len(self.records)
        budget = min(max(self.config.global_min_budget, self.config.global_initial_alpha * query.k), max_budget)
        rounds = 0
        final_items: list[tuple[int, float]] = []
        candidate_count = 0
        filtered_out = 0

        while budget <= max_budget:
            rounds += 1
            cand_ids, _ = self.index.knn_query(query.vector, budget)
            candidate_count = len(cand_ids)
            valid: list[tuple[int, float]] = []
            filtered_out = 0
            for rid in cand_ids:
                record = self.id_to_record[rid]
                if passes_filters(record, query):
                    valid.append((rid, l2(query.vector, record.vector)))
                else:
                    filtered_out += 1
                if len(valid) >= query.k:
                    break
            final_items = valid
            if len(valid) >= query.k or budget == max_budget:
                break
            budget = min(budget * 2, max_budget)

        ids, distances = merge_topk(final_items, query.k)
        return SearchResult(ids, distances, self._metadata(start, candidate_count, filtered_out, rounds, budget, len(final_items)))

    def _metadata(
        self,
        start: float,
        candidate_count: int,
        filtered_out: int,
        rounds: int,
        budget: int,
        exact_distance_count: int,
    ) -> dict:
        return {
            "algorithm": "global_filter",
            "visited_partitions": 0,
            "candidate_count": candidate_count,
            "filtered_out_count": filtered_out,
            "ann_expansion_rounds": rounds,
            "candidate_budget_final": budget,
            "filter_pass_rate": 0.0 if candidate_count == 0 else (candidate_count - filtered_out) / candidate_count,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "exact_distance_computations": exact_distance_count,
        }

    def stats(self) -> dict:
        return {
            "algorithm": "global_filter",
            "records": len(self.records),
            "hnsw_available": bool(self.index and self.index.available),
        }
