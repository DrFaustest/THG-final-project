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
        self.active_ids: set[int] = set()
        self.index: HnswVectorIndex | None = None
        self.dim: int | None = None
        self.rebuild_count = 0
        self.compaction_count = 0
        self.deleted_record_count = 0
        self.expired_record_count = 0

    def build(self, records: list[Record]) -> None:
        self.records = []
        self.id_to_record = {}
        self.active_ids = set()
        self.rebuild_count = 0
        self.compaction_count = 0
        self.deleted_record_count = 0
        self.expired_record_count = 0
        for record in records:
            if record.id in self.id_to_record:
                raise ValueError(f"Duplicate record id {record.id}")
            self.records.append(record)
            self.id_to_record[record.id] = record
            self.active_ids.add(record.id)
        self._rebuild_vector_index()

    def insert(self, record: Record) -> None:
        if record.id in self.id_to_record:
            raise ValueError(f"Duplicate record id {record.id}")
        self.records.append(record)
        self.id_to_record[record.id] = record
        self.active_ids.add(record.id)
        self._rebuild_vector_index()

    def delete(self, record_id: int) -> None:
        if record_id not in self.id_to_record:
            raise KeyError(record_id)
        if record_id in self.active_ids:
            self.active_ids.remove(record_id)
            self.deleted_record_count += 1
            self._maybe_rebuild_for_tombstones()

    def expire(self, before_time: int) -> int:
        expired = [
            record.id
            for record in self.records
            if record.id in self.active_ids and record.valid_to is not None and record.valid_to < before_time
        ]
        for record_id in expired:
            self.active_ids.remove(record_id)
        self.expired_record_count += len(expired)
        self._maybe_rebuild_for_tombstones()
        return len(expired)

    def _rebuild_vector_index(self, *, compact: bool = False) -> None:
        active_records = [record for record in self.records if record.id in self.active_ids]
        if compact:
            removed = len(self.records) - len(active_records)
            if removed > 0:
                self.records = active_records
                self.compaction_count += 1
        if not active_records:
            self.index = None
            self.dim = None
            return
        self.dim = int(active_records[0].vector.shape[0])
        ids = [record.id for record in active_records]
        vectors = np.stack([record.vector.astype(np.float32) for record in active_records])
        self.index = HnswVectorIndex(
            self.dim,
            ef_construction=self.config.hnsw_ef_construction,
            m=self.config.hnsw_m,
            ef_search=self.config.hnsw_ef_search,
        )
        self.index.build(ids, vectors)
        self.rebuild_count += 1

    def _maybe_rebuild_for_tombstones(self) -> None:
        if not self.records:
            return
        tombstone_ratio = 1.0 - (len(self.active_ids) / len(self.records))
        if tombstone_ratio > self.config.global_rebuild_tombstone_ratio:
            self._rebuild_vector_index(compact=True)

    def search(self, query: Query) -> SearchResult:
        validate_query(query)
        start = time.perf_counter()
        if self.index is None or query.k == 0 or not self.active_ids:
            return SearchResult([], [], self._metadata(start, 0, 0, 0, 0, 0))

        index_visible_size = self.index.size
        max_budget = min(self.config.global_max_budget or index_visible_size, index_visible_size)
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
                if rid in self.active_ids and passes_filters(record, query):
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
        return SearchResult(
            ids,
            distances,
            self._metadata(start, candidate_count, filtered_out, rounds, budget, len(final_items), index_visible_size),
        )

    def _metadata(
        self,
        start: float,
        candidate_count: int,
        filtered_out: int,
        rounds: int,
        budget: int,
        exact_distance_count: int,
        index_visible_size: int = 0,
    ) -> dict:
        return {
            "algorithm": "global_filter",
            "visited_partitions": 0,
            "candidate_count": candidate_count,
            "filtered_out_count": filtered_out,
            "ann_expansion_rounds": rounds,
            "candidate_budget_final": budget,
            "active_record_count": len(self.active_ids),
            "index_visible_size": index_visible_size,
            "filter_pass_rate": 0.0 if candidate_count == 0 else (candidate_count - filtered_out) / candidate_count,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "exact_distance_computations": exact_distance_count,
        }

    def stats(self) -> dict:
        return {
            "algorithm": "global_filter",
            "records": len(self.records),
            "active_records": len(self.active_ids),
            "tombstoned_records": len(self.records) - len(self.active_ids),
            "rebuild_count": self.rebuild_count,
            "compaction_count": self.compaction_count,
            "deleted_record_count": self.deleted_record_count,
            "expired_record_count": self.expired_record_count,
            "index_visible_size": self.index.size if self.index is not None else 0,
            "hnsw_available": bool(self.index and self.index.available),
        }
