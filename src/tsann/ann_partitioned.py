from dataclasses import dataclass, field
from collections import Counter
import time

import numpy as np

from tsann.config import IndexConfig
from tsann.distance import l2
from tsann.filters import passes_filters, validate_query
from tsann.indexes.base import BaseTemporalSubsetIndex
from tsann.indexes.hnsw_wrapper import HnswVectorIndex
from tsann.metrics import merge_topk
from tsann.partitioning.routing import CellKey, TimePriceRouter
from tsann.partitioning.scalar_buckets import FixedWidthPriceBucketizer
from tsann.partitioning.time_buckets import FixedTimeBucketizer
from tsann.types import Query, Record, SearchResult, SubsetEstimate


@dataclass
class PartitionCell:
    key: CellKey
    records: list[Record] = field(default_factory=list)
    index: HnswVectorIndex | None = None
    rebuild_count: int = 0
    compaction_count: int = 0
    active_record_count: int = 0
    min_valid_from: int | None = None
    max_valid_from: int | None = None
    min_valid_to: int | None = None
    max_valid_to: int | None = None
    open_ended_count: int = 0
    min_price: float | None = None
    max_price: float | None = None
    category_counts: Counter[int | None] = field(default_factory=Counter)

    @property
    def record_count(self) -> int:
        return len(self.records)

    def active_records(self, active_ids: set[int]) -> list[Record]:
        return [record for record in self.records if record.id in active_ids]

    def active_count(self, active_ids: set[int]) -> int:
        return self.active_record_count

    def inactive_count(self, active_ids: set[int]) -> int:
        return len(self.records) - self.active_record_count

    def add_active_record(self, record: Record) -> None:
        self.records.append(record)
        self.active_record_count += 1
        self._include_metadata(record)

    def refresh_metadata(self, active_ids: set[int]) -> None:
        self.active_record_count = 0
        self.min_valid_from = None
        self.max_valid_from = None
        self.min_valid_to = None
        self.max_valid_to = None
        self.open_ended_count = 0
        self.min_price = None
        self.max_price = None
        self.category_counts = Counter()
        for record in self.records:
            if record.id in active_ids:
                self.active_record_count += 1
                self._include_metadata(record)

    def can_intersect(self, query: Query) -> bool:
        if self.active_record_count == 0:
            return False
        if self.min_valid_from is not None and self.min_valid_from > query.t_end:
            return False
        if self.open_ended_count == 0 and self.max_valid_to is not None and self.max_valid_to < query.t_start:
            return False
        if self.min_price is not None and self.min_price > query.price_max:
            return False
        if self.max_price is not None and self.max_price < query.price_min:
            return False
        return True

    def _include_metadata(self, record: Record) -> None:
        self.min_valid_from = record.valid_from if self.min_valid_from is None else min(self.min_valid_from, record.valid_from)
        self.max_valid_from = record.valid_from if self.max_valid_from is None else max(self.max_valid_from, record.valid_from)
        if record.valid_to is None:
            self.open_ended_count += 1
        else:
            self.min_valid_to = record.valid_to if self.min_valid_to is None else min(self.min_valid_to, record.valid_to)
            self.max_valid_to = record.valid_to if self.max_valid_to is None else max(self.max_valid_to, record.valid_to)
        self.min_price = record.price if self.min_price is None else min(self.min_price, record.price)
        self.max_price = record.price if self.max_price is None else max(self.max_price, record.price)
        self.category_counts[record.category] += 1

    def compact(self, active_ids: set[int]) -> int:
        active_records = self.active_records(active_ids)
        removed = len(self.records) - len(active_records)
        if removed > 0:
            self.records = active_records
            self.compaction_count += 1
        self.refresh_metadata(active_ids)
        return removed

    def build_index(self, config: IndexConfig, active_ids: set[int], *, compact: bool = False) -> None:
        if compact:
            active_records = self.active_records(active_ids)
            self.compact(active_ids)
        else:
            active_records = self.active_records(active_ids)
        if not active_records or len(active_records) < config.partition_exact_threshold:
            self.index = None
            self.rebuild_count += 1
            return
        dim = int(active_records[0].vector.shape[0])
        self.index = HnswVectorIndex(
            dim,
            ef_construction=config.hnsw_ef_construction,
            m=config.hnsw_m,
            ef_search=config.hnsw_ef_search,
        )
        self.index.build(
            [record.id for record in active_records],
            np.stack([record.vector.astype(np.float32) for record in active_records]),
        )
        self.rebuild_count += 1


class PartitionFirstAnnIndex(BaseTemporalSubsetIndex):
    def __init__(self, config: IndexConfig | None = None) -> None:
        self.config = config or IndexConfig()
        self.router = TimePriceRouter(
            FixedTimeBucketizer(self.config.time_bucket_width),
            FixedWidthPriceBucketizer(self.config.price_bucket_width),
        )
        self.cells: dict[CellKey, PartitionCell] = {}
        self.records: list[Record] = []
        self.id_to_record: dict[int, Record] = {}
        self.active_ids: set[int] = set()
        self.deleted_record_count = 0
        self.expired_record_count = 0

    def build(self, records: list[Record]) -> None:
        self.cells = {}
        self.records = []
        self.id_to_record = {}
        self.active_ids = set()
        self.deleted_record_count = 0
        self.expired_record_count = 0
        for record in records:
            self._add_to_cell(record)
        for cell in self.cells.values():
            cell.build_index(self.config, self.active_ids)

    def insert(self, record: Record) -> None:
        self._add_to_cell(record)
        self.cells[self.router.key_for_record(record)].build_index(self.config, self.active_ids)

    def delete(self, record_id: int) -> None:
        if record_id not in self.id_to_record:
            raise KeyError(record_id)
        if record_id in self.active_ids:
            self.active_ids.remove(record_id)
            self.deleted_record_count += 1
            key = self.router.key_for_record(self.id_to_record[record_id])
            self.cells[key].refresh_metadata(self.active_ids)
            self._maybe_rebuild_cell(key)

    def expire(self, before_time: int) -> int:
        expired = [
            record.id
            for record in self.records
            if record.id in self.active_ids and record.valid_to is not None and record.valid_to < before_time
        ]
        affected_keys = {self.router.key_for_record(self.id_to_record[record_id]) for record_id in expired}
        for record_id in expired:
            self.active_ids.remove(record_id)
            self.cells[self.router.key_for_record(self.id_to_record[record_id])].refresh_metadata(self.active_ids)
        self.expired_record_count += len(expired)
        for key in affected_keys:
            self._maybe_rebuild_cell(key)
        return len(expired)

    def _add_to_cell(self, record: Record) -> None:
        if record.id in self.id_to_record:
            raise ValueError(f"Duplicate record id {record.id}")
        self.records.append(record)
        self.id_to_record[record.id] = record
        self.active_ids.add(record.id)
        key = self.router.key_for_record(record)
        cell = self.cells.setdefault(key, PartitionCell(key))
        cell.add_active_record(record)

    def _maybe_rebuild_cell(self, key: CellKey) -> None:
        cell = self.cells.get(key)
        if cell is None or not cell.records:
            return
        inactive_ratio = cell.inactive_count(self.active_ids) / len(cell.records)
        if inactive_ratio > self.config.partition_rebuild_tombstone_ratio:
            cell.build_index(self.config, self.active_ids, compact=True)

    def estimate_subset(self, query: Query) -> SubsetEstimate:
        keys = self._existing_intersecting_cells(query)
        cell_sizes = [self.cells[key].active_count(self.active_ids) for key in keys]
        subset_size = sum(self._estimate_cell_overlap(self.cells[key], query) for key in keys)
        avg = 0.0 if not cell_sizes else sum(cell_sizes) / len(cell_sizes)
        return SubsetEstimate(subset_size=subset_size, num_cells=len(keys), avg_cell_size=avg)

    def _estimate_cell_overlap(self, cell: PartitionCell, query: Query) -> float:
        if not cell.can_intersect(query):
            return 0.0
        time_fraction = _range_overlap_fraction(
            cell.min_valid_from,
            query.t_end if cell.open_ended_count else cell.max_valid_to,
            query.t_start,
            query.t_end,
        )
        price_fraction = _range_overlap_fraction(cell.min_price, cell.max_price, query.price_min, query.price_max)
        return cell.active_record_count * time_fraction * price_fraction

    def search(self, query: Query) -> SearchResult:
        validate_query(query)
        start = time.perf_counter()
        keys = self._existing_intersecting_cells(query)
        items: list[tuple[int, float]] = []
        candidate_count = 0
        filtered_out = 0
        exact_distance_count = 0

        for key in keys:
            cell = self.cells[key]
            if cell.index is None:
                for record in cell.records:
                    if record.id not in self.active_ids:
                        filtered_out += 1
                        continue
                    candidate_count += 1
                    if passes_filters(record, query):
                        items.append((record.id, l2(query.vector, record.vector)))
                        exact_distance_count += 1
                    else:
                        filtered_out += 1
                continue

            local_budget = min(len(cell.records), max(query.k * self.config.local_budget_factor, query.k))
            cand_ids, _ = cell.index.knn_query(query.vector, local_budget)
            candidate_count += len(cand_ids)
            for rid in cand_ids:
                record = self.id_to_record[rid]
                if rid in self.active_ids and passes_filters(record, query):
                    items.append((rid, l2(query.vector, record.vector)))
                    exact_distance_count += 1
                else:
                    filtered_out += 1

        ids, distances = merge_topk(items, query.k)
        latency_ms = (time.perf_counter() - start) * 1000
        return SearchResult(
            ids,
            distances,
            {
                "algorithm": "partition_ann",
                "visited_partitions": len(keys),
                "candidate_count": candidate_count,
                "filtered_out_count": filtered_out,
                "ann_expansion_rounds": 0,
                "latency_ms": latency_ms,
                "exact_distance_computations": exact_distance_count,
            },
        )

    def _existing_intersecting_cells(self, query: Query) -> list[CellKey]:
        end_time_bucket = self.router.time_bucketizer.bucket_id(query.t_end)
        requested_price_buckets = set(self.router.price_bucketizer.buckets_for_range(query.price_min, query.price_max))
        expanded: list[CellKey] = []
        for key in self.cells:
            if key.time_bucket > end_time_bucket:
                continue
            if key.price_bucket not in requested_price_buckets:
                continue
            if query.category is not None and key.category != query.category:
                continue
            if not self.cells[key].can_intersect(query):
                continue
            expanded.append(key)
        return sorted(expanded)

    def stats(self) -> dict:
        stored_records = sum(len(cell.records) for cell in self.cells.values())
        return {
            "algorithm": "partition_ann",
            "records": stored_records,
            "historical_records": len(self.records),
            "active_records": len(self.active_ids),
            "tombstoned_records": stored_records - len(self.active_ids),
            "cells": len(self.cells),
            "ann_cells": sum(1 for cell in self.cells.values() if cell.index is not None),
            "cell_rebuild_count": sum(cell.rebuild_count for cell in self.cells.values()),
            "compaction_count": sum(cell.compaction_count for cell in self.cells.values()),
            "deleted_record_count": self.deleted_record_count,
            "expired_record_count": self.expired_record_count,
            "open_ended_records": sum(cell.open_ended_count for cell in self.cells.values()),
            "category_histogram": dict(sum((cell.category_counts for cell in self.cells.values()), Counter())),
        }


def _range_overlap_fraction(
    envelope_min: int | float | None,
    envelope_max: int | float | None,
    query_min: int | float,
    query_max: int | float,
) -> float:
    if envelope_min is None or envelope_max is None:
        return 0.0
    if envelope_max < envelope_min:
        return 0.0
    overlap_min = max(envelope_min, query_min)
    overlap_max = min(envelope_max, query_max)
    if overlap_min > overlap_max:
        return 0.0
    envelope_width = max(float(envelope_max - envelope_min), 1.0)
    overlap_width = max(float(overlap_max - overlap_min), 1.0)
    return min(1.0, overlap_width / envelope_width)
