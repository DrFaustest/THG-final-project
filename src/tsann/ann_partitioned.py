from dataclasses import dataclass, field
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

    def build_index(self, config: IndexConfig) -> None:
        if not self.records or len(self.records) < config.partition_exact_threshold:
            self.index = None
            return
        dim = int(self.records[0].vector.shape[0])
        self.index = HnswVectorIndex(
            dim,
            ef_construction=config.hnsw_ef_construction,
            m=config.hnsw_m,
            ef_search=config.hnsw_ef_search,
        )
        self.index.build(
            [record.id for record in self.records],
            np.stack([record.vector.astype(np.float32) for record in self.records]),
        )


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

    def build(self, records: list[Record]) -> None:
        self.cells = {}
        self.records = []
        self.id_to_record = {}
        for record in records:
            self._add_to_cell(record)
        for cell in self.cells.values():
            cell.build_index(self.config)

    def insert(self, record: Record) -> None:
        self._add_to_cell(record)
        self.cells[self.router.key_for_record(record)].build_index(self.config)

    def _add_to_cell(self, record: Record) -> None:
        if record.id in self.id_to_record:
            raise ValueError(f"Duplicate record id {record.id}")
        self.records.append(record)
        self.id_to_record[record.id] = record
        key = self.router.key_for_record(record)
        self.cells.setdefault(key, PartitionCell(key)).records.append(record)

    def estimate_subset(self, query: Query) -> SubsetEstimate:
        keys = self._existing_intersecting_cells(query)
        cell_sizes = [len(self.cells[key].records) for key in keys]
        subset_size = sum(1 for key in keys for record in self.cells[key].records if passes_filters(record, query))
        avg = 0.0 if not cell_sizes else sum(cell_sizes) / len(cell_sizes)
        return SubsetEstimate(subset_size=subset_size, num_cells=len(keys), avg_cell_size=avg)

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
                if passes_filters(record, query):
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
        requested = self.router.intersect(query)
        if query.category is not None:
            return [key for key in requested if key in self.cells]
        expanded: list[CellKey] = []
        requested_no_category = {(key.time_bucket, key.price_bucket) for key in requested}
        for key in self.cells:
            if (key.time_bucket, key.price_bucket) in requested_no_category:
                expanded.append(key)
        return sorted(expanded)

    def stats(self) -> dict:
        return {
            "algorithm": "partition_ann",
            "records": len(self.records),
            "cells": len(self.cells),
            "ann_cells": sum(1 for cell in self.cells.values() if cell.index is not None),
        }
