from tsann.ann_global import GlobalAnnThenFilterIndex
from tsann.ann_partitioned import PartitionFirstAnnIndex
from tsann.config import IndexConfig
from tsann.indexes.base import BaseTemporalSubsetIndex
from tsann.oracle_exact import ExactFilteredOracle
from tsann.planner import RuleBasedPlanner
from tsann.types import Query, Record, SearchResult


class HybridPlannerIndex(BaseTemporalSubsetIndex):
    def __init__(self, config: IndexConfig | None = None) -> None:
        self.config = config or IndexConfig()
        self.oracle = ExactFilteredOracle()
        self.global_index = GlobalAnnThenFilterIndex(self.config)
        self.partition_index = PartitionFirstAnnIndex(self.config)
        self.planner = RuleBasedPlanner(self.config)

    def build(self, records: list[Record]) -> None:
        self.oracle.build(records)
        self.global_index.build(records)
        self.partition_index.build(records)

    def insert(self, record: Record) -> None:
        self.oracle.insert(record)
        self.global_index.insert(record)
        self.partition_index.insert(record)

    def search(self, query: Query) -> SearchResult:
        estimate = self.partition_index.estimate_subset(query)
        mode = self.planner.choose_mode(query, estimate)
        if mode == "exact":
            result = self.oracle.search(query)
        elif mode == "partition_ann":
            result = self.partition_index.search(query)
        else:
            result = self.global_index.search(query)
        result.metadata = {
            **result.metadata,
            "algorithm": "hybrid",
            "planner_mode": mode,
            "subset_size_estimate": estimate.subset_size,
            "estimate_num_cells": estimate.num_cells,
            "estimate_avg_cell_size": estimate.avg_cell_size,
        }
        return result

    def stats(self) -> dict:
        return {
            "algorithm": "hybrid",
            "exact": self.oracle.stats(),
            "global": self.global_index.stats(),
            "partitioned": self.partition_index.stats(),
        }
