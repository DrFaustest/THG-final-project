import time

from tsann.distance import l2
from tsann.filters import passes_filters, validate_query
from tsann.indexes.base import BaseTemporalSubsetIndex
from tsann.metrics import merge_topk
from tsann.types import Query, Record, SearchResult


class ExactFilteredOracle(BaseTemporalSubsetIndex):
    def __init__(self) -> None:
        self.records: list[Record] = []
        self.id_to_record: dict[int, Record] = {}
        self.active_ids: set[int] = set()

    def build(self, records: list[Record]) -> None:
        self.records = []
        self.id_to_record = {}
        self.active_ids = set()
        for record in records:
            self.insert(record)

    def insert(self, record: Record) -> None:
        if record.id in self.id_to_record:
            raise ValueError(f"Duplicate record id {record.id}")
        self.records.append(record)
        self.id_to_record[record.id] = record
        self.active_ids.add(record.id)

    def delete(self, record_id: int) -> None:
        if record_id not in self.id_to_record:
            raise KeyError(record_id)
        self.active_ids.discard(record_id)

    def expire(self, before_time: int) -> int:
        expired = [
            record.id
            for record in self.records
            if record.id in self.active_ids and record.valid_to is not None and record.valid_to < before_time
        ]
        for record_id in expired:
            self.active_ids.remove(record_id)
        return len(expired)

    def search(self, query: Query) -> SearchResult:
        validate_query(query)
        start = time.perf_counter()
        candidates = [
            (record.id, l2(query.vector, record.vector))
            for record in self.records
            if record.id in self.active_ids and passes_filters(record, query)
        ]
        ids, distances = merge_topk(candidates, query.k)
        latency_ms = (time.perf_counter() - start) * 1000
        return SearchResult(
            ids=ids,
            distances=distances,
            metadata={
                "algorithm": "exact",
                "visited_partitions": 0,
                "candidate_count": len(candidates),
                "filtered_out_count": len(self.active_ids) - len(candidates),
                "ann_expansion_rounds": 0,
                "latency_ms": latency_ms,
                "exact_distance_computations": len(candidates),
            },
        )

    def stats(self) -> dict:
        return {
            "algorithm": "exact",
            "records": len(self.records),
            "active_records": len(self.active_ids),
            "tombstoned_records": len(self.records) - len(self.active_ids),
        }
