import numpy as np

from tsann.ann_global import GlobalAnnThenFilterIndex
from tsann.ann_hybrid import HybridPlannerIndex
from tsann.ann_partitioned import PartitionFirstAnnIndex
from tsann.config import IndexConfig
from tsann.metrics import recall_at_k
from tsann.oracle_exact import ExactFilteredOracle
from tsann.types import Query, Record


def test_insert_makes_new_record_searchable() -> None:
    index = HybridPlannerIndex()
    index.build([])
    record = Record(10, np.array([1.0, 0.0], dtype=np.float32), 3, 4.0, None, None)
    index.insert(record)
    result = index.search(Query(np.array([1.0, 0.0], dtype=np.float32), 1, 0, 10, 0.0, 10.0))
    assert result.ids == [10]


def test_delete_removes_result_eligibility() -> None:
    vector = np.array([1.0, 0.0], dtype=np.float32)
    index = HybridPlannerIndex()
    index.build([Record(10, vector, 3, 4.0, None, None)])
    index.delete(10)
    result = index.search(Query(vector, 1, 0, 10, 0.0, 10.0))
    assert result.ids == []


def test_expire_removes_only_records_ending_before_cutoff() -> None:
    vector = np.array([1.0, 0.0], dtype=np.float32)
    records = [
        Record(1, vector, 0, 1.0, None, 4),
        Record(2, vector + 1, 0, 1.0, None, 5),
        Record(3, vector + 2, 0, 1.0, None, None),
    ]
    index = HybridPlannerIndex()
    index.build(records)
    assert index.expire(before_time=5) == 1
    result = index.search(Query(vector, 3, 0, 10, 0.0, 2.0))
    assert 1 not in result.ids
    assert {2, 3}.issubset(set(result.ids))


def test_delete_already_expired_record_is_idempotent() -> None:
    vector = np.array([1.0, 0.0], dtype=np.float32)
    index = HybridPlannerIndex()
    index.build([Record(1, vector, 0, 1.0, None, 2)])
    assert index.expire(before_time=3) == 1
    index.delete(1)
    result = index.search(Query(vector, 1, 0, 10, 0.0, 2.0))
    assert result.ids == []


def test_repeated_delete_on_inactive_record_is_idempotent() -> None:
    vector = np.array([1.0, 0.0], dtype=np.float32)
    index = HybridPlannerIndex()
    index.build([Record(1, vector, 0, 1.0, None, None)])
    index.delete(1)
    index.delete(1)
    result = index.search(Query(vector, 1, 0, 10, 0.0, 2.0))
    assert result.ids == []
    assert index.stats()["partitioned"]["deleted_record_count"] == 1


def test_expire_with_no_matching_records_returns_zero() -> None:
    vector = np.array([1.0, 0.0], dtype=np.float32)
    index = HybridPlannerIndex()
    index.build([Record(1, vector, 0, 1.0, None, 10)])
    assert index.expire(before_time=3) == 0
    result = index.search(Query(vector, 1, 0, 10, 0.0, 2.0))
    assert result.ids == [1]


def test_partition_cell_compacts_when_rebuild_threshold_crosses() -> None:
    config = IndexConfig(time_bucket_width=10, price_bucket_width=10.0, partition_rebuild_tombstone_ratio=0.25)
    records = [
        Record(i, np.array([float(i), 0.0], dtype=np.float32), 0, 1.0, None, None)
        for i in range(4)
    ]
    index = PartitionFirstAnnIndex(config)
    index.build(records)

    index.delete(0)
    index.delete(1)

    stats = index.stats()
    assert stats["compaction_count"] >= 1
    assert stats["tombstoned_records"] == 0
    assert stats["active_records"] == 2


def test_global_compacts_when_rebuild_threshold_crosses() -> None:
    config = IndexConfig(global_rebuild_tombstone_ratio=0.25)
    records = [
        Record(i, np.array([float(i), 0.0], dtype=np.float32), 0, 1.0, None, None)
        for i in range(4)
    ]
    index = GlobalAnnThenFilterIndex(config)
    index.build(records)

    index.delete(0)
    index.delete(1)

    stats = index.stats()
    assert stats["compaction_count"] >= 1
    assert stats["records"] == 2
    assert stats["active_records"] == 2
    assert stats["index_visible_size"] == 2


def test_global_partitioned_and_oracle_consistent_after_update_sequence() -> None:
    vector = np.array([0.0, 0.0], dtype=np.float32)
    records = [
        Record(1, vector, 0, 1.0, None, 4),
        Record(2, vector + 1, 1, 1.0, None, 10),
        Record(3, vector + 2, 2, 1.0, None, None),
    ]
    indexes = [ExactFilteredOracle(), GlobalAnnThenFilterIndex(), PartitionFirstAnnIndex()]
    for index in indexes:
        index.build(records)
        index.insert(Record(4, vector + 3, 3, 1.0, None, None))
        index.delete(2)
        assert index.expire(5) == 1

    query = Query(vector, 10, 0, 20, 0.0, 2.0)
    truth = indexes[0].search(query)
    assert truth.ids == [3, 4]
    for index in indexes[1:]:
        result = index.search(query)
        assert recall_at_k(result, truth, 10) == 1.0


def test_hybrid_handles_mixed_insert_delete_expire_sequence() -> None:
    vector = np.array([0.0, 0.0], dtype=np.float32)
    index = HybridPlannerIndex()
    index.build([Record(1, vector, 0, 1.0, None, 3)])
    index.insert(Record(2, vector + 1, 2, 1.0, None, 10))
    index.insert(Record(3, vector + 2, 2, 1.0, None, None))
    index.delete(2)
    assert index.expire(before_time=4) == 1
    result = index.search(Query(vector, 3, 0, 20, 0.0, 2.0))
    assert result.ids == [3]
