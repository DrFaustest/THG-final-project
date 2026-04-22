import numpy as np

from tsann.ann_hybrid import HybridPlannerIndex
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
