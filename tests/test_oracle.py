import numpy as np

from tsann.oracle_exact import ExactFilteredOracle
from tsann.types import Query, Record


def test_oracle_returns_exact_top_k_sorted() -> None:
    records = [
        Record(1, np.array([0.0, 0.0], dtype=np.float32), 1, 10.0, 0),
        Record(2, np.array([1.0, 0.0], dtype=np.float32), 1, 10.0, 0),
        Record(3, np.array([2.0, 0.0], dtype=np.float32), 1, 10.0, 0),
        Record(4, np.array([0.1, 0.0], dtype=np.float32), 5, 99.0, 0),
    ]
    index = ExactFilteredOracle()
    index.build(records)
    result = index.search(Query(np.array([0.0, 0.0], dtype=np.float32), 2, 0, 2, 0.0, 20.0, 0))
    assert result.ids == [1, 2]
    assert result.distances == sorted(result.distances)


def test_oracle_handles_k_larger_than_valid_subset() -> None:
    index = ExactFilteredOracle()
    index.build([Record(1, np.zeros(2, dtype=np.float32), 1, 10.0, None)])
    result = index.search(Query(np.zeros(2, dtype=np.float32), 10, 1, 1, 10.0, 10.0))
    assert result.ids == [1]
