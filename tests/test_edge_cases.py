import numpy as np

from tsann.ann_partitioned import PartitionFirstAnnIndex
from tsann.oracle_exact import ExactFilteredOracle
from tsann.types import Query, Record


def test_empty_subset_returns_empty_cleanly() -> None:
    records = [Record(1, np.zeros(2, dtype=np.float32), 1, 1.0, None, 1)]
    query = Query(np.zeros(2, dtype=np.float32), 5, 9, 10, 0.0, 1.0)
    for index in (ExactFilteredOracle(), PartitionFirstAnnIndex()):
        index.build(records)
        result = index.search(query)
        assert result.ids == []
        assert result.distances == []


def test_zero_k_returns_empty() -> None:
    index = ExactFilteredOracle()
    index.build([Record(1, np.zeros(2, dtype=np.float32), 1, 1.0, None, 1)])
    result = index.search(Query(np.zeros(2, dtype=np.float32), 0, 0, 10, 0.0, 2.0))
    assert result.ids == []
