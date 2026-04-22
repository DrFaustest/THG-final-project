import numpy as np

from tsann.ann_hybrid import HybridPlannerIndex
from tsann.types import Query, Record


def test_insert_makes_new_record_searchable() -> None:
    index = HybridPlannerIndex()
    index.build([])
    record = Record(10, np.array([1.0, 0.0], dtype=np.float32), 3, 4.0, None)
    index.insert(record)
    result = index.search(Query(np.array([1.0, 0.0], dtype=np.float32), 1, 0, 10, 0.0, 10.0))
    assert result.ids == [10]
