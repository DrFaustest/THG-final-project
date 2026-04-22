import numpy as np
import pytest

from tsann.filters import passes_filters, validate_query
from tsann.types import Query, Record


def test_passes_filters_inclusive_boundaries() -> None:
    record = Record(1, np.zeros(2, dtype=np.float32), timestamp=10, price=5.0, category=2)
    query = Query(np.zeros(2, dtype=np.float32), 3, 10, 10, 5.0, 5.0, 2)
    assert passes_filters(record, query)


def test_category_none_matches_any_category() -> None:
    record = Record(1, np.zeros(2, dtype=np.float32), timestamp=10, price=5.0, category=2)
    query = Query(np.zeros(2, dtype=np.float32), 3, 0, 20, 0.0, 10.0, None)
    assert passes_filters(record, query)


def test_validate_query_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        validate_query(Query(np.zeros(2, dtype=np.float32), 1, 5, 4, 0.0, 1.0))
    with pytest.raises(ValueError):
        validate_query(Query(np.zeros(2, dtype=np.float32), 1, 0, 4, 2.0, 1.0))
