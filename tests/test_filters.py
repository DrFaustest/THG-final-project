import numpy as np
import pytest

from tsann.filters import passes_filters, validate_query
from tsann.types import Query, Record


def test_passes_filters_inclusive_boundaries() -> None:
    record = Record(1, np.zeros(2, dtype=np.float32), valid_from=10, valid_to=10, price=5.0, category=2)
    query = Query(np.zeros(2, dtype=np.float32), 3, 10, 10, 5.0, 5.0, 2)
    assert passes_filters(record, query)


def test_category_none_matches_any_category() -> None:
    record = Record(1, np.zeros(2, dtype=np.float32), valid_from=10, valid_to=10, price=5.0, category=2)
    query = Query(np.zeros(2, dtype=np.float32), 3, 0, 20, 0.0, 10.0, None)
    assert passes_filters(record, query)


def test_interval_intersection_boundaries() -> None:
    vector = np.zeros(2, dtype=np.float32)
    assert passes_filters(Record(1, vector, valid_from=5, valid_to=10, price=1.0), Query(vector, 1, 10, 20, 0.0, 2.0))
    assert passes_filters(Record(2, vector, valid_from=20, valid_to=30, price=1.0), Query(vector, 1, 10, 20, 0.0, 2.0))
    assert not passes_filters(Record(3, vector, valid_from=1, valid_to=9, price=1.0), Query(vector, 1, 10, 20, 0.0, 2.0))
    assert not passes_filters(Record(4, vector, valid_from=21, valid_to=25, price=1.0), Query(vector, 1, 10, 20, 0.0, 2.0))


def test_open_ended_validity_matches_future_window() -> None:
    vector = np.zeros(2, dtype=np.float32)
    record = Record(1, vector, valid_from=5, valid_to=None, price=1.0)
    assert passes_filters(record, Query(vector, 1, 100, 120, 0.0, 2.0))


def test_validate_query_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        validate_query(Query(np.zeros(2, dtype=np.float32), 1, 5, 4, 0.0, 1.0))
    with pytest.raises(ValueError):
        validate_query(Query(np.zeros(2, dtype=np.float32), 1, 0, 4, 2.0, 1.0))
