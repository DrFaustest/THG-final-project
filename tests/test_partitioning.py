import numpy as np

from tsann.partitioning.routing import CellKey, TimePriceRouter
from tsann.partitioning.scalar_buckets import FixedWidthPriceBucketizer
from tsann.partitioning.time_buckets import FixedTimeBucketizer
from tsann.types import Query, Record


def test_time_bucket_intersection_is_inclusive() -> None:
    buckets = FixedTimeBucketizer(width=7)
    assert buckets.buckets_for_range(0, 14) == [0, 1, 2]


def test_router_record_key_and_query_intersection() -> None:
    router = TimePriceRouter(FixedTimeBucketizer(10), FixedWidthPriceBucketizer(5.0))
    record = Record(1, np.zeros(2, dtype=np.float32), timestamp=12, price=7.0, category=3)
    assert router.key_for_record(record) == CellKey(1, 1, 3)
    query = Query(np.zeros(2, dtype=np.float32), 1, 0, 10, 0.0, 5.0, 3)
    assert router.intersect(query) == [CellKey(0, 0, 3), CellKey(0, 1, 3), CellKey(1, 0, 3), CellKey(1, 1, 3)]
