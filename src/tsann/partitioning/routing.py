from dataclasses import dataclass

from tsann.partitioning.scalar_buckets import FixedWidthPriceBucketizer
from tsann.partitioning.time_buckets import FixedTimeBucketizer
from tsann.types import Query, Record


@dataclass(frozen=True, order=True)
class CellKey:
    time_bucket: int
    price_bucket: int
    category: int | None = None


class TimePriceRouter:
    def __init__(
        self,
        time_bucketizer: FixedTimeBucketizer,
        price_bucketizer: FixedWidthPriceBucketizer,
        *,
        include_category: bool = True,
    ) -> None:
        self.time_bucketizer = time_bucketizer
        self.price_bucketizer = price_bucketizer
        self.include_category = include_category

    def key_for_record(self, record: Record) -> CellKey:
        category = record.category if self.include_category else None
        return CellKey(
            self.time_bucketizer.bucket_id(record.valid_from),
            self.price_bucketizer.bucket_id(record.price),
            category,
        )

    def intersect(self, query: Query) -> list[CellKey]:
        category_values = [query.category] if self.include_category and query.category is not None else [None]
        cells: list[CellKey] = []
        for time_bucket in self.time_bucketizer.buckets_for_range(query.t_start, query.t_end):
            for price_bucket in self.price_bucketizer.buckets_for_range(query.price_min, query.price_max):
                for category in category_values:
                    cells.append(CellKey(time_bucket, price_bucket, category))
        return cells
