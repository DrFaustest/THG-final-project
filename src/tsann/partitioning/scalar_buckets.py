import math


class FixedWidthPriceBucketizer:
    def __init__(self, width: float) -> None:
        if width <= 0:
            raise ValueError("Price bucket width must be positive")
        self.width = width

    def bucket_id(self, price: float) -> int:
        return math.floor(price / self.width)

    def buckets_for_range(self, minimum: float, maximum: float) -> list[int]:
        if minimum > maximum:
            return []
        first = self.bucket_id(minimum)
        last = self.bucket_id(maximum)
        return list(range(first, last + 1))

    def bucket_range(self, bucket_id: int) -> tuple[float, float]:
        start = bucket_id * self.width
        return start, start + self.width

    def overlap_fraction(self, bucket_id: int, minimum: float, maximum: float) -> float:
        bucket_start, bucket_end = self.bucket_range(bucket_id)
        if minimum == maximum:
            return 1.0 if bucket_start <= minimum <= bucket_end else 0.0
        overlap_start = max(bucket_start, minimum)
        overlap_end = min(bucket_end, maximum)
        if overlap_start > overlap_end:
            return 0.0
        return min(1.0, (overlap_end - overlap_start) / self.width)
