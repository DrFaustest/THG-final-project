import math


class FixedTimeBucketizer:
    def __init__(self, width: int) -> None:
        if width <= 0:
            raise ValueError("Time bucket width must be positive")
        self.width = width

    def bucket_id(self, timestamp: int) -> int:
        return math.floor(timestamp / self.width)

    def buckets_for_range(self, start: int, end: int) -> list[int]:
        if start > end:
            return []
        first = self.bucket_id(start)
        last = self.bucket_id(end)
        return list(range(first, last + 1))

    def bucket_range(self, bucket_id: int) -> tuple[int, int]:
        start = bucket_id * self.width
        return start, start + self.width - 1

    def overlap_fraction(self, bucket_id: int, start: int, end: int) -> float:
        bucket_start, bucket_end = self.bucket_range(bucket_id)
        overlap_start = max(bucket_start, start)
        overlap_end = min(bucket_end, end)
        if overlap_start > overlap_end:
            return 0.0
        return (overlap_end - overlap_start + 1) / self.width
