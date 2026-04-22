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
