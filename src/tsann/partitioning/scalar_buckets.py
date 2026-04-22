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
