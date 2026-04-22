from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True, eq=False)
class Record:
    id: int
    vector: np.ndarray
    valid_from: int
    price: float
    category: Optional[int] = None
    valid_to: Optional[int] = None

    def __post_init__(self) -> None:
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("Record.valid_to must be >= Record.valid_from when set")

    @property
    def timestamp(self) -> int:
        return self.valid_from

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Record):
            return False
        return (
            self.id == other.id
            and np.array_equal(self.vector, other.vector)
            and self.valid_from == other.valid_from
            and self.valid_to == other.valid_to
            and self.price == other.price
            and self.category == other.category
        )


@dataclass(frozen=True, eq=False)
class Query:
    vector: np.ndarray
    k: int
    t_start: int
    t_end: int
    price_min: float
    price_max: float
    category: Optional[int] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Query):
            return False
        return (
            np.array_equal(self.vector, other.vector)
            and self.k == other.k
            and self.t_start == other.t_start
            and self.t_end == other.t_end
            and self.price_min == other.price_min
            and self.price_max == other.price_max
            and self.category == other.category
        )


@dataclass
class SearchResult:
    ids: list[int]
    distances: list[float]
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SubsetEstimate:
    subset_size: float
    num_cells: int
    avg_cell_size: float
    estimator: str = "metadata"
