import numpy as np

from tsann.indexes.brute_force import BruteForceVectorIndex

try:
    import hnswlib
except ImportError:  # pragma: no cover - depends on optional binary package availability.
    hnswlib = None


class HnswVectorIndex:
    def __init__(
        self,
        dim: int,
        *,
        space: str = "l2",
        ef_construction: int = 100,
        m: int = 16,
        ef_search: int = 100,
    ) -> None:
        self.dim = dim
        self.space = space
        self.ef_construction = ef_construction
        self.m = m
        self.ef_search = ef_search
        self._index = None
        self._fallback = BruteForceVectorIndex()
        self._ids: list[int] = []
        self.available = hnswlib is not None

    def build(self, ids: list[int], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        self._ids = list(ids)
        self._fallback.build(self._ids, matrix)
        if hnswlib is None or len(self._ids) == 0:
            self._index = None
            return
        self._index = hnswlib.Index(space=self.space, dim=self.dim)
        self._index.init_index(
            max_elements=len(self._ids),
            ef_construction=self.ef_construction,
            M=self.m,
        )
        self._index.add_items(matrix, np.asarray(self._ids, dtype=np.int64))
        self._index.set_ef(self.ef_search)

    def knn_query(self, vector: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        if k <= 0 or not self._ids:
            return [], []
        k = min(k, len(self._ids))
        if self._index is None:
            return self._fallback.knn_query(vector, k)
        labels, distances = self._index.knn_query(np.asarray(vector, dtype=np.float32), k=k)
        return [int(x) for x in labels[0]], [float(x) for x in distances[0]]
