import numpy as np

from tsann.distance import pairwise_l2


class BruteForceVectorIndex:
    def __init__(self) -> None:
        self.ids: list[int] = []
        self.vectors: np.ndarray | None = None

    def build(self, ids: list[int], vectors: np.ndarray) -> None:
        self.ids = list(ids)
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def knn_query(self, vector: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        if self.vectors is None or not self.ids or k <= 0:
            return [], []
        k = min(k, len(self.ids))
        distances = pairwise_l2(vector, self.vectors)
        order = np.argsort(distances, kind="stable")[:k]
        return [self.ids[int(i)] for i in order], [float(distances[int(i)]) for i in order]
