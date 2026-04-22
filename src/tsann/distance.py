import numpy as np


def as_float32_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected one-dimensional vector, got shape {arr.shape}")
    return arr


def l2(query: np.ndarray, vector: np.ndarray) -> float:
    diff = as_float32_vector(query) - as_float32_vector(vector)
    return float(np.linalg.norm(diff))


def l2_squared(query: np.ndarray, vector: np.ndarray) -> float:
    diff = as_float32_vector(query) - as_float32_vector(vector)
    return float(np.dot(diff, diff))


def pairwise_l2(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q = as_float32_vector(query)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected two-dimensional vector matrix, got shape {matrix.shape}")
    return np.linalg.norm(matrix - q[None, :], axis=1)
