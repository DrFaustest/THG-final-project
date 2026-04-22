from dataclasses import dataclass
from typing import Literal

import numpy as np

from tsann.types import Query, Record


@dataclass(frozen=True)
class SyntheticConfig:
    n: int = 10_000
    d: int = 64
    num_clusters: int = 10
    time_span: int = 365
    price_span: float = 100.0
    drift_strength: float = 0.1
    attr_corr: float = 0.5
    category_corr: float = 0.5
    noise_std: float = 0.05
    seed: int = 13
    cluster_size_mode: Literal["balanced", "zipf"] = "balanced"
    arrival_mode: Literal["uniform", "bursty", "drifting"] = "uniform"
    lifetime_min: int | None = None
    lifetime_max: int | None = None
    open_ended_fraction: float = 0.0


def generate_records(config: SyntheticConfig) -> list[Record]:
    rng = np.random.default_rng(config.seed)
    centroids = rng.normal(0, 1, size=(config.num_clusters, config.d)).astype(np.float32)
    drift_dirs = rng.normal(0, 1, size=(config.num_clusters, config.d)).astype(np.float32)
    drift_dirs /= np.linalg.norm(drift_dirs, axis=1, keepdims=True) + 1e-8
    base_prices = np.linspace(0, config.price_span, config.num_clusters, endpoint=False)

    clusters = _sample_clusters(rng, config.n, config.num_clusters, config.cluster_size_mode)
    timestamps = _sample_timestamps(rng, config.n, config.time_span, config.arrival_mode)
    records: list[Record] = []
    for rid, (cluster, timestamp) in enumerate(zip(clusters, timestamps)):
        time_fraction = 0.0 if config.time_span <= 1 else timestamp / (config.time_span - 1)
        drift = config.drift_strength * time_fraction * drift_dirs[cluster]
        noise = rng.normal(0, config.noise_std, size=config.d).astype(np.float32)
        vector = (centroids[cluster] + drift + noise).astype(np.float32)

        vector_signal = float(np.linalg.norm(vector)) % config.price_span
        random_price = float(rng.uniform(0, config.price_span))
        correlated_price = float(base_prices[cluster] + 0.1 * vector_signal)
        price = config.attr_corr * correlated_price + (1.0 - config.attr_corr) * random_price
        price = float(np.clip(price, 0.0, config.price_span))

        if rng.random() < config.category_corr:
            category = int(cluster % max(1, min(config.num_clusters, 10)))
        else:
            category = int(rng.integers(0, max(1, min(config.num_clusters, 10))))

        valid_to = _sample_valid_to(rng, int(timestamp), config)
        records.append(Record(rid, vector, int(timestamp), price, category, valid_to))
    return records


def generate_queries(
    records: list[Record],
    *,
    num_queries: int = 1_000,
    k: int = 10,
    time_selectivity: float = 0.05,
    price_selectivity: float = 0.1,
    seed: int = 99,
) -> list[Query]:
    if not records:
        return []
    rng = np.random.default_rng(seed)
    timestamps = np.asarray([record.valid_from for record in records])
    prices = np.asarray([record.price for record in records])
    min_time, max_time = int(timestamps.min()), int(timestamps.max())
    time_span = max(1, max_time - min_time + 1)
    price_min_all, price_max_all = float(prices.min()), float(prices.max())
    price_span = max(1e-6, price_max_all - price_min_all)
    time_width = max(0, int(round(time_span * time_selectivity)) - 1)
    price_width = price_span * price_selectivity

    queries: list[Query] = []
    for i in range(num_queries):
        anchor = records[int(rng.integers(0, len(records)))]
        query_family = rng.choice(["easy", "medium", "hard"], p=[0.2, 0.5, 0.3])
        vector = (anchor.vector + rng.normal(0, 0.02, size=anchor.vector.shape)).astype(np.float32)

        if query_family == "hard":
            t_start = int(rng.integers(min_time, max_time + 1))
            p_start = float(rng.uniform(price_min_all, price_max_all))
            category = None if i % 2 == 0 else anchor.category
        else:
            t_start = max(min_time, min(max_time, anchor.valid_from - time_width // 2))
            p_start = max(price_min_all, min(price_max_all, anchor.price - price_width / 2))
            category = anchor.category if query_family == "easy" else None

        t_end = min(max_time, t_start + time_width)
        p_end = min(price_max_all, p_start + price_width)
        queries.append(Query(vector, k, t_start, t_end, float(p_start), float(p_end), category))
    return queries


def _sample_clusters(rng: np.random.Generator, n: int, num_clusters: int, mode: str) -> np.ndarray:
    if mode == "zipf":
        ranks = np.arange(1, num_clusters + 1)
        probs = 1 / ranks
        probs = probs / probs.sum()
        return rng.choice(num_clusters, size=n, p=probs)
    return rng.integers(0, num_clusters, size=n)


def _sample_timestamps(rng: np.random.Generator, n: int, time_span: int, mode: str) -> np.ndarray:
    if mode == "bursty":
        centers = rng.integers(0, time_span, size=max(2, min(20, time_span)))
        chosen = rng.choice(centers, size=n)
        return np.clip(chosen + rng.normal(0, max(1, time_span * 0.02), size=n), 0, time_span - 1).astype(int)
    if mode == "drifting":
        return np.clip(np.floor(rng.beta(2, 1, size=n) * time_span), 0, time_span - 1).astype(int)
    return rng.integers(0, time_span, size=n)


def _sample_valid_to(rng: np.random.Generator, valid_from: int, config: SyntheticConfig) -> int | None:
    if config.lifetime_min is None or config.lifetime_max is None:
        return valid_from
    if rng.random() < config.open_ended_fraction:
        return None
    lifetime = int(rng.integers(config.lifetime_min, config.lifetime_max + 1))
    return min(config.time_span - 1, valid_from + lifetime)
