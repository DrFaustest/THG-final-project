import csv
import json
import math
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from pathlib import Path

from tsann.config import IndexConfig
from tsann.types import Query, SubsetEstimate


PLANNER_FEATURE_COLUMNS = [
    "subset_size",
    "num_cells",
    "avg_cell_size",
    "query_window_width",
    "price_range_width",
    "category_present",
    "active_records",
    "tombstoned_records",
    "tombstone_ratio",
    "open_ended_fraction",
    "fragmentation_score",
]


@dataclass(frozen=True)
class PlannerFeatures:
    subset_size: float
    num_cells: int
    avg_cell_size: float
    query_window_width: int
    price_range_width: float
    category_present: bool
    active_records: int
    tombstoned_records: int
    tombstone_ratio: float
    open_ended_fraction: float
    fragmentation_score: float

    def as_vector(self) -> list[float]:
        return [
            float(self.subset_size),
            float(self.num_cells),
            float(self.avg_cell_size),
            float(self.query_window_width),
            float(self.price_range_width),
            1.0 if self.category_present else 0.0,
            float(self.active_records),
            float(self.tombstoned_records),
            float(self.tombstone_ratio),
            float(self.open_ended_fraction),
            float(self.fragmentation_score),
        ]

    def to_metadata(self) -> dict:
        return {f"planner_feature_{key}": value for key, value in asdict(self).items()}


class RuleBasedPlanner:
    def __init__(self, config: IndexConfig | None = None) -> None:
        self.config = config or IndexConfig()

    def features_from(
        self,
        query: Query,
        estimate: SubsetEstimate,
        partition_stats: Mapping,
    ) -> PlannerFeatures:
        active_records = int(partition_stats.get("active_records", 0))
        tombstoned_records = int(partition_stats.get("tombstoned_records", 0))
        open_ended_records = int(partition_stats.get("open_ended_records", 0))
        total_records = max(1, active_records + tombstoned_records)
        tombstone_ratio = tombstoned_records / total_records
        open_ended_fraction = open_ended_records / max(1, active_records)
        fragmentation_score = estimate.num_cells / max(1.0, estimate.subset_size)
        return PlannerFeatures(
            subset_size=estimate.subset_size,
            num_cells=estimate.num_cells,
            avg_cell_size=estimate.avg_cell_size,
            query_window_width=query.t_end - query.t_start + 1,
            price_range_width=query.price_max - query.price_min,
            category_present=query.category is not None,
            active_records=active_records,
            tombstoned_records=tombstoned_records,
            tombstone_ratio=tombstone_ratio,
            open_ended_fraction=open_ended_fraction,
            fragmentation_score=fragmentation_score,
        )

    def choose_mode(
        self,
        query: Query,
        estimate: SubsetEstimate,
        features: PlannerFeatures | None = None,
    ) -> str:
        features = features or self.features_from(query, estimate, {})
        if estimate.subset_size <= self.config.planner_exact_threshold:
            return "exact"
        if features.tombstone_ratio > 0.40 and estimate.subset_size <= self.config.planner_exact_threshold * 2:
            return "exact"
        if (
            estimate.num_cells <= self.config.planner_max_cells_for_partition
            and estimate.avg_cell_size >= self.config.planner_min_avg_cell_size
        ):
            return "partition_ann"
        if (
            estimate.num_cells <= self.config.planner_max_cells_for_partition
            and features.fragmentation_score <= 2.0
            and estimate.subset_size <= self.config.planner_exact_threshold * 8
        ):
            return "partition_ann"
        if (
            features.open_ended_fraction >= 0.50
            and estimate.num_cells <= self.config.planner_max_cells_for_partition * 2
            and features.fragmentation_score <= 4.0
        ):
            return "partition_ann"
        if (
            features.category_present
            and estimate.num_cells <= self.config.planner_max_cells_for_partition * 2
            and features.open_ended_fraction < 0.75
        ):
            return "partition_ann"
        return "global_filter"


class NearestCentroidPlanner(RuleBasedPlanner):
    def __init__(
        self,
        centroids: dict[str, list[float]],
        means: list[float],
        scales: list[float],
        config: IndexConfig | None = None,
        fallback: RuleBasedPlanner | None = None,
    ) -> None:
        super().__init__(config)
        self.centroids = centroids
        self.means = means
        self.scales = scales
        self.fallback = fallback or RuleBasedPlanner(config)

    def choose_mode(
        self,
        query: Query,
        estimate: SubsetEstimate,
        features: PlannerFeatures | None = None,
    ) -> str:
        if not self.centroids:
            return self.fallback.choose_mode(query, estimate, features)
        features = features or self.features_from(query, estimate, {})
        vector = self._standardize(features.as_vector())
        return min(self.centroids, key=lambda mode: _squared_distance(vector, self.centroids[mode]))

    def predict_vector(self, vector: list[float]) -> str:
        if not self.centroids:
            raise ValueError("Planner has no centroids")
        standardized = self._standardize(vector)
        return min(self.centroids, key=lambda mode: _squared_distance(standardized, self.centroids[mode]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model_type": "nearest_centroid",
                    "feature_columns": PLANNER_FEATURE_COLUMNS,
                    "means": self.means,
                    "scales": self.scales,
                    "centroids": self.centroids,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, config: IndexConfig | None = None) -> "NearestCentroidPlanner":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("feature_columns") != PLANNER_FEATURE_COLUMNS:
            raise ValueError("Planner feature schema mismatch")
        return cls(
            centroids={key: [float(x) for x in values] for key, values in payload["centroids"].items()},
            means=[float(x) for x in payload["means"]],
            scales=[float(x) for x in payload["scales"]],
            config=config,
        )

    @classmethod
    def train_from_csv(
        cls,
        path: Path,
        *,
        config: IndexConfig | None = None,
        algorithm_filter: str = "hybrid",
        include_seeds: set[str] | None = None,
        exclude_seeds: set[str] | None = None,
    ) -> "NearestCentroidPlanner":
        rows: list[tuple[str, list[float]]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("algorithm") != algorithm_filter:
                    continue
                seed = row.get("seed", "")
                if include_seeds is not None and seed not in include_seeds:
                    continue
                if exclude_seeds is not None and seed in exclude_seeds:
                    continue
                label = row.get("best_mode", "")
                if not label:
                    continue
                rows.append((label, _row_to_vector(row)))
        if not rows:
            raise ValueError(f"No labeled planner rows found in {path}")
        means, scales = _standardization([vector for _, vector in rows])
        grouped: dict[str, list[list[float]]] = {}
        for label, vector in rows:
            grouped.setdefault(label, []).append(_standardize_vector(vector, means, scales))
        centroids = {
            label: [_mean([vector[i] for vector in vectors]) for i in range(len(PLANNER_FEATURE_COLUMNS))]
            for label, vectors in grouped.items()
        }
        return cls(centroids=centroids, means=means, scales=scales, config=config)

    def _standardize(self, vector: list[float]) -> list[float]:
        return _standardize_vector(vector, self.means, self.scales)


def _row_to_vector(row: Mapping[str, str]) -> list[float]:
    return [
        _float(row, "planner_feature_subset_size"),
        _float(row, "planner_feature_num_cells"),
        _float(row, "planner_feature_avg_cell_size"),
        _float(row, "planner_feature_query_window_width"),
        _float(row, "planner_feature_price_range_width"),
        _float(row, "planner_feature_category_present"),
        _float(row, "planner_feature_active_records"),
        _float(row, "planner_feature_tombstoned_records"),
        _float(row, "planner_feature_tombstone_ratio"),
        _float(row, "planner_feature_open_ended_fraction"),
        _float(row, "planner_feature_fragmentation_score"),
    ]


def _float(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    return 0.0 if value == "" else float(value)


def _standardization(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    means = [_mean([vector[i] for vector in vectors]) for i in range(len(vectors[0]))]
    scales: list[float] = []
    for i, mean in enumerate(means):
        variance = _mean([(vector[i] - mean) ** 2 for vector in vectors])
        scales.append(max(math.sqrt(variance), 1e-9))
    return means, scales


def _standardize_vector(vector: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [(value - means[i]) / scales[i] for i, value in enumerate(vector)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))
