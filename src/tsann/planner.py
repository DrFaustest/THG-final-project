from dataclasses import asdict, dataclass
from collections.abc import Mapping

from tsann.config import IndexConfig
from tsann.types import Query, SubsetEstimate


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
            features.category_present
            and estimate.num_cells <= self.config.planner_max_cells_for_partition * 2
            and features.open_ended_fraction < 0.75
        ):
            return "partition_ann"
        return "global_filter"
