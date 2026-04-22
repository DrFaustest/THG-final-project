from tsann.config import IndexConfig
from tsann.types import Query, SubsetEstimate


class RuleBasedPlanner:
    def __init__(self, config: IndexConfig | None = None) -> None:
        self.config = config or IndexConfig()

    def choose_mode(self, query: Query, estimate: SubsetEstimate) -> str:
        if estimate.subset_size <= self.config.planner_exact_threshold:
            return "exact"
        if (
            estimate.num_cells <= self.config.planner_max_cells_for_partition
            and estimate.avg_cell_size >= self.config.planner_min_avg_cell_size
        ):
            return "partition_ann"
        return "global_filter"
