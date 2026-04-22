from dataclasses import dataclass


@dataclass(frozen=True)
class IndexConfig:
    distance: str = "l2"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 100
    hnsw_ef_search: int = 100
    global_initial_alpha: int = 10
    global_min_budget: int = 50
    global_max_budget: int | None = None
    time_bucket_width: int = 7
    price_bucket_width: float = 10.0
    partition_exact_threshold: int = 2_000
    local_budget_factor: int = 5
    planner_exact_threshold: int = 5_000
    planner_max_cells_for_partition: int = 8
    planner_min_avg_cell_size: int = 2_000
