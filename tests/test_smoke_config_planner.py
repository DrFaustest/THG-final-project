import yaml

from tsann.config import IndexConfig
from tsann.planner import PlannerFeatures, RuleBasedPlanner
from tsann.types import Query, SubsetEstimate


def test_smoke_config_planner_can_choose_all_modes() -> None:
    with open("configs/smoke_grid.yaml", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    config = IndexConfig(**payload["index_config"])
    planner = RuleBasedPlanner(config)
    query = Query(vector=_DummyVector(), k=10, t_start=0, t_end=10, price_min=0.0, price_max=5.0)

    assert planner.choose_mode(query, SubsetEstimate(1, 1, 1.0), _features(subset_size=1, num_cells=1)) == "exact"
    assert (
        planner.choose_mode(
            query,
            SubsetEstimate(200, config.planner_max_cells_for_partition, config.planner_min_avg_cell_size),
            _features(
                subset_size=200,
                num_cells=config.planner_max_cells_for_partition,
                avg_cell_size=config.planner_min_avg_cell_size,
            ),
        )
        == "partition_ann"
    )
    assert planner.choose_mode(query, SubsetEstimate(200, 999, 1.0), _features(subset_size=200, num_cells=999)) == "global_filter"


def _features(subset_size: float, num_cells: int, avg_cell_size: float = 1.0) -> PlannerFeatures:
    return PlannerFeatures(
        subset_size=subset_size,
        num_cells=num_cells,
        avg_cell_size=avg_cell_size,
        query_window_width=10,
        price_range_width=5.0,
        category_present=False,
        active_records=1000,
        tombstoned_records=0,
        tombstone_ratio=0.0,
        open_ended_fraction=0.0,
        fragmentation_score=num_cells / max(1.0, subset_size),
    )


class _DummyVector:
    pass
