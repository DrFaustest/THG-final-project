import csv
from pathlib import Path

import numpy as np

from tsann.ann_hybrid import HybridPlannerIndex
from tsann.planner import NearestCentroidPlanner
from tsann.types import Query, Record


def test_nearest_centroid_planner_trains_saves_and_loads() -> None:
    workdir = Path("results/test_tmp/learned_planner")
    workdir.mkdir(parents=True, exist_ok=True)
    trace = workdir / "trace.csv"
    fieldnames = [
        "algorithm",
        "best_mode",
        "planner_feature_subset_size",
        "planner_feature_num_cells",
        "planner_feature_avg_cell_size",
        "planner_feature_query_window_width",
        "planner_feature_price_range_width",
        "planner_feature_category_present",
        "planner_feature_active_records",
        "planner_feature_tombstoned_records",
        "planner_feature_tombstone_ratio",
        "planner_feature_open_ended_fraction",
        "planner_feature_fragmentation_score",
    ]
    rows = [
        _row("exact", 10, 1, "exact"),
        _row("exact", 20, 2, "exact"),
        _row("partition_ann", 1000, 3, "partition_ann"),
        _row("partition_ann", 1200, 4, "partition_ann"),
    ]
    with trace.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    planner = NearestCentroidPlanner.train_from_csv(trace)
    output = workdir / "planner.json"
    planner.save(output)
    loaded = NearestCentroidPlanner.load(output)

    assert set(loaded.centroids) == {"exact", "partition_ann"}


def test_hybrid_can_use_injected_learned_planner() -> None:
    workdir = Path("results/test_tmp/learned_planner_hybrid")
    workdir.mkdir(parents=True, exist_ok=True)
    trace = workdir / "trace.csv"
    fieldnames = [
        "algorithm",
        "best_mode",
        "planner_feature_subset_size",
        "planner_feature_num_cells",
        "planner_feature_avg_cell_size",
        "planner_feature_query_window_width",
        "planner_feature_price_range_width",
        "planner_feature_category_present",
        "planner_feature_active_records",
        "planner_feature_tombstoned_records",
        "planner_feature_tombstone_ratio",
        "planner_feature_open_ended_fraction",
        "planner_feature_fragmentation_score",
    ]
    with trace.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(_row("exact", 1, 1, "exact"))
        writer.writerow(_row("global_filter", 1000, 20, "global_filter"))

    planner = NearestCentroidPlanner.train_from_csv(trace)
    vector = np.array([0.0, 0.0], dtype=np.float32)
    index = HybridPlannerIndex(planner=planner)
    index.build([Record(1, vector, 0, 1.0, None, None)])

    result = index.search(Query(vector, 1, 0, 10, 0.0, 2.0))

    assert result.metadata["planner_type"] == "NearestCentroidPlanner"
    assert result.ids == [1]


def _row(label: str, subset_size: float, num_cells: int, best_mode: str) -> dict[str, object]:
    return {
        "algorithm": "hybrid",
        "best_mode": best_mode,
        "planner_feature_subset_size": subset_size,
        "planner_feature_num_cells": num_cells,
        "planner_feature_avg_cell_size": max(1.0, subset_size / max(1, num_cells)),
        "planner_feature_query_window_width": 10,
        "planner_feature_price_range_width": 5.0,
        "planner_feature_category_present": 0,
        "planner_feature_active_records": 1000,
        "planner_feature_tombstoned_records": 0,
        "planner_feature_tombstone_ratio": 0.0,
        "planner_feature_open_ended_fraction": 0.0,
        "planner_feature_fragmentation_score": num_cells / max(1.0, subset_size),
    }
