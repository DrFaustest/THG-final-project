import csv
from pathlib import Path
from collections.abc import Mapping

from tsann.ann_global import GlobalAnnThenFilterIndex
from tsann.ann_hybrid import HybridPlannerIndex
from tsann.ann_partitioned import PartitionFirstAnnIndex
from tsann.config import IndexConfig
from tsann.generators import SyntheticConfig, generate_queries, generate_records
from tsann.metrics import recall_at_k, valid_result_rate
from tsann.oracle_exact import ExactFilteredOracle
from tsann.types import Query, Record


FIELDNAMES = [
    "workload",
    "seed",
    "query_id",
    "algorithm",
    "latency_ms",
    "recall_at_10",
    "valid_result_rate",
    "exact_subset_size",
    "subset_size_estimate",
    "subset_estimate_error",
    "estimate_num_cells",
    "estimate_avg_cell_size",
    "query_window_width",
    "price_range_width",
    "category_present",
    "candidate_count",
    "filtered_out_count",
    "visited_partitions",
    "ann_expansion_rounds",
    "exact_distance_computations",
    "planner_type",
    "planner_mode",
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
    "best_mode",
    "best_mode_latency_ms",
    "best_mode_recall_at_10",
    "active_records",
    "tombstoned_records",
    "tombstone_ratio",
    "rebuild_count",
    "cell_rebuild_count",
    "maintenance_rebuild_count",
    "maintenance_cell_rebuild_count",
    "compaction_count",
    "deleted_record_count",
    "expired_record_count",
    "open_ended_fraction",
    "index_visible_size",
]


def main() -> None:
    output = Path("results/csv/run_single.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    seed = 7
    records = generate_records(SyntheticConfig(n=2_000, d=32, num_clusters=8, seed=seed))
    queries = generate_queries(records, num_queries=100, k=10, seed=17)
    run_experiment(records, queries, output, workload="static_smoke", seed=seed)
    print(f"Wrote {output}")


def run_experiment(
    records: list[Record],
    queries: list[Query],
    output: Path,
    *,
    workload: str,
    seed: int,
    config: IndexConfig | None = None,
    append_records: list[Record] | None = None,
    delete_ids: list[int] | None = None,
    expire_before: int | None = None,
    append_output: bool = False,
) -> None:
    id_to_record = {record.id: record for record in records}
    for record in append_records or []:
        id_to_record[record.id] = record

    index_config = config or IndexConfig(partition_exact_threshold=100, planner_exact_threshold=300)
    algorithms = {
        "exact": ExactFilteredOracle(),
        "global_filter": GlobalAnnThenFilterIndex(index_config),
        "partition_ann": PartitionFirstAnnIndex(index_config),
        "hybrid": HybridPlannerIndex(index_config),
    }
    for algorithm in algorithms.values():
        algorithm.build(records)
        for record in append_records or []:
            algorithm.insert(record)
        for record_id in delete_ids or []:
            algorithm.delete(record_id)
        if expire_before is not None:
            algorithm.expire(expire_before)

    write_header = not append_output or not output.exists()
    with output.open("a" if append_output else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for query_id, query in enumerate(queries):
            truth = algorithms["exact"].search(query)
            exact_subset_size = int(truth.metadata.get("candidate_count", len(truth.ids)))
            estimate = algorithms["partition_ann"].estimate_subset(query)
            estimate_error = abs(estimate.subset_size - exact_subset_size) / max(1, exact_subset_size)
            results = {name: algorithm.search(query) for name, algorithm in algorithms.items()}
            recalls = {name: recall_at_k(result, truth, 10) for name, result in results.items()}
            best_mode = _best_mode(results, recalls)
            for name, algorithm in algorithms.items():
                result = results[name]
                stats = _maintenance_stats(algorithm.stats())
                writer.writerow(
                    {
                        "workload": workload,
                        "seed": seed,
                        "query_id": query_id,
                        "algorithm": name,
                        "latency_ms": result.metadata.get("latency_ms", 0.0),
                        "recall_at_10": recall_at_k(result, truth, 10),
                        "valid_result_rate": valid_result_rate(result, query, id_to_record),
                        "exact_subset_size": exact_subset_size,
                        "subset_size_estimate": estimate.subset_size,
                        "subset_estimate_error": estimate_error,
                        "estimate_num_cells": estimate.num_cells,
                        "estimate_avg_cell_size": estimate.avg_cell_size,
                        "query_window_width": query.t_end - query.t_start + 1,
                        "price_range_width": query.price_max - query.price_min,
                        "category_present": int(query.category is not None),
                        "candidate_count": result.metadata.get("candidate_count", 0),
                        "filtered_out_count": result.metadata.get("filtered_out_count", 0),
                        "visited_partitions": result.metadata.get("visited_partitions", 0),
                        "ann_expansion_rounds": result.metadata.get("ann_expansion_rounds", 0),
                        "exact_distance_computations": result.metadata.get("exact_distance_computations", 0),
                        "planner_type": result.metadata.get("planner_type", ""),
                        "planner_mode": result.metadata.get("planner_mode", ""),
                        "planner_feature_subset_size": result.metadata.get("planner_feature_subset_size", ""),
                        "planner_feature_num_cells": result.metadata.get("planner_feature_num_cells", ""),
                        "planner_feature_avg_cell_size": result.metadata.get("planner_feature_avg_cell_size", ""),
                        "planner_feature_query_window_width": result.metadata.get("planner_feature_query_window_width", ""),
                        "planner_feature_price_range_width": result.metadata.get("planner_feature_price_range_width", ""),
                        "planner_feature_category_present": int(result.metadata.get("planner_feature_category_present", False)),
                        "planner_feature_active_records": result.metadata.get("planner_feature_active_records", ""),
                        "planner_feature_tombstoned_records": result.metadata.get("planner_feature_tombstoned_records", ""),
                        "planner_feature_tombstone_ratio": result.metadata.get("planner_feature_tombstone_ratio", ""),
                        "planner_feature_open_ended_fraction": result.metadata.get("planner_feature_open_ended_fraction", ""),
                        "planner_feature_fragmentation_score": result.metadata.get("planner_feature_fragmentation_score", ""),
                        "best_mode": best_mode,
                        "best_mode_latency_ms": results[best_mode].metadata.get("latency_ms", 0.0),
                        "best_mode_recall_at_10": recalls[best_mode],
                        "active_records": stats["active_records"],
                        "tombstoned_records": stats["tombstoned_records"],
                        "tombstone_ratio": stats["tombstone_ratio"],
                        "rebuild_count": stats["rebuild_count"],
                        "cell_rebuild_count": stats["cell_rebuild_count"],
                        "maintenance_rebuild_count": stats["maintenance_rebuild_count"],
                        "maintenance_cell_rebuild_count": stats["maintenance_cell_rebuild_count"],
                        "compaction_count": stats["compaction_count"],
                        "deleted_record_count": stats["deleted_record_count"],
                        "expired_record_count": stats["expired_record_count"],
                        "open_ended_fraction": stats["open_ended_fraction"],
                        "index_visible_size": result.metadata.get("index_visible_size", ""),
                    }
                )


def _best_mode(results: dict[str, object], recalls: dict[str, float], recall_floor: float = 0.98) -> str:
    eligible = [
        (name, float(result.metadata.get("latency_ms", 0.0)))
        for name, result in results.items()
        if name != "hybrid" and recalls[name] >= recall_floor
    ]
    if not eligible:
        return max((name for name in recalls if name != "hybrid"), key=lambda name: recalls[name])
    return min(eligible, key=lambda item: item[1])[0]


def _maintenance_stats(stats: Mapping) -> dict[str, float]:
    if "partitioned" in stats:
        partitioned = stats["partitioned"]
        global_stats = stats["global"]
        active = float(partitioned.get("active_records", 0))
        tombstoned = float(partitioned.get("tombstoned_records", 0))
        open_ended = float(partitioned.get("open_ended_records", 0))
        return {
            "active_records": active,
            "tombstoned_records": tombstoned,
            "tombstone_ratio": tombstoned / max(1.0, active + tombstoned),
            "rebuild_count": float(global_stats.get("rebuild_count", 0)),
            "cell_rebuild_count": float(partitioned.get("cell_rebuild_count", 0)),
            "maintenance_rebuild_count": float(global_stats.get("maintenance_rebuild_count", 0)),
            "maintenance_cell_rebuild_count": float(partitioned.get("maintenance_cell_rebuild_count", 0)),
            "compaction_count": float(global_stats.get("compaction_count", 0)) + float(partitioned.get("compaction_count", 0)),
            "deleted_record_count": float(partitioned.get("deleted_record_count", 0)),
            "expired_record_count": float(partitioned.get("expired_record_count", 0)),
            "open_ended_fraction": open_ended / max(1.0, active),
        }
    active = float(stats.get("active_records", stats.get("records", 0)))
    tombstoned = float(stats.get("tombstoned_records", 0))
    open_ended = float(stats.get("open_ended_records", 0))
    return {
        "active_records": active,
        "tombstoned_records": tombstoned,
        "tombstone_ratio": tombstoned / max(1.0, active + tombstoned),
        "rebuild_count": float(stats.get("rebuild_count", 0)),
        "cell_rebuild_count": float(stats.get("cell_rebuild_count", 0)),
        "maintenance_rebuild_count": float(stats.get("maintenance_rebuild_count", 0)),
        "maintenance_cell_rebuild_count": float(stats.get("maintenance_cell_rebuild_count", 0)),
        "compaction_count": float(stats.get("compaction_count", 0)),
        "deleted_record_count": float(stats.get("deleted_record_count", 0)),
        "expired_record_count": float(stats.get("expired_record_count", 0)),
        "open_ended_fraction": open_ended / max(1.0, active),
    }


if __name__ == "__main__":
    main()
