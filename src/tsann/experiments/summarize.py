from pathlib import Path
import json


def main() -> None:
    path = Path("results/csv/run_grid.csv")
    if not path.exists():
        path = Path("results/csv/run_single.csv")
    if not path.exists():
        raise SystemExit(f"Missing {path}; run python -m tsann.experiments.run_single first")
    summary = _summarize(path)
    output = Path("results/reports/summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_summary(output, summary)
    json_output = Path("results/reports/summary.json")
    json_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    for group, row in summary.items():
        print(
            group,
            f"p50={row['latency_p50_ms']:.3f}ms",
            f"p95={row['latency_p95_ms']:.3f}ms",
            f"p99={row['latency_p99_ms']:.3f}ms",
            f"recall={row['mean_recall_at_10']:.3f}",
            f"valid={row['mean_valid_rate']:.3f}",
            f"est_err={row['mean_subset_estimate_error']:.3f}",
        )
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")


def _summarize(path: Path) -> dict[str, dict[str, float | str]]:
    import csv

    groups: dict[str, dict[str, list[float] | list[str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            workload = row.get("workload", "default")
            algorithm = row["algorithm"]
            group = groups.setdefault(
                f"{workload}:{algorithm}",
                {
                    "latency_ms": [],
                    "recall_at_10": [],
                    "valid_result_rate": [],
                    "subset_estimate_error": [],
                    "active_records": [],
                    "tombstoned_records": [],
                    "rebuild_count": [],
                    "cell_rebuild_count": [],
                    "maintenance_rebuild_count": [],
                    "maintenance_cell_rebuild_count": [],
                    "compaction_count": [],
                    "deleted_record_count": [],
                    "expired_record_count": [],
                    "planner_matched_best_mode": [],
                    "planner_latency_regret": [],
                    "planner_recall_gap": [],
                    "planner_mode": [],
                },
            )
            _append_float(group, "latency_ms", row)
            _append_float(group, "recall_at_10", row)
            _append_float(group, "valid_result_rate", row)
            _append_float(group, "subset_estimate_error", row)
            _append_float(group, "active_records", row)
            _append_float(group, "tombstoned_records", row)
            _append_float(group, "rebuild_count", row)
            _append_float(group, "cell_rebuild_count", row)
            _append_float(group, "maintenance_rebuild_count", row)
            _append_float(group, "maintenance_cell_rebuild_count", row)
            _append_float(group, "compaction_count", row)
            _append_float(group, "deleted_record_count", row)
            _append_float(group, "expired_record_count", row)
            _append_float(group, "planner_matched_best_mode", row)
            _append_float(group, "planner_latency_regret", row)
            _append_float(group, "planner_recall_gap", row)
            if row.get("planner_mode"):
                group["planner_mode"].append(row["planner_mode"])

    return {
        group_name: {
            "latency_p50_ms": _quantile(values["latency_ms"], 0.50),
            "latency_p95_ms": _quantile(values["latency_ms"], 0.95),
            "latency_p99_ms": _quantile(values["latency_ms"], 0.99),
            "mean_recall_at_10": _mean(values["recall_at_10"]),
            "mean_valid_rate": _mean(values["valid_result_rate"]),
            "mean_subset_estimate_error": _mean(values["subset_estimate_error"]),
            "mean_active_records": _mean(values["active_records"]),
            "mean_tombstoned_records": _mean(values["tombstoned_records"]),
            "max_rebuild_count": max(values["rebuild_count"], default=0.0),
            "max_cell_rebuild_count": max(values["cell_rebuild_count"], default=0.0),
            "max_maintenance_rebuild_count": max(values["maintenance_rebuild_count"], default=0.0),
            "max_maintenance_cell_rebuild_count": max(values["maintenance_cell_rebuild_count"], default=0.0),
            "max_compaction_count": max(values["compaction_count"], default=0.0),
            "max_deleted_record_count": max(values["deleted_record_count"], default=0.0),
            "max_expired_record_count": max(values["expired_record_count"], default=0.0),
            "mean_planner_match_rate": _mean(values["planner_matched_best_mode"]),
            "mean_planner_latency_regret": _mean(values["planner_latency_regret"]),
            "p95_planner_latency_regret": _quantile(values["planner_latency_regret"], 0.95),
            "mean_planner_recall_gap": _mean(values["planner_recall_gap"]),
            "planner_mode_counts": _mode_counts(values["planner_mode"]),
        }
        for group_name, values in groups.items()
    }


def _write_summary(path: Path, summary: dict[str, dict[str, float]]) -> None:
    import csv

    fieldnames = [
        "group",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "mean_recall_at_10",
        "mean_valid_rate",
        "mean_subset_estimate_error",
        "mean_active_records",
        "mean_tombstoned_records",
        "max_rebuild_count",
        "max_cell_rebuild_count",
        "max_maintenance_rebuild_count",
        "max_maintenance_cell_rebuild_count",
        "max_compaction_count",
        "max_deleted_record_count",
        "max_expired_record_count",
        "mean_planner_match_rate",
        "mean_planner_latency_regret",
        "p95_planner_latency_regret",
        "mean_planner_recall_gap",
        "planner_mode_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, row in summary.items():
            writer.writerow({"group": group, **row})


def _append_float(group: dict, key: str, row: dict[str, str]) -> None:
    value = row.get(key, "")
    if value == "":
        return
    group[key].append(float(value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _mode_counts(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


if __name__ == "__main__":
    main()
