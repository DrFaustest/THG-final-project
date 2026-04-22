import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ALGORITHMS = ["exact", "global_filter", "partition_ann", "hybrid"]


def main() -> None:
    args = _parse_args()
    rows = _read_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _plot_latency(rows, args.output_dir / "latency_by_workload.png")
    _plot_recall(rows, args.output_dir / "recall_by_workload.png")
    _plot_estimate_error(rows, args.output_dir / "subset_estimate_error.png")
    _plot_planner_modes(rows, args.output_dir / "planner_mode_counts.png")
    _plot_planner_regret(rows, args.output_dir / "planner_regret_by_workload.png")
    _plot_maintenance(rows, args.output_dir / "maintenance_metrics.png")
    print(f"Wrote figures to {args.output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report figures from experiment CSV output.")
    parser.add_argument("--input", type=Path, default=Path("results/csv/run_grid.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing {path}; run python -m tsann.experiments.run_grid first")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plot_latency(rows: list[dict[str, str]], output: Path) -> None:
    grouped = _group(rows, "latency_ms")
    workloads = _workloads(rows)
    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(workloads) * 0.8), 10), sharex=True)
    for axis, quantile, title in zip(axes, [0.50, 0.95, 0.99], ["p50 latency", "p95 latency", "p99 latency"]):
        series = {
            algorithm: [_quantile(grouped.get((workload, algorithm), []), quantile) for workload in workloads]
            for algorithm in ALGORITHMS
        }
        _grouped_bars(axis, workloads, series)
        axis.set_ylabel("ms")
        axis.set_title(title)
    axes[-1].tick_params(axis="x", labelrotation=35)
    axes[0].legend(ncols=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_recall(rows: list[dict[str, str]], output: Path) -> None:
    workloads = _workloads(rows)
    grouped = _group(rows, "recall_at_10")
    series = {
        algorithm: [_mean(grouped.get((workload, algorithm), [])) for workload in workloads]
        for algorithm in ALGORITHMS
    }
    fig, axis = plt.subplots(figsize=(max(10, len(workloads) * 0.8), 5))
    _grouped_bars(axis, workloads, series)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("mean recall@10")
    axis.set_title("Recall by workload")
    axis.tick_params(axis="x", labelrotation=35)
    axis.legend(ncols=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_estimate_error(rows: list[dict[str, str]], output: Path) -> None:
    workloads = _workloads(rows)
    by_workload = {
        workload: [
            _float(row, "subset_estimate_error")
            for row in rows
            if row["workload"] == workload and row["algorithm"] == "hybrid"
        ]
        for workload in workloads
    }
    fig, axis = plt.subplots(figsize=(max(10, len(workloads) * 0.8), 5))
    axis.bar(range(len(workloads)), [_mean(by_workload[workload]) for workload in workloads], color="#4c78a8")
    axis.set_xticks(range(len(workloads)), workloads, rotation=35, ha="right")
    axis.set_ylabel("mean relative error")
    axis.set_title("Subset estimate error by workload")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_planner_modes(rows: list[dict[str, str]], output: Path) -> None:
    workloads = _workloads(rows)
    modes = ["exact", "partition_ann", "global_filter"]
    counts = {workload: Counter() for workload in workloads}
    for row in rows:
        if row["algorithm"] == "hybrid" and row.get("planner_mode"):
            counts[row["workload"]][row["planner_mode"]] += 1
    bottoms = [0] * len(workloads)
    fig, axis = plt.subplots(figsize=(max(10, len(workloads) * 0.8), 5))
    colors = {"exact": "#4c78a8", "partition_ann": "#59a14f", "global_filter": "#f28e2b"}
    for mode in modes:
        values = [counts[workload][mode] for workload in workloads]
        axis.bar(range(len(workloads)), values, bottom=bottoms, label=mode, color=colors[mode])
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_xticks(range(len(workloads)), workloads, rotation=35, ha="right")
    axis.set_ylabel("query count")
    axis.set_title("Hybrid planner mode counts")
    axis.legend(ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_planner_regret(rows: list[dict[str, str]], output: Path) -> None:
    workloads = _workloads(rows)
    hybrid_rows = [row for row in rows if row["algorithm"] == "hybrid"]
    series = {
        "mean_latency_regret": [
            _mean([_float(row, "planner_latency_regret") for row in hybrid_rows if row["workload"] == workload])
            for workload in workloads
        ],
        "p95_latency_regret": [
            _quantile([_float(row, "planner_latency_regret") for row in hybrid_rows if row["workload"] == workload], 0.95)
            for workload in workloads
        ],
        "mean_recall_gap": [
            _mean([_float(row, "planner_recall_gap") for row in hybrid_rows if row["workload"] == workload])
            for workload in workloads
        ],
    }
    fig, axis = plt.subplots(figsize=(max(10, len(workloads) * 0.8), 5))
    _grouped_bars(axis, workloads, series)
    axis.set_ylabel("relative regret / recall gap")
    axis.set_title("Hybrid planner regret by workload")
    axis.tick_params(axis="x", labelrotation=35)
    axis.legend(ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_maintenance(rows: list[dict[str, str]], output: Path) -> None:
    workloads = _workloads(rows)
    hybrid_rows = [row for row in rows if row["algorithm"] == "hybrid"]
    metrics = {
        "maintenance_rebuild_count": [
            _max_for_workload(hybrid_rows, workload, "maintenance_rebuild_count") for workload in workloads
        ],
        "maintenance_cell_rebuild_count": [
            _max_for_workload(hybrid_rows, workload, "maintenance_cell_rebuild_count") for workload in workloads
        ],
        "compaction_count": [_max_for_workload(hybrid_rows, workload, "compaction_count") for workload in workloads],
    }
    fig, axis = plt.subplots(figsize=(max(10, len(workloads) * 0.8), 5))
    _grouped_bars(axis, workloads, metrics)
    axis.set_ylabel("max count")
    axis.set_title("Maintenance metrics by workload")
    axis.tick_params(axis="x", labelrotation=35)
    axis.legend(ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _group(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str], list[float]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["workload"], row["algorithm"]), []).append(_float(row, metric))
    return grouped


def _grouped_bars(axis, labels: list[str], series: dict[str, list[float]]) -> None:
    width = 0.8 / max(1, len(series))
    x_values = list(range(len(labels)))
    offsets = [(-0.4 + width / 2) + idx * width for idx in range(len(series))]
    for offset, (name, values) in zip(offsets, series.items()):
        axis.bar([x + offset for x in x_values], values, width=width, label=name)
    axis.set_xticks(x_values, labels, rotation=35, ha="right")


def _workloads(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["workload"] for row in rows})


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return 0.0 if value == "" else float(value)


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


def _max_for_workload(rows: list[dict[str, str]], workload: str, metric: str) -> float:
    return max((_float(row, metric) for row in rows if row["workload"] == workload), default=0.0)


if __name__ == "__main__":
    main()
