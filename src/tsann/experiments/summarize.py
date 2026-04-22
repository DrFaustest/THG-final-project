from pathlib import Path


def main() -> None:
    path = Path("results/csv/run_single.csv")
    if not path.exists():
        raise SystemExit(f"Missing {path}; run python -m tsann.experiments.run_single first")
    summary = _summarize(path)
    output = Path("results/reports/summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_summary(output, summary)
    for algorithm, row in summary.items():
        print(
            algorithm,
            f"p50={row['latency_p50_ms']:.3f}ms",
            f"p95={row['latency_p95_ms']:.3f}ms",
            f"recall={row['mean_recall_at_10']:.3f}",
            f"valid={row['mean_valid_rate']:.3f}",
        )
    print(f"Wrote {output}")


def _summarize(path: Path) -> dict[str, dict[str, float]]:
    import csv

    groups: dict[str, dict[str, list[float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = groups.setdefault(
                row["algorithm"],
                {"latency_ms": [], "recall_at_10": [], "valid_result_rate": []},
            )
            group["latency_ms"].append(float(row["latency_ms"]))
            group["recall_at_10"].append(float(row["recall_at_10"]))
            group["valid_result_rate"].append(float(row["valid_result_rate"]))

    return {
        algorithm: {
            "latency_p50_ms": _quantile(values["latency_ms"], 0.50),
            "latency_p95_ms": _quantile(values["latency_ms"], 0.95),
            "mean_recall_at_10": _mean(values["recall_at_10"]),
            "mean_valid_rate": _mean(values["valid_result_rate"]),
        }
        for algorithm, values in groups.items()
    }


def _write_summary(path: Path, summary: dict[str, dict[str, float]]) -> None:
    import csv

    fieldnames = ["algorithm", "latency_p50_ms", "latency_p95_ms", "mean_recall_at_10", "mean_valid_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for algorithm, row in summary.items():
            writer.writerow({"algorithm": algorithm, **row})


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


if __name__ == "__main__":
    main()
