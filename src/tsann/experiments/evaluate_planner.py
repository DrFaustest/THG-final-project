import argparse
import csv
import json
from pathlib import Path

from tsann.planner import NearestCentroidPlanner, _row_to_vector


def main() -> None:
    args = _parse_args()
    trace_path = args.trace
    model_path = args.model
    if not trace_path.exists():
        raise SystemExit(f"Missing {trace_path}; run python -m tsann.experiments.run_grid first")
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}; run python -m tsann.experiments.train_planner first")

    planner = NearestCentroidPlanner.load(model_path)
    rows = _read_rows(
        trace_path,
        include_seeds=set(args.include_seed) if args.include_seed else None,
        exclude_seeds=set(args.exclude_seed) if args.exclude_seed else None,
    )
    predictions = []
    for key, group in rows.items():
        hybrid = group.get("hybrid")
        if hybrid is None:
            continue
        predicted = planner.predict_vector(_row_to_vector(hybrid))
        best = hybrid["best_mode"]
        predicted_latency = float(group.get(predicted, hybrid)["latency_ms"])
        best_latency = float(group[best]["latency_ms"])
        predictions.append(
            {
                "key": key,
                "predicted": predicted,
                "best": best,
                "correct": predicted == best,
                "latency_regret_ms": max(0.0, predicted_latency - best_latency),
            }
        )

    accuracy = sum(1 for row in predictions if row["correct"]) / max(1, len(predictions))
    mean_regret = sum(row["latency_regret_ms"] for row in predictions) / max(1, len(predictions))
    report = {
        "num_queries": len(predictions),
        "accuracy": accuracy,
        "mean_latency_regret_ms": mean_regret,
    }
    output = args.output
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nearest-centroid planner against best-mode trace labels.")
    parser.add_argument("--trace", type=Path, default=Path("results/csv/run_grid.csv"))
    parser.add_argument("--model", type=Path, default=Path("results/reports/nearest_centroid_planner.json"))
    parser.add_argument("--output", type=Path, default=Path("results/reports/nearest_centroid_planner_eval.json"))
    parser.add_argument("--include-seed", action="append", default=[])
    parser.add_argument("--exclude-seed", action="append", default=[])
    return parser.parse_args()


def _read_rows(
    path: Path,
    *,
    include_seeds: set[str] | None = None,
    exclude_seeds: set[str] | None = None,
) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    groups: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = row["seed"]
            if include_seeds is not None and seed not in include_seeds:
                continue
            if exclude_seeds is not None and seed in exclude_seeds:
                continue
            key = (row["workload"], row["seed"], row["query_id"])
            groups.setdefault(key, {})[row["algorithm"]] = row
    return groups


if __name__ == "__main__":
    main()
