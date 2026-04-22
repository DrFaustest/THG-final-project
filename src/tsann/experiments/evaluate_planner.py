import csv
import json
from pathlib import Path

from tsann.planner import NearestCentroidPlanner, _row_to_vector


def main() -> None:
    trace_path = Path("results/csv/run_grid.csv")
    model_path = Path("results/reports/nearest_centroid_planner.json")
    if not trace_path.exists():
        raise SystemExit(f"Missing {trace_path}; run python -m tsann.experiments.run_grid first")
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}; run python -m tsann.experiments.train_planner first")

    planner = NearestCentroidPlanner.load(model_path)
    rows = _read_rows(trace_path)
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
    output = Path("results/reports/nearest_centroid_planner_eval.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}")


def _read_rows(path: Path) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    groups: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["workload"], row["seed"], row["query_id"])
            groups.setdefault(key, {})[row["algorithm"]] = row
    return groups


if __name__ == "__main__":
    main()
