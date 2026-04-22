from pathlib import Path

from tsann.planner import NearestCentroidPlanner


def main() -> None:
    trace_path = Path("results/csv/run_grid.csv")
    if not trace_path.exists():
        raise SystemExit(f"Missing {trace_path}; run python -m tsann.experiments.run_grid first")
    output = Path("results/reports/nearest_centroid_planner.json")
    planner = NearestCentroidPlanner.train_from_csv(trace_path)
    planner.save(output)
    print(f"Trained nearest-centroid planner with modes: {', '.join(sorted(planner.centroids))}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
