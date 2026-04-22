import argparse
from pathlib import Path

from tsann.planner import NearestCentroidPlanner


def main() -> None:
    args = _parse_args()
    trace_path = args.trace
    if not trace_path.exists():
        raise SystemExit(f"Missing {trace_path}; run python -m tsann.experiments.run_grid first")
    output = args.output
    planner = NearestCentroidPlanner.train_from_csv(
        trace_path,
        include_seeds=set(args.include_seed) if args.include_seed else None,
        exclude_seeds=set(args.exclude_seed) if args.exclude_seed else None,
    )
    planner.save(output)
    print(f"Trained nearest-centroid planner with modes: {', '.join(sorted(planner.centroids))}")
    print(f"Wrote {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nearest-centroid planner from labeled experiment traces.")
    parser.add_argument("--trace", type=Path, default=Path("results/csv/run_grid.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/reports/nearest_centroid_planner.json"))
    parser.add_argument("--include-seed", action="append", default=[])
    parser.add_argument("--exclude-seed", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    main()
