import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsann.experiments import evaluate_planner, plot_results, run_grid, summarize, train_planner


def main() -> None:
    args = _parse_args()
    _run("grid experiments", run_grid.main, ["--config", str(args.config)])
    _run("summary reports", summarize.main, [])
    if not args.skip_planner:
        train_args = ["--trace", str(args.trace), "--output", str(args.model)]
        eval_args = ["--trace", str(args.trace), "--model", str(args.model), "--output", str(args.planner_eval)]
        if args.holdout_seed:
            for seed in args.holdout_seed:
                train_args.extend(["--exclude-seed", seed])
                eval_args.extend(["--include-seed", seed])
        _run("planner training", train_planner.main, train_args)
        _run("planner evaluation", evaluate_planner.main, eval_args)
    _run("figure generation", plot_results.main, ["--input", str(args.trace), "--output-dir", str(args.figures)])
    print("Full experiment suite complete.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full temporal subset ANN experiment pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke_grid.yaml"))
    parser.add_argument("--trace", type=Path, default=Path("results/csv/run_grid.csv"))
    parser.add_argument("--model", type=Path, default=Path("results/reports/nearest_centroid_planner.json"))
    parser.add_argument("--planner-eval", type=Path, default=Path("results/reports/nearest_centroid_planner_eval.json"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--holdout-seed", action="append", default=["29"])
    parser.add_argument("--skip-planner", action="store_true")
    return parser.parse_args()


def _run(label: str, fn, argv: list[str]) -> None:
    print(f"==> Running {label}")
    previous = sys.argv
    try:
        sys.argv = [label, *argv]
        fn()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
