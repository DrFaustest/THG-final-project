import argparse
from pathlib import Path

import yaml

from tsann.config import IndexConfig
from tsann.experiments.run_single import run_experiment
from tsann.generators import SyntheticConfig, generate_queries, generate_records
from tsann.types import Record


def main() -> None:
    args = _parse_args()
    grid = _load_grid(args.config)
    output = Path(grid.get("output", "results/csv/run_grid.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    index_config = IndexConfig(**grid.get("index_config", {}))
    seeds = grid["seeds"]
    dimensions = grid["dimensions"]
    dataset_sizes = grid.get("dataset_sizes", [None])
    num_queries = int(grid.get("num_queries", 40))
    k = int(grid.get("k", 10))

    for regime in grid["regimes"]:
        base_config = regime["synthetic"]
        operations = regime.get("operations", {})
        for seed in seeds:
            for dimension in dimensions:
                for dataset_size in dataset_sizes:
                    synthetic = _synthetic_config(base_config, seed, dimension, dataset_size)
                    _run_regime(output, index_config, regime["name"], synthetic, operations, num_queries, k)
    print(f"Wrote {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configurable temporal subset ANN benchmark grid.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke_grid.yaml"))
    return parser.parse_args()


def _load_grid(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing grid config: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _synthetic_config(base: dict, seed: int, dimension: int, dataset_size: int | None) -> SyntheticConfig:
    payload = dict(base)
    payload["seed"] = seed
    payload["d"] = dimension
    if dataset_size is not None:
        payload["n"] = dataset_size
    return SyntheticConfig(**payload)


def _run_regime(
    output: Path,
    index_config: IndexConfig,
    workload: str,
    config: SyntheticConfig,
    operations: dict,
    num_queries: int,
    k: int,
) -> None:
    records = generate_records(config)
    queries = generate_queries(records, num_queries=num_queries, k=k, seed=config.seed + config.d + config.n)
    append_records = _append_records(records, config, operations)
    delete_ids = None
    if "delete_fraction" in operations:
        delete_count = int(len(records) * float(operations["delete_fraction"]))
        delete_ids = [record.id for record in records[:delete_count]]
    workload_name = f"{workload}_n{config.n}_d{config.d}"
    run_experiment(
        records,
        queries,
        output,
        workload=workload_name,
        seed=config.seed,
        config=index_config,
        append_records=append_records,
        delete_ids=delete_ids,
        expire_before=operations.get("expire_before"),
        append_output=True,
    )


def _append_records(records: list[Record], config: SyntheticConfig, operations: dict) -> list[Record] | None:
    append_count = int(operations.get("append_count", 0))
    if append_count <= 0:
        return None
    generated = generate_records(
        SyntheticConfig(
            n=append_count,
            d=config.d,
            num_clusters=config.num_clusters,
            time_span=config.time_span,
            price_span=config.price_span,
            drift_strength=config.drift_strength,
            attr_corr=config.attr_corr,
            category_corr=config.category_corr,
            noise_std=config.noise_std,
            seed=config.seed + 1,
            cluster_size_mode=config.cluster_size_mode,
            arrival_mode=config.arrival_mode,
            lifetime_min=config.lifetime_min,
            lifetime_max=config.lifetime_max,
            open_ended_fraction=config.open_ended_fraction,
        )
    )
    max_id = max(record.id for record in records)
    return [
        Record(
            max_id + 1 + offset,
            record.vector,
            record.valid_from,
            record.price,
            record.category,
            record.valid_to,
        )
        for offset, record in enumerate(generated)
    ]


if __name__ == "__main__":
    main()
