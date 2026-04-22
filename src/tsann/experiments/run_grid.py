from pathlib import Path
from dataclasses import replace

from tsann.config import IndexConfig
from tsann.experiments.run_single import run_experiment
from tsann.generators import SyntheticConfig, generate_queries, generate_records


def main() -> None:
    output = Path("results/csv/run_grid.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    index_config = IndexConfig(partition_exact_threshold=80, planner_exact_threshold=250)
    seeds = [11, 29]
    dimensions = [32, 64]
    regimes = [
        ("static", SyntheticConfig(n=1_200, d=32, num_clusters=8, seed=0), None),
        (
            "short_lifetime_expire",
            SyntheticConfig(n=1_200, d=32, num_clusters=8, seed=0, lifetime_min=1, lifetime_max=8),
            {"expire_before": 120},
        ),
        (
            "long_lifetime_expire",
            SyntheticConfig(n=1_200, d=32, num_clusters=8, seed=0, lifetime_min=120, lifetime_max=300),
            {"expire_before": 120},
        ),
        (
            "open_ended_heavy",
            SyntheticConfig(
                n=1_200,
                d=32,
                num_clusters=8,
                seed=0,
                lifetime_min=20,
                lifetime_max=80,
                open_ended_fraction=0.7,
            ),
            {"expire_before": 120},
        ),
        ("append_delete", SyntheticConfig(n=900, d=32, num_clusters=8, seed=0), {"append_records": True, "delete": True}),
    ]

    for workload, base_config, operations in regimes:
        for seed in seeds:
            for dimension in dimensions:
                config = replace(base_config, seed=seed, d=dimension)
                _run_regime(output, index_config, workload, config, operations or {})
    print(f"Wrote {output}")


def _run_regime(
    output: Path,
    index_config: IndexConfig,
    workload: str,
    config: SyntheticConfig,
    operations: dict,
) -> None:
    records = generate_records(config)
    queries = generate_queries(records, num_queries=40, k=10, seed=config.seed + config.d + 100)
    append_records = None
    delete_ids = None
    workload_name = f"{workload}_d{config.d}"
    if operations.get("append_records"):
        append_records = generate_records(
            SyntheticConfig(n=300, d=config.d, num_clusters=config.num_clusters, seed=config.seed + 1)
        )
        max_id = max(record.id for record in records)
        append_records = [
            type(record)(
                max_id + 1 + offset,
                record.vector,
                record.valid_from,
                record.price,
                record.category,
                record.valid_to,
            )
            for offset, record in enumerate(append_records)
        ]
    if operations.get("delete"):
        delete_ids = [record.id for record in records[: len(records) // 10]]
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


if __name__ == "__main__":
    main()
