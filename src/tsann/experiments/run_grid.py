from pathlib import Path

from tsann.config import IndexConfig
from tsann.experiments.run_single import run_experiment
from tsann.generators import SyntheticConfig, generate_queries, generate_records


def main() -> None:
    output = Path("results/csv/run_grid.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    index_config = IndexConfig(partition_exact_threshold=100, planner_exact_threshold=300)
    regimes = [
        ("static", SyntheticConfig(n=2_000, d=32, num_clusters=8, seed=11), None),
        (
            "short_lifetime_expire",
            SyntheticConfig(n=2_000, d=32, num_clusters=8, seed=12, lifetime_min=1, lifetime_max=8),
            {"expire_before": 120},
        ),
        (
            "long_lifetime_expire",
            SyntheticConfig(n=2_000, d=32, num_clusters=8, seed=13, lifetime_min=120, lifetime_max=300),
            {"expire_before": 120},
        ),
        (
            "open_ended_heavy",
            SyntheticConfig(
                n=2_000,
                d=32,
                num_clusters=8,
                seed=14,
                lifetime_min=20,
                lifetime_max=80,
                open_ended_fraction=0.7,
            ),
            {"expire_before": 120},
        ),
        ("append_delete", SyntheticConfig(n=1_500, d=32, num_clusters=8, seed=15), {"append": True, "delete": True}),
    ]

    for workload, config, operations in regimes:
        records = generate_records(config)
        queries = generate_queries(records, num_queries=80, k=10, seed=config.seed + 100)
        kwargs = operations or {}
        append_records = None
        delete_ids = None
        if kwargs.get("append"):
            append_records = generate_records(
                SyntheticConfig(n=500, d=config.d, num_clusters=config.num_clusters, seed=config.seed + 1)
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
        if kwargs.get("delete"):
            delete_ids = [record.id for record in records[: len(records) // 10]]
        run_experiment(
            records,
            queries,
            output,
            workload=workload,
            seed=config.seed,
            config=index_config,
            append_records=append_records,
            delete_ids=delete_ids,
            expire_before=kwargs.get("expire_before"),
            append=True,
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
