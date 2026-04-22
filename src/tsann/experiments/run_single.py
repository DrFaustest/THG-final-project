import csv
from pathlib import Path

from tsann.ann_global import GlobalAnnThenFilterIndex
from tsann.ann_hybrid import HybridPlannerIndex
from tsann.ann_partitioned import PartitionFirstAnnIndex
from tsann.config import IndexConfig
from tsann.generators import SyntheticConfig, generate_queries, generate_records
from tsann.metrics import recall_at_k, valid_result_rate
from tsann.oracle_exact import ExactFilteredOracle


def main() -> None:
    output = Path("results/csv/run_single.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = generate_records(SyntheticConfig(n=2_000, d=32, num_clusters=8, seed=7))
    queries = generate_queries(records, num_queries=100, k=10, seed=17)
    id_to_record = {record.id: record for record in records}

    config = IndexConfig(partition_exact_threshold=100, planner_exact_threshold=300)
    algorithms = {
        "exact": ExactFilteredOracle(),
        "global_filter": GlobalAnnThenFilterIndex(config),
        "partition_ann": PartitionFirstAnnIndex(config),
        "hybrid": HybridPlannerIndex(config),
    }
    for algorithm in algorithms.values():
        algorithm.build(records)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "algorithm",
                "latency_ms",
                "recall_at_10",
                "valid_result_rate",
                "candidate_count",
                "filtered_out_count",
                "visited_partitions",
                "ann_expansion_rounds",
                "planner_mode",
            ],
        )
        writer.writeheader()
        for query_id, query in enumerate(queries):
            truth = algorithms["exact"].search(query)
            for name, algorithm in algorithms.items():
                result = algorithm.search(query)
                writer.writerow(
                    {
                        "query_id": query_id,
                        "algorithm": name,
                        "latency_ms": result.metadata.get("latency_ms", 0.0),
                        "recall_at_10": recall_at_k(result, truth, 10),
                        "valid_result_rate": valid_result_rate(result, query, id_to_record),
                        "candidate_count": result.metadata.get("candidate_count", 0),
                        "filtered_out_count": result.metadata.get("filtered_out_count", 0),
                        "visited_partitions": result.metadata.get("visited_partitions", 0),
                        "ann_expansion_rounds": result.metadata.get("ann_expansion_rounds", 0),
                        "planner_mode": result.metadata.get("planner_mode", ""),
                    }
                )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
