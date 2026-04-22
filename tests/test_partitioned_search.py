from tsann.ann_partitioned import PartitionFirstAnnIndex
from tsann.config import IndexConfig
from tsann.generators import SyntheticConfig, generate_records
from tsann.metrics import recall_at_k
from tsann.oracle_exact import ExactFilteredOracle
from tsann.types import Query


def test_partitioned_search_matches_oracle_when_cells_exact() -> None:
    records = generate_records(SyntheticConfig(n=300, d=8, num_clusters=4, seed=2))
    query = Query(records[0].vector, 10, 0, 364, 0.0, 100.0, None)
    config = IndexConfig(partition_exact_threshold=10_000)
    oracle = ExactFilteredOracle()
    partitioned = PartitionFirstAnnIndex(config)
    oracle.build(records)
    partitioned.build(records)
    truth = oracle.search(query)
    result = partitioned.search(query)
    assert recall_at_k(result, truth, 10) == 1.0
    assert result.metadata["visited_partitions"] > 0
