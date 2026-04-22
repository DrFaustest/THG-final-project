from tsann.ann_partitioned import PartitionFirstAnnIndex
from tsann.config import IndexConfig
from tsann.generators import SyntheticConfig, generate_records
from tsann.metrics import recall_at_k
from tsann.oracle_exact import ExactFilteredOracle
from tsann.types import Query, Record
import numpy as np


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


def test_partitioned_routing_finds_old_open_ended_record() -> None:
    vector = np.array([0.0, 0.0], dtype=np.float32)
    records = [Record(1, vector, valid_from=0, valid_to=None, price=10.0)]
    index = PartitionFirstAnnIndex(IndexConfig(time_bucket_width=10, price_bucket_width=10.0))
    index.build(records)

    result = index.search(Query(vector, 1, t_start=100, t_end=110, price_min=0.0, price_max=20.0))

    assert result.ids == [1]
    assert result.metadata["visited_partitions"] == 1


def test_partitioned_routing_prunes_expired_only_cells() -> None:
    vector = np.array([0.0, 0.0], dtype=np.float32)
    records = [
        Record(1, vector + 1, valid_from=0, valid_to=5, price=1.0),
        Record(2, vector, valid_from=0, valid_to=None, price=11.0),
    ]
    index = PartitionFirstAnnIndex(IndexConfig(time_bucket_width=10, price_bucket_width=10.0))
    index.build(records)

    result = index.search(Query(vector, 2, t_start=100, t_end=110, price_min=0.0, price_max=20.0))

    assert result.ids == [2]
    assert result.metadata["visited_partitions"] == 1
