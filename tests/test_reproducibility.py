import numpy as np

from tsann.generators import SyntheticConfig, generate_queries, generate_records


def test_synthetic_generation_is_deterministic() -> None:
    config = SyntheticConfig(n=20, d=4, num_clusters=3, seed=123)
    left = generate_records(config)
    right = generate_records(config)
    assert [record.timestamp for record in left] == [record.timestamp for record in right]
    assert [record.price for record in left] == [record.price for record in right]
    assert np.allclose(left[0].vector, right[0].vector)


def test_query_generation_is_deterministic() -> None:
    records = generate_records(SyntheticConfig(n=20, d=4, num_clusters=3, seed=123))
    left = generate_queries(records, num_queries=5, seed=456)
    right = generate_queries(records, num_queries=5, seed=456)
    assert left == right
