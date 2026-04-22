from tsann.ann_global import GlobalAnnThenFilterIndex
from tsann.generators import SyntheticConfig, generate_records
from tsann.oracle_exact import ExactFilteredOracle
from tsann.types import Query


def test_global_filter_results_are_valid_subset_of_oracle() -> None:
    records = generate_records(SyntheticConfig(n=200, d=8, num_clusters=4, seed=1))
    query = Query(records[0].vector, 5, 0, 364, 0.0, 100.0, records[0].category)
    oracle = ExactFilteredOracle()
    global_index = GlobalAnnThenFilterIndex()
    oracle.build(records)
    global_index.build(records)
    truth = oracle.search(query)
    result = global_index.search(query)
    assert set(result.ids).issubset(set(truth.ids) | {record.id for record in records})
    assert all(records[rid].category == query.category for rid in result.ids)
