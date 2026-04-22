from tsann.ann_hybrid import HybridPlannerIndex
from tsann.config import IndexConfig
from tsann.generators import SyntheticConfig, generate_records
from tsann.types import Query


def test_hybrid_uses_exact_for_tiny_subset() -> None:
    records = generate_records(SyntheticConfig(n=100, d=8, num_clusters=4, seed=3))
    query = Query(records[0].vector, 5, records[0].timestamp, records[0].timestamp, records[0].price, records[0].price)
    hybrid = HybridPlannerIndex(IndexConfig(planner_exact_threshold=500))
    hybrid.build(records)
    result = hybrid.search(query)
    assert result.metadata["planner_mode"] == "exact"
    assert result.metadata["algorithm"] == "hybrid"
