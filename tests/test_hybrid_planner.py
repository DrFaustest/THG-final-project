from tsann.ann_hybrid import HybridPlannerIndex
import tsann.ann_partitioned as ann_partitioned
from tsann.ann_partitioned import PartitionFirstAnnIndex
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
    assert "planner_feature_tombstone_ratio" in result.metadata
    assert "planner_feature_open_ended_fraction" in result.metadata


def test_hybrid_planner_features_reflect_churn() -> None:
    records = generate_records(
        SyntheticConfig(n=100, d=8, num_clusters=4, seed=30, lifetime_min=10, lifetime_max=30, open_ended_fraction=0.5)
    )
    hybrid = HybridPlannerIndex(IndexConfig(global_rebuild_tombstone_ratio=1.0, partition_rebuild_tombstone_ratio=1.0))
    hybrid.build(records)
    for record in records[:20]:
        hybrid.delete(record.id)
    query = Query(records[-1].vector, 5, 0, 364, 0.0, 100.0)

    result = hybrid.search(query)

    assert result.metadata["planner_feature_tombstoned_records"] == 20
    assert result.metadata["planner_feature_tombstone_ratio"] > 0.0
    assert result.metadata["planner_feature_open_ended_fraction"] > 0.0


def test_subset_estimate_does_not_scan_records(monkeypatch) -> None:
    records = generate_records(SyntheticConfig(n=100, d=8, num_clusters=4, seed=4))
    query = Query(records[0].vector, 5, 0, 364, 0.0, 100.0)
    index = PartitionFirstAnnIndex()
    index.build(records)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("planner estimate must not call per-record filters")

    monkeypatch.setattr(ann_partitioned, "passes_filters", fail_if_called)
    estimate = index.estimate_subset(query)
    assert estimate.subset_size > 0
    assert estimate.estimator == "metadata"
