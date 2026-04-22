# Temporal Subset ANN

Starter research prototype for interval-valid temporal subset approximate nearest neighbor search with one primary scalar predicate and an optional category filter.

Current scope:

- interval-valid temporal subset ANN
- one primary numeric scalar predicate, currently `price`
- optional categorical filter
- exact, global post-filter, partition-first, rule-based hybrid planning, and an initial nearest-centroid learned planner
- logical delete/expiration with threshold-triggered index rebuilds
- interval-aware cell pruning from maintained temporal and scalar metadata

Not yet supported:

- robust learned planner selection beyond the initial nearest-centroid prototype
- generalized multi-scalar heterogeneous routing
- dual interval indexes over both `valid_from` and `valid_to`
- production-grade incremental HNSW maintenance

Implemented algorithms:

- `ExactFilteredOracle`: exact flat scan ground truth.
- `GlobalAnnThenFilterIndex`: global ANN candidates followed by exact filtering.
- `PartitionFirstAnnIndex`: time and price bucket routing, with exact fallback for small cells.
- `HybridPlannerIndex`: rule-based planner choosing exact, partitioned, or global search.

The HNSW dependency is optional at runtime. If `hnswlib` is unavailable, the wrapper falls back to brute-force candidate generation so the tests and harness still run.

Records use interval validity:

- `valid_from` is the first logical time where a record is eligible.
- `valid_to=None` means the record remains active indefinitely.
- A query window matches a record when the intervals intersect.
- `delete(record_id)` and `expire(before_time)` logically invalidate records before threshold-triggered index rebuilds.

## Quick Start

```powershell
python -m pip install -e .
python -m pytest
python -m tsann.experiments.run_single
python -m tsann.experiments.run_grid --config configs/smoke_grid.yaml
python -m tsann.experiments.summarize
python -m tsann.experiments.plot_results
python -m tsann.experiments.train_planner
python -m tsann.experiments.evaluate_planner
```

Or run the full smoke experiment suite:

```powershell
python main.py
```

For HNSW-backed ANN experiments instead of the brute-force fallback:

```powershell
python -m pip install -e ".[hnsw]"
```

## Output

Experiment CSV files are written under `results/csv/`. Each row is per query and includes latency, recall, exact subset size, subset estimate error, chosen planner mode, candidate counts, partitions touched, tombstone counts, deleted/expired counts, compaction counts, rebuild counts, expansion rounds, and a `best_mode` label for planner training. Run-level CSV and JSON summaries are written under `results/reports/`. Report figures are written under `results/figures/`.

`plot_results` generates:

- `latency_by_workload.png`
- `recall_by_workload.png`
- `subset_estimate_error.png`
- `planner_mode_counts.png`
- `maintenance_metrics.png`

`configs/smoke_grid.yaml` includes small workloads for static, append/delete, short-lived expiration, long-lived expiration, and mostly open-ended intervals. It is intentionally small enough for development runs. `configs/research_grid.example.yaml` shows a larger multi-seed, multi-dimension, multi-scale matrix for paper-oriented runs.

For held-out planner evaluation, reserve one or more seeds during training and evaluate only those seeds:

```powershell
python -m tsann.experiments.train_planner --exclude-seed 29
python -m tsann.experiments.evaluate_planner --include-seed 29
```
