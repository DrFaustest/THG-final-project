# Temporal Subset ANN

Starter research prototype for filtered approximate nearest neighbor search with temporal and scalar predicates.

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
python -m tsann.experiments.run_grid
python -m tsann.experiments.summarize
```

For HNSW-backed ANN experiments instead of the brute-force fallback:

```powershell
python -m pip install -e ".[hnsw]"
```

## Output

Experiment CSV files are written under `results/csv/`. Each row is per query and includes latency, recall, chosen planner mode, candidate counts, partitions touched, and expansion rounds.
