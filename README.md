# Temporal Subset ANN

Starter research prototype for filtered approximate nearest neighbor search with temporal and scalar predicates.

Implemented algorithms:

- `ExactFilteredOracle`: exact flat scan ground truth.
- `GlobalAnnThenFilterIndex`: global ANN candidates followed by exact filtering.
- `PartitionFirstAnnIndex`: time and price bucket routing, with exact fallback for small cells.
- `HybridPlannerIndex`: rule-based planner choosing exact, partitioned, or global search.

The HNSW dependency is optional at runtime. If `hnswlib` is unavailable, the wrapper falls back to brute-force candidate generation so the tests and harness still run.

## Quick Start

```powershell
python -m pip install -e .
python -m pytest
python -m tsann.experiments.run_single
python -m tsann.experiments.run_grid
python -m tsann.experiments.summarize
```

## Output

Experiment CSV files are written under `results/csv/`. Each row is per query and includes latency, recall, chosen planner mode, candidate counts, partitions touched, and expansion rounds.
