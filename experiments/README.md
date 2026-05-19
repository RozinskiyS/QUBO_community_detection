# Experiments

Each script is an independent entry point that produces a CSV in
`../results/`. All runners use `concurrent.futures.ProcessPoolExecutor`
with the `spawn` start method; `--workers N` controls parallelism.

| Script | Purpose | Output | Wall time (8 workers) |
|---|---|---|---|
| `run_hp_sweep.py` | HP sweep on Karate, Polbooks, LFR-μ0.1: `lr × α_ortho × epochs × 5 shots` | `hp_sweep.csv` | ~15 min |
| `run_extended_shots.py` | Formula 5 (multi-class), 7 graphs × 2 configs × 20 shots | `extended_shots.csv` | ~30 min |
| `run_extended_k2.py` | Formula 6 (k=2), 7 binary graphs × 20 shots | `extended_k2.csv` | ~10 min |
| `run_hierarchical.py` | Hierarchical, 13 graphs × 3 criteria × 5 seeds | `hierarchical.csv` | ~3 hours |

All runners support `--quick` for a minimal smoke test (3–5 runs total).

## Typical usage

```bash
# Sanity check
python experiments/run_hp_sweep.py --quick

# Full sweep
python experiments/run_hp_sweep.py --workers 8

# Only specific graphs
python experiments/run_hierarchical.py --workers 8 \
    --graphs karate dolphins polbooks_binary
```

## Re-using cached graphs

The first runner that touches a graph downloads it (via `src/data_loaders.py`)
to `../data/raw/`, then pickles a pre-processed version to `../data/cache/`.
Subsequent runners reuse the cache. To force a rebuild, delete the relevant
`.pkl` file from `../data/cache/`.
