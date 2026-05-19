# Library code

Three modules. None depends on the others except through public APIs.

## `qubo_modularity.py`

Construction of QUBO matrices from Wang et al. 2024.

```python
from qubo_modularity import (
    gen_q_dict_modularity_2community,  # formula 6, single-bit, k=2
    gen_q_dict_modularity_multi,       # formula 5, one-hot, k>=2
)
```

Both return a `defaultdict(int)` mapping `(i, j) -> float` with
upper-triangular storage. The multi-class variant linearises the
`(vertex, community)` double index by `i = v * k + c`.

Run `python qubo_modularity.py` to execute three structural unit tests.

## `data_loaders.py`

13 graph loaders with a uniform interface:

```python
from data_loaders import ALL_LOADERS
G, true_labels, k_true = ALL_LOADERS['karate']()
```

Each loader downloads (once) to `../data/raw/`, normalises node IDs to
`0..n-1`, and returns `(networkx.Graph, dict[node -> int], k_true)`. The
loaders are pure functions and idempotent.

Datasets included: karate, dolphins, football, polbooks (multi & binary),
polblogs, email-Eu-core, three LFR variants, three SBM variants. See the
table in `../README.md` for sizes.

## `hierarchical_qignn.py`

The hierarchical algorithm — recursive binary splitting via the bipartition
QIGNN (formula 6).

```python
from hierarchical_qignn import evaluate_hierarchical
res = evaluate_hierarchical(
    G, true_labels,
    criterion='modularity_gain',   # or 'target_k' or 'min_size'
    n_shots=10, epochs=3000, seed=42,
)
# res = {'mod': float, 'nmi': float, 'k_found': int, 'time': float,
#        'n_decisions': int, 'history': list[dict]}
```

The `history` field carries one dict per accepted/rejected split, useful
for plotting the recursion tree.

Three stopping criteria:
- `modularity_gain`: accept a split iff global modularity strictly increases (Newman 2006).
- `target_k`: keep splitting (largest-cluster-first) until exactly `target_k` clusters exist.
- `min_size`: accept a split iff both sides have ≥ `min_size` vertices.
