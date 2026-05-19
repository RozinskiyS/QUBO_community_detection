"""Hierarchical QIGNN — recursive binary splitting via formula-6 (k=2).

Top-level algorithm:
  1. Apply k=2 QIGNN to the whole graph -> {A, B}.
  2. For each subgraph, apply a stopping criterion. If the criterion accepts,
     recursively split. Otherwise treat as a leaf community.
  3. Final partition = leaves of the binary recursion tree.

Three split criteria (chosen at top level):
  modularity_gain  — Newman 2006: split iff *global* modularity strictly grows.
  target_k         — keep splitting (largest-cluster-first) until exactly
                     target_k clusters exist.
  min_size         — split iff both sides would have >= min_size nodes.

split_once(...) trains the formula-6 ResSAGE+sigmoid network n_shots times on
the given subgraph and returns the bitstring with the best *local subgraph*
modularity. Locally is correct because the GNN has no view of the rest of G;
the global stopping rule is enforced by hierarchical_split.
"""
from __future__ import annotations
import os
import time
import warnings
import heapq
from collections import defaultdict
from itertools import chain

import numpy as np
import networkx as nx


# Suppress noisy warnings inside workers.
warnings.filterwarnings('ignore')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


# Lazy heavy-import wrapper so that this module imports cheaply on the main
# process; workers will materialize the imports inside split_once.
def _heavy_imports():
    import torch
    import dgl
    import torch.nn as nn
    import torch.nn.functional as F
    from dgl.nn.pytorch import SAGEConv
    return torch, dgl, nn, F, SAGEConv


# -----------------------------------------------------------------------------
def _build_net_k2(n_nodes, dim_emb, hidden, dropout, lr, dtype, device):
    """ResSAGE with single sigmoid output — formula 6 setup."""
    torch, dgl, nn, F, SAGEConv = _heavy_imports()

    class SAGEResBlock(nn.Module):
        def __init__(self, ic, oc, fd=0.):
            super().__init__()
            self.sage1 = SAGEConv(ic, oc, aggregator_type='mean',
                                   feat_drop=fd, bias=False)
            self.bn1 = nn.BatchNorm1d(oc)
            self.sage2 = SAGEConv(ic, oc, aggregator_type='pool',
                                   feat_drop=fd, bias=False)
            self.bn2 = nn.BatchNorm1d(oc)
            self.relu = nn.LeakyReLU()

        def forward(self, g, x, ew=None):
            return self.relu(self.bn1(self.sage1(g, x, ew))
                             + self.bn2(self.sage2(g, x, ew)))

    class ResSAGE(nn.Module):
        def __init__(self, in_f, hd, dr, dev):
            super().__init__()
            self.dr = dr
            self.layers = nn.ModuleList()
            cur = in_f
            for h in [hd] if isinstance(hd, int) else hd:
                self.layers.append(SAGEResBlock(cur, h).to(dev))
                self.layers.append(nn.LeakyReLU())
                cur = h
            self.layers.append(SAGEConv(cur, 1, aggregator_type='mean').to(dev))

        def forward(self, g, h, h0, ew=None):
            h = torch.cat([h, h0], 1)
            for layer, norm in zip(self.layers[:-1][::2],
                                    self.layers[:-1][1::2]):
                h = norm(layer(g, h, ew))
            h = F.dropout(h, p=self.dr)
            h0n = self.layers[-1](g, h, ew)
            return torch.sigmoid(h0n), h0n

    in_feats = dim_emb + 1 + 4 * dim_emb
    net = ResSAGE(in_feats, hidden, dropout, device).type(dtype).to(device)
    embed = nn.Embedding(n_nodes, dim_emb).type(dtype).to(device)
    optimizer = torch.optim.Adam(chain(net.parameters(), embed.parameters()),
                                  lr=lr)
    return net, embed, optimizer


def _loss_k2(probs, Q_mat, epoch):
    p_flat = probs.reshape(-1)
    qubo = p_flat @ Q_mat @ p_flat
    annealing = (epoch / 1e4) * (p_flat * (1 - p_flat)).abs().sum()
    return qubo + annealing


def _train_one_shot_k2(G_sub, shot_seed, epochs, lr=0.014,
                        dim_emb=10, hidden=50, dropout=0.5,
                        prob_threshold=0.5):
    """Train formula-6 QIGNN once on G_sub. Returns (bitstring_array, mod_local)."""
    torch, dgl, nn, F, SAGEConv = _heavy_imports()
    import random as _random
    torch.set_num_threads(1)

    _random.seed(shot_seed)
    np.random.seed(shot_seed)
    torch.manual_seed(shot_seed)

    nodes = list(G_sub.nodes())
    n = len(nodes)

    # Compute modularity matrix B and Q = -B (formula 6).
    A = nx.to_numpy_array(G_sub, nodelist=nodes, weight=None)
    deg = A.sum(axis=1)
    m = G_sub.number_of_edges()
    if m == 0:
        # No edges — trivial all-same-cluster solution.
        return np.zeros(n, dtype=int), 0.0
    B = A - np.outer(deg, deg) / (2.0 * m)
    Q = -B

    device = torch.device('cpu')
    dtype = torch.float32
    Q_t = torch.tensor(Q, dtype=dtype, device=device)

    # Build a simple-int-labeled subgraph for DGL (DGL needs 0..n-1 ids).
    G_int = nx.convert_node_labels_to_integers(G_sub)
    g_dgl = dgl.from_networkx(G_int).to(device)

    net, embed, opt = _build_net_k2(n, dim_emb=dim_emb, hidden=hidden,
                                     dropout=dropout, lr=lr, dtype=dtype,
                                     device=device)

    edge_weight_full = (Q_t - torch.diag(torch.diag(Q_t))) / 2
    edge_weight_full = edge_weight_full + edge_weight_full.T
    src, dst = g_dgl.edges()
    edge_weight = edge_weight_full[src, dst]

    inputs = torch.rand((n, dim_emb), dtype=dtype, device=device)
    pr = nx.pagerank(nx.Graph(G_int))
    walk = torch.zeros((n, 2 * dim_emb), dtype=dtype, device=device)
    for v, val in pr.items():
        walk[v, :] = val
    inputs = torch.cat([inputs, torch.ones_like(inputs),
                        torch.ones_like(inputs), walk], 1)

    h0 = torch.zeros(n, 1, device=device, dtype=dtype)

    best_bits = np.zeros(n, dtype=int)
    best_neg_loss = -_loss_k2(torch.zeros(n, dtype=dtype, device=device), Q_t, 0).item()
    prev_loss = 1.0
    bad_count = 0
    patience = max(200, epochs // 4)
    tol = 1e-4

    for epoch in range(epochs):
        probs, h0 = net(g_dgl, inputs, h0.detach(), edge_weight)
        probs_flat = probs.squeeze(-1)
        loss = _loss_k2(probs_flat, Q_t, epoch)
        lv = loss.detach().item()

        with torch.no_grad():
            bits = (probs_flat.detach() >= prob_threshold).long()
            # Score by NEGATIVE local QUBO loss (continuous-form for direct
            # comparison with best_neg_loss thresholding).
            neg_loss = -_loss_k2(bits.float(), Q_t, 0).item()
            if neg_loss > best_neg_loss:
                best_neg_loss = neg_loss
                best_bits = bits.detach().cpu().numpy().astype(int)

        if (abs(lv - prev_loss) <= tol) or ((lv - prev_loss) > 0):
            bad_count += 1
        else:
            bad_count = 0
        if bad_count >= patience:
            break
        prev_loss = lv

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        opt.step()

    # Local subgraph modularity (with the integer-relabeled graph; equivalent).
    groups = defaultdict(set)
    for i, lbl in enumerate(best_bits):
        groups[int(lbl)].add(i)
    comms = [frozenset(s) for s in groups.values() if len(s) > 0]
    mod_local = 0.0 if len(comms) <= 1 else float(nx.community.modularity(G_int, comms))
    return best_bits, mod_local


# -----------------------------------------------------------------------------
def split_once(G_sub, n_shots=10, epochs=3000, base_seed=42):
    """Split G_sub into 2 communities using formula-6 QIGNN.

    Picks the bitstring with best LOCAL subgraph modularity over n_shots.

    Parameters
    ----------
    G_sub : networkx.Graph
        Subgraph to split. Node IDs are arbitrary (will be preserved in the
        returned label dict).
    n_shots : int
    epochs : int
        Per-shot training budget.
    base_seed : int
        First shot uses base_seed, then base_seed+1, ...

    Returns
    -------
    labels : dict {node_id (original): 0 or 1}
    mod_local_after : float
        Modularity of the (A, B) split on G_sub itself.
    elapsed : float
    """
    t0 = time.time()
    nodes = list(G_sub.nodes())
    n = len(nodes)

    if n <= 1:
        return {v: 0 for v in nodes}, 0.0, time.time() - t0
    if n == 2:
        # Trivially two singletons.
        labels = {nodes[0]: 0, nodes[1]: 1}
        try:
            mod = float(nx.community.modularity(G_sub,
                       [{nodes[0]}, {nodes[1]}]))
        except Exception:
            mod = 0.0
        return labels, mod, time.time() - t0

    # Adaptive epoch budget: tiny subgraphs need fewer epochs.
    if n < 30:
        epochs_eff = min(epochs, 1000)
    elif n < 100:
        epochs_eff = min(epochs, 2000)
    else:
        epochs_eff = epochs

    best_bits = None
    best_mod = -1e9
    for shot in range(n_shots):
        bits, mod = _train_one_shot_k2(G_sub, base_seed + shot, epochs_eff)
        if mod > best_mod:
            best_mod = mod
            best_bits = bits
    # best_bits is an int array indexed 0..n-1 (in node-iteration order of G_sub)
    labels = {nodes[i]: int(best_bits[i]) for i in range(n)}
    return labels, best_mod, time.time() - t0


# -----------------------------------------------------------------------------
def _communities_from_partition(partition):
    """Convert {node: cluster_id} -> list[set[node]] (one set per cluster)."""
    groups = defaultdict(set)
    for v, cid in partition.items():
        groups[cid].add(v)
    return [frozenset(s) for s in groups.values() if len(s) > 0]


def hierarchical_split(G, criterion='modularity_gain',
                        min_size=5, target_k=None,
                        max_depth=10, n_shots=10, epochs=3000,
                        verbose=False, seed=42):
    """Recursively split G via formula-6 binary QIGNN.

    Returns (final_partition, history) where:
        final_partition : dict {node_id: 0..k-1}
        history         : list of dicts (one per accepted/rejected split)
    """
    nodes_all = list(G.nodes())
    partition = {v: 0 for v in nodes_all}
    next_id = 1

    # Queue items: (priority_or_filler, nodes, depth, current_cid)
    if criterion == 'target_k':
        if target_k is None or target_k < 1:
            raise ValueError('target_k criterion requires target_k>=1')
        queue = []
        heapq.heappush(queue,
                       (-len(nodes_all), 0, frozenset(nodes_all), 0, 0))
        # tie-breaker counter
        tcount = 1
    else:
        queue = [(frozenset(nodes_all), 0, 0)]

    history = []
    skipped_clusters = []  # frozenset of nodes that we decided not to split

    while queue:
        if criterion == 'target_k':
            neg_size, _, nodes, depth, cur_cid = heapq.heappop(queue)
            cur_size = -neg_size
        else:
            nodes, depth, cur_cid = queue.pop(0)
            cur_size = len(nodes)

        # ---- Trivial-stop checks (apply to all criteria) ----
        if cur_size <= 1:
            skipped_clusters.append(nodes)
            history.append(dict(depth=depth, n=cur_size, decision='trivial-leaf',
                                 reason='size<=1', mod_before=None,
                                 mod_after=None, time=0.0))
            continue
        if depth >= max_depth:
            skipped_clusters.append(nodes)
            history.append(dict(depth=depth, n=cur_size, decision='depth-cap',
                                 reason=f'depth>={max_depth}',
                                 mod_before=None, mod_after=None, time=0.0))
            continue

        # ---- Per-criterion fast checks before training ----
        if criterion == 'min_size' and cur_size < 2 * min_size:
            skipped_clusters.append(nodes)
            history.append(dict(depth=depth, n=cur_size, decision='no-split',
                                 reason=f'size<{2*min_size} (min_size={min_size})',
                                 mod_before=None, mod_after=None, time=0.0))
            continue

        if criterion == 'target_k' and next_id >= target_k:
            # We already have target_k clusters in `partition`; current one and
            # everything in queue stays as-is.
            skipped_clusters.append(nodes)
            for item in queue:
                if criterion == 'target_k':
                    skipped_clusters.append(item[2])
                else:
                    skipped_clusters.append(item[0])
            queue.clear()
            history.append(dict(depth=depth, n=cur_size, decision='target-k-reached',
                                 reason=f'next_id>={target_k}',
                                 mod_before=None, mod_after=None, time=0.0))
            continue

        # ---- Run one split ----
        G_sub = G.subgraph(nodes).copy()
        labels, mod_local, t_split = split_once(G_sub, n_shots=n_shots,
                                                 epochs=epochs,
                                                 base_seed=seed + 1000 * depth + cur_cid)
        side_a = frozenset(v for v, lbl in labels.items() if lbl == 0)
        side_b = frozenset(v for v, lbl in labels.items() if lbl == 1)

        # GNN may collapse to single side -> abandon split.
        if len(side_a) == 0 or len(side_b) == 0:
            skipped_clusters.append(nodes)
            history.append(dict(depth=depth, n=cur_size, decision='collapsed',
                                 reason='one-sided GNN output',
                                 mod_before=None, mod_after=None, time=t_split))
            continue

        # ---- Apply criterion-specific post-split decision ----
        if criterion == 'min_size':
            if len(side_a) < min_size or len(side_b) < min_size:
                skipped_clusters.append(nodes)
                history.append(dict(depth=depth, n=cur_size, decision='no-split',
                                     reason=f'side sizes {len(side_a)}/{len(side_b)} < {min_size}',
                                     mod_before=None, mod_after=None,
                                     time=t_split))
                continue

        elif criterion == 'modularity_gain':
            comms_before = _communities_from_partition(partition)
            mod_before = float(nx.community.modularity(G, comms_before))
            new_partition = dict(partition)
            for v in side_b:
                new_partition[v] = next_id
            comms_after = _communities_from_partition(new_partition)
            mod_after = float(nx.community.modularity(G, comms_after))
            if mod_after <= mod_before + 1e-9:
                skipped_clusters.append(nodes)
                history.append(dict(depth=depth, n=cur_size, decision='no-split',
                                     reason=f'mod_gain {mod_after-mod_before:+.4f}<=0',
                                     mod_before=mod_before, mod_after=mod_after,
                                     time=t_split))
                continue
            history.append(dict(depth=depth, n=cur_size, decision='split',
                                 reason=f'mod_gain {mod_after-mod_before:+.4f}>0',
                                 mod_before=mod_before, mod_after=mod_after,
                                 time=t_split))

        elif criterion == 'target_k':
            history.append(dict(depth=depth, n=cur_size, decision='split',
                                 reason='target-k progressing',
                                 mod_before=None, mod_after=None, time=t_split))
        else:
            raise ValueError(f'unknown criterion {criterion}')

        # ---- Commit the split ----
        for v in side_b:
            partition[v] = next_id
        new_cid = next_id
        next_id += 1

        if criterion == 'target_k':
            heapq.heappush(queue, (-len(side_a), tcount, side_a, depth + 1, cur_cid))
            tcount += 1
            heapq.heappush(queue, (-len(side_b), tcount, side_b, depth + 1, new_cid))
            tcount += 1
        else:
            queue.append((side_a, depth + 1, cur_cid))
            queue.append((side_b, depth + 1, new_cid))

    # ---- Renumber 0..k-1 ----
    unique_ids = sorted(set(partition.values()))
    remap = {old: new for new, old in enumerate(unique_ids)}
    final_partition = {v: remap[partition[v]] for v in partition}

    if verbose:
        print(f'  hierarchical_split({criterion}): {len(unique_ids)} clusters '
              f'after {len(history)} decisions')

    return final_partition, history


# -----------------------------------------------------------------------------
def evaluate_hierarchical(G, true_labels, criterion='modularity_gain',
                           min_size=5, target_k=None, max_depth=10,
                           n_shots=10, epochs=3000, seed=42):
    """Run hierarchical_split and report headline metrics."""
    from sklearn.metrics import normalized_mutual_info_score
    nodes = list(G.nodes())
    t0 = time.time()
    final_partition, history = hierarchical_split(
        G, criterion=criterion, min_size=min_size,
        target_k=target_k, max_depth=max_depth,
        n_shots=n_shots, epochs=epochs, seed=seed)
    elapsed = time.time() - t0

    # Modularity of final partition
    comms = _communities_from_partition(final_partition)
    mod = 0.0 if len(comms) <= 1 else float(nx.community.modularity(G, comms))
    truth = np.array([true_labels[v] for v in nodes])
    pred = np.array([final_partition[v] for v in nodes])
    nmi = float(normalized_mutual_info_score(truth, pred))
    n_used = len(comms)

    return {
        'mod': mod, 'nmi': nmi, 'k_found': n_used,
        'time': elapsed, 'n_decisions': len(history),
        'history': history,
    }


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Smoke test on Karate.
    print('Smoke test: Karate club, criterion=modularity_gain')
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
    true = {i: 0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes()}
    res = evaluate_hierarchical(G, true, criterion='modularity_gain',
                                  n_shots=3, epochs=1000, seed=42)
    print(f'  mod={res["mod"]:.4f} nmi={res["nmi"]:.4f} k_found={res["k_found"]} '
          f'time={res["time"]:.1f}s decisions={res["n_decisions"]}')
    for h in res['history']:
        print(f'    depth={h["depth"]} n={h["n"]:<4} decision={h["decision"]:<14} '
              f'reason={h["reason"]}')
