"""Formula-6 (k=2) runner on 7 binary benchmarks with 20 shots each
(10 for polblogs).

QUBO: H(x) = -sum_{v,w} B_{vw} x_v x_w  ==>  Q = -B (Wang et al. 2024 eq. 6).
Net: ResSAGE with number_classes=1, sigmoid output. Loss: p^T Q p + annealing.
No constraint, no regularization (formula 6 doesn't need any).
Bitstring: probs >= 0.5.

Streams output by graph: prepares cache, runs all shots for one graph in
parallel (up to 8 workers), prints summary, saves CSV incrementally. If a
graph fails entirely, it is logged and skipped.
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

CACHE_DIR = os.path.join(REPO_ROOT, 'data', 'cache')
RES_DIR = os.path.join(REPO_ROOT, 'results')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


GRAPH_ORDER = [
    'karate',
    'dolphins',
    'polbooks_binary',
    'SBM_n200_p0.3_q0.05',
    'SBM_n200_p0.2_q0.15',
    'SBM_n500_p0.2_q0.05',
    'polblogs',
]

EPOCHS = 3000
LR = 0.014
DIM_EMB = 10
HIDDEN = 50
DROPOUT = 0.5
PROB_THRESHOLD = 0.5
BASE_SEED = 42


def prepare_graph_cache(graph_names):
    """Cache (G, true_labels, B, m) for each binary benchmark in data/cache/."""
    import networkx as nx
    from data_loaders import (load_karate, load_dolphins, load_polblogs,
                              load_polbooks_binary, generate_sbm)
    loaders = {
        'karate':                load_karate,
        'dolphins':              load_dolphins,
        'polbooks_binary':       load_polbooks_binary,
        'polblogs':              load_polblogs,
        'SBM_n200_p0.3_q0.05':   lambda: generate_sbm(n=200, p_in=0.3, p_out=0.05),
        'SBM_n200_p0.2_q0.15':   lambda: generate_sbm(n=200, p_in=0.2, p_out=0.15),
        'SBM_n500_p0.2_q0.05':   lambda: generate_sbm(n=500, p_in=0.2, p_out=0.05),
    }
    for name in graph_names:
        # k2 cache lives under a separate filename so the multi-class cache from
        # the previous experiment isn't reused for binary loaders (e.g. karate's
        # k_true is 2 in both, but we still want a clean rebuild here).
        out = os.path.join(CACHE_DIR, f'{name}__k2.pkl')
        if os.path.exists(out):
            print(f'  cache hit: {name}')
            continue
        print(f'  building cache: {name}...')
        try:
            G, lbls, k_true = loaders[name]()
        except Exception as e:
            print(f'  !! failed to load {name}: {e}')
            continue
        nodes = list(G.nodes())
        A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
        deg = A.sum(axis=1)
        m = G.number_of_edges()
        B = A - np.outer(deg, deg) / (2.0 * m)
        Q = -B  # formula 6
        payload = {
            'G': G, 'true_labels': lbls, 'k_true': int(k_true),
            'nodes': nodes, 'B': B, 'Q': Q, 'm': m,
        }
        with open(out, 'wb') as f:
            pickle.dump(payload, f)
        print(f'    cached {name}: n={len(nodes)} m={m}')


# Worker side ------------------------------------------------------------------

def _heavy_imports():
    import torch
    import dgl
    import torch.nn as nn
    import torch.nn.functional as F
    from dgl.nn.pytorch import SAGEConv
    return torch, dgl, nn, F, SAGEConv


_GRAPH_CACHE = {}


def _load_cached_graph(name):
    if name in _GRAPH_CACHE:
        return _GRAPH_CACHE[name]
    with open(os.path.join(CACHE_DIR, f'{name}__k2.pkl'), 'rb') as f:
        payload = pickle.load(f)
    _GRAPH_CACHE[name] = payload
    return payload


def _build_net_k2(n_nodes, dim_emb, hidden, dropout, lr, dtype, device):
    """ResSAGE with sigmoid head (number_classes=1) — Wang formula 6 setup."""
    torch, dgl, nn, F, SAGEConv = _heavy_imports()
    from itertools import chain

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

    in_feats = dim_emb + 1 * 1 + 4 * dim_emb
    net = ResSAGE(in_feats, hidden, dropout, device).type(dtype).to(device)
    embed = nn.Embedding(n_nodes, dim_emb).type(dtype).to(device)
    optimizer = torch.optim.Adam(chain(net.parameters(), embed.parameters()),
                                  lr=lr)
    return net, embed, optimizer


def _loss_k2(probs, Q_mat, epoch):
    """Formula-6 loss: probs^T Q probs + (epoch/1e4) * sum |p(1-p)|.

    probs is a flat (n,) tensor in [0,1].
    """
    p_flat = probs.reshape(-1)
    qubo = p_flat @ Q_mat @ p_flat
    annealing = (epoch / 1e4) * (p_flat * (1 - p_flat)).abs().sum()
    return qubo + annealing


def run_one_shot(args):
    """Single (graph, shot_seed) trial. Worker entry point."""
    graph_name, shot_seed = args

    warnings.filterwarnings('ignore')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    import torch as _t
    _t.set_num_threads(1)

    t_start = time.time()
    try:
        torch, dgl, nn, F, SAGEConv = _heavy_imports()
        import random as _random
        import networkx as nx
        from sklearn.metrics import normalized_mutual_info_score
        from collections import defaultdict

        _random.seed(shot_seed); np.random.seed(shot_seed); torch.manual_seed(shot_seed)

        payload = _load_cached_graph(graph_name)
        G = payload['G']
        nodes = payload['nodes']
        true_labels = payload['true_labels']
        Q = payload['Q']
        n = len(nodes)

        device = torch.device('cpu')
        dtype = torch.float32
        Q_t = torch.tensor(Q, dtype=dtype, device=device)
        g_dgl = dgl.from_networkx(G).to(device)

        net, embed, opt = _build_net_k2(n, dim_emb=DIM_EMB, hidden=HIDDEN,
                                         dropout=DROPOUT, lr=LR, dtype=dtype,
                                         device=device)

        # Edge weight = symmetric off-diagonal Q on the actual edges.
        edge_weight_full = (Q_t - torch.diag(torch.diag(Q_t))) / 2
        edge_weight_full = edge_weight_full + edge_weight_full.T
        src, dst = g_dgl.edges()
        edge_weight = edge_weight_full[src, dst]

        inputs = torch.rand((n, DIM_EMB), dtype=dtype, device=device)
        pr = nx.pagerank(nx.Graph(G))
        walk = torch.zeros((n, 2 * DIM_EMB), dtype=dtype, device=device)
        for v, val in pr.items():
            walk[v, :] = val
        inputs = torch.cat([inputs, torch.ones_like(inputs),
                            torch.ones_like(inputs), walk], 1)

        h0 = torch.zeros(n, 1, device=device, dtype=dtype)

        best_bitstring = torch.zeros(n, dtype=torch.long, device=device)
        best_loss = _loss_k2(best_bitstring.float(), Q_t, 0).item()
        best_sums = -best_loss
        best_epoch = 0

        prev_loss = 1.0
        bad_count = 0
        patience = 1000
        tol = 1e-4

        for epoch in range(EPOCHS):
            probs, h0 = net(g_dgl, inputs, h0.detach(), edge_weight)
            probs_flat = probs.squeeze(-1)
            loss = _loss_k2(probs_flat, Q_t, epoch)
            lv = loss.detach().item()

            with torch.no_grad():
                bitstring = (probs_flat.detach() >= PROB_THRESHOLD).long()
                if lv < best_loss:
                    sums = -_loss_k2(bitstring.float(), Q_t, 0).item()
                    if sums > best_sums:
                        best_loss = max(lv, -sums)
                        best_bitstring = bitstring
                        best_sums = sums
                        best_epoch = epoch

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

        bits = best_bitstring.detach().cpu().numpy().astype(int)
        groups = defaultdict(set)
        for v, lbl in zip(nodes, bits):
            groups[int(lbl)].add(v)
        comms = [frozenset(s) for s in groups.values() if len(s) > 0]
        mod = 0.0 if len(comms) <= 1 else float(nx.community.modularity(G, comms))
        truth = np.array([true_labels[v] for v in nodes])
        nmi = float(normalized_mutual_info_score(truth, bits))
        # collapse: one community holding >= 80% of nodes (unsplit prediction)
        counts = np.bincount(bits, minlength=2)
        collapsed = bool(counts.max() / n >= 0.8)
        used_k = len(comms)
        elapsed = time.time() - t_start

        return {
            'graph': graph_name, 'shot_seed': shot_seed,
            'mod': mod, 'nmi': nmi, 'used_k': used_k,
            'collapse': collapsed, 'best_epoch': best_epoch,
            'time': elapsed, 'error': '',
        }
    except Exception as e:
        return {
            'graph': graph_name, 'shot_seed': shot_seed,
            'mod': float('nan'), 'nmi': float('nan'), 'used_k': 0,
            'collapse': False, 'best_epoch': -1,
            'time': time.time() - t_start,
            'error': f'{type(e).__name__}: {e}\n{traceback.format_exc()[:400]}',
        }


def run_graph(graph_name, n_shots, max_workers=8):
    jobs = [(graph_name, BASE_SEED + i) for i in range(n_shots)]
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    results = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = [pool.submit(run_one_shot, job) for job in jobs]
        for fut in as_completed(futures):
            results.append(fut.result())
    return pd.DataFrame(results)


def compute_louvain_for_graph(graph_name):
    """Run Louvain on the cached graph; returns (mod, nmi, k_found, time)."""
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score
    payload = _load_cached_graph(graph_name)
    G = payload['G']; nodes = payload['nodes']
    true_labels = payload['true_labels']
    t0 = time.time()
    comms = nx.community.louvain_communities(G, seed=BASE_SEED)
    elapsed = time.time() - t0
    mod = float(nx.community.modularity(G, comms))
    node_to_c = {v: i for i, c in enumerate(comms) for v in c}
    pred = [node_to_c[v] for v in nodes]
    truth = [true_labels[v] for v in nodes]
    nmi = float(normalized_mutual_info_score(truth, pred))
    return {'method': 'Louvain', 'mod': mod, 'nmi': nmi,
            'k_found': len(comms), 'time': elapsed}


def print_graph_summary(graph_name, df_graph, louv, elapsed, remaining):
    payload = _load_cached_graph(graph_name)
    n = len(payload['nodes']); k_true = int(payload['k_true'])

    bar = '=' * 80
    print()
    print(bar)
    print(f"Graph: {graph_name} (n={n}, k_true={k_true})")
    mins = int(elapsed // 60); secs = int(elapsed % 60)
    print(f"Elapsed: {mins} min {secs} sec | Remaining: {remaining} graphs")
    print('-' * 80)
    print(f"{'Method':<18} {'mod_best':>9} {'mod_mean±std':>16} "
          f"{'nmi_best':>9} {'stable_rate':>12}")

    if louv is not None:
        print(f"{'Louvain (ref)':<18} {louv['mod']:>9.4f} {'-':>16} "
              f"{louv['nmi']:>9.4f} {'-':>12}")

    sub = df_graph[df_graph['mod'].notna()]
    if len(sub) == 0:
        print('QIGNN k=2          (no successful runs)')
    else:
        mods = sub['mod'].values
        # stable_rate (strict): fraction of shots with mod >= 0.8 * mod_best
        if mods.max() > 0:
            stable = float((mods >= 0.8 * mods.max()).mean())
        else:
            stable = float((mods == mods.max()).mean())
        print(f"{'QIGNN k=2':<18} "
              f"{mods.max():>9.4f} "
              f"{mods.mean():>+7.4f}±{mods.std():.4f} "
              f"{sub['nmi'].max():>9.4f} "
              f"{stable:>12.2f}")

    err_count = int(df_graph['error'].astype(bool).sum())
    if err_count:
        print(f"!! {err_count} runs errored")
    print(bar)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', default=os.path.join(RES_DIR, 'extended_k2.csv'))
    parser.add_argument('--louvain-out',
                        default=os.path.join(RES_DIR, 'extended_k2_louvain.csv'))
    parser.add_argument('--graphs', nargs='+', default=GRAPH_ORDER)
    parser.add_argument('--shots', type=int, default=20,
                        help='shots for non-polblogs graphs (polblogs=10)')
    args = parser.parse_args()

    print(f'Workers: {args.workers}, out: {args.out}, '
          f'graphs: {args.graphs}')
    print('Preparing graph cache...')
    t0 = time.time()
    prepare_graph_cache(args.graphs)
    print(f'  cache ready in {time.time()-t0:.1f}s')

    all_results = []
    louvain_rows = []
    start = time.time()
    for i, graph_name in enumerate(args.graphs):
        n_shots = 10 if graph_name == 'polblogs' else args.shots
        cache_path = os.path.join(CACHE_DIR, f'{graph_name}__k2.pkl')
        if not os.path.exists(cache_path):
            print(f"\n>>> SKIPPING {graph_name}: cache failed to build")
            continue

        # Louvain reference (cheap; computed in main process)
        louv = compute_louvain_for_graph(graph_name)
        louv_row = dict(louv); louv_row['graph'] = graph_name
        louvain_rows.append(louv_row)
        pd.DataFrame(louvain_rows).to_csv(args.louvain_out, index=False)

        print(f"\n>>> Starting graph {i+1}/{len(args.graphs)}: "
              f"{graph_name} ({n_shots} shots)...")
        sys.stdout.flush()

        try:
            df_graph = run_graph(graph_name, n_shots, max_workers=args.workers)
        except Exception as e:
            print(f'  !! graph {graph_name} crashed: {e}')
            continue

        all_results.append(df_graph)
        pd.concat(all_results, ignore_index=True).to_csv(args.out, index=False)

        print_graph_summary(graph_name, df_graph, louv,
                            elapsed=time.time() - start,
                            remaining=len(args.graphs) - i - 1)

    print(f"\n>>> All graphs done. Final results: {args.out}")
    print(f"Total wall time: {(time.time()-start)/60:.1f} min")


if __name__ == '__main__':
    main()
