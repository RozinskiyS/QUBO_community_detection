"""Hyperparameter sweep for QIGNN multi-class on 3 graphs.

Sweep grid (24 configurations per graph after the strategic cuts):
  lr     : [0.001, 0.005, 0.014]      (dropped 0.05 — riskiest)
  alpha  : [0.0, 0.1, 0.5, 1.0]       (dropped 0.05 — too close to 0.1)
  epochs : [3000, 10000]

3 graphs × 24 configs × 5 shots = 360 runs.
Parallelism: ProcessPoolExecutor with `spawn` start method, max 4 workers.

Usage:
    python3.12 sweep_runner.py --quick   # 1 config × 3 graphs sanity check
    python3.12 sweep_runner.py           # full sweep
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

# Defer heavy imports for `--quick`/main flow; workers will import them.
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

CACHE_DIR = os.path.join(REPO_ROOT, 'data', 'cache')
RES_DIR = os.path.join(REPO_ROOT, 'results')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


GRAPH_SPECS = {
    'karate':         {'k_test': 5},
    'polbooks':       {'k_test': 3},
    'lfr_n200_mu0.1': {'k_test': 5},
}


# ---------------------------------------------------------------------------
def prepare_graph_cache():
    """Run once in main process: load all 3 graphs, store as pickle so workers
    can read fast without regenerating LFR or re-parsing GML."""
    import networkx as nx
    from data_loaders import load_karate, load_polbooks, generate_lfr

    loaders = {
        'karate':         load_karate,
        'polbooks':       load_polbooks,
        'lfr_n200_mu0.1': lambda: generate_lfr(n=200, mu=0.1),
    }
    for name, fn in loaders.items():
        out = os.path.join(CACHE_DIR, f'{name}.pkl')
        if os.path.exists(out):
            print(f'  cache hit: {name}')
            continue
        G, lbls, k_true = fn()
        # Pre-compute the modularity matrix once.
        nodes = list(G.nodes())
        A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
        deg = A.sum(axis=1)
        m = G.number_of_edges()
        B = A - np.outer(deg, deg) / (2.0 * m)
        payload = {
            'G': G, 'true_labels': lbls, 'k_true': k_true,
            'nodes': nodes, 'B': B, 'm': m,
            'P_pen': 1.5 * float(np.max(np.abs(B))),
        }
        with open(out, 'wb') as f:
            pickle.dump(payload, f)
        print(f'  cached: {name}  n={len(nodes)} m={m} k_true={k_true}')


# ---------------------------------------------------------------------------
# Worker side -- everything below is what each worker runs.
# Module-level imports happen once per worker (after `spawn` re-imports).
# ---------------------------------------------------------------------------

def _heavy_imports():
    """Lazily import torch+dgl on first call inside a worker."""
    import torch  # noqa: F401
    import dgl    # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401
    from dgl.nn.pytorch import SAGEConv  # noqa: F401
    return torch, dgl, nn, F, SAGEConv


_GRAPH_CACHE = {}     # local-to-worker memoisation


def _load_cached_graph(name):
    if name in _GRAPH_CACHE:
        return _GRAPH_CACHE[name]
    with open(os.path.join(CACHE_DIR, f'{name}.pkl'), 'rb') as f:
        payload = pickle.load(f)
    _GRAPH_CACHE[name] = payload
    return payload


def _build_net(n_nodes, k, dim_emb, hidden, dropout, lr, dtype, device):
    torch, dgl, nn, F, SAGEConv = _heavy_imports()
    from itertools import chain

    class SAGEResBlockMulti(nn.Module):
        def __init__(self, ic, oc, fd=0.):
            super().__init__()
            self.sage1 = SAGEConv(ic, oc, aggregator_type='mean', feat_drop=fd, bias=False)
            self.bn1 = nn.BatchNorm1d(oc)
            self.sage2 = SAGEConv(ic, oc, aggregator_type='pool', feat_drop=fd, bias=False)
            self.bn2 = nn.BatchNorm1d(oc)
            self.relu = nn.LeakyReLU()

        def forward(self, g, x, ew=None):
            return self.relu(self.bn1(self.sage1(g, x, ew)) + self.bn2(self.sage2(g, x, ew)))

    class ResSAGEMulti(nn.Module):
        def __init__(self, in_f, hd, kk, dr, dev):
            super().__init__()
            self.dr = dr
            self.layers = nn.ModuleList()
            cur = in_f
            for h in [hd] if isinstance(hd, int) else hd:
                self.layers.append(SAGEResBlockMulti(cur, h).to(dev))
                self.layers.append(nn.LeakyReLU())
                cur = h
            self.layers.append(SAGEConv(cur, kk, aggregator_type='mean').to(dev))

        def forward(self, g, h, h0, ew=None):
            h = torch.cat([h, h0], 1)
            for layer, norm in zip(self.layers[:-1][::2], self.layers[:-1][1::2]):
                h = norm(layer(g, h, ew))
            h = F.dropout(h, p=self.dr)
            h0n = self.layers[-1](g, h, ew)
            return F.softmax(h0n, dim=1), h0n

    in_feats = dim_emb + 1 * k + 4 * dim_emb
    net = ResSAGEMulti(in_feats, hidden, k, dropout, device).type(dtype).to(device)
    embed = nn.Embedding(n_nodes, dim_emb).type(dtype).to(device)
    optimizer = torch.optim.Adam(chain(net.parameters(), embed.parameters()), lr=lr)
    return net, embed, optimizer


def _structured_loss(P, B_t, P_pen, m_edges, alpha_ortho, epoch, n, k):
    torch, *_ = _heavy_imports()
    B_diag = torch.diag(B_t)
    B_off = B_t - torch.diag(B_diag)
    diag_per_v = -B_diag / (2.0 * m_edges) - P_pen
    diag_term = (diag_per_v * (P ** 2).sum(dim=1)).sum()
    trace_off = (P.T @ B_off @ P).diagonal().sum()
    mod_off = -trace_off / (2.0 * m_edges)
    rs = P.sum(dim=1); rsq = (P ** 2).sum(dim=1)
    constr = P_pen * (rs ** 2 - rsq).sum()
    qubo = diag_term + mod_off + constr

    p = P.reshape(-1)
    annealing = (epoch / 1e4) * (p * (1 - p)).abs().sum()

    if alpha_ortho > 0:
        PtP = P.T @ P
        PtP_n = PtP / (torch.norm(PtP, p='fro') + 1e-10)
        I_n = torch.eye(k, device=P.device, dtype=P.dtype) / np.sqrt(k)
        ortho = torch.norm(PtP_n - I_n, p='fro') ** 2
    else:
        ortho = torch.tensor(0.0, device=P.device, dtype=P.dtype)

    return qubo + annealing + alpha_ortho * ortho


def run_one_config(args):
    """Worker entry point.

    args = (graph_name, k, lr, alpha_ortho, epochs, shot_seed)
    Returns a dict of metrics.
    """
    warnings.filterwarnings('ignore')
    # Each worker is a separate process: pin to 1 OMP thread so that
    # 8 workers × 1 thread fits comfortably in 14 cores (M4 Pro) without
    # OMP/MKL contention.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    import torch as _t
    _t.set_num_threads(1)
    graph_name, k, lr, alpha_ortho, epochs, shot_seed = args
    t_start = time.time()
    try:
        torch, dgl, nn, F, SAGEConv = _heavy_imports()
        import random as _random
        import networkx as nx
        from sklearn.metrics import normalized_mutual_info_score

        _random.seed(shot_seed); np.random.seed(shot_seed); torch.manual_seed(shot_seed)

        payload = _load_cached_graph(graph_name)
        G = payload['G']
        nodes = payload['nodes']
        true_labels = payload['true_labels']
        B = payload['B']
        m_edges = payload['m']
        P_pen = payload['P_pen']
        n = len(nodes)

        device = torch.device('cpu')
        dtype = torch.float32
        B_t = torch.tensor(B, dtype=dtype, device=device)
        g_dgl = dgl.from_networkx(G).to(device)

        net, embed, opt = _build_net(n, k, dim_emb=10, hidden=50, dropout=0.5,
                                     lr=lr, dtype=dtype, device=device)

        # Edge weights from -B over actual graph edges.
        src, dst = g_dgl.edges()
        edge_weight = (-B_t)[src, dst]

        inputs = torch.rand((n, 10), dtype=dtype, device=device)
        pr = nx.pagerank(nx.Graph(G))
        walk = torch.zeros((n, 20), dtype=dtype, device=device)
        for v, val in pr.items():
            walk[v, :] = val
        inputs = torch.cat([inputs, torch.ones_like(inputs), torch.ones_like(inputs), walk], 1)

        h0 = torch.zeros(n, k, device=device, dtype=dtype)

        # initial best is "everyone in community 0"
        X0 = torch.zeros(n, k, dtype=dtype, device=device); X0[:, 0] = 1.0
        with torch.no_grad():
            best_proj_loss = _structured_loss(X0, B_t, P_pen, m_edges,
                                              alpha_ortho, 0, n, k).item()
        best_assignment = np.zeros(n, dtype=int)
        best_epoch = 0

        prev_loss = 1.0
        bad_count = 0
        patience = 1000
        tol = 1e-4

        for epoch in range(epochs):
            probs, h0 = net(g_dgl, inputs, h0.detach(), edge_weight)
            loss = _structured_loss(probs, B_t, P_pen, m_edges,
                                    alpha_ortho, epoch, n, k)
            lv = loss.detach().item()

            with torch.no_grad():
                assn = probs.argmax(dim=1)
                X_proj = torch.zeros_like(probs)
                X_proj.scatter_(1, assn.unsqueeze(1), 1.0)
                pl = _structured_loss(X_proj, B_t, P_pen, m_edges,
                                      alpha_ortho, 0, n, k).item()
                if pl < best_proj_loss:
                    best_proj_loss = pl
                    best_assignment = assn.detach().cpu().numpy().astype(int)
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

        # Evaluate
        from collections import defaultdict
        groups = defaultdict(set)
        for v, lbl in zip(nodes, best_assignment):
            groups[int(lbl)].add(v)
        comms = [frozenset(s) for s in groups.values() if len(s) > 0]
        mod = 0.0 if len(comms) <= 1 else float(nx.community.modularity(G, comms))
        truth = np.array([true_labels[v] for v in nodes])
        nmi = float(normalized_mutual_info_score(truth, best_assignment))
        counts = np.bincount(best_assignment, minlength=k)
        collapsed = bool(counts.max() / n >= 0.8)
        n_used = len(comms)
        elapsed = time.time() - t_start

        return {
            'graph': graph_name, 'k': k, 'lr': lr, 'alpha_ortho': alpha_ortho,
            'epochs': epochs, 'shot': shot_seed, 'best_epoch': best_epoch,
            'modularity': mod, 'nmi': nmi, 'n_used': n_used,
            'collapsed': collapsed, 'time': elapsed, 'error': '',
        }
    except Exception as e:
        return {
            'graph': graph_name, 'k': k, 'lr': lr, 'alpha_ortho': alpha_ortho,
            'epochs': epochs, 'shot': shot_seed, 'best_epoch': -1,
            'modularity': float('nan'), 'nmi': float('nan'), 'n_used': 0,
            'collapsed': False, 'time': time.time() - t_start,
            'error': f'{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}',
        }


# ---------------------------------------------------------------------------
def build_grid(quick: bool):
    if quick:
        # 3 graphs × 1 config × 1 shot = 3 runs (sanity)
        configs = []
        for g, spec in GRAPH_SPECS.items():
            configs.append((g, spec['k_test'], 0.014, 0.1, 500, 42))
        return configs

    LRS = [0.001, 0.005, 0.014]
    ALPHAS = [0.0, 0.1, 0.5, 1.0]
    EPOCHS = [3000, 10000]
    SHOTS = [42, 43, 44, 45, 46]

    configs = []
    for g, spec in GRAPH_SPECS.items():
        for lr in LRS:
            for a in ALPHAS:
                for ep in EPOCHS:
                    for sd in SHOTS:
                        configs.append((g, spec['k_test'], lr, a, ep, sd))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='sanity check (3 runs)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', default=os.path.join(RES_DIR, 'hp_sweep.csv'))
    parser.add_argument('--budget-min', type=float, default=18.0,
                        help='abort accepting new tasks past this many minutes')
    args = parser.parse_args()

    print(f'Budget: {args.budget_min} min, workers: {args.workers}, '
          f'out: {args.out}')
    print('Preparing graph cache (one-time)...')
    t_pre = time.time()
    prepare_graph_cache()
    print(f'  cache ready in {time.time()-t_pre:.1f}s')

    configs = build_grid(quick=args.quick)
    print(f'Configurations to run: {len(configs)}'
          + (' (quick)' if args.quick else ''))

    t0 = time.time()
    rows = []
    deadline = t0 + args.budget_min * 60
    aborted = 0

    import multiprocessing as mp
    ctx = mp.get_context('spawn')

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        future_map = {pool.submit(run_one_config, cfg): cfg for cfg in configs}
        for i, fut in enumerate(as_completed(future_map), start=1):
            try:
                row = fut.result()
            except Exception as e:
                row = {'error': f'{type(e).__name__}: {e}'}
            rows.append(row)
            elapsed = time.time() - t0

            if i % 20 == 0 or i == len(future_map):
                avg = elapsed / i
                eta = avg * (len(future_map) - i)
                print(f'  [{i}/{len(future_map)}] elapsed={elapsed/60:.1f}min '
                      f'avg={avg:.1f}s ETA={eta/60:.1f}min')

            if time.time() > deadline:
                aborted = sum(1 for f in future_map if not f.done())
                print(f'  >>> budget exceeded ({elapsed/60:.1f}min). '
                      f'Cancelling {aborted} pending tasks. <<<')
                for f in future_map:
                    if not f.done():
                        f.cancel()
                break

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    elapsed = time.time() - t0
    print(f'\nWrote {args.out} ({len(df)} rows). '
          f'Total wall time: {elapsed/60:.1f}min. Aborted: {aborted}.')

    # Errors summary
    err = df[df['error'].astype(bool) & (df['error'] != '')]
    if len(err) > 0:
        print(f'\n!! {len(err)} rows had errors:')
        print(err.head(5).to_string())


if __name__ == '__main__':
    main()
