"""Extended-shots experiment: 20 shots (10 for email_eu_core) on 7 graphs × 2 configs.

Configs:
  baseline  -> alpha_ortho=0.0
  ortho     -> alpha_ortho=0.1
Per-graph k = k_true. lr=0.014, epochs=3000.

Streams output by graph: prepares cache up front, then for each graph runs all
shots × configs (parallel inside the graph), prints summary, saves CSV
incrementally.
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
    'polbooks',
    'football',
    'lfr_n200_mu0.1',
    'lfr_n200_mu0.3',
    'lfr_n500_mu0.3',
    'email_eu_core',
]

CONFIGS = [
    ('baseline', {'alpha_ortho': 0.0}),
    ('ortho',    {'alpha_ortho': 0.1}),
]

EPOCHS = 3000
LR = 0.014
BASE_SEED = 42


def prepare_graph_cache(graph_names):
    import networkx as nx
    from data_loaders import (load_karate, load_football, load_polbooks,
                              load_email_eu_core, generate_lfr)
    loaders = {
        'karate':         load_karate,
        'football':       load_football,
        'polbooks':       load_polbooks,
        'lfr_n200_mu0.1': lambda: generate_lfr(n=200, mu=0.1),
        'lfr_n200_mu0.3': lambda: generate_lfr(n=200, mu=0.3),
        'lfr_n500_mu0.3': lambda: generate_lfr(n=500, mu=0.3),
        'email_eu_core':  load_email_eu_core,
    }
    for name in graph_names:
        out = os.path.join(CACHE_DIR, f'{name}.pkl')
        if os.path.exists(out):
            print(f'  cache hit: {name}')
            continue
        print(f'  building cache: {name}...')
        G, lbls, k_true = loaders[name]()
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
        print(f'    cached {name}: n={len(nodes)} m={m} k_true={k_true}')


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


def run_one_shot(args):
    """Single (graph, config, seed) trial. Worker entry point."""
    graph_name, config_name, shot_seed = args

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

        cfg = dict(CONFIGS)[config_name]
        alpha_ortho = cfg['alpha_ortho']

        _random.seed(shot_seed); np.random.seed(shot_seed); torch.manual_seed(shot_seed)

        payload = _load_cached_graph(graph_name)
        G = payload['G']
        nodes = payload['nodes']
        true_labels = payload['true_labels']
        B = payload['B']
        m_edges = payload['m']
        P_pen = payload['P_pen']
        k = int(payload['k_true'])
        n = len(nodes)

        device = torch.device('cpu')
        dtype = torch.float32
        B_t = torch.tensor(B, dtype=dtype, device=device)
        g_dgl = dgl.from_networkx(G).to(device)

        net, embed, opt = _build_net(n, k, dim_emb=10, hidden=50, dropout=0.5,
                                     lr=LR, dtype=dtype, device=device)

        src, dst = g_dgl.edges()
        edge_weight = (-B_t)[src, dst]

        inputs = torch.rand((n, 10), dtype=dtype, device=device)
        pr = nx.pagerank(nx.Graph(G))
        walk = torch.zeros((n, 20), dtype=dtype, device=device)
        for v, val in pr.items():
            walk[v, :] = val
        inputs = torch.cat([inputs, torch.ones_like(inputs), torch.ones_like(inputs), walk], 1)

        h0 = torch.zeros(n, k, device=device, dtype=dtype)

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

        for epoch in range(EPOCHS):
            probs, h0 = net(g_dgl, inputs, h0.detach(), edge_weight)
            loss = _structured_loss(probs, B_t, P_pen, m_edges, alpha_ortho,
                                    epoch, n, k)
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

        groups = defaultdict(set)
        for v, lbl in zip(nodes, best_assignment):
            groups[int(lbl)].add(v)
        comms = [frozenset(s) for s in groups.values() if len(s) > 0]
        mod = 0.0 if len(comms) <= 1 else float(nx.community.modularity(G, comms))
        truth = np.array([true_labels[v] for v in nodes])
        nmi = float(normalized_mutual_info_score(truth, best_assignment))
        counts = np.bincount(best_assignment, minlength=k)
        collapsed = bool(counts.max() / n >= 0.8)
        used_k = len(comms)
        elapsed = time.time() - t_start

        return {
            'graph': graph_name, 'config': config_name, 'shot_seed': shot_seed,
            'k': k, 'mod': mod, 'nmi': nmi, 'used_k': used_k,
            'collapse': collapsed, 'best_epoch': best_epoch,
            'time': elapsed, 'error': '',
        }
    except Exception as e:
        return {
            'graph': graph_name, 'config': config_name, 'shot_seed': shot_seed,
            'k': -1, 'mod': float('nan'), 'nmi': float('nan'), 'used_k': 0,
            'collapse': False, 'best_epoch': -1,
            'time': time.time() - t_start,
            'error': f'{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}',
        }


def run_graph(graph_name, n_shots, max_workers=8):
    jobs = [(graph_name, cfg_name, BASE_SEED + i)
            for cfg_name, _ in CONFIGS
            for i in range(n_shots)]
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    results = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = [pool.submit(run_one_shot, job) for job in jobs]
        for fut in as_completed(futures):
            results.append(fut.result())
    return pd.DataFrame(results)


def load_baselines():
    fp = os.path.join(RES_DIR, 'baselines.csv')
    if os.path.exists(fp):
        return pd.read_csv(fp)
    return pd.DataFrame()


def print_graph_summary(graph_name, df_graph, baselines_df, elapsed, remaining):
    with open(os.path.join(CACHE_DIR, f'{graph_name}.pkl'), 'rb') as f:
        payload = pickle.load(f)
    n = len(payload['nodes']); k_true = int(payload['k_true'])

    bar = '=' * 76
    print()
    print(bar)
    print(f"Graph: {graph_name} (n={n}, k_true={k_true})")
    mins = int(elapsed // 60); secs = int(elapsed % 60)
    print(f"Elapsed: {mins} min {secs} sec | Remaining graphs: {remaining}")
    print('-' * 76)
    print(f"{'Method':<18} {'mod_best':>9} {'mod_mean±std':>16} "
          f"{'nmi_best':>9} {'stable_rate':>12}")

    louv = baselines_df[(baselines_df['graph'] == graph_name) &
                        (baselines_df['method'] == 'Louvain')]
    if len(louv):
        r = louv.iloc[0]
        print(f"{'Louvain (ref)':<18} {r['mod']:>9.4f} {'-':>16} "
              f"{r['nmi']:>9.4f} {'-':>12}")

    for cfg_name, _ in CONFIGS:
        sub = df_graph[df_graph['config'] == cfg_name]
        sub = sub[sub['mod'].notna()]
        if len(sub) == 0:
            print(f"QIGNN {cfg_name:<11} (no successful runs)")
            continue
        mods = sub['mod'].values
        stable = 1.0 - sub['collapse'].mean()
        label = f'QIGNN {cfg_name}'
        print(f"{label:<18} "
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
    parser.add_argument('--out', default=os.path.join(RES_DIR, 'extended_shots.csv'))
    parser.add_argument('--graphs', nargs='+', default=GRAPH_ORDER)
    parser.add_argument('--shots', type=int, default=20,
                        help='shots for non-email graphs (email always 10)')
    args = parser.parse_args()

    print(f'Workers: {args.workers}, out: {args.out}, '
          f'graphs: {args.graphs}, shots: {args.shots} (email=10)')
    print('Preparing graph cache...')
    t0 = time.time()
    prepare_graph_cache(args.graphs)
    print(f'  cache ready in {time.time()-t0:.1f}s')

    baselines_df = load_baselines()

    all_results = []
    start = time.time()
    for i, graph_name in enumerate(args.graphs):
        n_shots = 10 if graph_name == 'email_eu_core' else args.shots
        n_jobs = n_shots * len(CONFIGS)
        print(f"\n>>> Starting graph {i+1}/{len(args.graphs)}: "
              f"{graph_name} ({n_jobs} jobs)...")
        sys.stdout.flush()

        df_graph = run_graph(graph_name, n_shots, max_workers=args.workers)
        all_results.append(df_graph)

        pd.concat(all_results, ignore_index=True).to_csv(args.out, index=False)

        print_graph_summary(graph_name, df_graph, baselines_df,
                            elapsed=time.time() - start,
                            remaining=len(args.graphs) - i - 1)

    print(f"\n>>> All graphs done. Final results: {args.out}")
    print(f"Total wall time: {(time.time()-start)/60:.1f} min")


if __name__ == '__main__':
    main()
