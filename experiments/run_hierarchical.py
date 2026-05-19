"""Hierarchical QIGNN runner: 3 criteria × 5 seeds across many graphs.

For each graph (in light->heavy order) we run:
  - criterion='modularity_gain'  (Newman 2006 stopping rule, auto-k)
  - criterion='target_k', target_k=k_true
  - criterion='min_size', min_size=ceil(sqrt(n))
each with seeds [42..46]. The 15 (criterion, seed) jobs of one graph are
parallelized over up to N workers; graphs themselves are processed sequentially
so the streaming output makes sense.

Each job runs `hierarchical_qignn.evaluate_hierarchical` (sequential splits
inside) and reports modularity, NMI, k_found, time, history depth.
"""
from __future__ import annotations
import argparse
import math
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
    'polbooks',
    'football',
    'lfr_n200_mu0.1',
    'lfr_n200_mu0.3',
    'SBM_n200_p0.3_q0.05',
    'SBM_n200_p0.2_q0.15',
    'lfr_n500_mu0.3',
    'SBM_n500_p0.2_q0.05',
    'polblogs',
    'email_eu_core',
]

CRITERIA = ['modularity_gain', 'target_k', 'min_size']

DEFAULT_N_SHOTS = 10
DEFAULT_EPOCHS = 3000
DEFAULT_SEEDS = list(range(42, 47))   # 5 seeds


def _load_graph_pickle(name):
    """Look in data/cache for either k2 or multi cache; fall back to loaders."""
    for fname in (f'{name}__k2.pkl', f'{name}.pkl'):
        p = os.path.join(CACHE_DIR, fname)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                payload = pickle.load(f)
            return payload
    return None


def prepare_graph_cache(graph_names):
    """Ensure each graph has at least one pickled cache containing G + true_labels."""
    import networkx as nx
    from data_loaders import (load_karate, load_football, load_polbooks,
                              load_email_eu_core, generate_lfr,
                              load_dolphins, load_polblogs,
                              load_polbooks_binary, generate_sbm)
    loaders = {
        'karate':                load_karate,
        'football':              load_football,
        'polbooks':              load_polbooks,
        'lfr_n200_mu0.1':        lambda: generate_lfr(n=200, mu=0.1),
        'lfr_n200_mu0.3':        lambda: generate_lfr(n=200, mu=0.3),
        'lfr_n500_mu0.3':        lambda: generate_lfr(n=500, mu=0.3),
        'email_eu_core':         load_email_eu_core,
        'dolphins':              load_dolphins,
        'polblogs':              load_polblogs,
        'polbooks_binary':       load_polbooks_binary,
        'SBM_n200_p0.3_q0.05':   lambda: generate_sbm(n=200, p_in=0.3, p_out=0.05),
        'SBM_n200_p0.2_q0.15':   lambda: generate_sbm(n=200, p_in=0.2, p_out=0.15),
        'SBM_n500_p0.2_q0.05':   lambda: generate_sbm(n=500, p_in=0.2, p_out=0.05),
    }
    for name in graph_names:
        if _load_graph_pickle(name) is not None:
            print(f'  cache hit: {name}')
            continue
        print(f'  building cache: {name}...')
        try:
            G, lbls, k_true = loaders[name]()
        except Exception as e:
            print(f'  !! failed to load {name}: {e}')
            continue
        nodes = list(G.nodes())
        # Use the same shape as k2 cache (no Q precomputation needed; subgraphs
        # compute their own Q in split_once).
        payload = {
            'G': G, 'true_labels': lbls, 'k_true': int(k_true),
            'nodes': nodes,
        }
        out = os.path.join(CACHE_DIR, f'{name}__k2.pkl')
        with open(out, 'wb') as f:
            pickle.dump(payload, f)
        print(f'    cached {name}: n={len(nodes)} m={G.number_of_edges()} '
              f'k_true={k_true}')


# -----------------------------------------------------------------------------
def run_one_job(args):
    """args = (graph_name, criterion, seed). Returns a dict of metrics."""
    graph_name, criterion, seed = args
    t_start = time.time()
    warnings.filterwarnings('ignore')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    try:
        from hierarchical_qignn import evaluate_hierarchical
        payload = _load_graph_pickle(graph_name)
        if payload is None:
            raise RuntimeError(f'no cache for {graph_name}')
        G = payload['G']
        true_labels = payload['true_labels']
        k_true = int(payload['k_true'])
        n = G.number_of_nodes()

        kwargs = dict(criterion=criterion, n_shots=DEFAULT_N_SHOTS,
                      epochs=DEFAULT_EPOCHS, seed=seed,
                      max_depth=12)
        if criterion == 'target_k':
            kwargs['target_k'] = k_true
        elif criterion == 'min_size':
            kwargs['min_size'] = max(2, int(math.ceil(math.sqrt(n))))
        # modularity_gain uses defaults

        res = evaluate_hierarchical(G, true_labels, **kwargs)

        # Compress history into compact per-depth metrics.
        depths = [h['depth'] for h in res['history']]
        max_d = max(depths) if depths else 0
        n_split = sum(1 for h in res['history'] if h['decision'] == 'split')
        n_collapsed = sum(1 for h in res['history'] if h['decision'] == 'collapsed')

        return {
            'graph': graph_name, 'criterion': criterion, 'seed': seed,
            'n': n, 'k_true': k_true,
            'mod': res['mod'], 'nmi': res['nmi'], 'k_found': res['k_found'],
            'time': res['time'], 'n_decisions': res['n_decisions'],
            'n_splits': n_split, 'n_collapsed': n_collapsed,
            'max_depth': max_d,
            'min_size_used': kwargs.get('min_size', None),
            'target_k_used': kwargs.get('target_k', None),
            'wall_time': time.time() - t_start,
            'error': '',
        }
    except Exception as e:
        return {
            'graph': graph_name, 'criterion': criterion, 'seed': seed,
            'n': -1, 'k_true': -1,
            'mod': float('nan'), 'nmi': float('nan'), 'k_found': 0,
            'time': time.time() - t_start, 'n_decisions': 0,
            'n_splits': 0, 'n_collapsed': 0, 'max_depth': 0,
            'min_size_used': None, 'target_k_used': None,
            'wall_time': time.time() - t_start,
            'error': f'{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}',
        }


def run_graph_jobs(graph_name, criteria, seeds, max_workers=8):
    jobs = [(graph_name, c, s) for c in criteria for s in seeds]
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    results = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = [pool.submit(run_one_job, j) for j in jobs]
        for fut in as_completed(futures):
            results.append(fut.result())
    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
def compute_louvain_for_graph(graph_name):
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score
    payload = _load_graph_pickle(graph_name)
    G = payload['G']; nodes = payload['nodes']; tl = payload['true_labels']
    t0 = time.time()
    comms = nx.community.louvain_communities(G, seed=42)
    elapsed = time.time() - t0
    mod = float(nx.community.modularity(G, comms))
    n2c = {v: i for i, c in enumerate(comms) for v in c}
    pred = [n2c[v] for v in nodes]; truth = [tl[v] for v in nodes]
    nmi = float(normalized_mutual_info_score(truth, pred))
    return {'method': 'Louvain', 'mod': mod, 'nmi': nmi,
            'k_found': len(comms), 'time': elapsed,
            'graph': graph_name}


def lookup_multi_class_baseline(graph_name):
    """Pull formula-5 (multi-class, k=k_true, baseline strategy) result for the
    same graph from results/extended_shots.csv if available. Returns dict or None.
    """
    fp = os.path.join(RES_DIR, 'extended_shots.csv')
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp)
    df = df[df['error'].fillna('').astype(str).str.strip().str.len() == 0]
    sub = df[(df['graph'] == graph_name) & (df['config'] == 'baseline')]
    if not len(sub):
        return None
    return {'method': 'QIGNN-multi (formula 5)',
            'mod_best': sub['mod'].max(), 'mod_mean': sub['mod'].mean(),
            'mod_std': sub['mod'].std(),  'nmi_best': sub['nmi'].max(),
            'k_found_avg': sub['used_k'].mean(),
            'time_avg': sub['time'].mean(), 'graph': graph_name}


def print_graph_summary(graph_name, df_graph, louv, multi5,
                         elapsed, remaining):
    payload = _load_graph_pickle(graph_name)
    n = len(payload['nodes']) if payload else -1
    k_true = int(payload['k_true']) if payload else -1
    mins = int(elapsed // 60); secs = int(elapsed % 60)
    bar = '=' * 90
    print()
    print(bar)
    print(f"Graph: {graph_name} (n={n}, k_true={k_true})")
    print(f"Elapsed: {mins} min {secs} sec | Remaining: {remaining} graphs")
    print('-' * 90)
    print(f"{'Method':<32} {'mod_best':>9} {'nmi_best':>9} {'k_found':>8} "
          f"{'time':>8}")

    if louv is not None:
        print(f"{'Louvain (ref)':<32} {louv['mod']:>9.4f} {louv['nmi']:>9.4f} "
              f"{louv['k_found']:>8d} {louv['time']:>7.2f}s")
    if multi5 is not None:
        print(f"{'QIGNN-multi (formula 5)':<32} {multi5['mod_best']:>9.4f} "
              f"{multi5['nmi_best']:>9.4f} {multi5['k_found_avg']:>8.1f} "
              f"{multi5['time_avg']:>7.2f}s")

    sub_clean = df_graph[df_graph['error'].fillna('').astype(str).str.strip().str.len() == 0]
    for c in CRITERIA:
        s = sub_clean[sub_clean['criterion'] == c]
        if not len(s):
            print(f"{'Hierarchical (' + c + ')':<32} (no data)")
            continue
        mod_best = s['mod'].max()
        nmi_best = s['nmi'].max()
        k_avg = s['k_found'].mean()
        t_avg = s['time'].mean()
        print(f"{'Hierarchical (' + c + ')':<32} {mod_best:>9.4f} {nmi_best:>9.4f} "
              f"{k_avg:>8.1f} {t_avg:>7.2f}s")

    err_count = int(df_graph['error'].astype(bool).sum())
    if err_count:
        print(f"!! {err_count} jobs errored")
    print(bar)
    sys.stdout.flush()


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--graphs', nargs='+', default=GRAPH_ORDER)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--out', default=os.path.join(RES_DIR, 'hierarchical.csv'))
    parser.add_argument('--louvain-out',
                        default=os.path.join(RES_DIR, 'hierarchical_louvain.csv'))
    parser.add_argument('--criteria', nargs='+', default=CRITERIA)
    args = parser.parse_args()

    print(f'Workers: {args.workers}, graphs: {args.graphs}')
    print(f'Seeds: {args.seeds}, criteria: {args.criteria}')
    print('Preparing graph cache...')
    t0 = time.time()
    prepare_graph_cache(args.graphs)
    print(f'  cache ready in {time.time()-t0:.1f}s')

    all_results = []
    louv_rows = []
    start = time.time()
    for i, graph_name in enumerate(args.graphs):
        if _load_graph_pickle(graph_name) is None:
            print(f"\n>>> SKIPPING {graph_name}: cache not available")
            continue

        louv = compute_louvain_for_graph(graph_name)
        louv_rows.append(louv)
        pd.DataFrame(louv_rows).to_csv(args.louvain_out, index=False)
        multi5 = lookup_multi_class_baseline(graph_name)

        n_jobs = len(args.criteria) * len(args.seeds)
        print(f"\n>>> Starting graph {i+1}/{len(args.graphs)}: "
              f"{graph_name} ({n_jobs} jobs = {len(args.criteria)} criteria × "
              f"{len(args.seeds)} seeds)...")
        sys.stdout.flush()

        try:
            df_graph = run_graph_jobs(graph_name, args.criteria, args.seeds,
                                       max_workers=args.workers)
        except Exception as e:
            print(f'  !! graph {graph_name} crashed: {e}')
            continue
        all_results.append(df_graph)
        pd.concat(all_results, ignore_index=True).to_csv(args.out, index=False)

        print_graph_summary(graph_name, df_graph, louv, multi5,
                             elapsed=time.time() - start,
                             remaining=len(args.graphs) - i - 1)

    print(f"\n>>> All graphs done. Final results: {args.out}")
    print(f"Total wall time: {(time.time()-start)/60:.1f} min")


if __name__ == '__main__':
    main()
