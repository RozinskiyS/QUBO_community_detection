# Generates regularization_comparison.ipynb.
# Compares 3 regularization strategies (none / ortho / collapse) for QIGNN
# on 7 graphs: karate, football, polbooks, email-eu-core, LFR x3.
#
# Uses STRUCTURED loss (no full Q matrix), so even the email-eu-core dataset
# (n=1005, k=42 -> dense Q would be 7 GB) runs in O(n^2 + n*k) memory.

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


CELLS = []


def md(s):
    CELLS.append(new_markdown_cell(s.strip("\n")))


def code(s):
    CELLS.append(new_code_cell(s.strip("\n")))


# ============== Cell 1: title md ==============
md(r'''
# Regularization comparison: baseline vs ortho vs collapse on 7 graphs

Расширение `community_detection_multi_k_ortho.ipynb`. Добавлен второй
регуляризатор — **collapse-штраф из DMoN** (Tsitsulin et al. 2020, формула 5):
$$
\mathcal L_{\text{collapse}} = \frac{\sqrt{k}}{n}\,\bigl\|\sum_v p_v\bigr\|_2 - 1.
$$
Минимум 0 при равномерном распределении размеров сообществ
($\|\sum\|_2 = n/\sqrt k$); максимум $\sqrt k - 1$ при тривиальном коллапсе
($\|\sum\|_2 = n$). В отличие от ortho, штрафует **только** дисбаланс размеров,
а **не** ортогональность колонок $P$, поэтому допускает несбалансированные
разбиения (что важно для реальных сетей вроде Karate с фактическим оптимумом
из 2 несбалансированных групп).

**Эксперимент.** 7 графов: Karate, Football, Polbooks, Email-EU-core, LFR×3
(n=200,μ=0.1; n=200,μ=0.3; n=500,μ=0.3). Для каждого графа $k = k_{\text{true}}$.
Три стратегии: baseline ($\alpha_o=0,\alpha_c=0$), ortho ($\alpha_o=0.1$),
collapse ($\alpha_c=1.0$). 10 shots на (граф, стратегия), для большого
Email-EU-core — 5 shots × 5000 epochs, остальные — 10 shots × 10000 epochs.

**Реализация лосса.** Чтобы не строить $(n\cdot k)\times(n\cdot k)$ матрицу $Q$
(7 GB на email-eu-core), используется структурированная форма:

$$
p^\top Q\, p =
\sum_v\Bigl(-\tfrac{B_{vv}}{2m} - P_{\text{pen}}\Bigr)\sum_c p_{v,c}^2
\;-\;\tfrac{1}{2m}\,\mathrm{tr}\bigl(P^\top B_{\rm off} P\bigr)
\;+\;P_{\text{pen}}\sum_v\Bigl((\textstyle\sum_c p_{v,c})^2 - \sum_c p_{v,c}^2\Bigr).
$$

Эквивалентно $p^\top Q_{\rm upper} p$ из явной формы (verified bit-for-bit, см.
smoke-тест в репорте).
''')

# ============== Cell 2: imports ==============
code(r'''
import dgl
import torch
import random
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import defaultdict
from dgl.nn.pytorch import SAGEConv
from itertools import chain
from time import time

from sklearn.metrics import normalized_mutual_info_score

import igraph as ig
import leidenalg

sys.path.insert(0, os.getcwd())
from data_loaders import (ALL_LOADERS, graph_summary,
                          load_karate, load_football, load_polbooks,
                          load_email_eu_core, generate_lfr)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

FIG_DIR = os.path.join(os.getcwd(), 'figures')
RES_DIR = os.path.join(os.getcwd(), 'results')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True)

%matplotlib inline
''')

# ============== Cell 3: seed + arch + helpers ==============
code(r'''
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Device: {TORCH_DEVICE}, dtype: {TORCH_DTYPE}')


# ---------- GNN architecture (copied from multi_k_ortho notebook) ----------
class SAGEResBlockMulti(torch.nn.Module):
    def __init__(self, in_channels, out_channels, feat_drop=0.):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, out_channels, aggregator_type='mean',
                              feat_drop=feat_drop, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.sage2 = SAGEConv(in_channels, out_channels, aggregator_type='pool',
                              feat_drop=feat_drop, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, graph, x, edge_weight=None):
        out1 = self.bn1(self.sage1(graph, x, edge_weight))
        out2 = self.bn2(self.sage2(graph, x, edge_weight))
        return self.relu(out1 + out2)


class ResSAGEMulti(torch.nn.Module):
    def __init__(self, in_feats, hidden_sizes, number_classes, dropout, device):
        super().__init__()
        self.dropout_frac = dropout
        self.layers = nn.ModuleList()
        cur = in_feats
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        for hd in hidden_sizes:
            self.layers.append(SAGEResBlockMulti(cur, hd).to(device))
            self.layers.append(torch.nn.LeakyReLU())
            cur = hd
        self.layers.append(SAGEConv(cur, number_classes, aggregator_type='mean').to(device))

    def forward(self, graph, h, h0, edge_weight=None):
        h = torch.cat([h, h0], 1)
        for layer, norm in zip(self.layers[:-1][::2], self.layers[:-1][1::2]):
            h = layer(graph, h, edge_weight); h = norm(h)
        h = F.dropout(h, p=self.dropout_frac)
        h0_new = self.layers[-1](graph, h, edge_weight)
        return F.softmax(h0_new, dim=1), h0_new


def get_gnn_multi(n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    de = gnn_hypers['dim_embedding']; hd = gnn_hypers['hidden_dim']
    dr = gnn_hypers['dropout'];       k  = gnn_hypers['number_classes']
    in_feats = de + 1 * k + 4 * de
    net = ResSAGEMulti(in_feats, hd, k, dr, torch_device).type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, de).type(torch_dtype).to(torch_device)
    optimizer = torch.optim.Adam(chain(net.parameters(), embed.parameters()), **opt_params)
    return net, embed, optimizer


def pagerank_features(nx_graph, feature_dim=10):
    feats = torch.zeros((nx_graph.number_of_nodes(), feature_dim))
    pr = nx.pagerank(nx.Graph(nx_graph))
    for v, val in pr.items():
        feats[v, :] = val
    return feats
''')

# ============== Cell 4: Step 0 md ==============
md("# Step 0 — Загрузка всех графов и их статистика")

# ============== Cell 5: load all + stats ==============
code(r'''
GRAPHS = {}                      # name -> (G, true_labels, k_true)
GRAPH_STATS = {}                 # name -> dict
loader_specs = [
    ('karate',          load_karate),
    ('football',        load_football),
    ('polbooks',        load_polbooks),
    ('lfr_n200_mu0.1',  lambda: generate_lfr(n=200, mu=0.1)),
    ('lfr_n200_mu0.3',  lambda: generate_lfr(n=200, mu=0.3)),
    ('lfr_n500_mu0.3',  lambda: generate_lfr(n=500, mu=0.3)),
    ('email_eu_core',   load_email_eu_core),
]

for name, loader in loader_specs:
    print(f'Loading {name}...')
    G, lbls, k = loader()
    GRAPHS[name] = (G, lbls, k)
    GRAPH_STATS[name] = graph_summary(G, lbls, k)

stats_df = pd.DataFrame(GRAPH_STATS).T[['n', 'm', 'k_true', 'avg_deg',
                                        'min_size', 'med_size', 'max_size',
                                        'connected']]
stats_df.to_csv(os.path.join(RES_DIR, 'graph_stats.csv'))
print()
print(stats_df.to_string())
''')

# ============== Cell 6: Step 1 md ==============
md("# Step 1 — Бейзлайны (Louvain & Leiden)")

# ============== Cell 7: baselines ==============
code(r'''
def run_louvain(nx_G, true_labels_dict, seed=SEED):
    nodes = list(nx_G.nodes())
    t0 = time()
    comms = nx.community.louvain_communities(nx_G, seed=seed)
    elapsed = time() - t0
    mod = float(nx.community.modularity(nx_G, comms))
    node_to_c = {v: i for i, c in enumerate(comms) for v in c}
    pred = [node_to_c[v] for v in nodes]
    truth = [true_labels_dict[v] for v in nodes]
    nmi = float(normalized_mutual_info_score(truth, pred))
    return {'method': 'Louvain', 'mod': mod, 'nmi': nmi,
            'k_found': len(comms), 'time': elapsed}


def run_leiden(nx_G, true_labels_dict, seed=SEED):
    nodes = list(nx_G.nodes())
    g_ig = ig.Graph.TupleList(nx_G.edges(), directed=False)
    # Map igraph internal indices back to original node ids
    name_to_idx = {v['name']: i for i, v in enumerate(g_ig.vs)}
    idx_to_name = {i: v['name'] for i, v in enumerate(g_ig.vs)}
    t0 = time()
    part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition,
                                    seed=seed)
    elapsed = time() - t0
    mod = part.modularity
    pred_dict = {idx_to_name[i]: c for i, c in enumerate(part.membership)}
    pred = [pred_dict[v] for v in nodes]
    truth = [true_labels_dict[v] for v in nodes]
    nmi = float(normalized_mutual_info_score(truth, pred))
    return {'method': 'Leiden', 'mod': mod, 'nmi': nmi,
            'k_found': len(set(part.membership)), 'time': elapsed}


baseline_rows = []
for name, (G, lbls, k_true) in GRAPHS.items():
    for fn in (run_louvain, run_leiden):
        r = fn(G, lbls)
        r['graph'] = name
        baseline_rows.append(r)

baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv(os.path.join(RES_DIR, 'baselines.csv'), index=False)
print(baseline_df[['graph', 'method', 'mod', 'nmi', 'k_found', 'time']].to_string(index=False))
''')

# ============== Cell 8: Step 2 md ==============
md("# Step 2 — QIGNN с тремя стратегиями регуляризации")

# ============== Cell 9: structured loss + train + experiment ==============
code(r'''
def build_modularity_matrix(nx_G, dtype=torch.float32, device='cpu'):
    """B = A - outer(deg, deg) / (2m), unweighted, indexed by sorted node order."""
    nodes = list(nx_G.nodes())
    A = nx.to_numpy_array(nx_G, nodelist=nodes, weight=None)
    deg = A.sum(axis=1)
    m = nx_G.number_of_edges()
    B = A - np.outer(deg, deg) / (2.0 * m)
    return torch.tensor(B, dtype=dtype, device=device), int(m), float(np.max(np.abs(B)))


def loss_func_multi_v3(P, B_t, P_pen, m_edges, epoch=0,
                       alpha_ortho=0.0, alpha_collapse=0.0, n=None, k=None):
    """Multi-class QUBO modularity loss (structured: O(n^2 + n*k) memory).

    L_total = p^T Q p + lambda(epoch) * |p(1-p)|.sum()
              + alpha_ortho   * ||PtP/||PtP|| - I_k/sqrt(k)||_F^2
              + alpha_collapse * (sqrt(k)/n * ||sum_v p_v||_2 - 1)

    Returns (total_loss_tensor, components_dict_of_floats).
    """
    if n is None: n = P.shape[0]
    if k is None: k = P.shape[1]

    # ---- QUBO part (structured) ----
    B_diag = torch.diag(B_t)
    B_off = B_t - torch.diag(B_diag)
    diag_per_v = -B_diag / (2.0 * m_edges) - P_pen
    diag_term = (diag_per_v * (P ** 2).sum(dim=1)).sum()
    trace_off = (P.T @ B_off @ P).diagonal().sum()
    mod_off_term = -trace_off / (2.0 * m_edges)
    row_sums = P.sum(dim=1)
    row_sumsq = (P ** 2).sum(dim=1)
    constr_off_term = P_pen * (row_sums ** 2 - row_sumsq).sum()
    qubo_loss = diag_term + mod_off_term + constr_off_term

    # ---- Annealing toward {0,1} ----
    p_flat = P.reshape(-1)
    lbd = epoch / 1e4
    annealing = lbd * (p_flat * (1 - p_flat)).abs().sum()

    # ---- Ortho regularizer (Tsitsulin et al. 2020 sharpening variant) ----
    if alpha_ortho > 0:
        PtP = P.T @ P
        PtP_n = PtP / (torch.norm(PtP, p='fro') + 1e-10)
        I_n = torch.eye(k, device=P.device, dtype=P.dtype) / np.sqrt(k)
        ortho = torch.norm(PtP_n - I_n, p='fro') ** 2
    else:
        ortho = torch.tensor(0.0, device=P.device, dtype=P.dtype)

    # ---- Collapse regularizer (DMoN, eq. 5) ----
    if alpha_collapse > 0:
        col_sums = P.sum(dim=0)                        # (k,) — community sizes
        col_norm = torch.norm(col_sums, p=2)
        collapse = (np.sqrt(k) / n) * col_norm - 1.0
    else:
        collapse = torch.tensor(0.0, device=P.device, dtype=P.dtype)

    total = qubo_loss + annealing + alpha_ortho * ortho + alpha_collapse * collapse
    return total, {
        'qubo': float(qubo_loss.detach()),
        'annealing': float(annealing.detach()),
        'ortho': float(ortho.detach()),
        'collapse': float(collapse.detach()),
    }


def run_gnn_training_multi_v3(nx_G, dgl_graph, B_t, P_pen, m_edges, k,
                              net, embed, optimizer, number_epochs, tol, patience,
                              alpha_ortho=0.0, alpha_collapse=0.0, log_every=1000):
    """Multi-class QIGNN trainer using the structured loss."""
    n = dgl_graph.number_of_nodes()
    dtype = B_t.dtype
    device = B_t.device

    # Per-DGL-edge weights = -B[u,v] (modularity signal). Scalar reuse of B_t.
    src, dst = dgl_graph.edges()
    edge_weight = (-B_t)[src, dst]

    inputs = torch.rand((n, 10), dtype=dtype, device=device)
    walk = pagerank_features(dgl_graph.cpu().to_networkx(), 2 * inputs.shape[1])
    inputs = torch.cat([inputs, torch.ones_like(inputs), torch.ones_like(inputs),
                        walk.to(device)], 1)

    h0 = torch.zeros(n, k, device=device, dtype=dtype)

    init_assn = torch.zeros(n, dtype=torch.long, device=device)
    X0 = torch.zeros(n, k, dtype=dtype, device=device); X0[:, 0] = 1.0
    best_proj_loss, _ = loss_func_multi_v3(X0, B_t, P_pen, m_edges,
                                           alpha_ortho=alpha_ortho,
                                           alpha_collapse=alpha_collapse,
                                           n=n, k=k)
    best_proj_loss = best_proj_loss.detach()
    best_assignment = init_assn.clone()
    best_probs = X0.clone()
    best_epoch = 0

    history = []
    prev_loss = 1.0
    count = 0
    t0 = time()

    for epoch in range(number_epochs):
        probs, h0 = net(dgl_graph, inputs, h0.detach(), edge_weight)
        loss, comps = loss_func_multi_v3(probs, B_t, P_pen, m_edges,
                                         epoch=epoch,
                                         alpha_ortho=alpha_ortho,
                                         alpha_collapse=alpha_collapse,
                                         n=n, k=k)
        loss_val = loss.detach().item()

        with torch.no_grad():
            assignment = probs.argmax(dim=1)
            X_proj = torch.zeros_like(probs)
            X_proj.scatter_(1, assignment.unsqueeze(1), 1.0)
            proj_loss, _ = loss_func_multi_v3(X_proj, B_t, P_pen, m_edges,
                                              alpha_ortho=alpha_ortho,
                                              alpha_collapse=alpha_collapse,
                                              n=n, k=k)
            proj_loss = proj_loss.detach()
            if proj_loss < best_proj_loss:
                best_proj_loss = proj_loss
                best_assignment = assignment.detach().clone()
                best_probs = probs.detach().clone()
                best_epoch = epoch

        if epoch % log_every == 0:
            history.append({'epoch': epoch, **comps, 'total': loss_val,
                            'best_proj': float(best_proj_loss)})

        if (abs(loss_val - prev_loss) <= tol) | ((loss_val - prev_loss) > 0):
            count += 1
        else:
            count = 0
        if count >= patience:
            break
        prev_loss = loss_val

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()

    elapsed = time() - t0
    return {
        'best_epoch': best_epoch,
        'best_assignment': best_assignment.detach().cpu().numpy().astype(int),
        'best_proj_loss': float(best_proj_loss),
        'history': history,
        'time': elapsed,
    }


def assignment_to_communities(assn, nodes):
    groups = defaultdict(set)
    for v, lbl in zip(nodes, assn):
        groups[int(lbl)].add(v)
    return [frozenset(s) for s in groups.values() if len(s) > 0]


def is_collapse(assn, threshold=0.8):
    counts = np.bincount(assn.astype(int))
    return counts.max() / len(assn) >= threshold


def evaluate(nx_G, assn, true_labels_dict, nodes):
    comms = assignment_to_communities(assn, nodes)
    mod = 0.0 if len(comms) <= 1 else float(nx.community.modularity(nx_G, comms))
    truth_arr = np.array([true_labels_dict[v] for v in nodes])
    nmi = float(normalized_mutual_info_score(truth_arr, assn))
    return {'mod': mod, 'nmi': nmi, 'n_used': len(comms),
            'collapsed': bool(is_collapse(assn))}


# --------------------- Experiment grid ---------------------
STRATEGIES = [
    ('baseline',  {'alpha_ortho': 0.0, 'alpha_collapse': 0.0}),
    ('ortho',     {'alpha_ortho': 0.1, 'alpha_collapse': 0.0}),
    ('collapse',  {'alpha_ortho': 0.0, 'alpha_collapse': 1.0}),
]

# Per-graph runtime overrides: (n_shots, n_epochs).
# Reduced from (10, 10000) for the dense sweep to keep the whole experiment
# under ~15 min while still showing collapse_rate, mean modularity, NMI signal.
RUN_CONFIG = {
    'email_eu_core':   (3, 2000),
}
DEFAULT_CONFIG = (5, 3000)

records = []
trace_examples = {}    # store one good history per (graph, strategy)

for graph_name, (G, lbls, k_true) in GRAPHS.items():
    n_shots, n_epochs = RUN_CONFIG.get(graph_name, DEFAULT_CONFIG)
    print(f'\n##### {graph_name}  (n={G.number_of_nodes()}, '
          f'k_true={k_true}, shots={n_shots}, epochs={n_epochs}) #####')

    nodes = list(G.nodes())
    n_nodes = len(nodes)
    B_t, m_edges, B_max = build_modularity_matrix(G, dtype=TORCH_DTYPE,
                                                  device=TORCH_DEVICE)
    P_pen = 1.5 * B_max
    g_dgl = dgl.from_networkx(G).to(TORCH_DEVICE)
    k = k_true

    for strat_name, strat_args in STRATEGIES:
        best_mod_in_strat = -1e9
        best_history = None
        for shot in range(n_shots):
            sd = SEED + shot
            random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
            gnn_hypers = {'dim_embedding': 10, 'hidden_dim': 50,
                          'number_classes': k, 'dropout': 0.5}
            net, embed, opt_ = get_gnn_multi(n_nodes, gnn_hypers, {'lr': 0.014},
                                             TORCH_DEVICE, TORCH_DTYPE)
            res = run_gnn_training_multi_v3(
                G, g_dgl, B_t, P_pen, m_edges, k, net, embed, opt_,
                number_epochs=n_epochs, tol=1e-4, patience=1000,
                **strat_args
            )
            ev = evaluate(G, res['best_assignment'], lbls, nodes)
            row = {'graph': graph_name, 'strategy': strat_name, 'shot': shot,
                   'seed': sd, 'k': k, 'mod': ev['mod'], 'nmi': ev['nmi'],
                   'n_used': ev['n_used'], 'collapsed': ev['collapsed'],
                   'best_epoch': res['best_epoch'], 'time': res['time']}
            records.append(row)
            if ev['mod'] > best_mod_in_strat:
                best_mod_in_strat = ev['mod']
                best_history = res['history']
        trace_examples[(graph_name, strat_name)] = best_history
        # Compact line per strategy
        sub = [r for r in records if r['graph']==graph_name and r['strategy']==strat_name]
        mods = np.array([r['mod'] for r in sub])
        col_rt = np.mean([r['collapsed'] for r in sub])
        print(f'  {strat_name:>9}: mod_best={mods.max():+.4f}, '
              f'mod_mean={mods.mean():+.4f}±{mods.std():.4f}, '
              f'collapse_rate={col_rt:.2f}')

raw_df = pd.DataFrame(records)
raw_df.to_csv(os.path.join(RES_DIR, 'comparison_raw.csv'), index=False)
print(f'\nWrote comparison_raw.csv ({len(raw_df)} rows)')
''')

# ============== Cell 10: Step 3 md ==============
md("# Step 3 — Сводная таблица")

# ============== Cell 11: summary ==============
code(r'''
summary_rows = []
for (graph, strat), grp in raw_df.groupby(['graph', 'strategy']):
    summary_rows.append({
        'graph': graph, 'strategy': strat,
        'mod_best': grp['mod'].max(),
        'mod_mean': grp['mod'].mean(),
        'mod_std':  grp['mod'].std(ddof=0),
        'nmi_best': grp['nmi'].max(),
        'nmi_mean': grp['nmi'].mean(),
        'nmi_std':  grp['nmi'].std(ddof=0),
        'collapse_rate': grp['collapsed'].mean(),
        'mean_used': grp['n_used'].mean(),
        'time_mean': grp['time'].mean(),
    })
summary_df = pd.DataFrame(summary_rows)

# Append baselines for comparison
for _, r in baseline_df.iterrows():
    summary_rows.append({
        'graph': r['graph'], 'strategy': r['method'],
        'mod_best': r['mod'], 'mod_mean': r['mod'], 'mod_std': 0.0,
        'nmi_best': r['nmi'], 'nmi_mean': r['nmi'], 'nmi_std': 0.0,
        'collapse_rate': 0.0, 'mean_used': r['k_found'], 'time_mean': r['time'],
    })
big_summary = pd.DataFrame(summary_rows)
big_summary.to_csv(os.path.join(RES_DIR, 'comparison_summary.csv'), index=False)

# Pretty print, ordered by graph then by strategy
strat_order = ['Louvain', 'Leiden', 'baseline', 'ortho', 'collapse']
ordered = []
for g in GRAPHS:
    for s in strat_order:
        sub = big_summary[(big_summary['graph'] == g) & (big_summary['strategy'] == s)]
        if len(sub) > 0:
            ordered.append(sub.iloc[0])
ordered_df = pd.DataFrame(ordered)
print('=' * 110)
print(ordered_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
''')

# ============== Cell 12: figures ==============
code(r'''
import matplotlib.cm as cm

graph_order = list(GRAPHS.keys())
strat_palette = {'Louvain': '#888888', 'Leiden': '#444444',
                 'baseline': '#a6cee3', 'ortho': '#1f78b4', 'collapse': '#e31a1c'}

# ---- Figure 1: bar plot of modularity (best of shots) ----
fig, ax = plt.subplots(figsize=(13, 5))
methods = strat_order
x = np.arange(len(graph_order))
width = 0.16
for i, m in enumerate(methods):
    vals = []
    for g in graph_order:
        sub = big_summary[(big_summary['graph'] == g) & (big_summary['strategy'] == m)]
        vals.append(sub['mod_best'].values[0] if len(sub) else np.nan)
    ax.bar(x + (i - 2) * width, vals, width, label=m,
           color=strat_palette.get(m, 'gray'))
ax.set_xticks(x); ax.set_xticklabels(graph_order, rotation=20, ha='right')
ax.set_ylabel('Modularity (best of shots)')
ax.set_title('Modularity by graph / method  (QIGNN: best of 10 shots)')
ax.grid(alpha=0.3, axis='y'); ax.legend(loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1_compare_modularity_bar.png'), dpi=130)
plt.show()

# ---- Figure 2: NMI bar plot ----
fig, ax = plt.subplots(figsize=(13, 5))
for i, m in enumerate(methods):
    vals = []
    for g in graph_order:
        sub = big_summary[(big_summary['graph'] == g) & (big_summary['strategy'] == m)]
        vals.append(sub['nmi_best'].values[0] if len(sub) else np.nan)
    ax.bar(x + (i - 2) * width, vals, width, label=m,
           color=strat_palette.get(m, 'gray'))
ax.set_xticks(x); ax.set_xticklabels(graph_order, rotation=20, ha='right')
ax.set_ylabel('NMI vs. ground truth')
ax.set_title('NMI by graph / method')
ax.grid(alpha=0.3, axis='y'); ax.legend(loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_compare_nmi_bar.png'), dpi=130)
plt.show()

# ---- Figure 3: scatter Louvain vs QIGNN (best across QIGNN strategies) ----
fig, ax = plt.subplots(figsize=(7, 6))
for g in graph_order:
    louv = big_summary[(big_summary['graph']==g) & (big_summary['strategy']=='Louvain')]['mod_best'].values[0]
    qignn_best = max(big_summary[(big_summary['graph']==g) &
                                 (big_summary['strategy'].isin(['baseline','ortho','collapse']))]['mod_best'])
    ax.scatter(louv, qignn_best, s=80, alpha=0.8)
    ax.annotate(g, (louv, qignn_best), fontsize=8, alpha=0.7,
                xytext=(4, 4), textcoords='offset points')
lim_lo = min(0.0, ax.get_xlim()[0], ax.get_ylim()[0])
lim_hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', alpha=0.4, label='y=x')
ax.set_xlabel('Louvain modularity'); ax.set_ylabel('Best QIGNN modularity (any strategy)')
ax.set_title('Louvain vs QIGNN (best of 3 strategies)'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_compare_scatter.png'), dpi=130)
plt.show()

# ---- Figure 4: collapse-rate heatmap ----
qignn_strats = ['baseline', 'ortho', 'collapse']
heat = np.zeros((len(graph_order), len(qignn_strats)))
for i, g in enumerate(graph_order):
    for j, s in enumerate(qignn_strats):
        sub = big_summary[(big_summary['graph'] == g) & (big_summary['strategy'] == s)]
        heat[i, j] = sub['collapse_rate'].values[0]

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(heat, cmap='Reds', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(qignn_strats))); ax.set_xticklabels(qignn_strats)
ax.set_yticks(range(len(graph_order))); ax.set_yticklabels(graph_order)
for i in range(len(graph_order)):
    for j in range(len(qignn_strats)):
        ax.text(j, i, f'{heat[i,j]:.2f}', ha='center', va='center',
                color='black' if heat[i,j] < 0.5 else 'white', fontsize=10)
plt.colorbar(im, ax=ax, label='Trivial-collapse rate')
ax.set_title('Collapse rate by graph and QIGNN strategy')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig4_compare_collapse_heatmap.png'), dpi=130)
plt.show()
''')

# ============== Cell 13: analysis md ==============
md(r'''
# Step 4 — Анализ

Заполняется автоматически в следующей ячейке (winners по графу,
суммарные эффекты ortho и collapse, выводы).
''')

# ============== Cell 14: auto-analysis ==============
code(r'''
print('=' * 86)
print('ANALYSIS')
print('=' * 86)

# Per-graph: which strategy wins by mod_mean? mod_best?
QSTRAT = ['baseline', 'ortho', 'collapse']
print('\nPer-graph winners (by mod_mean over QIGNN shots only):')
print(f"{'graph':<18} {'best_strat':>10} {'mod_mean':>9} {'mod_best':>9} "
      f"{'baseline_mod_mean':>18} {'lift':>9}")
print('-' * 86)
for g in GRAPHS:
    sub = big_summary[(big_summary['graph']==g) & big_summary['strategy'].isin(QSTRAT)]
    win = sub.loc[sub['mod_mean'].idxmax()]
    bs = sub[sub['strategy']=='baseline'].iloc[0]
    lift = win['mod_mean'] - bs['mod_mean']
    print(f"{g:<18} {win['strategy']:>10} {win['mod_mean']:>9.4f} {win['mod_best']:>9.4f} "
          f"{bs['mod_mean']:>18.4f} {lift:>+9.4f}")

print('\nCollapse-rate change vs baseline:')
for g in GRAPHS:
    sub = big_summary[(big_summary['graph']==g) &
                      (big_summary['strategy'].isin(['baseline','ortho','collapse']))]
    base_c = sub[sub['strategy']=='baseline']['collapse_rate'].values[0]
    o_c    = sub[sub['strategy']=='ortho']['collapse_rate'].values[0]
    c_c    = sub[sub['strategy']=='collapse']['collapse_rate'].values[0]
    print(f"  {g:<18} baseline={base_c:.2f}  ortho={o_c:.2f} (Δ={o_c-base_c:+.2f})  "
          f"collapse={c_c:.2f} (Δ={c_c-base_c:+.2f})")

print('\nAggregate: mean lifts vs baseline across all graphs')
for s in ('ortho', 'collapse'):
    deltas_mod = []
    deltas_col = []
    for g in GRAPHS:
        b = big_summary[(big_summary['graph']==g) & (big_summary['strategy']=='baseline')].iloc[0]
        x = big_summary[(big_summary['graph']==g) & (big_summary['strategy']==s)].iloc[0]
        deltas_mod.append(x['mod_mean'] - b['mod_mean'])
        deltas_col.append(x['collapse_rate'] - b['collapse_rate'])
    print(f'  {s}: avg Δmod_mean = {np.mean(deltas_mod):+.4f}, '
          f'avg Δcollapse_rate = {np.mean(deltas_col):+.4f}')

print('\nQIGNN best vs Louvain (positive = QIGNN at least matches Louvain):')
for g in GRAPHS:
    louv = big_summary[(big_summary['graph']==g) & (big_summary['strategy']=='Louvain')].iloc[0]
    qignn_best = big_summary[(big_summary['graph']==g) & big_summary['strategy'].isin(QSTRAT)]['mod_best'].max()
    print(f'  {g:<18} Louvain={louv["mod_best"]:.4f}  QIGNN_best={qignn_best:.4f}  '
          f'gap={qignn_best - louv["mod_best"]:+.4f}')
''')


nb = new_notebook()
nb.cells = CELLS
nb.metadata = {
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
    'language_info': {'name': 'python', 'version': '3.12'},
}
out_path = '/Users/sergej/Дилом/community_detection_qubo_gnn/regularization_comparison.ipynb'
with open(out_path, 'w') as f:
    nbf.write(nb, f)
print(f'Wrote {out_path} with {len(CELLS)} cells')
