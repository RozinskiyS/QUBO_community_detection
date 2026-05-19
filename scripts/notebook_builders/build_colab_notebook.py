# Generates a single self-contained notebook for Google Colab + A100 GPU.
# Embeds all helpers (no external .py files needed): data loaders, QUBO
# formulations, ResSAGEMulti architecture, loss, training, analysis.
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

CELLS = []
def md(s): CELLS.append(new_markdown_cell(s.strip("\n")))
def code(s): CELLS.append(new_code_cell(s.strip("\n")))


# ============================================================
# Cell 1: title md
# ============================================================
md(r'''
# Community Detection with GNN via QUBO — Colab + GPU edition

Self-contained ноутбук для запуска на Google Colab (рекомендуется A100/V100).
Воспроизводит и расширяет локальный эксперимент из `regularization_comparison.ipynb`
с **полным budget'ом** (10 shots × 10000 epochs) и опциональным HP sweep'ом
+ scaling-разделом на больших LFR.

**Что внутри:**
- Section 0 — установка (DGL под текущий torch+CUDA на Colab) и проверка GPU
- Section 1 — все embedded helpers: загрузка 7 графов, QUBO modularity
  (k=2 и multi-k через формулы (5),(6) Wang et al. 2024), GNN-архитектура
  (ResSAGEMulti с softmax), structured loss (без формирования (n*k)² матрицы),
  training-цикл с component-логированием
- Section 2 — baseline (Louvain, Leiden) на 7 графах
- Section 3 — main experiment: 3 стратегии регуляризации (baseline / ortho / collapse)
  × 7 графов × 10 shots × 10000 epochs (для email-eu-core: 5 shots × 5000)
- Section 4 — HP sweep (опционально, ~30 мин на A100): lr × alpha × epochs
  на 3 ключевых графах
- Section 5 — scaling: LFR n=500, 1000, 2000 (опционально)
- Section 6 — таблицы, графики, выводы

Все выходы пишутся в `./results/` и `./figures/`. CSV можно скачать в конце через
`google.colab.files.download(...)`.
''')


# ============================================================
# Cell 2: install
# ============================================================
code(r'''
# ===== Section 0a: dependency install =====
import sys, subprocess, importlib

def pip(*args, q=True):
    cmd = [sys.executable, '-m', 'pip', 'install'] + (['-q'] if q else []) + list(args)
    subprocess.check_call(cmd)

# torch is pre-installed on Colab; we just inspect.
import torch
print('torch:', torch.__version__, '| cuda available:', torch.cuda.is_available(),
      '| cuda version:', torch.version.cuda)

# Install DGL matched to the running torch+cuda. DGL hosts wheels at data.dgl.ai
# for combinations torch-X.Y / cu1ZZ. On Colab you typically land on a recent
# torch (2.4–2.6) with CUDA 12.1+. We try the matching index first; if no wheel
# exists, fall back to PyPI (CPU build) which still works for our small graphs.
def install_dgl():
    try:
        import dgl  # noqa: F401
        from dgl.nn.pytorch import SAGEConv  # noqa: F401
        print('dgl already installed:', dgl.__version__)
        return
    except Exception:
        pass

    tv = torch.__version__.split('+')[0]
    mm = '.'.join(tv.split('.')[:2])  # "2.4"
    cuda = torch.version.cuda or ''
    # cu123 / cu121 / cu118 — pick first three digits compact
    cu = ''.join(cuda.split('.'))[:3] if cuda else ''
    candidate_urls = []
    if cu:
        candidate_urls.append(f'https://data.dgl.ai/wheels/torch-{mm}/cu{cu}/repo.html')
        candidate_urls.append(f'https://data.dgl.ai/wheels/torch-{mm}/repo.html')
    candidate_urls.append('https://data.dgl.ai/wheels/repo.html')
    for url in candidate_urls:
        try:
            print(f'  trying DGL index {url}')
            pip('dgl', '-f', url)
            import dgl  # noqa: F401
            from dgl.nn.pytorch import SAGEConv  # noqa: F401
            print('  -> installed dgl', dgl.__version__)
            return
        except Exception as e:
            print(f'    failed: {e}')
    # last-resort PyPI (often CPU-only build)
    pip('dgl')
    import dgl  # noqa: F401
    print('  -> PyPI dgl', dgl.__version__)

install_dgl()

# Other lightweight deps
pip('python-louvain', 'leidenalg', 'python-igraph')
print('Installs done.')
''')


# ============================================================
# Cell 3: imports + GPU check
# ============================================================
code(r'''
# ===== Section 0b: imports & device =====
import os, io, gzip, zipfile, urllib.request, time, random, pickle, math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from itertools import chain
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import dgl
from dgl.nn.pytorch import SAGEConv

import igraph as ig
import leidenalg

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
print(f'Device: {DEVICE}  | dgl: {dgl.__version__}  | torch: {torch.__version__}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = '/content/data/raw' if os.path.exists('/content') else 'data/raw'
RES_DIR = '/content/results' if os.path.exists('/content') else 'results'
FIG_DIR = '/content/figures' if os.path.exists('/content') else 'figures'
for d in (DATA_DIR, RES_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

%matplotlib inline
''')


# ============================================================
# Cell 4: data loaders (embedded)
# ============================================================
code(r'''
# ===== Section 1a: data loaders (embedded copy of data_loaders.py) =====
def _download(url, dest, timeout=60):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f'  downloading {url}')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, 'wb') as f:
        f.write(r.read())


def _ensure_zip(url, zip_path, extract_dir, sentinel='.gml'):
    _download(url, zip_path)
    os.makedirs(extract_dir, exist_ok=True)
    has = any(fn.endswith(sentinel) for _, _, files in os.walk(extract_dir) for fn in files)
    if not has:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)


def load_karate():
    G = nx.karate_club_graph()
    G = nx.relabel.convert_node_labels_to_integers(G, label_attribute='orig_id')
    lbls = {i: 0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes}
    return G, lbls, 2


def _load_gml_graph(name, url):
    base = os.path.join(DATA_DIR, name)
    zip_p = os.path.join(base, f'{name}.zip')
    _ensure_zip(url, zip_p, base)
    gml = None
    for r, _, files in os.walk(base):
        for fn in files:
            if fn.endswith('.gml'):
                gml = os.path.join(r, fn); break
    with open(gml, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    idx = text.find('graph')
    if idx > 0: text = text[idx:]
    G = nx.parse_gml(io.StringIO(text), label='id')
    G = nx.Graph(G)
    G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
    return G


def load_football():
    G = _load_gml_graph('football', 'http://www-personal.umich.edu/~mejn/netdata/football.zip')
    lbls = {i: int(G.nodes[i]['value']) for i in G.nodes}
    return G, lbls, len(set(lbls.values()))


def load_polbooks():
    G = _load_gml_graph('polbooks', 'http://www-personal.umich.edu/~mejn/netdata/polbooks.zip')
    m = {'l': 0, 'n': 1, 'c': 2}
    lbls = {i: m[G.nodes[i]['value']] for i in G.nodes}
    return G, lbls, 3


def load_email_eu_core():
    base = os.path.join(DATA_DIR, 'email_eu_core')
    e_gz = os.path.join(base, 'email-Eu-core.txt.gz')
    l_gz = os.path.join(base, 'email-Eu-core-department-labels.txt.gz')
    _download('https://snap.stanford.edu/data/email-Eu-core.txt.gz', e_gz)
    _download('https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz', l_gz)
    G = nx.Graph()
    with gzip.open(e_gz, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            u, v = map(int, line.split())
            if u != v: G.add_edge(u, v)
    raw = {}
    with gzip.open(l_gz, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            n_id, dept = line.split()
            raw[int(n_id)] = int(dept)
    if not nx.is_connected(G):
        gcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc).copy()
    mp = {old: new for new, old in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mp)
    lbls = {mp[o]: raw.get(o, -1) for o in mp}
    lbls = {n: l for n, l in lbls.items() if l >= 0}
    return G, lbls, len(set(lbls.values()))


def generate_lfr(n, mu, tau1=3.0, tau2=1.5, average_degree=10,
                 min_community=20, seed=42, max_tries=8):
    last = None
    for t in range(max_tries):
        try:
            G = nx.LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu,
                                       average_degree=average_degree,
                                       min_community=min_community, seed=seed + t)
            G.remove_edges_from(nx.selfloop_edges(G))
            G = nx.Graph(G)
            G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
            comm_to_idx = {}
            lbls = {}
            for v in G.nodes():
                comm = frozenset(G.nodes[v]['community'])
                if comm not in comm_to_idx:
                    comm_to_idx[comm] = len(comm_to_idx)
                lbls[v] = comm_to_idx[comm]
            return G, lbls, len(comm_to_idx)
        except (nx.ExceededMaxIterations, RuntimeError) as e:
            last = e; continue
    raise RuntimeError(f'LFR failed: {last}')


def graph_summary(G, lbls, k_true):
    n = G.number_of_nodes(); m = G.number_of_edges()
    deg = [d for _, d in G.degree()]
    sizes = sorted([sum(1 for v, c in lbls.items() if c == cc)
                    for cc in set(lbls.values())])
    return dict(n=n, m=m, k_true=k_true, avg_deg=sum(deg)/max(n,1),
                min_size=sizes[0], med_size=int(np.median(sizes)),
                max_size=sizes[-1], connected=nx.is_connected(G))


print('data_loaders ready')
''')


# ============================================================
# Cell 5: QUBO + GNN architecture
# ============================================================
code(r'''
# ===== Section 1b: QUBO modularity + GNN architecture =====

# ---- Modularity matrix (used by both 2-comm and multi formulations) ----
def modularity_matrix(G):
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    deg = A.sum(axis=1)
    m = G.number_of_edges()
    B = A - np.outer(deg, deg) / (2.0 * m)
    return B, m, nodes


def gen_q_dict_modularity_2community(G):
    """Wang eq.(6) collapsed:  H(x) = -sum_{v,w} B_{vw} x_v x_w  =>  Q = -B."""
    B, _, _ = modularity_matrix(G)
    Q = -B
    n = B.shape[0]
    Q_dic = defaultdict(int)
    for i in range(n):
        for j in range(n):
            v = float(Q[i, j])
            if v != 0.0:
                Q_dic[(i, j)] = v
    return Q_dic


def gen_q_dict_modularity_multi(G, k, P=None):
    """Wang eq.(5): linearised i = v*k + c, returns upper-tri Q dict.
    NOTE: we never call this for n*k > ~5000 — instead use structured loss below
    (saves the (n*k)² dense matrix; on email-eu-core that's 7 GB)."""
    B, m, _ = modularity_matrix(G)
    n = B.shape[0]; two_m = 2.0 * m
    if P is None: P = 10.0 * float(np.max(np.abs(B)))
    Q = defaultdict(int)
    for v in range(n):
        Bvv = float(B[v, v])
        for c in range(k):
            i = v*k + c
            Q[(i, i)] += -Bvv / two_m - P
    for v in range(n):
        for w in range(v+1, n):
            Bvw = float(B[v, w])
            if Bvw == 0.0: continue
            coef = -Bvw / m
            for c in range(k):
                i = v*k + c; j = w*k + c
                Q[(i, j)] += coef
    for v in range(n):
        for c1 in range(k):
            for c2 in range(c1+1, k):
                i = v*k + c1; j = v*k + c2
                Q[(i, j)] += 2.0 * P
    Q = {k_: v for k_, v in Q.items() if v != 0}
    return Q


def linearize_index(v, c, k):  return v*k + c
def delinearize_index(i, k):   return i // k, i % k


# ---- GNN architecture ----
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
    def __init__(self, in_f, hd, k, dr, dev):
        super().__init__()
        self.dr = dr
        self.layers = nn.ModuleList()
        cur = in_f
        for h in [hd] if isinstance(hd, int) else hd:
            self.layers.append(SAGEResBlockMulti(cur, h).to(dev))
            self.layers.append(nn.LeakyReLU())
            cur = h
        self.layers.append(SAGEConv(cur, k, aggregator_type='mean').to(dev))

    def forward(self, g, h, h0, ew=None):
        h = torch.cat([h, h0], 1)
        for layer, norm in zip(self.layers[:-1][::2], self.layers[:-1][1::2]):
            h = norm(layer(g, h, ew))
        h = F.dropout(h, p=self.dr)
        h0n = self.layers[-1](g, h, ew)
        return F.softmax(h0n, dim=1), h0n


def get_gnn_multi(n_nodes, k, lr, dim_emb=10, hidden=50, dropout=0.5,
                  device=DEVICE, dtype=DTYPE):
    in_feats = dim_emb + 1*k + 4*dim_emb
    net = ResSAGEMulti(in_feats, hidden, k, dropout, device).type(dtype).to(device)
    embed = nn.Embedding(n_nodes, dim_emb).type(dtype).to(device)
    opt = torch.optim.Adam(chain(net.parameters(), embed.parameters()), lr=lr)
    return net, embed, opt


def pagerank_features(G, dim=10):
    feats = torch.zeros((G.number_of_nodes(), dim))
    pr = nx.pagerank(nx.Graph(G))
    for v, val in pr.items():
        feats[v, :] = val
    return feats


# ---- Structured loss (no full Q matrix) ----
def loss_func_multi(P, B_t, P_pen, m_e, epoch=0, alpha_ortho=0.0, alpha_collapse=0.0):
    """Total = QUBO + annealing + α_ortho·L_ortho + α_collapse·L_collapse.

    QUBO computed without forming the (n*k)² matrix:
        diag    : sum_v (-B_vv/(2m) - P_pen) · sum_c p_{v,c}^2
        mod_off : -trace(P^T B_off P) / (2m)
        constr  :  P_pen · sum_v ((sum_c p_{v,c})^2 - sum_c p_{v,c}^2)
    Equivalent bit-for-bit to p^T Q_upper p (verified locally).
    """
    n, k = P.shape
    B_diag = torch.diag(B_t)
    B_off = B_t - torch.diag(B_diag)
    diag_per_v = -B_diag / (2.0 * m_e) - P_pen
    diag_term = (diag_per_v * (P**2).sum(dim=1)).sum()
    trace_off = (P.T @ B_off @ P).diagonal().sum()
    mod_off = -trace_off / (2.0 * m_e)
    rs = P.sum(dim=1); rsq = (P**2).sum(dim=1)
    constr_off = P_pen * (rs**2 - rsq).sum()
    qubo = diag_term + mod_off + constr_off

    p = P.reshape(-1)
    annealing = (epoch / 1e4) * (p * (1 - p)).abs().sum()

    if alpha_ortho > 0:
        PtP = P.T @ P
        PtP_n = PtP / (torch.norm(PtP, p='fro') + 1e-10)
        I_n = torch.eye(k, device=P.device, dtype=P.dtype) / math.sqrt(k)
        ortho = torch.norm(PtP_n - I_n, p='fro') ** 2
    else:
        ortho = torch.tensor(0.0, device=P.device, dtype=P.dtype)

    if alpha_collapse > 0:
        col_sums = P.sum(dim=0)
        collapse = (math.sqrt(k) / n) * torch.norm(col_sums, p=2) - 1.0
    else:
        collapse = torch.tensor(0.0, device=P.device, dtype=P.dtype)

    total = qubo + annealing + alpha_ortho * ortho + alpha_collapse * collapse
    return total, dict(qubo=float(qubo.detach()),
                       annealing=float(annealing.detach()),
                       ortho=float(ortho.detach()),
                       collapse=float(collapse.detach()))


def run_gnn_training_multi(G, g_dgl, B_t, P_pen, m_e, k,
                            net, embed, optimizer,
                            number_epochs, tol=1e-4, patience=1000,
                            alpha_ortho=0.0, alpha_collapse=0.0, log_every=1000):
    n = g_dgl.number_of_nodes()
    dtype = B_t.dtype; device = B_t.device

    src, dst = g_dgl.edges()
    edge_weight = (-B_t)[src, dst]

    inputs = torch.rand((n, 10), dtype=dtype, device=device)
    walk = pagerank_features(G, 2 * inputs.shape[1]).to(device).type(dtype)
    inputs = torch.cat([inputs, torch.ones_like(inputs), torch.ones_like(inputs), walk], 1)
    h0 = torch.zeros(n, k, device=device, dtype=dtype)

    init_assn = torch.zeros(n, dtype=torch.long, device=device)
    X0 = torch.zeros(n, k, dtype=dtype, device=device); X0[:, 0] = 1.0
    with torch.no_grad():
        best_proj_loss = loss_func_multi(X0, B_t, P_pen, m_e, alpha_ortho=alpha_ortho,
                                         alpha_collapse=alpha_collapse)[0].detach()
    best_assignment = init_assn.clone()
    best_epoch = 0
    history = []

    prev_loss = 1.0; bad = 0
    t0 = time.time()
    for epoch in range(number_epochs):
        probs, h0 = net(g_dgl, inputs, h0.detach(), edge_weight)
        loss, comps = loss_func_multi(probs, B_t, P_pen, m_e, epoch=epoch,
                                      alpha_ortho=alpha_ortho, alpha_collapse=alpha_collapse)
        lv = loss.detach().item()

        with torch.no_grad():
            assn = probs.argmax(dim=1)
            X_proj = torch.zeros_like(probs)
            X_proj.scatter_(1, assn.unsqueeze(1), 1.0)
            pl, _ = loss_func_multi(X_proj, B_t, P_pen, m_e,
                                    alpha_ortho=alpha_ortho, alpha_collapse=alpha_collapse)
            pl = pl.detach()
            if pl < best_proj_loss:
                best_proj_loss = pl
                best_assignment = assn.detach().clone()
                best_epoch = epoch

        if epoch % log_every == 0:
            history.append(dict(epoch=epoch, **comps, total=lv,
                                best_proj=float(best_proj_loss)))

        if (abs(lv - prev_loss) <= tol) or ((lv - prev_loss) > 0):
            bad += 1
        else:
            bad = 0
        if bad >= patience: break
        prev_loss = lv

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()

    return dict(best_epoch=best_epoch,
                best_assignment=best_assignment.detach().cpu().numpy().astype(int),
                history=history, time=time.time() - t0)


def assignment_to_communities(assn, nodes):
    g = defaultdict(set)
    for v, l in zip(nodes, assn):
        g[int(l)].add(v)
    return [frozenset(s) for s in g.values() if len(s) > 0]


def is_collapse(assn, thresh=0.8):
    return np.bincount(assn.astype(int)).max() / len(assn) >= thresh


def evaluate(G, assn, lbls, nodes):
    comms = assignment_to_communities(assn, nodes)
    mod = 0.0 if len(comms) <= 1 else float(nx.community.modularity(G, comms))
    truth = np.array([lbls[v] for v in nodes])
    nmi = float(normalized_mutual_info_score(truth, assn))
    return dict(mod=mod, nmi=nmi, n_used=len(comms), collapsed=bool(is_collapse(assn)))


print('QUBO + GNN architecture ready')
''')


# ============================================================
# Cell 6: load all 7 graphs
# ============================================================
md("# Section 2 — Load all 7 graphs")

code(r'''
loaders = {
    'karate':         load_karate,
    'football':       load_football,
    'polbooks':       load_polbooks,
    'lfr_n200_mu0.1': lambda: generate_lfr(n=200, mu=0.1),
    'lfr_n200_mu0.3': lambda: generate_lfr(n=200, mu=0.3),
    'lfr_n500_mu0.3': lambda: generate_lfr(n=500, mu=0.3),
    'email_eu_core':  load_email_eu_core,
}
GRAPHS = {}
GRAPH_STATS = {}
for name, fn in loaders.items():
    print(f'loading {name}...')
    G, lbls, k = fn()
    GRAPHS[name] = (G, lbls, k)
    GRAPH_STATS[name] = graph_summary(G, lbls, k)

stats_df = pd.DataFrame(GRAPH_STATS).T[['n','m','k_true','avg_deg',
                                        'min_size','med_size','max_size','connected']]
stats_df.to_csv(os.path.join(RES_DIR, 'graph_stats.csv'))
print()
print(stats_df)
''')


# ============================================================
# Cell 7: baselines (Louvain + Leiden)
# ============================================================
md("# Section 3 — Baselines (Louvain & Leiden)")

code(r'''
def run_louvain(G, lbls, seed=SEED):
    nodes = list(G.nodes())
    t0 = time.time()
    comms = nx.community.louvain_communities(G, seed=seed)
    el = time.time() - t0
    mod = float(nx.community.modularity(G, comms))
    n2c = {v: i for i, c in enumerate(comms) for v in c}
    pred = [n2c[v] for v in nodes]
    truth = [lbls[v] for v in nodes]
    return dict(method='Louvain', mod=mod,
                nmi=float(normalized_mutual_info_score(truth, pred)),
                k_found=len(comms), time=el)


def run_leiden(G, lbls, seed=SEED):
    nodes = list(G.nodes())
    g_ig = ig.Graph.TupleList(G.edges(), directed=False)
    n2i = {v['name']: i for i, v in enumerate(g_ig.vs)}
    i2n = {i: v['name'] for i, v in enumerate(g_ig.vs)}
    t0 = time.time()
    part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition, seed=seed)
    el = time.time() - t0
    pd_ = {i2n[i]: c for i, c in enumerate(part.membership)}
    pred = [pd_[v] for v in nodes]
    truth = [lbls[v] for v in nodes]
    return dict(method='Leiden', mod=part.modularity,
                nmi=float(normalized_mutual_info_score(truth, pred)),
                k_found=len(set(part.membership)), time=el)


bl = []
for name, (G, lbls, _) in GRAPHS.items():
    for fn in (run_louvain, run_leiden):
        r = fn(G, lbls); r['graph'] = name; bl.append(r)
baseline_df = pd.DataFrame(bl)
baseline_df.to_csv(os.path.join(RES_DIR, 'baselines.csv'), index=False)
print(baseline_df[['graph','method','mod','nmi','k_found','time']].to_string(index=False))
''')


# ============================================================
# Cell 8: main experiment (configurable)
# ============================================================
md("# Section 4 — Main experiment: 3 strategies × 7 graphs × 10 shots × 10000 epochs")

code(r'''
# Tweak these to control runtime ===========================================
N_SHOTS_DEFAULT = 10
EPOCHS_DEFAULT  = 10000
# Per-graph overrides for the largest dataset:
RUN_CONFIG = {'email_eu_core': dict(n_shots=5, epochs=5000)}
# =========================================================================

STRATEGIES = [
    ('baseline',  dict(alpha_ortho=0.0, alpha_collapse=0.0)),
    ('ortho',     dict(alpha_ortho=0.1, alpha_collapse=0.0)),
    ('collapse',  dict(alpha_ortho=0.0, alpha_collapse=1.0)),
]

records = []
trace_examples = {}

t_global = time.time()
for graph_name, (G, lbls, k_true) in GRAPHS.items():
    cfg = RUN_CONFIG.get(graph_name, dict(n_shots=N_SHOTS_DEFAULT, epochs=EPOCHS_DEFAULT))
    n_shots, epochs = cfg['n_shots'], cfg['epochs']
    print(f'\n##### {graph_name}  (n={G.number_of_nodes()} k_true={k_true} '
          f'shots={n_shots} epochs={epochs}) #####')

    nodes = list(G.nodes()); n_nodes = len(nodes)
    B_np, m_e, _ = modularity_matrix(G)
    B_t = torch.tensor(B_np, dtype=DTYPE, device=DEVICE)
    P_pen = 1.5 * float(np.max(np.abs(B_np)))
    g_dgl = dgl.from_networkx(G).to(DEVICE)
    k = k_true

    for strat_name, strat_args in STRATEGIES:
        best_mod = -1e9; best_hist = None
        for shot in range(n_shots):
            sd = SEED + shot
            random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
            if DEVICE.type == 'cuda':
                torch.cuda.manual_seed_all(sd)
            net, embed, opt = get_gnn_multi(n_nodes, k, lr=0.014)
            res = run_gnn_training_multi(G, g_dgl, B_t, P_pen, m_e, k, net, embed, opt,
                                          number_epochs=epochs, **strat_args)
            ev = evaluate(G, res['best_assignment'], lbls, nodes)
            records.append(dict(graph=graph_name, strategy=strat_name, shot=shot,
                                seed=sd, k=k, best_epoch=res['best_epoch'],
                                **ev, time=res['time']))
            if ev['mod'] > best_mod:
                best_mod, best_hist = ev['mod'], res['history']
        trace_examples[(graph_name, strat_name)] = best_hist
        sub = [r for r in records if r['graph']==graph_name and r['strategy']==strat_name]
        mods = np.array([r['mod'] for r in sub])
        col = np.mean([r['collapsed'] for r in sub])
        print(f'  {strat_name:>9}: mod_best={mods.max():+.4f}, '
              f'mod_mean={mods.mean():+.4f}±{mods.std():.4f}, collapse={col:.2f}')

raw_df = pd.DataFrame(records)
raw_df.to_csv(os.path.join(RES_DIR, 'comparison_raw.csv'), index=False)
print(f'\nTotal time: {(time.time()-t_global)/60:.1f} min')
''')


# ============================================================
# Cell 9: summary table + plots
# ============================================================
md("# Section 5 — Summary table & plots")

code(r'''
# Aggregate
summary_rows = []
for (g, s), grp in raw_df.groupby(['graph', 'strategy']):
    summary_rows.append(dict(graph=g, strategy=s,
        mod_best=grp['mod'].max(), mod_mean=grp['mod'].mean(), mod_std=grp['mod'].std(ddof=0),
        nmi_best=grp['nmi'].max(), nmi_mean=grp['nmi'].mean(), nmi_std=grp['nmi'].std(ddof=0),
        collapse_rate=grp['collapsed'].mean(), mean_used=grp['n_used'].mean(),
        time_mean=grp['time'].mean()))
for _, r in baseline_df.iterrows():
    summary_rows.append(dict(graph=r['graph'], strategy=r['method'],
        mod_best=r['mod'], mod_mean=r['mod'], mod_std=0.0,
        nmi_best=r['nmi'], nmi_mean=r['nmi'], nmi_std=0.0,
        collapse_rate=0.0, mean_used=r['k_found'], time_mean=r['time']))
big_summary = pd.DataFrame(summary_rows)
big_summary.to_csv(os.path.join(RES_DIR, 'comparison_summary.csv'), index=False)

strat_order = ['Louvain', 'Leiden', 'baseline', 'ortho', 'collapse']
graph_order = list(GRAPHS.keys())
print('=' * 110)
for g in graph_order:
    for s in strat_order:
        sub = big_summary[(big_summary['graph']==g) & (big_summary['strategy']==s)]
        if len(sub):
            r = sub.iloc[0]
            print(f"{g:<18} {s:>10} mod_best={r['mod_best']:.4f} "
                  f"mod_mean={r['mod_mean']:.4f}±{r['mod_std']:.4f} "
                  f"nmi_best={r['nmi_best']:.4f} collapse={r['collapse_rate']:.2f} "
                  f"used={r['mean_used']:.1f} t={r['time_mean']:.1f}s")
    print()
''')


# ============================================================
# Cell 10: plots
# ============================================================
code(r'''
strat_palette = {'Louvain': '#888', 'Leiden': '#444',
                 'baseline': '#a6cee3', 'ortho': '#1f78b4', 'collapse': '#e31a1c'}

# Fig1: modularity bar
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(graph_order)); width = 0.16
for i, m_ in enumerate(strat_order):
    vals = []
    for g in graph_order:
        s = big_summary[(big_summary['graph']==g) & (big_summary['strategy']==m_)]
        vals.append(s['mod_best'].values[0] if len(s) else np.nan)
    ax.bar(x + (i-2)*width, vals, width, label=m_, color=strat_palette.get(m_, 'gray'))
ax.set_xticks(x); ax.set_xticklabels(graph_order, rotation=20, ha='right')
ax.set_ylabel('Modularity (best of shots)'); ax.legend(loc='lower right')
ax.set_title('Modularity by graph / method'); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1_modularity_bar.png'), dpi=130)
plt.show()

# Fig2: NMI bar
fig, ax = plt.subplots(figsize=(13, 5))
for i, m_ in enumerate(strat_order):
    vals = []
    for g in graph_order:
        s = big_summary[(big_summary['graph']==g) & (big_summary['strategy']==m_)]
        vals.append(s['nmi_best'].values[0] if len(s) else np.nan)
    ax.bar(x + (i-2)*width, vals, width, label=m_, color=strat_palette.get(m_, 'gray'))
ax.set_xticks(x); ax.set_xticklabels(graph_order, rotation=20, ha='right')
ax.set_ylabel('NMI vs ground truth'); ax.legend(loc='lower right')
ax.set_title('NMI by graph / method'); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_nmi_bar.png'), dpi=130)
plt.show()

# Fig3: collapse-rate heatmap
qstrats = ['baseline', 'ortho', 'collapse']
heat = np.zeros((len(graph_order), len(qstrats)))
for i, g in enumerate(graph_order):
    for j, s in enumerate(qstrats):
        sub = big_summary[(big_summary['graph']==g) & (big_summary['strategy']==s)]
        heat[i, j] = sub['collapse_rate'].values[0] if len(sub) else np.nan
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(heat, cmap='Reds', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(qstrats))); ax.set_xticklabels(qstrats)
ax.set_yticks(range(len(graph_order))); ax.set_yticklabels(graph_order)
for i in range(len(graph_order)):
    for j in range(len(qstrats)):
        ax.text(j, i, f'{heat[i,j]:.2f}', ha='center', va='center',
                color='white' if heat[i,j] > 0.5 else 'black')
plt.colorbar(im, ax=ax, label='Trivial-collapse rate')
ax.set_title('Collapse rate by graph and QIGNN strategy')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_collapse_heatmap.png'), dpi=130)
plt.show()
''')


# ============================================================
# Cell 11: optional HP sweep
# ============================================================
md(r'''
# Section 6 — (опционально) HP sweep на 3 ключевых графах

Закомментирован по умолчанию: ~30–40 мин на A100.
Раскомментируй блок ниже, чтобы запустить расширенный sweep.
''')

code(r'''
RUN_HP_SWEEP = False   # set True to enable

if RUN_HP_SWEEP:
    LRS    = [0.001, 0.005, 0.014, 0.05]
    ALPHAS = [0.0, 0.05, 0.1, 0.5, 1.0]
    EPOCHS = [3000, 10000]
    SHOTS  = [SEED + i for i in range(10)]
    HP_GRAPHS = {
        'karate':         {'k_test': 5},
        'polbooks':       {'k_test': 3},
        'lfr_n200_mu0.1': {'k_test': 5},
    }
    rows = []
    t_hp = time.time()
    for graph_name, spec in HP_GRAPHS.items():
        G, lbls, _ = GRAPHS[graph_name]
        nodes = list(G.nodes()); n = len(nodes); k = spec['k_test']
        B_np, m_e, _ = modularity_matrix(G)
        B_t = torch.tensor(B_np, dtype=DTYPE, device=DEVICE)
        P_pen = 1.5 * float(np.max(np.abs(B_np)))
        g_dgl = dgl.from_networkx(G).to(DEVICE)
        for lr in LRS:
            for a in ALPHAS:
                for ep in EPOCHS:
                    for sd in SHOTS:
                        random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
                        if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(sd)
                        net, embed, opt = get_gnn_multi(n, k, lr=lr)
                        res = run_gnn_training_multi(G, g_dgl, B_t, P_pen, m_e, k,
                                                      net, embed, opt, number_epochs=ep,
                                                      alpha_ortho=a, alpha_collapse=0.0)
                        ev = evaluate(G, res['best_assignment'], lbls, nodes)
                        rows.append(dict(graph=graph_name, k=k, lr=lr,
                                         alpha_ortho=a, epochs=ep, shot=sd,
                                         **ev, time=res['time']))
        print(f'  {graph_name} done')
    hp_df = pd.DataFrame(rows)
    hp_df.to_csv(os.path.join(RES_DIR, 'hp_sweep_gpu.csv'), index=False)
    print(f'HP sweep wall time: {(time.time()-t_hp)/60:.1f} min, rows={len(hp_df)}')
else:
    print('Skipped HP sweep (set RUN_HP_SWEEP=True to run).')
''')


# ============================================================
# Cell 12: optional scaling experiment
# ============================================================
md(r'''
# Section 7 — (опционально) Scaling experiment на больших LFR

Включи `RUN_SCALING=True` чтобы прогнать LFR n=500/1000/2000.
А100 справляется за ~10–15 мин. Покажет насколько GNN-time vs Louvain-time
расходятся с ростом n.
''')

code(r'''
RUN_SCALING = False

if RUN_SCALING:
    sizes = [500, 1000, 2000]
    rows = []
    t_sc = time.time()
    for n in sizes:
        G, lbls, k_true = generate_lfr(n=n, mu=0.3, average_degree=15,
                                       min_community=30, seed=SEED)
        nodes = list(G.nodes())
        # Louvain timing
        t0 = time.time()
        comms = nx.community.louvain_communities(G, seed=SEED)
        louv_t = time.time() - t0
        louv_mod = float(nx.community.modularity(G, comms))
        # QIGNN timing (single shot, baseline)
        B_np, m_e, _ = modularity_matrix(G)
        B_t = torch.tensor(B_np, dtype=DTYPE, device=DEVICE)
        P_pen = 1.5 * float(np.max(np.abs(B_np)))
        g_dgl = dgl.from_networkx(G).to(DEVICE)
        net, embed, opt = get_gnn_multi(n, k_true, lr=0.014)
        res = run_gnn_training_multi(G, g_dgl, B_t, P_pen, m_e, k_true,
                                      net, embed, opt, number_epochs=5000)
        ev = evaluate(G, res['best_assignment'], lbls, nodes)
        rows.append(dict(n=n, k_true=k_true, louv_mod=louv_mod, louv_t=louv_t,
                         qignn_mod=ev['mod'], qignn_nmi=ev['nmi'],
                         qignn_t=res['time']))
        print(f'  n={n}: Louvain mod={louv_mod:.4f} ({louv_t:.2f}s)  '
              f'QIGNN mod={ev["mod"]:.4f} ({res["time"]:.1f}s)')
    sc_df = pd.DataFrame(rows)
    sc_df.to_csv(os.path.join(RES_DIR, 'scaling_lfr.csv'), index=False)
    print(f'\nScaling wall time: {(time.time()-t_sc)/60:.1f} min')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(sc_df['n'], sc_df['louv_mod'], 'o-', label='Louvain')
    axes[0].plot(sc_df['n'], sc_df['qignn_mod'], 's-', label='QIGNN baseline')
    axes[0].set_xlabel('n (graph size)'); axes[0].set_ylabel('Modularity')
    axes[0].grid(alpha=0.3); axes[0].legend(); axes[0].set_title('Quality vs n')
    axes[1].plot(sc_df['n'], sc_df['louv_t'], 'o-', label='Louvain')
    axes[1].plot(sc_df['n'], sc_df['qignn_t'], 's-', label='QIGNN baseline')
    axes[1].set_xlabel('n (graph size)'); axes[1].set_ylabel('Runtime (s)')
    axes[1].set_yscale('log'); axes[1].grid(alpha=0.3); axes[1].legend()
    axes[1].set_title('Runtime vs n (log scale)')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_scaling.png'), dpi=130)
    plt.show()
else:
    print('Skipped scaling (set RUN_SCALING=True to run).')
''')


# ============================================================
# Cell 13: download artifacts
# ============================================================
md("# Section 8 — Download all CSV/PNG (Colab only)")

code(r'''
import shutil
zip_out = '/content/results_bundle.zip' if os.path.exists('/content') else 'results_bundle.zip'
with zipfile.ZipFile(zip_out, 'w', zipfile.ZIP_DEFLATED) as zf:
    for d in (RES_DIR, FIG_DIR):
        for r, _, files in os.walk(d):
            for fn in files:
                p = os.path.join(r, fn)
                zf.write(p, os.path.relpath(p, os.path.dirname(d)))
print(f'Wrote {zip_out} ({os.path.getsize(zip_out)/1e6:.2f} MB)')

try:
    from google.colab import files  # type: ignore
    files.download(zip_out)
except Exception:
    print('Not on Colab — skip auto-download. The bundle is at:', zip_out)
''')


nb = new_notebook()
nb.cells = CELLS
nb.metadata = {
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
    'language_info': {'name': 'python', 'version': '3.10'},
    'colab': {'provenance': [], 'gpuType': 'A100'},
    'accelerator': 'GPU',
}
out_path = '/Users/sergej/Дилом/community_detection_qubo_gnn/colab_full.ipynb'
with open(out_path, 'w') as f:
    nbf.write(nb, f)
print(f'Wrote {out_path} with {len(CELLS)} cells')
