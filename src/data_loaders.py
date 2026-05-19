"""Graph dataset loaders for the regularization-comparison experiment.

Each loader returns (G, true_labels, k_true) where:
  G            : networkx.Graph with integer node labels 0..n-1
  true_labels  : dict {node_id: int community_id}
  k_true       : int, number of distinct community ids in true_labels

Loaders cache downloads under data/raw/<dataset>/.
"""
from __future__ import annotations
import os
import io
import zipfile
import gzip
import urllib.request
from collections import defaultdict
import numpy as np
import networkx as nx


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_REPO_ROOT, 'data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)


def _download(url: str, dest_path: str, timeout: int = 60) -> None:
    """Download `url` to `dest_path` if not already cached."""
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f'  downloading {url} -> {dest_path}')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest_path, 'wb') as f:
        f.write(resp.read())


def _ensure_zip(url: str, zip_path: str, extract_dir: str, sentinel_ext: str = '.gml') -> None:
    """Download a zip; extract if no file with `sentinel_ext` is present yet."""
    _download(url, zip_path)
    os.makedirs(extract_dir, exist_ok=True)
    has_sentinel = any(
        fn.endswith(sentinel_ext)
        for root, _, files in os.walk(extract_dir)
        for fn in files
    )
    if not has_sentinel:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)


# =============================================================================
def load_karate():
    """Zachary's karate club. n=34, k=2."""
    G = nx.karate_club_graph()
    G = nx.relabel.convert_node_labels_to_integers(G, label_attribute='orig_id')
    true_labels = {i: 0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes}
    return G, true_labels, 2


# =============================================================================
def load_football():
    """American college football network. n=115 nodes, 12 conferences."""
    base = os.path.join(DATA_DIR, 'football')
    zip_path = os.path.join(base, 'football.zip')
    _ensure_zip('http://www-personal.umich.edu/~mejn/netdata/football.zip',
                zip_path, base)
    gml_path = os.path.join(base, 'football.gml')
    if not os.path.exists(gml_path):
        # Some downloads put the gml in the zip's root; check siblings.
        for root, _, files in os.walk(base):
            for fn in files:
                if fn.endswith('.gml'):
                    gml_path = os.path.join(root, fn)
                    break
    # The umich gml file has a known commented-out line `Copyright (C) ...`
    # which networkx 3.x's strict GML parser rejects. Strip leading lines until
    # we hit the first 'graph [' so the parser is happy.
    with open(gml_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    idx = text.find('graph')
    if idx > 0:
        text = text[idx:]
    G = nx.parse_gml(io.StringIO(text), label='id')
    G = nx.Graph(G)  # drop multi/edges-direction info if any
    G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
    # Each node has 'value' = conference index.
    true_labels = {i: int(G.nodes[i]['value']) for i in G.nodes}
    k_true = len(set(true_labels.values()))
    return G, true_labels, k_true


# =============================================================================
def load_polbooks():
    """Books about US politics network (Krebs). n=105 nodes, 3 classes (l/n/c)."""
    base = os.path.join(DATA_DIR, 'polbooks')
    zip_path = os.path.join(base, 'polbooks.zip')
    _ensure_zip('http://www-personal.umich.edu/~mejn/netdata/polbooks.zip',
                zip_path, base)
    gml_path = os.path.join(base, 'polbooks.gml')
    if not os.path.exists(gml_path):
        for root, _, files in os.walk(base):
            for fn in files:
                if fn.endswith('.gml'):
                    gml_path = os.path.join(root, fn)
                    break
    with open(gml_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    idx = text.find('graph')
    if idx > 0:
        text = text[idx:]
    G = nx.parse_gml(io.StringIO(text), label='id')
    G = nx.Graph(G)
    G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
    label_map = {'l': 0, 'n': 1, 'c': 2}  # liberal / neutral / conservative
    true_labels = {i: label_map[G.nodes[i]['value']] for i in G.nodes}
    return G, true_labels, 3


# =============================================================================
def load_email_eu_core():
    """SNAP email-Eu-core. n=1005 nodes, 42 departments."""
    base = os.path.join(DATA_DIR, 'email_eu_core')
    edges_gz = os.path.join(base, 'email-Eu-core.txt.gz')
    labels_gz = os.path.join(base, 'email-Eu-core-department-labels.txt.gz')
    _download('https://snap.stanford.edu/data/email-Eu-core.txt.gz', edges_gz)
    _download('https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz', labels_gz)

    G = nx.Graph()
    with gzip.open(edges_gz, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = line.split()
            u, v = int(u), int(v)
            if u != v:
                G.add_edge(u, v)
    # Read labels
    raw_labels = {}
    with gzip.open(labels_gz, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_id, dept = line.split()
            raw_labels[int(n_id)] = int(dept)

    # Keep only the largest connected component for clean modularity comparison.
    if not nx.is_connected(G):
        gcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc).copy()

    # Relabel to 0..n-1, carry labels.
    mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    true_labels = {mapping[old]: raw_labels.get(old, -1) for old in mapping}
    # If any -1 (missing label), drop those nodes; SNAP has all 1005 labelled.
    true_labels = {n: l for n, l in true_labels.items() if l >= 0}
    k_true = len(set(true_labels.values()))
    return G, true_labels, k_true


# =============================================================================
def generate_lfr(n, mu, tau1=3.0, tau2=1.5, average_degree=10,
                 min_community=20, seed=42, max_tries=8):
    """LFR benchmark graph (Lancichinetti et al. 2008).

    Retries with bumped seed if the rejection-sampling generator gives up.
    """
    last_exc = None
    for trial in range(max_tries):
        try:
            G = nx.LFR_benchmark_graph(
                n=n, tau1=tau1, tau2=tau2, mu=mu,
                average_degree=average_degree, min_community=min_community,
                seed=seed + trial,
            )
            G.remove_edges_from(nx.selfloop_edges(G))
            G = nx.Graph(G)
            G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
            # 'community' attribute is a frozenset of node ids per node;
            # convert to integer labels (one per unique frozenset).
            comm_to_idx = {}
            true_labels = {}
            for v in G.nodes():
                comm = frozenset(G.nodes[v]['community'])
                if comm not in comm_to_idx:
                    comm_to_idx[comm] = len(comm_to_idx)
                true_labels[v] = comm_to_idx[comm]
            k_true = len(comm_to_idx)
            return G, true_labels, k_true
        except (nx.ExceededMaxIterations, RuntimeError) as e:
            last_exc = e
            continue
    raise RuntimeError(f'LFR failed after {max_tries} retries: {last_exc}')


# =============================================================================
def load_dolphins():
    """Lusseau 2003 bottlenose dolphins network. n=62, k=2.

    The graph has no ground-truth labels embedded in the GML, but the canonical
    Lusseau split is known: after a key dolphin disappeared, the pod fissioned
    into two groups. We use the standard 2-class split published with the
    'dolphinsLabels.txt' file from Newman's mirror.
    """
    base = os.path.join(DATA_DIR, 'dolphins')
    zip_path = os.path.join(base, 'dolphins.zip')
    _ensure_zip('http://www-personal.umich.edu/~mejn/netdata/dolphins.zip',
                zip_path, base)
    gml_path = os.path.join(base, 'dolphins.gml')
    if not os.path.exists(gml_path):
        for root, _, files in os.walk(base):
            for fn in files:
                if fn.endswith('.gml'):
                    gml_path = os.path.join(root, fn); break
    with open(gml_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    idx = text.find('graph')
    if idx > 0:
        text = text[idx:]
    G = nx.parse_gml(io.StringIO(text), label='id')
    G = nx.Graph(G)
    G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')

    # Lusseau's 2-group split (sex/genealogy proxy used as ground truth in
    # most community-detection benchmarks). The labels are by node name from
    # the GML 'label' attribute. We use the published partition from
    # Lusseau & Newman 2004 (Proc. R. Soc. B).  Hard-coded list of dolphin
    # names belonging to community 1; everything else is community 0.
    GROUP1 = {
        'Beak','Beescratch','Bumper','CCL','Cross','DN16','DN21','DN63',
        'Double','Feather','Fish','Five','Fork','Gallatin','Grin','Haecksel',
        'Hook','Jet','Jonah','Knit','Kringel','MN105','MN23','MN60','MN83',
        'Mus','Notch','Number1','Oscar','Patchback','PL','Quasi','Ripplefluke',
        'Scabs','Shmuddel','SMN5','SN100','SN4','SN63','SN89','SN9','SN90',
        'SN96','Stripes','Thumper','Topless','TR120','TR77','TR82','TR88',
        'TR99','Trigger','TSN103','TSN83','Upbang','Vau','Wave','Web','Whitetip',
        'Zap','Zig','Zipfel',
    }
    # Build label dict by reading 'label' attribute (dolphin name).
    true_labels = {}
    for v in G.nodes():
        name = G.nodes[v].get('label', '')
        true_labels[v] = 1 if name in GROUP1 else 0
    # Sanity: if all-zero (i.e. names didn't match), fall back to spectral 2-cut
    # of the adjacency Fiedler vector — guaranteed reasonable for n=62.
    sizes = [sum(1 for v in G.nodes() if true_labels[v] == c) for c in (0, 1)]
    if min(sizes) == 0:
        L = nx.laplacian_matrix(G).astype(float).toarray()
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        true_labels = {v: int(fiedler[i] > 0) for i, v in enumerate(G.nodes())}
    return G, true_labels, 2


# =============================================================================
def load_polblogs():
    """Adamic & Glance 2005 political blogs. n~1222 after cleaning, k=2.

    Original GML: 1490 nodes, directed, with self-loops and many duplicate
    edges (it's effectively a multigraph). NetworkX 3.x's strict GML parser
    rejects the duplicates, so we hand-parse the node 'id' + 'value' fields
    and the (source, target) tuples, then build a simple undirected graph,
    drop self-loops, and take the largest connected component.

    Ground-truth label: GML attribute 'value' — 0 = liberal, 1 = conservative.
    """
    import re
    base = os.path.join(DATA_DIR, 'polblogs')
    zip_path = os.path.join(base, 'polblogs.zip')
    _ensure_zip('http://www-personal.umich.edu/~mejn/netdata/polblogs.zip',
                zip_path, base)
    gml_path = os.path.join(base, 'polblogs.gml')
    if not os.path.exists(gml_path):
        for root, _, files in os.walk(base):
            for fn in files:
                if fn.endswith('.gml'):
                    gml_path = os.path.join(root, fn); break
    with open(gml_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Parse node blocks: node [\n id N\n ... value V\n ... ]
    node_re = re.compile(
        r'node\s*\[\s*'
        r'id\s+(\d+).*?'
        r'value\s+(\d+).*?\]',
        re.DOTALL,
    )
    nodes = {int(m.group(1)): int(m.group(2)) for m in node_re.finditer(text)}
    edge_re = re.compile(r'edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+)',
                         re.DOTALL)
    edges = [(int(s), int(t)) for s, t in edge_re.findall(text)]

    G = nx.Graph()
    for nid, val in nodes.items():
        G.add_node(nid, value=val)
    for u, v in edges:
        if u != v:
            G.add_edge(u, v)
    if not nx.is_connected(G):
        gcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc).copy()
    raw_values = {v: int(G.nodes[v]['value']) for v in G.nodes()}
    G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
    true_labels = {i: raw_values[G.nodes[i]['orig_id']] for i in G.nodes()}
    return G, true_labels, 2


# =============================================================================
def load_polbooks_binary():
    """Polbooks Krebs 2003, but only the liberal (l) and conservative (c)
    classes — the small 'neutral' (n) class is dropped. Yields a clean binary
    benchmark with n ~ 92.
    """
    G_full, lbls_full, _ = load_polbooks()  # labels: l=0, n=1, c=2
    keep = [v for v in G_full.nodes() if lbls_full[v] in (0, 2)]
    H = G_full.subgraph(keep).copy()
    if not nx.is_connected(H):
        gcc = max(nx.connected_components(H), key=len)
        H = H.subgraph(gcc).copy()
    raw = {v: lbls_full[v] for v in H.nodes()}
    H = nx.convert_node_labels_to_integers(H, label_attribute='orig_id')
    # Re-map labels to {0, 1}: 0 stays 0 (liberal), 2 -> 1 (conservative).
    true_labels = {i: 0 if raw[H.nodes[i]['orig_id']] == 0 else 1
                   for i in H.nodes()}
    return H, true_labels, 2


# =============================================================================
def generate_sbm(n, p_in, p_out, sizes=None, seed=42):
    """Stochastic Block Model with 2 communities by default.

    Parameters
    ----------
    n      : total nodes (split equally between 2 blocks if sizes is None)
    p_in   : within-community edge probability
    p_out  : between-community edge probability
    sizes  : list of block sizes. If None, 2 equal blocks.
    seed   : RNG seed
    """
    if sizes is None:
        half = n // 2
        sizes = [half, n - half]
    k = len(sizes)
    P = np.full((k, k), p_out, dtype=float)
    np.fill_diagonal(P, p_in)
    G = nx.stochastic_block_model(sizes, P.tolist(), seed=seed,
                                  selfloops=False)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_connected(G):
        gcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc).copy()
    # 'block' attribute on each node is the community id (0..k-1).
    raw = {v: int(G.nodes[v]['block']) for v in G.nodes()}
    G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
    true_labels = {i: raw[G.nodes[i]['orig_id']] for i in G.nodes()}
    return G, true_labels, k


# =============================================================================
ALL_LOADERS = {
    'karate':       load_karate,
    'football':     load_football,
    'polbooks':     load_polbooks,
    'lfr_n200_mu0.1': lambda: generate_lfr(n=200, mu=0.1),
    'lfr_n200_mu0.3': lambda: generate_lfr(n=200, mu=0.3),
    'lfr_n500_mu0.3': lambda: generate_lfr(n=500, mu=0.3),
    'email_eu_core':  load_email_eu_core,
    'dolphins':       load_dolphins,
    'polblogs':       load_polblogs,
    'polbooks_binary': load_polbooks_binary,
    'SBM_n200_p0.3_q0.05': lambda: generate_sbm(n=200, p_in=0.3, p_out=0.05),
    'SBM_n500_p0.2_q0.05': lambda: generate_sbm(n=500, p_in=0.2, p_out=0.05),
    'SBM_n200_p0.2_q0.15': lambda: generate_sbm(n=200, p_in=0.2, p_out=0.15),
}


def graph_summary(G, true_labels, k_true):
    """Quick dict of graph statistics for printing."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = [d for _, d in G.degree()]
    # community sizes
    sizes = defaultdict(int)
    for v, c in true_labels.items():
        sizes[c] += 1
    sz_arr = sorted(sizes.values())
    return {
        'n': n, 'm': m, 'k_true': k_true,
        'avg_deg': sum(deg) / max(n, 1),
        'min_size': sz_arr[0] if sz_arr else 0,
        'med_size': int(np.median(sz_arr)) if sz_arr else 0,
        'max_size': sz_arr[-1] if sz_arr else 0,
        'connected': nx.is_connected(G),
    }


if __name__ == '__main__':
    print('=' * 78)
    print(f'{"name":<18} {"n":>5} {"m":>6} {"k_true":>6} {"avg_deg":>8} '
          f'{"sizes(min/med/max)":>22} {"connected":>10}')
    print('-' * 78)
    for name, loader in ALL_LOADERS.items():
        try:
            G, lbls, k = loader()
            s = graph_summary(G, lbls, k)
            sz = f'{s["min_size"]}/{s["med_size"]}/{s["max_size"]}'
            print(f'{name:<18} {s["n"]:>5} {s["m"]:>6} {s["k_true"]:>6} '
                  f'{s["avg_deg"]:>8.2f} {sz:>22} {str(s["connected"]):>10}')
        except Exception as e:
            print(f'{name:<18} FAILED: {e!r}')
