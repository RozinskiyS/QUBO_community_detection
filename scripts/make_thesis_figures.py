"""Generate the three remaining publication-ready figures for the thesis:
  fig_4_1_sample_graphs.png   — 2x4 panel of network draws coloured by ground truth
  fig_4_7_cross_method_summary.png — 13×5 mod/NMI heatmap across all methods
  fig_4_8_scalability.png    — runtime vs n log-log scatter with O(n)/O(n^2) refs

Run:
    python3.12 make_thesis_figures.py
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES_DIR = os.path.join(HERE, 'results')
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Publication-quality defaults
mpl.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# =============================================================================
# Figure 4.1 — sample graph panel (2×4 networks)
# =============================================================================
def figure_sample_graphs():
    from data_loaders import (load_karate, load_dolphins, load_polbooks_binary,
                                load_polbooks, load_football, generate_lfr,
                                generate_sbm, load_polblogs)
    print('[fig 4.1] loading 8 graphs...')
    panels = [
        ('Karate',                load_karate,                'kk'),
        ('Dolphins',              load_dolphins,              'kk'),
        ('Polbooks (binary)',     load_polbooks_binary,       'kk'),
        ('Polbooks (k=3)',        load_polbooks,              'kk'),
        ('Football',              load_football,              'kk'),
        ('LFR (n=200, μ=0.1)',    lambda: generate_lfr(n=200, mu=0.1), 'kk'),
        ('SBM (n=200, p=0.3, q=0.05)', lambda: generate_sbm(n=200, p_in=0.3, p_out=0.05), 'kk'),
        ('Polblogs',              load_polblogs,              'spring'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for ax, (title, loader, layout) in zip(axes, panels):
        t0 = time.time()
        G, lbls, k_true = loader()
        n = G.number_of_nodes()
        m = G.number_of_edges()

        # Pick layout (kamada_kawai for small, spring for polblogs)
        if layout == 'spring':
            print(f'  [fig 4.1] {title}: spring_layout(n={n}) ...')
            pos = nx.spring_layout(G, k=0.3, iterations=80, seed=42)
        else:
            print(f'  [fig 4.1] {title}: kamada_kawai_layout(n={n}) ...')
            pos = nx.kamada_kawai_layout(G)

        # Colour by ground truth — tab10 for k<=10, tab20 otherwise.
        # Sample by integer index (so tab10[0]=blue, tab10[1]=orange, etc.)
        # — using fractional sampling collapses k=2 to two shades of blue.
        cmap_name = 'tab10' if k_true <= 10 else 'tab20'
        cmap = mpl.colormaps[cmap_name]
        unique_ids = sorted(set(lbls.values()))
        id_to_idx = {c: i for i, c in enumerate(unique_ids)}
        node_colors = [cmap(id_to_idx[lbls[v]] % cmap.N) for v in G.nodes()]

        # Node sizes proportional to degree (clamped); edges darker than
        # the previous lightgray for better contrast at print resolution.
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        if n > 500:
            sizes = [max(4, 20 * degrees[v] / max_deg) for v in G.nodes()]
            edge_alpha = 0.30; edge_width = 0.4
        elif n > 100:
            sizes = [max(20, 90 * degrees[v] / max_deg) for v in G.nodes()]
            edge_alpha = 0.55; edge_width = 0.7
        else:
            sizes = [max(60, 200 * degrees[v] / max_deg) for v in G.nodes()]
            edge_alpha = 0.75; edge_width = 0.9

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#333333',
                                width=edge_width, alpha=edge_alpha)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                node_size=sizes, edgecolors='black',
                                linewidths=0.4)

        ax.set_title(f'{title}  (n={n}, m={m}, k={k_true})', fontsize=12,
                     pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        print(f'    {title} drawn in {time.time()-t0:.1f}s')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_4_1_sample_graphs.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig 4.1] saved {out}')


# =============================================================================
# Figure 4.7 — cross-method summary heatmap
# =============================================================================
def figure_cross_method_summary():
    print('[fig 4.7] loading aggregated CSVs...')

    base = pd.read_csv(os.path.join(RES_DIR, 'baselines.csv'))
    base.columns = [c.strip() for c in base.columns]
    # Add Louvain rows from extended_k2_louvain (binary graphs missing in
    # baselines) and hierarchical_louvain (SBMs etc. missing in baselines).
    for fp in ['extended_k2_louvain.csv', 'hierarchical_louvain.csv']:
        p = os.path.join(RES_DIR, fp)
        if os.path.exists(p):
            extra = pd.read_csv(p)
            base = pd.concat([base, extra], ignore_index=True)
    # Keep first occurrence per (method, graph)
    base = base.drop_duplicates(subset=['method', 'graph'], keep='first')

    f5_agg = pd.read_csv(os.path.join(RES_DIR, 'extended_shots_agg.csv'))
    # Use the +ortho config (sweet spot identified in HP sweep)
    f5_ortho = f5_agg[f5_agg['config'] == 'ortho'].copy()

    f6_agg = pd.read_csv(os.path.join(RES_DIR, 'extended_k2_agg.csv'))
    # f6_agg is already aggregated per-graph; it does not have a 'config' column.

    # Hierarchical: take best across criteria (mod_best & nmi_best) per graph
    h_raw = pd.read_csv(os.path.join(RES_DIR, 'hierarchical.csv'))
    if 'error' in h_raw.columns:
        h_raw = h_raw[h_raw['error'].fillna('').astype(str).str.strip().str.len() == 0]
    h_per_graph = (h_raw.groupby('graph')
                       .agg(mod_best=('mod', 'max'),
                             nmi_best=('nmi', 'max'))
                       .reset_index())

    # Final list of 13 graphs (rows of the heatmap)
    graph_order = ['karate', 'dolphins', 'polbooks_binary', 'polbooks',
                    'football', 'lfr_n200_mu0.1', 'lfr_n200_mu0.3',
                    'SBM_n200_p0.3_q0.05', 'SBM_n200_p0.2_q0.15',
                    'lfr_n500_mu0.3', 'SBM_n500_p0.2_q0.05',
                    'polblogs', 'email_eu_core']

    methods = ['Louvain', 'Leiden', 'Formula-5\n+ortho',
               'Formula-6\nk=2', 'Hierarchical\n(best of 3)']

    mod_mat = np.full((len(graph_order), len(methods)), np.nan)
    nmi_mat = np.full((len(graph_order), len(methods)), np.nan)

    def lookup_baseline(g, method_name):
        sub = base[(base['graph'] == g) & (base['method'] == method_name)]
        if len(sub):
            return float(sub.iloc[0]['mod']), float(sub.iloc[0]['nmi'])
        return np.nan, np.nan

    for i, g in enumerate(graph_order):
        # Louvain
        m, n_ = lookup_baseline(g, 'Louvain')
        mod_mat[i, 0], nmi_mat[i, 0] = m, n_
        # Leiden
        m, n_ = lookup_baseline(g, 'Leiden')
        mod_mat[i, 1], nmi_mat[i, 1] = m, n_
        # Formula-5 +ortho
        sub = f5_ortho[f5_ortho['graph'] == g]
        if len(sub):
            mod_mat[i, 2] = float(sub.iloc[0]['mod_best'])
            nmi_mat[i, 2] = float(sub.iloc[0]['nmi_best'])
        # Formula-6 k=2
        sub = f6_agg[f6_agg['graph'] == g]
        if len(sub):
            mod_mat[i, 3] = float(sub.iloc[0]['mod_best'])
            nmi_mat[i, 3] = float(sub.iloc[0]['nmi_best'])
        # Hierarchical
        sub = h_per_graph[h_per_graph['graph'] == g]
        if len(sub):
            mod_mat[i, 4] = float(sub.iloc[0]['mod_best'])
            nmi_mat[i, 4] = float(sub.iloc[0]['nmi_best'])

    # Custom teal-yellow colormap: low values → light yellow, high → dark teal
    teal_cmap = LinearSegmentedColormap.from_list(
        'teal_yellow_dark',
        ['#ffffe0', '#a8e1d2', '#3d9d8f', '#0d4d4d'])

    fig, axes = plt.subplots(1, 2, figsize=(17, 9),
                              gridspec_kw={'wspace': 0.05})
    for idx, (ax, mat, title) in enumerate(zip(axes,
                                [mod_mat, nmi_mat],
                                ['Modularity (best of N shots)',
                                 'NMI vs ground truth (best of N shots)'])):
        masked = np.ma.masked_invalid(mat)
        norm = Normalize(vmin=0, vmax=1.0)
        im = ax.imshow(masked, cmap=teal_cmap, aspect='auto', norm=norm)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=10)
        if idx == 0:
            # Left panel: standard left-side y-tick labels
            ax.set_yticks(range(len(graph_order)))
            ax.set_yticklabels(graph_order, fontsize=10)
        else:
            # Right panel: hide y-tick labels (same row order as left)
            ax.set_yticks(range(len(graph_order)))
            ax.set_yticklabels([])

        # Cell annotations: number, bold if it's the row max
        for i in range(mat.shape[0]):
            row = mat[i, :]
            row_max = np.nanmax(row) if np.any(~np.isnan(row)) else None
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    txt = '—'; fw = 'normal'; col = '#666'
                else:
                    txt = f'{v:.2f}'
                    is_best = row_max is not None and abs(v - row_max) < 1e-9
                    fw = 'bold' if is_best else 'normal'
                    col = 'white' if v > 0.5 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=9.5, fontweight=fw, color=col)

        ax.set_title(title, fontsize=12, pad=10)
        # Light gridlines between cells
        ax.set_xticks(np.arange(-.5, len(methods), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(graph_order), 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
        ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Add colorbar only to the rightmost panel (shared colormap range).
        if idx == 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
            cbar.outline.set_visible(False)

    out = os.path.join(FIG_DIR, 'fig_4_7_cross_method_summary.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig 4.7] saved {out}')


# =============================================================================
# Figure 4.8 — runtime vs n scalability
# =============================================================================
def figure_scalability():
    print('[fig 4.8] preparing scalability data...')

    # Louvain times per graph (combine baselines + hierarchical_louvain to get
    # the broadest coverage)
    louv = pd.read_csv(os.path.join(RES_DIR, 'baselines.csv'))
    louv = louv[louv['method'] == 'Louvain']
    extras = []
    for fp in ['extended_k2_louvain.csv', 'hierarchical_louvain.csv']:
        p = os.path.join(RES_DIR, fp)
        if os.path.exists(p):
            extras.append(pd.read_csv(p))
    if extras:
        louv = pd.concat([louv] + extras, ignore_index=True)
        louv = louv[louv['method'] == 'Louvain']
        louv = louv.drop_duplicates(subset=['graph'], keep='first')

    # Formula-6 k=2 mean time per graph
    k2_raw = pd.read_csv(os.path.join(RES_DIR, 'extended_k2.csv'))
    if 'error' in k2_raw.columns:
        k2_raw = k2_raw[k2_raw['error'].fillna('').astype(str).str.strip().str.len() == 0]
    k2_t = k2_raw.groupby('graph')['time'].mean().reset_index()

    # Hierarchical mean time per graph (averaged across criteria & seeds)
    h_raw = pd.read_csv(os.path.join(RES_DIR, 'hierarchical.csv'))
    if 'error' in h_raw.columns:
        h_raw = h_raw[h_raw['error'].fillna('').astype(str).str.strip().str.len() == 0]
    h_t = h_raw.groupby('graph')['time'].mean().reset_index()

    # n per graph: hierarchical.csv has 'n', baselines doesn't, but we can pull
    # from graph_stats.csv plus our own knowledge.
    n_map = {}
    for _, r in h_raw.iterrows():
        n_map[r['graph']] = int(r['n'])
    # Backfill missing graphs from graph_stats.csv if present
    gs_path = os.path.join(RES_DIR, 'graph_stats.csv')
    if os.path.exists(gs_path):
        gs = pd.read_csv(gs_path, index_col=0)
        for g in gs.index:
            n_map.setdefault(g, int(gs.loc[g, 'n']))
    # Hard-coded fallbacks for graphs not in any CSV
    n_map.setdefault('SBM_n200_p0.3_q0.05', 200)
    n_map.setdefault('SBM_n200_p0.2_q0.15', 200)
    n_map.setdefault('SBM_n500_p0.2_q0.05', 500)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Louvain
    L_pts = []
    for _, r in louv.iterrows():
        if r['graph'] in n_map:
            L_pts.append((n_map[r['graph']], float(r['time']), r['graph']))
    L_pts.sort()
    if L_pts:
        ax.scatter([p[0] for p in L_pts], [p[1] for p in L_pts],
                    s=70, c='#1f77b4', marker='o', edgecolor='black',
                    linewidth=0.5, label='Louvain', zorder=3)

    # Formula-6 k=2
    k2_pts = []
    for _, r in k2_t.iterrows():
        if r['graph'] in n_map:
            k2_pts.append((n_map[r['graph']], float(r['time']), r['graph']))
    k2_pts.sort()
    if k2_pts:
        ax.scatter([p[0] for p in k2_pts], [p[1] for p in k2_pts],
                    s=80, c='#2ca02c', marker='s', edgecolor='black',
                    linewidth=0.5, label='Formula-6 (k=2)', zorder=3)

    # Hierarchical
    h_pts = []
    for _, r in h_t.iterrows():
        if r['graph'] in n_map:
            h_pts.append((n_map[r['graph']], float(r['time']), r['graph']))
    h_pts.sort()
    if h_pts:
        ax.scatter([p[0] for p in h_pts], [p[1] for p in h_pts],
                    s=90, c='#d62728', marker='^', edgecolor='black',
                    linewidth=0.5, label='Hierarchical', zorder=3)

    # Reference curves: O(n) and O(n^2). Anchor them through the median
    # Louvain point so they match the actual data range.
    ns_ref = np.logspace(np.log10(30), np.log10(1500), 80)
    if L_pts:
        anchor_n = sorted([p[0] for p in L_pts])[len(L_pts) // 2]
        anchor_t = sorted([p[1] for p in L_pts])[len(L_pts) // 2]
    else:
        anchor_n, anchor_t = 100, 0.01
    on_curve = anchor_t * (ns_ref / anchor_n)
    on2_curve = anchor_t * (ns_ref / anchor_n) ** 2
    ax.plot(ns_ref, on_curve, '--', color='#888', lw=1.0, label=r'$O(n)$ ref',
             zorder=1)
    ax.plot(ns_ref, on2_curve, ':', color='#888', lw=1.0, label=r'$O(n^2)$ ref',
             zorder=1)

    # Annotate selected points across all methods
    annotate_set = {'karate', 'polbooks', 'polblogs', 'email_eu_core'}
    annotated = set()
    for pts, marker_color in [(L_pts, '#1f77b4'),
                               (k2_pts, '#2ca02c'),
                               (h_pts, '#d62728')]:
        for n, t, g in pts:
            if g in annotate_set and (g, marker_color) not in annotated:
                ax.annotate(g, xy=(n, t), xytext=(8, -2),
                            textcoords='offset points', fontsize=9,
                            color=marker_color, alpha=0.95,
                            ha='left', va='top')
                annotated.add((g, marker_color))

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(20, 2000)
    ax.set_xlabel('Number of nodes  $n$', fontsize=11)
    ax.set_ylabel('Wall-clock time per run  (seconds)', fontsize=11)
    ax.set_title('Runtime scalability: classical vs QIGNN-based methods',
                  fontsize=12, pad=10)
    ax.grid(which='major', alpha=0.30, linestyle='-')
    ax.grid(which='minor', alpha=0.15, linestyle=':')
    ax.legend(loc='upper left', frameon=False, ncol=1)

    out = os.path.join(FIG_DIR, 'fig_4_8_scalability.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig 4.8] saved {out}')


# =============================================================================
if __name__ == '__main__':
    figure_sample_graphs()
    figure_cross_method_summary()
    figure_scalability()
    print('\nAll three thesis figures generated:')
    for f in ['fig_4_1_sample_graphs.png',
              'fig_4_7_cross_method_summary.png',
              'fig_4_8_scalability.png']:
        p = os.path.join(FIG_DIR, f)
        if os.path.exists(p):
            kb = os.path.getsize(p) / 1024
            print(f'  {f}  ({kb:.0f} KB)')
