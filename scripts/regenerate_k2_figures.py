"""Regenerate the three k=2 (formula 6) figures with bigger fonts for the
A4 thesis (textwidth ~16 cm).

Inputs:
  results/extended_k2.csv
  results/extended_k2_louvain.csv
  data/cache/<graph>__k2.pkl  (for n / k_true info)

Outputs (overwrites in place):
  figures/fig_k2_1_boxplots.png
  figures/fig_k2_2_gap_to_louvain.png
  figures/fig_k2_3_stability.png
"""
from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, 'results')
FIG_DIR = os.path.join(HERE, 'figures')
CACHE_DIR = os.path.join(HERE, 'data', 'cache')


# Bigger fonts for A4 print rendering.
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})


def stable_rate_strict(mods, frac=0.8):
    if len(mods) == 0:
        return 0.0
    if mods.max() <= 0:
        return float((mods == mods.max()).mean())
    return float((mods >= frac * mods.max()).mean())


# Load data ====================================================================
df = pd.read_csv(os.path.join(RES_DIR, 'extended_k2.csv'))
if 'error' in df.columns:
    err_mask = df['error'].fillna('').astype(str).str.strip().str.len() > 0
    df = df[~err_mask].reset_index(drop=True)

louv = pd.read_csv(os.path.join(RES_DIR, 'extended_k2_louvain.csv'))

graph_order = ['karate', 'dolphins', 'polbooks_binary',
               'SBM_n200_p0.3_q0.05', 'SBM_n200_p0.2_q0.15',
               'SBM_n500_p0.2_q0.05', 'polblogs']

GINFO = {}
for g in df['graph'].unique():
    p = os.path.join(CACHE_DIR, f'{g}__k2.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            payload = pickle.load(f)
        GINFO[g] = {'n': len(payload['nodes']), 'k_true': int(payload['k_true'])}
    else:
        GINFO[g] = {'n': -1, 'k_true': -1}

agg = (df.groupby('graph')
         .agg(mod_best=('mod', 'max'), mod_mean=('mod', 'mean'),
              mod_std=('mod', 'std'),  nmi_best=('nmi', 'max'),
              nmi_mean=('nmi', 'mean'),
              n_shots=('shot_seed', 'count'))
         .reset_index())
sm = df.groupby('graph')['mod'].apply(lambda s: stable_rate_strict(s.values))
agg['stable_rate'] = agg['graph'].map(sm.to_dict())


# fig_k2_1_boxplots.png ========================================================
def fig_k2_1():
    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    axes = axes.flatten()
    for ax, g in zip(axes, graph_order):
        if g not in df['graph'].values:
            ax.axis('off'); ax.set_title(f'{g}\n(no data)', fontsize=16)
            continue
        sub = df[df['graph'] == g]['mod'].values
        bp = ax.boxplot([sub], labels=['QIGNN k=2'], patch_artist=True,
                        widths=0.55, showmeans=True)
        bp['boxes'][0].set_facecolor('#1f78b4')
        # Slightly thicker median/whisker lines for print readability
        for line in bp['medians'] + bp['whiskers'] + bp['caps']:
            line.set_linewidth(1.4)
        ax.scatter([1 + np.random.uniform(-0.05, 0.05) for _ in sub], sub,
                    alpha=0.7, s=32, color='black', zorder=3)
        L = louv[louv['graph'] == g]
        if len(L):
            louv_mod = L.iloc[0]['mod']
            ax.axhline(louv_mod, color='red', ls='--', lw=1.6,
                        label=f'Louvain ({louv_mod:.3f})')
            ax.legend(loc='lower left', fontsize=12, frameon=False)
        info = GINFO.get(g, {'n': '?'})
        ax.set_title(f'{g}\n(n={info["n"]})', fontsize=16)
        ax.set_ylabel('Modularity', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(alpha=0.3, axis='y')
    for ax in axes[len(graph_order):]:
        ax.axis('off')
    fig.suptitle('Formula-6 (k=2): modularity distribution over 20 shots',
                  fontsize=18, y=1.0)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_k2_1_boxplots.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


# fig_k2_2_gap_to_louvain.png =================================================
def fig_k2_2():
    gnames = [g for g in graph_order if g in agg['graph'].values]
    gap_mod = [agg[agg['graph']==g].iloc[0]['mod_best'] - louv[louv['graph']==g].iloc[0]['mod']
               for g in gnames]
    gap_nmi = [agg[agg['graph']==g].iloc[0]['nmi_best'] - louv[louv['graph']==g].iloc[0]['nmi']
               for g in gnames]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    x = np.arange(len(gnames))
    for ax, gaps, ylabel, title in [
        (axes[0], gap_mod, 'mod_best(QIGNN-k=2) − mod(Louvain)',
         'Modularity gap to Louvain (20-shots best)'),
        (axes[1], gap_nmi, 'nmi_best(QIGNN-k=2) − nmi(Louvain)',
         'NMI gap to Louvain (20-shots best)'),
    ]:
        colors = ['#33a02c' if v >= 0 else '#e31a1c' for v in gaps]
        ax.bar(x, gaps, color=colors, edgecolor='black', linewidth=0.6)
        ax.axhline(0, color='black', lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(gnames, rotation=25, ha='right', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(alpha=0.3, axis='y')
        # Annotate values atop each bar
        for xi, v in zip(x, gaps):
            ax.text(xi, v + (0.005 if v >= 0 else -0.005),
                    f'{v:+.3f}', ha='center',
                    va='bottom' if v >= 0 else 'top',
                    fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_k2_2_gap_to_louvain.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


# fig_k2_3_stability.png =======================================================
def fig_k2_3():
    gnames = [g for g in graph_order if g in agg['graph'].values]
    stable_rates = [agg[agg['graph']==g].iloc[0]['stable_rate'] for g in gnames]
    stds = [agg[agg['graph']==g].iloc[0]['mod_std'] for g in gnames]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    x = np.arange(len(gnames))

    ax = axes[0]
    ax.bar(x, stable_rates, color='#1f78b4', edgecolor='black', linewidth=0.6)
    ax.axhline(0.5, color='red', ls='--', lw=1.5, label='unstable threshold')
    ax.set_xticks(x); ax.set_xticklabels(gnames, rotation=25, ha='right',
                                         fontsize=12)
    ax.set_ylabel('stable_rate (fraction of shots within 80% of best)',
                   fontsize=14)
    ax.set_title('Reproducibility of formula-6 peak', fontsize=16)
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=12, frameon=False)
    ax.tick_params(axis='y', labelsize=12)
    for xi, v in zip(x, stable_rates):
        ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=11)

    ax = axes[1]
    ax.bar(x, stds, color='#a6cee3', edgecolor='black', linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(gnames, rotation=25, ha='right',
                                         fontsize=12)
    ax.set_ylabel('std(modularity) over 20 shots', fontsize=14)
    ax.set_title('Shot-to-shot dispersion (lower → more stable)', fontsize=16)
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='y', labelsize=12)
    for xi, v in zip(x, stds):
        ax.text(xi, v + max(stds)*0.02, f'{v:.4f}', ha='center', va='bottom',
                 fontsize=11)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_k2_3_stability.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    fig_k2_1()
    fig_k2_2()
    fig_k2_3()
    print('\nAll three k=2 figures regenerated with bigger fonts (A4-friendly).')
