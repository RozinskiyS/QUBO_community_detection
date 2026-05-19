# Notebooks

Pre-executed analysis notebooks with outputs and figures visible. They
follow the narrative arc of the thesis chapters 4.2–4.6.

| # | Notebook | Thesis section | Source CSV |
|---|---|---|---|
| 01 | `01_karate_k2.ipynb` | §4.2 (warm-up, k=2 on Karate) | inline |
| 02 | `02_multi_class.ipynb` | §4.3 (first multi-class run) | inline |
| 03 | `03_ortho_regularization.ipynb` | §4.4 (α-sweep on Karate) | `ortho_experiment_*.csv` |
| 04 | `04_regularization_comparison.ipynb` | §4.4 (cross-graph) | `comparison_*.csv` |
| 05 | `05_hp_sweep.ipynb` | §4.5 (HP sweep heatmaps) | `hp_sweep.csv` |
| 06 | `06_extended_shots_formula5.ipynb` | §4.5 (20-shot extended) | `extended_shots.csv` |
| 07 | `07_extended_k2_formula6.ipynb` | §4.2 (k=2 on 7 binary graphs) | `extended_k2.csv` |
| 08 | `08_hierarchical.ipynb` | §4.6 (hierarchical, 13 graphs) | `hierarchical.csv` |

## Reference notebooks

`reference/pi_gnn_maxcut_reference.ipynb` — original PI-GNN implementation
on MaxCut (Schuetz et al. 2022), used as the architectural starting point.

`reference/colab_full.ipynb` — self-contained Colab edition of the
regularization-comparison experiment (GPU-ready).

## Re-executing

The notebooks read CSVs from `../results/`. To re-execute, just open in
Jupyter and run all cells; no Python path tweaks needed since the imports
are inline.
