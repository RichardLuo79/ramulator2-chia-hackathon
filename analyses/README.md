# Analysis

`paper_results.ipynb` regenerates the paper's data figures (Figs. 2–9) and Tables
4–5; it is committed without outputs. `scripts/reproduce-figures` executes it in
the analysis environment, offline. The figures are drawn by `paper_figures.py`
(also runnable on its own: `python analyses/paper_figures.py`) from the CSVs in
`results/paper/`; the tables come from `scripts/astra_ablation/` and
`scripts/multicore_metrics/`.
