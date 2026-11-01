# Scripts

Run these from the repository root; each takes `--help`. Examples are in
`REPRODUCING.md`.

- `setup`: install a Python environment or build the simulator.
- `fetch-data`: download and check the simulation inputs.
- `reproduce-figures`: regenerate Figs. 2–9 and Tables 4–5.
- `evaluate`: run one model or baseline on one workload, without an LLM.
- `run-campaign`: start or resume a CHIA search (needs `--allow-paid`).
- `verify-artifact`: check every file against `results/manifests/release.json`.

`build_frontend.py` builds ChampSim.

| Directory | Use | Data |
|---|---|---|
| `astra_ablation/` | GPT-6 Astra ablation variants and Table 4 | `results/foundation/astra-ablation/` |
| `multicore_metrics/` | Table 5: Wasserstein-1 (`pairfree.py`), paired MAE_Req (`score.py`), constant predictor (`pair_stats.py`), `table5.py` | `results/extensions/multicore/metrics/` |
