# Results

- `paper/`: the values plotted in each paper figure (Figs. 2–9), one CSV per
  figure.
- `foundation/`: per-case training, validation and test results of the
  campaigns and baselines (`campaigns.json`: GPT-6 Astra, DeepSeek V4.1 Flash and
  the baselines; `single-core-opus-gemini.json`: Opus 5.5 and Gemini 3.8 Flash),
  LLM usage (`single-core-usage.json`) and the GPT-6 Astra ablation
  (`astra-ablation/`; Table 4).
- `diagnostics/`: every repetition of the standalone speed measurements
  (`dram-speed.json`: GPT-6 Astra, DeepSeek and the baselines;
  `dram-speed-opus-gemini.json`: Opus and Gemini), each with the oracle measured
  on the same host.
- `extensions/`: gem5 and DRAM-configuration results (`transfer.json`,
  `opus-gem5.json`, `opus-gemini-{by-group,per-case}.csv`, `hardware-grid/`),
  the multicore test results (`multicore/`) and the multicore request metrics
  (`multicore/metrics/`; Table 5).
- `models/`: frozen model sources, parameters and identities (Table 3).
- `manifests/`: input identities (`inputs.json`, read by `scripts/fetch-data`),
  model endpoints (`endpoints.json`) and file checksums (`release.json`, checked
  by `scripts/verify-artifact`).
- `figures/`, `tables/`: written by `scripts/reproduce-figures` (ignored by Git).
