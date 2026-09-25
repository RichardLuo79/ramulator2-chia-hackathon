# Paper figure reproduction

`chia_campaign_results.ipynb` reproduces Figures 1–8 from the paper using the
data in `results/paper/paper-v1/`.

Run from the artifact root:

```sh
python3 scripts/setup --component analysis
.venv-analysis/bin/python scripts/reproduce-figures
```

Execution is offline and requires no simulator, campaign package, credentials,
paper checkout, trace, or raw observation archive. The command exports eight
PDF/SVG/PNG figures and their CSV source tables to `analyses/figures/`. It checks
all CSV values against the paper and all PNG hashes against the reference
figures. Use the pinned analysis environment for identical rendering.

## Inputs and code

The bundle manifest maps each figure to its caption, data, plotting function,
reference tables, and checksums. The four JSON inputs contain per-case
measurements, model identities, promotion decisions, speed repetitions, and
token accounting. Model sources and additional results are in `results/`.

`paper_notebook.py` assembles the notebook from `paper_plots.py` and
`paper_workflow.py`; it embeds their actual plotting function source, not cached
images. `paper_evidence.py` and `paper_checks.py` validate inputs before plotting.
`import_paper_snapshot.py` is an operator-only importer for an explicitly supplied
paper checkout and a new version name. It is never invoked during reproduction.
`campaign_results_notebook.py` forwards notebook generation to the paper builder.

## Reading the values

Core-cycle error is absolute percentage error in execution cycles. Normalized
request error is matched-read latency MAE divided by the mean latency of all
recorded eligible oracle reads, including unpaired reads. Workloads have equal
weight; multicore case errors first give each core equal weight. Four- and
eight-core mixes remain separate.

Standalone speed uses five-repetition median request rates and the same-host
oracle for each timing batch. The figure combines speedup ratios, not raw
timings across hosts. gem5 reports simulated elapsed-time accuracy only, grouped
by SE/FS; per-case stopping conditions remain in the evidence.

Usage covers rounds 1–10. Dollar values are estimated API-equivalent LLM costs,
not invoices. GPT-6 Astra and DeepSeek V4.1 Flash have known subtotals with
additional unknown usage (†); Gemini 3.8 Flash has a cost interval. Opus 5.5 uses
main-agent invocation accounting. GPT-6 Astra and Opus 5.5 used subscriptions.

Missing or changed inputs fail explicitly. No unavailable result is imputed.
