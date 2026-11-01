# Reproducing the results

Run every command from the repository root; each one takes `--help`.

| Command | What it does |
|---|---|
| `scripts/setup` | Installs a Python environment or builds the simulator |
| `scripts/fetch-data` | Downloads and checks the simulation inputs |
| `scripts/reproduce-figures` | Regenerates Figs. 2–9 and Tables 4–5 |
| `scripts/evaluate` | Runs one model or baseline on one workload, without an LLM |
| `scripts/run-campaign` | Runs a new CHIA search (calls paid LLM services) |
| `scripts/verify-artifact` | Checks every file against `results/manifests/release.json` |

If the machine is already part of a Ray cluster, run `export RAY_ADDRESS=local`
first.

## Figures and tables

```sh
python3 scripts/setup --component analysis
.venv-analysis/bin/python scripts/reproduce-figures
```

This runs `analyses/paper_results.ipynb`. It draws Figs. 2–9 from the CSVs in
`results/paper/` into `results/figures/` and writes Tables 4 and 5 to
`results/tables/`. With the pinned packages, the figure PDFs are byte-identical
to the paper's.

## Running the simulations

Build the simulator, then fetch the inputs:

```sh
python3 scripts/setup --component campaign
.venv-campaign/bin/python scripts/setup --component runtime --workers 8
for cores in 1 4 8; do
  .venv-campaign/bin/python scripts/setup --component champsim --cores $cores --workers 8
done
.venv-campaign/bin/python scripts/fetch-data --download
```

[results/INPUTS.md](results/INPUTS.md) explains what `fetch-data` downloads.

Run a model and the oracle on the same workload:

```sh
.venv-campaign/bin/python scripts/evaluate --case validation-c1-bwaves \
  --model astra_single_core --output .work/bwaves-astra
.venv-campaign/bin/python scripts/evaluate --case validation-c1-bwaves \
  --model oracle --output .work/bwaves-oracle
```

Each run prints its result and saves it to
`<output>/measurement/measurement.json`; the core-cycle error is the model's
`per_core_cycles` relative to the oracle's. Rerunning a frozen model or baseline
reproduces the recorded results exactly. Give every run a new output directory.
The models are `astra_single_core`, `astra_multicore`, `deepseek_single_core`,
`deepseek_multicore`, `gemini_single_core`, `gemini_multicore`,
`opus_single_core` and `opus_multicore`; the baselines are `fixedlat`, `md1`,
`wmg1` and `mess`. The workloads are the cases in
`chia-loop/configs/astra.json`.

Each ChampSim core runs 2M warmup instructions and then 20M measured
instructions. Cores that finish early keep running as background traffic until
the last core finishes. Warmup bypasses Ramulator.

## DRAM configurations (Fig. 8)

Add `--organization` and `--queue-entries` to run at one of the nine
configurations:

```sh
.venv-campaign/bin/python scripts/evaluate --case test-c1-ligra_radii \
  --organization DDR5_16Gb_x16 --queue-entries 32 \
  --model gemini_single_core --output .work/ligra-x16-q32-gemini
```

The organizations are `DDR5_16Gb_x8` (the default), `DDR5_8Gb_x8` and
`DDR5_16Gb_x16`; the queue depths are 32, 64 (the default) and 128. Changing the
organization also changes the capacity and timings, and the placement map is
rebuilt to fit. The per-case results of all models at all nine configurations
are in `results/extensions/` (`transfer.json`, `opus-gemini-per-case.csv` and
`hardware-grid/`).

## Speed (Fig. 5)

```sh
.venv-campaign/bin/python scripts/setup --component speed-inputs
.venv-campaign/bin/python scripts/evaluate --suite speed --model oracle \
  --pattern random --read-percent 75 --interval 16 --cpu 2 \
  --output .work/speed-oracle-random
```

Use an otherwise idle machine and pick `--cpu` from the CPUs you are allowed to
use. Fig. 5 covers streaming and random traffic, 100%, 75% and 50% reads and
arrival intervals of 1, 4, 16, 64 and 256 DRAM cycles. Each point is the median
of five runs of 100k warmup and 5M measured requests, run one at a time on one
pinned CPU. Each speedup compares a model with the oracle measured on the same
machine. `results/diagnostics/` has every repetition.

## gem5 (Fig. 7)

1. Build gem5 v25.1 with the Ramulator wrapper:
   [ramulator/resources/gem5_wrappers/BUILD.md](ramulator/resources/gem5_wrappers/BUILD.md).
2. Build the PolyBench kernels and get the full-system images and checkpoints:
   [chia-loop/src/ramulator_chia/eval/gem5/PUBLIC_SUITE.md](chia-loop/src/ramulator_chia/eval/gem5/PUBLIC_SUITE.md).
3. Run each workload with `chia-loop/src/ramulator_chia/eval/gem5/roi_study.py`,
   passing `--memory-config`, `--ramulator-python` and either `--binary` (SE)
   or `--checkpoint` and `--resources` (FS).

The paper uses the O3CPU model, 2M warmup instructions inside the region of
interest, then up to 2B instructions or the end of the region, polling mode
(`event_driven=False`, `clock_edge_first=False`) and an eight-hour time limit per
run. The recorded results are in `results/extensions/`.

## Table 4: GPT-6 Astra ablation

Each of the twelve variants is the frozen GPT-6 Astra single-core model with one
mechanism removed (variant 11 instead changes how the channel clock starts). All
were run on the 24 single-core test workloads. These commands regenerate the
variants, the control expectations and the tables from the recorded runs:

```sh
python3 -I scripts/astra_ablation/make_variants.py \
  --model results/models/astra_single_core --out .work/ablation/variants
python3 -I scripts/astra_ablation/extract_expected.py --artifact . --out .work/ablation/expected_control.json
python3 -I scripts/astra_ablation/summarize.py --run results/foundation/astra-ablation/run \
  --variants results/foundation/astra-ablation/variants/variants.json \
  --expected results/foundation/astra-ablation/expected_control.json --out .work/ablation/tables
```

`results/foundation/astra-ablation/run/comparisons.json` has the per-case scores
of all 288 runs.

## Table 5: multicore metrics

```sh
python3 -I scripts/multicore_metrics/table5.py . .work/table5
```

This writes `per_mix.csv`, `summary.csv` and `table5.csv` from the per-mix data
in `results/extensions/multicore/metrics/`. The Wasserstein-1 column is the
distance between a model's and the oracle's read-latency distributions on each
core, divided by the mean oracle read latency and averaged over cores and the
eight test mixes. The "+1 cycle" row is the oracle with its DRAM read latency
raised by one cycle.

To compute the distance for your own runs, list each mix's oracle and model
controller traces and admission windows (from `measurement/measurement.json`
in the `scripts/evaluate` output) in a `jobs.json`, then run:

```sh
.venv-campaign/bin/python -I scripts/multicore_metrics/pairfree.py jobs.json pairfree.json 8
```

`score.py` computes MAE<sub>Req</sub> and `pair_stats.py` the per-core constant
predictor.

## New searches

Set up LLM access as described in the README. DeepSeek and Gemini also need the
context-compaction environment: `python3 scripts/setup --component context`.

```sh
.venv-campaign/bin/python scripts/run-campaign --config chia-loop/configs/astra.json \
  --tariff chia-loop/configs/tariffs/astra.json --runtime .work/runtime \
  --output .work/new-astra --allow-paid
.venv-campaign/bin/python scripts/run-campaign --config chia-loop/configs/opus.json \
  --tariff chia-loop/configs/tariffs/opus.json --runtime .work/runtime \
  --output .work/new-opus --allow-paid
.venv-campaign/bin/python scripts/run-campaign --config chia-loop/configs/deepseek.json \
  --tariff chia-loop/configs/tariffs/deepseek.json --runtime .work/runtime \
  --context-compaction chia-loop/configs/compaction/deepseek.json \
  --output .work/new-deepseek --allow-paid
.venv-campaign/bin/python scripts/run-campaign --config chia-loop/configs/gemini.json \
  --tariff chia-loop/configs/tariffs/gemini.json --runtime .work/runtime \
  --context-compaction chia-loop/configs/compaction/gemini.json \
  --vertex-execution-project <project-id> --output .work/new-gemini --allow-paid
```

Each search runs 10 single-core iterations and then 5 multicore iterations. Add
`--evaluate` to test the selected models on the test workloads afterwards. To
summarize a search as Markdown and JSON:

```sh
.venv-campaign/bin/python -m ramulator_chia.framework.staged_report .work/new-astra \
  --output .work/new-astra-report.md
```

The agents see their training workloads by name, get only metrics/stats for the
validation workloads, and never see the test workloads during search. The tariff
files hold the prices behind the cost estimates in Fig. 6, as of September 2026.
