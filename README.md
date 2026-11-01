# Ramulator–CHIA loop research artifact

Artifact of *Synthesizing Fast and Accurate Immediate-Response DRAM Latency
Model with Agents* (A³ Workshop at MICRO 2026). It contains the CHIA loop that
synthesizes DRAM latency models with LLM agents, the simulators that evaluate
them, the frozen models from the paper, and the data and scripts behind every
figure and table.

## Layout

```
ramulator/   Ramulator 2 (cycle-level oracle, baselines), ChampSim and gem5 integrations
chia-loop/   ramulator_chia package: campaign loop, evaluation, configs
results/     frozen models, plotted values and per-case results, input manifests
analyses/    notebook and plotting code for the paper's figures and tables
scripts/     entry points; each takes --help
```

## Requirements

- Linux x86-64 and Python 3.12 (enough for the figures and tables).
- For simulation:
  - GCC 13, CMake and GNU Make;
  - zlib, liblzma, bzip2, fmt, CLI11 and nlohmann-json; on Ubuntu 24.04:
    `apt install build-essential cmake zlib1g-dev liblzma-dev libbz2-dev libfmt-dev libcli11-dev nlohmann-json3-dev`;
  - a kernel with Landlock and seccomp (models run sandboxed);
  - the workload traces and placement maps, which `scripts/fetch-data --download`
    fetches (see [results/INPUTS.md](results/INPUTS.md)).
- For gem5 (Fig. 7): gem5 v25.1 and its kernels and disk images (see
  [REPRODUCING.md](REPRODUCING.md#gem5-fig-7)).
- For new searches: access to the LLMs (see [below](#adapting-to-your-machinesystem)).
- Python packages are pinned in `analyses/requirements.lock.txt` and
  `chia-loop/requirements*.lock.txt`; `scripts/setup` installs them.

## Run

**Figures and tables** (offline, no simulator build):

```sh
python3 scripts/setup --component analysis
.venv-analysis/bin/python scripts/reproduce-figures
python3 scripts/verify-artifact    # checks every file against results/manifests/release.json
```

**Evaluate a frozen model** (no LLM). Build the simulator, fetch the inputs,
then run one case:

```sh
python3 scripts/setup --component campaign
.venv-campaign/bin/python scripts/setup --component runtime --workers 8
for cores in 1 4 8; do
  .venv-campaign/bin/python scripts/setup --component champsim --cores $cores --workers 8
done
.venv-campaign/bin/python scripts/fetch-data --download
.venv-campaign/bin/python scripts/evaluate --case validation-c1-bwaves \
  --model astra_single_core --output .work/bwaves-astra
```

`--model oracle` gives the cycle-level reference. Every model and baseline also
runs at the other DRAM configurations (`--organization`, `--queue-entries`);
`scripts/evaluate --help` lists them.

**Run a new search** (calls LLM APIs):

```sh
.venv-campaign/bin/python scripts/run-campaign --config chia-loop/configs/astra.json \
  --tariff chia-loop/configs/tariffs/astra.json --runtime .work/runtime \
  --output .work/new-astra --allow-paid
```

Configs for all four LLMs are in `chia-loop/configs/`; DeepSeek and Gemini also
take `--context-compaction` (see [REPRODUCING.md](REPRODUCING.md#new-searches)).

## Adapting to your machine/system

- **Cores:** `--workers N` sets build parallelism; `--evaluation-workers N`
  (1–120) sets how many simulations a campaign runs at once. Speed
  measurements run on the one CPU given by `--cpu`.
- **Paths:** `--data-root` (default `.data`), `--work-root` (`.work`) and
  `--runtime` (`.work/runtime`).
- **Ray:** the code uses its own single-node Ray instance. If the machine is
  already part of a Ray cluster (`ray start`), first run
  `export RAY_ADDRESS=local`; otherwise some steps try to join that cluster and
  hang.
- **LLM access** (new searches only), read from the launching environment:
  - `astra.json`: Codex CLI (`codex` on `PATH` or `CHIA_CODEX_BIN`), login in
    `~/.codex/auth.json` or `CHIA_CODEX_AUTH`.
  - `opus.json`: Claude Code (`claude` or `CHIA_CLAUDE_BIN`), OAuth token file
    `~/.claude/chia-oauth-token` or `CHIA_CLAUDE_AUTH`.
  - `deepseek.json`: `DEEPSEEK_API_KEY`.
  - `gemini.json`: Vertex AI with Google application-default credentials;
    pass your project with `--vertex-execution-project`.

## Results

| Paper | Plotted values in `results/paper/` | Per-case results in `results/` |
|---|---|---|
| Fig. 2: per-workload test errors | `single-core-test.csv` | `foundation/campaigns.json`, `foundation/single-core-opus-gemini.json` |
| Fig. 3: mean test errors | `single-core-test-means.csv` | as Fig. 2 |
| Fig. 4: errors during the loop | `convergence.csv` | as Fig. 2 |
| Fig. 5: speedup | `speed.csv` | `diagnostics/` |
| Fig. 6: LLM usage | `usage.csv` | `foundation/single-core-usage.json` |
| Fig. 7: gem5 | `gem5.csv` | `extensions/` |
| Fig. 8: DRAM configurations | `hardware.csv` | `extensions/` |
| Fig. 9: multicore | `multicore.csv` | `foundation/campaigns.json`, `extensions/multicore/` |
| Table 3: model mechanisms | – | `models/<model>/`: `model.cpp`, `parameters.json`, `identity.json` |
| Table 4: GPT-6 Astra ablation | – | `foundation/astra-ablation/` |
| Table 5: multicore metrics | – | `extensions/multicore/metrics/` |

`<model>_single_core` is the model after the 10 single-core iterations,
`<model>_multicore` after 5 more multicore iterations.

Where each command writes:

- `reproduce-figures`: Figs. 2–9 (PDF/SVG/PNG) in `results/figures/`, Tables 4
  and 5 (CSV) in `results/tables/`.
- `evaluate`: prints the result; measurement files go to `<output>/measurement/`.
- `run-campaign`: everything under `--output`; `status.json` shows the current
  stage.

## More

- [REPRODUCING.md](REPRODUCING.md): full protocols (speed, gem5, hardware
  configurations, ablation, multicore metrics).
- [results/INPUTS.md](results/INPUTS.md): inputs and how to obtain them.

## AI use

Claude Opus 5.5 was used to clean up and organize this artifact for release.
