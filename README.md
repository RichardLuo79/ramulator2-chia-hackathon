# Ramulator–CHIA loop research artifact

Artifact submission for the CHIA Hackathon at A3 Workshop at MICRO 2026.

## Start here

| Looking for | Go to |
|---|---|
| The paper's figures | [Executed notebook](analyses/chia_campaign_results.ipynb) · [PDF/SVG/PNG exports and CSV source tables](analyses/figures/) |
| Final synthesized model implementations | [Single-core source links below](#synthesized-model-source-code) · [All frozen model snapshots](results/models/) |
| Data needed to reproduce the figures | [Paper evidence bundle](results/paper/paper-v1/) · [Figure-to-input inventory](results/paper/paper-v1/manifest.json) |
| Evaluation and campaign instructions | [Reproduction guide](REPRODUCING.md) · [Command index](scripts/README.md) |

## Reproduce the paper figures

Use Python 3.12 on Linux. From the repository root:

```sh
python3 scripts/setup --component analysis
.venv-analysis/bin/python scripts/reproduce-figures
```

The [notebook](analyses/chia_campaign_results.ipynb) reproduces Figures 1–8. It covers the workflow,
single-core accuracy, search progression, simulation speed, token usage/cost,
gem5 accuracy, DRAM configuration transfer, and multicore accuracy.

Setup installs [pinned analysis dependencies](analyses/requirements.lock.txt).
After setup, reproduction is offline: no credentials, raw traces, observation
archives, simulator build, or paper checkout is needed. Outputs are written to
[analyses/figures/](analyses/figures/) as PDF/SVG/PNG figures and CSV source tables. See the
[analysis guide](analyses/README.md) for metric definitions and accounting notes.

## Synthesized model source code

These are the frozen models used after **10 single-core CHIA iterations**.
Use each implementation with its accompanying parameters; the identity record
provides the source and parameter hashes.

| Campaign | C++ implementation | Parameters | Identity |
|---|---|---|---|
| Gemini 3.8 Flash | [model.cpp](results/models/gemini_single_core/model.cpp) | [parameters.json](results/models/gemini_single_core/parameters.json) | [identity.json](results/models/gemini_single_core/identity.json) |
| DeepSeek V4.1 Flash | [model.cpp](results/models/deepseek_single_core/model.cpp) | [parameters.json](results/models/deepseek_single_core/parameters.json) | [identity.json](results/models/deepseek_single_core/identity.json) |
| GPT-6 Astra | [model.cpp](results/models/astra_single_core/model.cpp) | [parameters.json](results/models/astra_single_core/parameters.json) | [identity.json](results/models/astra_single_core/identity.json) |
| Opus 5.5 | [model.cpp](results/models/opus_single_core/model.cpp) | [parameters.json](results/models/opus_single_core/parameters.json) | [identity.json](results/models/opus_single_core/identity.json) |

The multicore-stage source snapshots included here are
[GPT-6 Astra](results/models/astra_multicore/) and
[DeepSeek V4.1 Flash](results/models/deepseek_multicore/), each after five additional
iterations. A stage's final model may have been introduced earlier: DeepSeek
V4.1 Flash's single-core model was introduced in iteration 7, and its multicore
model in iteration 12. The end-of-stage selection and introducing iteration are
distinct.

For implementation context, see the [model interface](ramulator/src/ramulator/controller/atomic_model/api.h),
[common seed](chia-loop/native/model/seed.cpp), and
[synthesis contract](chia-loop/src/ramulator_chia/framework/prompts/staged_v1/task.md).

## Measurements and artifacts

The [paper snapshot](results/paper/paper-v1/) is the input to the current notebook:

| Evidence | Contents |
|---|---|
| [paper-evidence.json](results/paper/paper-v1/paper-evidence.json) | Single-core per-workload results, headline values, model identities, and oracle-equivalence records |
| [paper-convergence.json](results/paper/paper-v1/paper-convergence.json) | Training/validation histories and promotion decisions for iterations 1–10 |
| [paper-additions.json](results/paper/paper-v1/paper-additions.json) | Standalone speed repetitions and same-host oracle batches; token/cost accounting; simulator configuration |
| [paper-extensions.json](results/paper/paper-v1/paper-extensions.json) | gem5, DRAM configuration-transfer, and multicore results with coverage and provenance |
| [tables/](results/paper/paper-v1/tables/) · [reference-tables/](results/paper/paper-v1/reference-tables/) | The paper's configuration/mechanisms tables and exact figure-data CSV references |

The snapshot includes the four single-core campaigns, Opus 5.5 gem5 accuracy,
Gemini 3.8 Flash multicore results, and the separate Opus 5.5/Gemini 3.8 Flash
speed batch. Gemini 3.8 Flash gem5 and Opus 5.5 multicore results are not included.
Speed measurements use each batch's own oracle; dollar values are estimated
API-equivalent costs, with recorded uncertainty, rather than invoices.

[results/README.md](results/README.md) indexes the measurements, additional
analyses, and source identities. Large observations and campaign logs are indexed by
the [archive manifest](results/manifests/archives.json); they are not required to
plot the paper. For simulator inputs, see the [acquisition guide](results/INPUTS.md)
and [checksummed input inventory](results/manifests/inputs.json).

## Reproduction levels

1. **Figures:** [reproduce-figures](scripts/reproduce-figures) uses the compact
   checked-in evidence, without rerunning experiments.
2. **Frozen evaluation:** [evaluate](scripts/evaluate) runs a specified model and
   protocol after rebuilding the simulator and supplying verified inputs.
3. **Fresh search:** [run-campaign](scripts/run-campaign) uses the
   [campaign configurations](chia-loop/configs/) and
   [stage prompts](chia-loop/src/ramulator_chia/framework/prompts/staged_v1/).
   It requires explicit provider credentials and `--allow-paid`. This reproduces
   the procedure, not necessarily the same generated models.

Installation, figure reproduction, frozen-model evaluation, and ordinary tests
do not call an LLM or provision cloud resources. Full commands are in
[REPRODUCING.md](REPRODUCING.md). To verify the checked-in release files before
making local changes, run `python3 scripts/verify-artifact`.

## Evaluate a published model

Use native Ubuntu 24.04; Docker and cloud VMs are optional. Install the system
prerequisites in [REPRODUCING.md](REPRODUCING.md#environment), then:

```sh
python3 scripts/setup --component campaign
.venv-campaign/bin/python scripts/setup --component runtime --workers 8
.venv-campaign/bin/python scripts/setup --component champsim --cores 1 --workers 8
.venv-campaign/bin/python scripts/fetch-data --download --config chia-loop/configs/opus.json
.venv-campaign/bin/python scripts/evaluate --config chia-loop/configs/opus.json \
  --case validation-c1-bwaves --model opus_single_core --with-oracle --output .work/bwaves-opus
```

The downloader uses the official [DPC4 traces](https://github.com/CMU-SAFARI/DPC4#workload-traces).
It resumes partial transfers, checks the exact instruction prefixes, and rebuilds
the placement maps. It downloads only the selected configuration's inputs.
`--with-oracle` also measures the reference and writes `comparison.json` with
core-cycle and request errors, coverage, and tails. Omitting it measures only
the chosen model. Neither mode uses an LLM.

## Run a new campaign

For the 10+5 templates, also build ChampSim with `--cores 4` and `--cores 8`,
then prepare inputs with that template. Opus's template has ten single-core
rounds only. Follow [provider setup](chia-loop/PROVIDERS.md) before checking:

```sh
.venv-campaign/bin/python scripts/run-campaign \
  --config chia-loop/configs/opus.json --output .work/my-opus \
  --tariff chia-loop/configs/tariffs/opus.json --evaluation-workers 4 --check
```

`--check` makes no provider requests and creates no campaign. It checks local
credentials, dependencies, inputs, builds, prompts, and isolation. After it
passes, replace `--check` with `--allow-paid --evaluate` to start search and
evaluate the final selection on the hidden test cohort. Reuse the same command
and output directory to resume. See [progress, recovery, and reports](REPRODUCING.md#progress-preservation-and-reports).

The pipeline is **build → prepare inputs → check → search → final test → report**.
The harness manages rounds and accounting; vendored CHIA handles conversations;
protected tools expose training data and anonymous validation; local Ray workers
run ChampSim–Ramulator and measure accuracy. Agents cannot read the published
results in this repository. A fresh campaign writes its own evidence and never
updates the paper notebook's data.

## Repository layout

| Directory | Contents |
|---|---|
| [ramulator/](ramulator/) | Simulator sources, frontend integrations, baseline implementations, and upstream notices |
| [chia-loop/](chia-loop/) | Campaign/evaluation harness, prompts, configurations, seed, and tests |
| [results/](results/) | Frozen model sources, measurements, and evidence manifests |
| [analyses/](analyses/) | Paper notebook, plotting code, and figure exports |
| [scripts/](scripts/) | Setup, input verification, reproduction, evaluation, and campaign entry points |

See [CITATION.cff](CITATION.cff) for citation metadata,
[LICENSE](LICENSE) for licensing, and [THIRD_PARTY.md](THIRD_PARTY.md) for upstream
source pins and notices.
