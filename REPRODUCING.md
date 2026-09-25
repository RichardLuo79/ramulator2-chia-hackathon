# Reproducing the artifact

## Environment

On Ubuntu 24.04 x86-64, install:

```sh
sudo apt update
sudo apt install build-essential gcc-13 g++-13 cmake make git patch curl \
  python3.12 python3.12-venv python3-dev zlib1g-dev liblzma-dev \
  libseccomp2 pkg-config xz-utils gzip bzip2 libbz2-dev unzip time util-linux
```

Campaign isolation requires **Landlock ABI 6** and seccomp (Linux 6.12 or newer
with both enabled). Ubuntu 24.04's original 6.8 kernel is insufficient; use a
supported newer HWE kernel. `run-campaign --check` tests the boundary and refuses
an incompatible host. Do not disable isolation to pass the check.

Allow roughly 12 GiB for download fragments, compressed prefixes, and preparation
scratch, plus environments and builds. The downloader leaves 32 GiB free by
default. Campaign evidence needs additional space and grows with the number of
evaluations. Start with four evaluation workers; each simulation has an 8-GiB
address-space limit. Choose workers for available RAM and CPU affinity rather
than treating the maximum of 120 as a recommended setting.

Use Python 3.12 and GCC 13.3 with optimized builds (`-O3 -DNDEBUG`).

The analysis environment is deliberately separate. Installers need network
access (or a prepopulated wheel cache); **notebook execution does not**.
The lock files pin Python dependencies, including transitive packages.

## Six commands

Run these from the artifact root. Each supports `--help`.

| Entry point | Use |
|---|---|
| `scripts/setup` | Install the analysis/campaign environment or build the native runtime/ChampSim |
| `scripts/fetch-data` | Download/verify configured trace prefixes, rebuild placement maps, or verify/extract an archive |
| `scripts/reproduce-figures` | Execute the eight-figure paper notebook and export figures/source tables |
| `scripts/evaluate` | Evaluate a frozen case or a synthetic diagnostic, with no provider access |
| `scripts/run-campaign` | Read-only `--check`, or start/resume with explicit paid opt-in |
| `scripts/verify-artifact` | Check the release file inventory |

### Figures only

```sh
python3 scripts/setup --component analysis
.venv-analysis/bin/python scripts/reproduce-figures
python3 scripts/verify-artifact
```

The notebook reproduces the paper's eight figures. Metric definitions and cost
accounting are documented in `analyses/README.md`; per-case data and coverage
are in `results/paper/paper-v1/`. Exports include figures, CSV source tables,
and a checksum manifest.

### Native runtime and ChampSim

Install system build prerequisites through your system package manager, then:

```sh
python3 scripts/setup --component campaign
.venv-campaign/bin/python scripts/setup --component runtime --workers 8
.venv-campaign/bin/python scripts/setup --component champsim --cores 1 --workers 8
.venv-campaign/bin/python scripts/setup --component champsim --cores 4 --workers 8
.venv-campaign/bin/python scripts/setup --component champsim --cores 8 --workers 8
```

`ramulator/integrations/champsim/source` contains the ChampSim source and
integration patches. Each build records its source, compiler, flags, and linked
libraries. Evaluation caches require a matching build identity.

Download the configured inputs and regenerate their frozen placement maps:

```sh
.venv-campaign/bin/python scripts/fetch-data --download --config chia-loop/configs/astra.json
.venv-campaign/bin/python scripts/fetch-data --config chia-loop/configs/astra.json
.venv-campaign/bin/python scripts/evaluate \
  --case validation-c1-bwaves --model astra_single_core \
  --with-oracle --output .work/my-bwaves-astra
```

`--with-oracle` runs the reference using the same frontend receipt and writes
verified core-cycle/request accuracy, pairing coverage, and tails to
`comparison.json`. Omit it for one measurement only. Other choices include
`astra_multicore`, `deepseek_single_core`, `deepseek_multicore`,
`opus_single_core`, `gemini_single_core`, `fixedlat`, `md1`, `wmg1`, `mess`.
Use a new output directory for each attempt; completed evidence is not
overwritten. A missing input is an error, never a workload substitution.

ChampSim uses 2M warmup + 20M measured instructions **per core**, a warmup
barrier, frozen physical placement, continuous background contention and
admission-window request accounting. Warmup bypasses Ramulator; cores that finish
their measurement windows continue generating background traffic.

### Synthetic and transfer workflows

`scripts/evaluate --suite lat-tp` runs one suite point; `--suite speed`
runs one standalone offered-stream measurement. Their `--help` lists explicit
pattern, load and measurement options. These are DRAM diagnostics, not
application accuracy or paired-request error measurements. Timing scripts must
be run on a quiet machine; the artifact does not claim host-speed parity across
platforms.

```sh
.venv-campaign/bin/python scripts/setup --component speed-inputs
.venv-campaign/bin/python scripts/evaluate --suite lat-tp --model oracle \
  --read-percent 100 --nop 10000 --output .work/lat-tp-oracle-low-load
.venv-campaign/bin/python scripts/evaluate --suite speed --model oracle \
  --pattern random --read-percent 75 --interval 16 --cpu 2 \
  --output .work/speed-oracle-random
```

Choose `--cpu` from the machine's allowed affinity mask. Shorter `--requests`
are explicit diagnostics and are not labeled as the published 5M-request window.

The separate Opus/Gemini speed study uses `oracle`, `opus_single_core`, and
`gemini_single_core`: both patterns, 100/75/50% reads, intervals 1/4/16/64/256,
and five fresh repetitions of 100k warmup + 5M measured requests. Run one job
at a time on the same pinned CPU. Complete all 90 configurations once before
adding later repetitions, interleaving model order. Reproduce the reference on
that same host; do not divide new timings by the archived oracle from another
machine. Initialization, input generation, and model compilation are outside
the measured boundary; do not run unrelated builds or transfers concurrently.
Use the single-point executor above for each point and repetition. The
`ramulator_chia.eval.dram_speed` helpers provide matrix ordering and receipt
aggregation.

For ChampSim hardware transfer, use `--organization` and `--queue-entries`.
The native allocator derives a capacity-correct placement map from the same
logical pages. Capacity/geometry/timing change together; this is not an isolated
bank-count experiment.

gem5 is separately built from the pin in
`ramulator/resources/gem5_wrappers/BUILD.md`. The transfer driver is
`chia-loop/src/ramulator_chia/eval/gem5/roi_study.py`; it takes an explicit
`--memory-config`, `--ramulator-python` and either `--binary` (SE) or
`--checkpoint` plus `--resources` (FS). Run it with `gem5.opt`, **O3CPU**,
2M in-ROI warmup then up to 2B additional committed instructions or ROI end.
The published protocol uses polling: `event_driven=False` and
`clock_edge_first=False`.
Use an eight-hour process watchdog; retain but do not score partial results.
Guest images/checkpoints are supplied separately.

### Fresh campaigns

Configure the chosen provider using [PROVIDERS.md](chia-loop/PROVIDERS.md).
Templates are `astra.json`, `deepseek.json`, `gemini.json`, and `opus.json`.
Gemini requires `CHIA_VERTEX_PROJECT`; Gemini and DeepSeek also require
`python3 scripts/setup --component context` for their isolated ADK summarizer.

```sh
.venv-campaign/bin/python scripts/run-campaign --help
.venv-campaign/bin/python scripts/run-campaign \
  --config chia-loop/configs/astra.json --output .work/new-astra \
  --runtime .work/runtime --tariff chia-loop/configs/tariffs/astra.json --check
.venv-campaign/bin/python scripts/run-campaign \
  --config chia-loop/configs/astra.json --output .work/new-astra \
  --runtime .work/runtime --tariff chia-loop/configs/tariffs/astra.json --allow-paid --evaluate
```

Supply the launcher's explicit runtime/tariff/provider options shown by help.
The example configuration uses the paper's scientific settings. Provider
availability and pricing must be checked by the operator;
bundled tariff records are dated accounting inputs, not current price quotes.
For a different provider, create a new configuration; do not rewrite an active
campaign's identity.

Training is named, validation is anonymous numerical feedback, and test evidence
is hidden during search. The protected agent receives a staged allowlisted
workspace—not this repository's published `results/`. Credentials stay outside
the artifact and agent-visible paths. Keep the existing independent review,
mandatory integrity checks, bounded provider retries, 30-minute optional
diagnostic limits and uncapped full evaluations. Fresh searches reproduce the
procedure, not a deterministic sequence of LLM outputs.

`--check` does not refresh credentials, verify remote account permissions, test
model availability, or authorize billing. Its isolation probe uses disposable
local files, not a campaign database. A live provider smoke test is a separate,
explicitly authorized action; ordinary setup and tests cannot perform one.

### Progress, preservation, and reports

Keep the driver in a persistent terminal (for example `tmux`) for long runs.
Read `.work/new-astra/status.json` for its current phase. The output contains:

| Path | Contents |
|---|---|
| `config.json`, `resolved-launch-config.json` | Frozen scientific settings and prepared input identities |
| `campaign.sqlite` | Committed rounds, selections, decisions, attempts, and accounting |
| `candidates/`, `agents/` | Candidate source snapshots and the current working draft |
| `events/`, `native-evidence/` | Tool exchanges, receipts, and private native continuation state |
| `measurements/`, `evaluations/` | Completed measurements, metrics, and failure evidence |
| `source/`, `operations/` | Source/build provenance and operational amendments |

Exact child paths are also recorded in the receipts. Summaries, review decisions,
and usage are retained with the corresponding round. Native state is private:
do not upload a campaign directory as a public artifact or copy credentials into it.

To resume, run the same launch command with the same config, runtime, output,
tariff, and compaction policy. Completed steps are reused only when identities
match. If search finished without final testing, rerun with `--evaluate`.
Unknown or ambiguous attempts stop for inspection; do not delete their receipts
or mark them successful to force a retry. An interrupted provider call may have
unknown usage even when no response was saved.

Before moving a run, stop its driver, wait for its workers to settle, and retain
the complete output plus input/build manifests. Never run two owners of one
campaign. Do not copy a live SQLite file blindly; the existing
`ramulator_chia.framework.archive.backup_sqlite` helper produces a consistent
snapshot. Keep runtime/input paths stable during a resume.

Export a report without inference or reevaluation:

```sh
.venv-campaign/bin/python -m ramulator_chia.framework.staged_report \
  .work/new-astra --output .work/new-astra-report.md
```

This creates Markdown and companion JSON with endpoint metrics, coverage,
promotion decisions, and usage. Use a new report filename for each snapshot.

## External evidence

`results/manifests/archives.json` lists ZIP64 evidence bundles and their checksums.
To extract a supplied bundle, pass its directory to `scripts/fetch-data`; see
`--help` for the archive options. Automatic archive download is not configured.
Figure reproduction does not require these bundles. Observation files are
identified by SHA256; campaign log exports exclude credentials and private
session state.

## Tests

```sh
.venv-campaign/bin/python -m pytest chia-loop/tests -q
.venv-analysis/bin/python -m pytest analyses/tests -q
```

Optional native fixtures use `CHIA_NATIVE_TEST_RUNTIME` and
`BASELINE_REUSE_RUNTIME`. `scripts/qualify_artifact.py` performs only the
representative full-window cases listed in its reference inventory.

The root pytest configuration restricts ordinary discovery to the artifact's
offline fixtures. Vendored upstream live-service tests and fixture-capture
utilities are not part of this command and must not be run indiscriminately.
