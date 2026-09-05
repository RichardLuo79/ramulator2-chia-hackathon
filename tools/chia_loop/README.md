# CHIA: evolving immediate-response DRAM controllers

This branch contains a clean fixed-delay Atomic seed, the protected evaluation
harness, four comparison models, and a native CHIA/Ray evolution graph.
See [project status and results](../../doc/chia_hackathon_review.md) for the
scientific scope and current Gemini results. Setup conversations, unrelated
DRAM timing audits, licensed workload traces, and raw interaction logs are not
part of this repository.

## Current experiment

One invocation runs exactly one Gemini backend through Google Cloud ADC.
Gemini 3.1 Pro Preview and Gemini 3.8 Flash are separate model runs, compared
only in downstream analysis. Each run starts clean, uses HIGH thinking and
65,536 maximum output tokens, and stops at its configured iteration or
spending/safety limit. Defaults are five evaluated designs and USD 50;
larger limits require explicit run configuration and spending authorization.
Failed drafts are repairable within an iteration.
Every draft, API attempt, tool result, build, compliance decision, and score is
retained locally; no hidden source repair or automatic paid resume occurs.

Create a `STOP` file in the run root to stop safely after any in-flight
generation settles and before another generation/evaluation starts. A fresh
restart after an audited infrastructure abort can use
`prepare_gemini.py --carry-budget-from OLD_RUN`; financial charges are carried
forward without importing model sources or feedback. The USD 50 authorization
is not replenished by restarting. This option requires that the aborted run
evaluated no candidate and that all generation usage is settled.

Training uses mcf/lbm; final testing uses milc/soplex/GemsFDTD/fotonik3d, only
after that run's selection is frozen. The historical shared execution waited
for both selections; this is retained in its provenance, not imposed as a
dependency between new runs. Every case executes 20 million issued
instructions per core, from a cold start through complete drain. The real
runner rejects shorter windows. It checks trace length, disjoint input hashes,
oracle DRAM traffic, exact request pairing, and callback integrity. The
published comparison models are FixedLat, MD1, Sniper WMG1, and MESS.

An independent earlier feasibility design and all earlier campaigns are
unavailable to optimization agents. Only their own run history, selected
public source files, and training diagnostics are exposed. This isolation is
implemented by the tool adapter and runtime, not merely requested in a prompt.

## Setup

Use Linux with Landlock ABI >= 3, libseccomp, a C++20 compiler, CMake, and Python
3.10 or newer. The recorded experiments use Python 3.12. Configure the external
trace paths described in [the evaluator guide](../eval/README.md); workload
files are not redistributed. Six trace families are needed for real trials.

```sh
python3 -m venv .venv
.venv/bin/pip install -e . -r tools/chia_loop/requirements.txt
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"
cmake --build build-bench --parallel 12
export PYTHONPATH="$PWD/python:$PWD/tools:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT
export RAMULATOR_TRACES=/path/to/traces
gcloud auth application-default login
```

Preparation runs the optimized oracle, four comparisons, seed, full-window
normal/isolated-runner parity checks, unit tests, and actual loaded-DSO isolation
and model-parameter probes. It refreshes ADC but makes no paid generation calls.

```sh
.venv/bin/python tools/chia_loop/prepare_gemini.py --root eval_out/chia/PRO_RUN_ID --model pro --workers 6
.venv/bin/python tools/chia_loop/gemini_loop.py --root eval_out/chia/PRO_RUN_ID --model pro
```

Use a different root and `--model flash` for a Flash run. Root names are stable
run IDs and must be unique within a comparison. Do not reuse a run directory
for another model or trial. Preparation accepts `--max-iterations`, `--usd-cap`,
and `--cpus`; limits are pinned before generation and the runner reads them.
Do not change an initialized ledger's cap or replenish it by restarting.
Concurrent runs must have CPU budgets totaling at most 12. For example, two
independent runs may each use `--cpus 6 --workers 6`; each gets its own local
Ray instance. Do not launch two default 12-CPU runners concurrently.
Each run has its own preparation, state, budget, interactions, CHIA profile,
freeze, final-test results, and stop/failure status. A comparison owns none of
those and cannot promote candidates or transfer budget between runs. Even when
one run finishes first, its results must not become feedback for the other.

The next authorized trials use separate fresh roots with
`--max-iterations 25 --usd-cap 100 --cpus 6 --workers 6`. These change search
duration and its spending allowance, not the 20M-instruction window, HIGH
thinking, per-iteration repair limits, atomicity contract, or Pareto rule.

A completed build emits a `compliance_review_needed` event and waits for a
reviewer. Review the exact candidate source and its explanation against the
frozen contract: immutable admission-time departures, bounded causal state,
no explicit command scheduler, no hidden data access, generic parameters, and
physically interpretable rules. Do not supply model changes or accuracy hints.
Record a decision, for example:

```sh
.venv/bin/python tools/chia_loop/review_candidate.py /path/to/draft \
  --approve --reviewer REVIEWER_NAME --reviewer-kind agent \
  --reason "Source-specific justification of compliance."
```

Use `--reject` with a precise contract violation instead when needed. A rejected
draft receives that feedback and can be repaired by its proposing model. This
external compliance supervision is part of the protocol; the system is not
claimed to be an unsupervised single-model search.

## Editable interface and limits

Agents submit complete `includes` and `code` region bodies as JSON. The trusted
runner preserves all surrounding code. Model state, helpers, parameter defaults
and validation, `init_model()`, and `predict_departure()` are editable.
`model_param(name, default, minimum, maximum)` declares a numeric model
parameter during initialization. Optional public overrides use
`model_parameters=["name=value"]`; a run uses one shared configuration,
never workload-specific values. `controller_config()` supplies the resolved
buffer sizes and write-drain watermarks read-only. The DRAM specification is
also readable. Admission, callbacks, time, mapping, and observations stay frozen.

Per evaluated iteration, runaway guards allow 48 model turns, 192 inspections,
12 submitted drafts, and 60 API attempts including one transient retry per
turn. The final two turns reserve a submission/repair opportunity. Source
search, 1,200-line paging, final statistics, paired extremes, and 200-row
time/type-filtered training slices are available. No shell or arbitrary file
tool is provided. Open-loop/synthetic tools exist in the wider harness but
are not yet exposed through this real-agent adapter.

Input is counted before dispatch, with a 900,000-token limit and an 8 MB
transport guard; conversation content and returned signatures are preserved.
The crash-safe ledger reserves before each generation, includes thinking in
output usage, uses conservative model-specific tariffs, and retains unknown
transport-failure reservations. Known non-200 HTTP responses are unbilled
under Google's pricing policy. Estimates are not Cloud Billing invoices.

Builds use explicit `-O3`; total evaluation/build parallelism is at most 12.
Per process: 4 GiB address space, 180 CPU seconds for compilation, 600 CPU
seconds for simulation, and 8 GiB per simulator output file. These are
infrastructure guards, not accuracy thresholds.

## Results, audit, and storage

```sh
.venv/bin/python tools/chia_loop/analyze_gemini.py eval_out/chia/NEW_RUN
.venv/bin/python tools/chia_loop/audit_gemini_campaign.py eval_out/chia/NEW_RUN
.venv/bin/python tools/chia_loop/export_review.py eval_out/chia/NEW_RUN doc/results/NEW_RUN
```

After separately exporting two completed runs, create a comparison that
references their checksummed summaries:

```sh
.venv/bin/python tools/chia_loop/compare_reviews.py \
  doc/results/PRO_RUN_ID/summary.json doc/results/FLASH_RUN_ID/summary.json \
  --destination doc/results/COMPARISON_ID
```

The comparison checks matching setup, input/protocol identities, and source
checksums. Its table labels every row by run ID. It contains no joint budget,
search lineage, or promotion history. Repeated runs of the same model are also
supported; five iterations within one run are not five independent trials.

The run directory holds frozen execution/source snapshots, exact manifests,
provider ledgers, source lineage, CHIA execution profiles, per-workload metrics,
and plots. Raw traces are gzip-compressed immediately after each successful
simulation, checksum-verified, and read transparently by metrics/diagnostics.
Failed traces have separate archives and cannot become scoring evidence.
Large finalized interactions/logs/profiles are compressed too.

```sh
.venv/bin/python tools/eval/archive_results.py verify eval_out/chia/NEW_RUN/archive_manifest.json
.venv/bin/python tools/eval/archive_results.py restore eval_out/chia/NEW_RUN/archive_manifest.json
.venv/bin/python tools/chia_loop/artifacts.py restore eval_out/chia/NEW_RUN/aux_archive_manifest.json
```

Restoration is optional. Do not restore whole archives merely to compute
metrics. Generated data and interaction logs stay outside Git; a compact
review summary and selected result tables/plots can be curated separately.
The exporter requires a completed, audited run and a new destination. It
copies only selected sources, numeric summaries, and allowlisted tables/plots,
not the full run manifest, proposal explanations, or interaction records.
The runner never commits or pushes.

Generation retries are bounded to one retry per model turn, within the same
API-attempt and spending caps. The v7 runner recognizes transient HTTP responses
and HTTPX network, timeout, and remote-protocol failures. A lost response can
still be billed: its full reservation remains charged, and the retry needs a
separate affordable reservation before dispatch. Invalid local requests,
authentication errors, and exhausted retry limits stop the search. An operator
STOP is checked before every attempt. SDK-hidden retries remain disabled.
This policy improves resilience; it does not establish the cause of an upstream
disconnect or guarantee that a repeated request will succeed.

An early search stop can still produce a completed artifact record: the runner
freezes the incumbent, evaluates it on the final test, and verifies archives.
Always inspect `evaluated_designs` and the stopped trajectory entry; a completed
record does not mean that the configured iteration ceiling was reached. Once
test evaluation begins, do not resume that run for optimization. An authorized
fresh run must start clean and preserve any carried-forward budget charges.

The completed 2026-09-05 evidence used an older shared execution directory.
The reader supports that layout without rewriting its manifests, ledgers,
interactions, or compressed traces. Its export contains one `runs/RUN_ID/`
record per model, each with its own selected source, accounting, trajectory,
and numerical tables; the top-level `summary.json` is only a comparison with
references. Shared oracle/baseline trace archives are retained once. Shared
artifact counts must not be reported as each model run's sample count.

If evaluation completed but post-run archival failed, fix the specific storage
problem, then run `finalize_artifacts.py RUN_ROOT` and repeat the analysis/audit.
This command does not repeat evaluations or make generation calls. Failed-run
traces stay in separate archives with explicit ineligibility for metrics.
`archive_results.py recover-duplicate MANIFEST RAW_KEY DUPLICATE_GZIP` can restore
a damaged archive only from an exact match to its original compressed and raw
checksums; it preserves the damaged file and writes a recovery record. Never
replace expected checksums to make a damaged or changed result pass verification.

The original `atomic_loop.py --config tools/chia_loop/smoke.json` remains a
non-LLM plumbing test with a deliberately short ROI. It is not the real-model
experiment and its accuracy is not used as scientific evidence.
