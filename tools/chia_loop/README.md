# CHIA: evolving immediate-response DRAM controllers

This branch contains a clean fixed-delay Atomic seed, the protected evaluation
harness, four comparison models, and a native CHIA/Ray evolution graph.
See [project status and results](../../doc/chia_hackathon_review.md) for the
scientific scope and current Gemini results. Setup conversations, unrelated
DRAM timing audits, licensed workload traces, and raw interaction logs are not
part of this repository.

## Current experiment

One invocation runs exactly one proposing model and effort. Gemini 3.1 Pro
Preview and Gemini 3.8 Flash use Google Cloud ADC; the isolated
[Astra backend](../../doc/chia_codex_cli.md) and
[Fable backend](../../doc/chia_claude_cli.md) use their native CLI subscription
logins. Runs are compared only in downstream analysis and do not exchange
candidates or conversations. An explicitly requested same-run continuation is
distinct from a fresh experiment.

Gemini uses HIGH thinking and 65,536 maximum output tokens. Its generic defaults
are five evaluated designs and USD 50; current rich-DDR5 runs were explicitly
configured for 20 designs and USD 100 each. Astra and Fable support independent
xhigh/max runs with iteration-only guards and API-equivalent usage accounting.
All backends retain finite operational limits, recorded automatic compliance
review, and repairable drafts. There is no implicit model or billing fallback.
Every draft, API attempt, tool result, build, compliance decision, and score is
retained locally; no hidden source repair occurs. The supervisor can recover
the same run automatically under its unchanged, frozen protocol and budget.
See [unattended operation](../../doc/chia_unattended_runs.md).

Create a `STOP` file in the run root to stop safely after any in-flight
generation settles and before another generation/evaluation starts. A fresh
restart after an audited infrastructure abort can use
`prepare_gemini.py --carry-budget-from OLD_RUN`; financial charges are carried
forward without importing model sources or feedback. The USD 50 authorization
is not replenished by restarting. This option requires that the aborted run
evaluated no candidate and that all generation usage is settled.

Historical runs used mcf/lbm for training and
milc/soplex/GemsFDTD/fotonik3d for final testing, only
after that run's selection was frozen. New preparations use the expanded DDR5
profile described below. The historical shared execution waited
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

## Expanded DDR5 evaluation

New Gemini, Astra and Fable preparations accept `--evaluation-config PATH`, defaulting
to [ddr5_frontend_transfer_v1.json](configs/ddr5_frontend_transfer_v1.json).
This contains 8 SimpleO3 training families, 8 disjoint test families, 6 ChampSim
traces, and 12 gem5 SE programs. DDR5 organization/timing, candidate parameters,
and four comparison models stay fixed across frontends. SimpleO3 remains the
only training frontend. Neither transfer scores nor transfer traces enter an
agent prompt, tool response, compliance review, or promotion decision.

The profile and input identities are frozen before generation. Alias hashes,
application families across SPEC versions/intervals, available instruction
counts, and training oracle traffic are checked. Legacy roots without a
profile retain their historical 2/4 split; they are not silently upgraded.
Changing cohorts requires a new campaign, not an in-place continuation with
old scores. See [the full protocol](../../doc/chia_ddr5_frontend_transfer.md).

After selection freezes, `transfer.py` loads the exact selected DSO against the
same trusted O3 runtime, plus the unchanged seed, oracle, FixedLat, MD1, WMG1,
and MESS. ChampSim uses 2M warmup + 20M measured instructions; gem5 executes
whole benchmark programs to normal exit with an O3 CPU. Transfer request
errors are controller-level, unlike SimpleO3 logical LLC-request errors.
Reports keep frontend results separate and label incomplete pairing as
diagnostic, not an eligible exact-coverage headline score.

Each frontend writes `transfer/reports/FRONTEND.json`, `.md`, `.svg`, and
`FRONTEND_per_workload.csv`. Full results preserve latency MAE, signed drift,
paired P99, signed extremes, sample counts, and bidirectional coverage. Traces
are immediately checksum-verified and gzip-compressed; readers need no bulk
decompression. Completed cases are reused only when their identities match.

Validate without a model backend or login:

```sh
PYTHONPATH=python:tools:. python -m tools.chia_loop.validate_evaluation \
  --root eval_out/chia/NEW_VALIDATION_ID --workers 4 --transfer-smoke
```

This evaluates all 16 SimpleO3 cases and one **full-length** case per transfer
frontend, with the fixed seed standing in for a frozen selection. It is a
pipeline fixture, not an evolution campaign or an accuracy claim. Omit
`--transfer-smoke` for the full external matrix. No model calls are made.

## Optional synthetic diagnostics and ablations

New Gemini/Astra/Fable preparations and their applicable queues accept `--loop-config PATH`.
The [default profile](configs/loop_default_v1.json) enables an agent-callable
synthetic diagnostic tool; [the ablation profile](configs/loop_no_synthetic_v1.json)
disables it. The same configuration controls training trace/statistic access,
automatic score/history feedback, source visibility, parent selection and
per-iteration search limits. Effective settings are fully resolved and frozen
before generation. Correctness, isolation, exact scoring and accounting cannot
be disabled.

Synthetic experiments reuse only the generic generator in this branch, testing
the parent's exact optimized DSO against the oracle. They provide trace-backed
training evidence, never promotion scores or held-out feedback. Traces are
compressed and checksummed immediately. No independent controller findings or
model-specific scenarios are imported. See the
[instrument and ablation guide](../../doc/chia_diagnostics_and_ablations.md)
for the tool schema, metric definitions, configuration keys and no-LLM native
validation command. Existing campaigns require their frozen protocol; this
feature starts with fresh runs, not an in-place upgrade.

## Setup

Use Linux with Landlock ABI >= 3, libseccomp, a C++20 compiler, CMake, and Python
3.10 or newer. The recorded experiments use Python 3.12. Configure the external
trace paths described in [the evaluator guide](../eval/README.md); workload
files are not redistributed. The selected evaluation profile determines the
required traces, frontend binaries, and gem5 benchmark programs; unavailable
inputs fail preparation rather than silently shrinking the matrix.

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
.venv/bin/python tools/chia_loop/supervise_gemini.py --root eval_out/chia/PRO_RUN_ID --model pro
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

The earlier 25-iteration trials used separate fresh roots with
`--max-iterations 25 --usd-cap 100 --cpus 6 --workers 6`. These change search
duration and its spending allowance, not the 20M-instruction window, HIGH
thinking, per-iteration repair limits, atomicity contract, or Pareto rule.
The expanded frontend-transfer profile is not a new paid-run authorization.

A completed build triggers an isolated API compliance review automatically.
Both proposing backends use the same Gemini 3.1 Pro reviewer and frozen
`prompts/compliance_v1.md` rubric. The reviewer gets one source and its technical
explanation, not history, scores, other runs, or tools. Its calls count inside
this run's budget and are labeled separately in the ledger and audit.
Pass requires a source-bound affirmative decision on every rule. Reject and
uncertain decisions become repair feedback; they never silently pass or wait
for a human approval file. Only the proposing model edits its own draft.
`review_candidate.py` remains for historical externally reviewed protocols;
v8 does not accept such a file as an automatic review. This is unattended
multi-role evaluation, not a claim of formal correctness or no LLM reviewer.

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
12 submitted drafts, and 60 generation attempts including proposer, reviewer, and
retries. The final two turns reserve a submission/repair opportunity. Source
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

Builds use explicit `-O3`; new runners on this checkout share twelve CPU leases.
Preparation/other evaluation jobs must also stay within the overall 12-core allowance.
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

Generation uses up to two immediate retries with increasing delays and jitter,
then a persisted cooldown. Recovery is bounded by eight transient failures per
operation, one hour of continuous outage, 24 hours per run, three abrupt-process
restarts, and the existing attempt/spending caps. New client connections are
used for new attempts. Count-token failures also enter operational recovery.
Lost responses retain their full reservations; every retry must be separately
affordable. Saved complete responses are replayed without a new API call.
SDK-hidden retries remain disabled. None of this proves a disconnect was
Google's fault or guarantees eventual service availability.

Operational failure pauses training without freezing or testing the incumbent.
Only iteration completion, explicit no-change, or declared budget/search guards
can end selection. Authentication, integrity, disk-space and persistent service
failures stop safely for investigation. Resumption verifies protocol, prompts,
configuration, dependency versions, runtime and candidate identities. Completed
builds/evaluations are reused only with matching identities and verified traces.
Use `supervise_gemini.py --root RUN --model pro --resume` to reattach to the
same v8 run; it cannot upgrade historical runs in place or replenish a budget.

Once a source is frozen, recovery may finish that source's final evaluation or
archival, but must never re-enter optimization. Historical v6/v7 interruptions
were frozen and tested; those remain unchanged and require a fresh experiment,
with previously incurred charges preserved under the original authorization.

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
