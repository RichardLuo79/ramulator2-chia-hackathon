# CHIA: evolving immediate-response DRAM controllers

This branch contains a clean fixed-delay Atomic seed, the protected evaluation
harness, four comparison models, and a native CHIA/Ray evolution graph.
See [project status and results](../../doc/chia_hackathon_review.md) for the
scientific scope and current Gemini results. Setup conversations, unrelated
DRAM timing audits, licensed workload traces, and raw interaction logs are not
part of this repository.

## Current experiment

The real backend compares Gemini 3.1 Pro Preview with Gemini 3.8 Flash through
Google Cloud ADC. Each independent arm starts clean, uses HIGH thinking and
65,536 maximum output tokens, and stops after five evaluated designs or its
USD 50 spending/safety limit. Failed drafts are repairable within an iteration.
Every draft, API attempt, tool result, build, compliance decision, and score is
retained locally; no hidden source repair or automatic paid resume occurs.

Training uses mcf/lbm; final testing uses milc/soplex/GemsFDTD/fotonik3d, only
after both selections are frozen. Every case executes 20 million issued
instructions per core, from a cold start through complete drain. The real
runner rejects shorter windows. It checks trace length, disjoint input hashes,
oracle DRAM traffic, exact request pairing, and callback integrity. The
published comparison models are FixedLat, MD1, Sniper WMG1, and MESS.

An independent earlier feasibility design and all earlier campaigns are
unavailable to optimization agents. Only their own campaign history, selected
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
.venv/bin/python tools/chia_loop/prepare_gemini.py --root eval_out/chia/NEW_RUN --workers 6
.venv/bin/python tools/chia_loop/gemini_loop.py --root eval_out/chia/NEW_RUN
```

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
`model_parameters=["name=value"]`; a campaign uses one shared configuration,
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
```

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
The runner never commits or pushes.

The original `atomic_loop.py --config tools/chia_loop/smoke.json` remains a
non-LLM plumbing test with a deliberately short ROI. It is not the real-model
experiment and its accuracy is not used as scientific evidence.
