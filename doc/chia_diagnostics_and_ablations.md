# CHIA training diagnostics and feature ablations

The Gemini, Astra and Fable pipelines expose the same optional synthetic diagnostic
instrument. It reuses this branch's generic `SyntheticPattern` generator and
DRAM-layout helper, not the independent feasibility controller, its findings,
or model-specific diagnostic cases. The agent chooses its experiments from
documented generator parameters. Nothing from another campaign is agent input.

Current implementations are Gemini v11, Astra v7, and Fable v1. Use fresh campaign
roots. Existing completed runs, frozen protocol snapshots, and scores are not
upgraded or recomputed by this change.

## Configuration layers

All backend preparations and the Astra/Fable queues accept `--loop-config PATH`.
The default is
[`loop_default_v1.json`](../tools/chia_loop/configs/loop_default_v1.json).
Profiles accept partial overrides of the strictly validated defaults in
[`loop_config.py`](../tools/chia_loop/loop_config.py). Unknown fields, wrong
types, and out-of-range limits fail preparation rather than being ignored.

| Configuration | Controls |
| --- | --- |
| `--evaluation-config` | Workload families, full instruction windows, DDR5 evaluation and post-freeze frontend transfer |
| `--loop-config` | Agent-visible evidence, optional diagnostic tools, parent selection, per-iteration search and synthetic limits |
| Backend CLI / run configuration | Model or Astra effort, total evaluated iterations, authorized usage guard, authentication and CPU allocation |
| Frozen prompts and protocol | Scientific contract, explanation requirements, mandatory validity checks, scoring and accounting |

Preparation saves **fully resolved** `loop_config.json` and
`loop_config_identity.json`, not merely the overrides. Their hashes, effective
settings, source protocol and prompts enter the run manifest before generation.
The delayed Astra queue pins the source profile and its resolved settings before
waiting, then verifies them again before launch. Changing settings requires a
new run; scientific continuation rejects different loop/evaluation profiles.
Historical roots without this profile retain their historical interface in
read-only/configuration helpers; their saved protocol is required to resume.

## Selectable features

All switches below default to `true` for new runs. Enforcement is in the
dispatcher and prompt assembly, not just a request to the model.

| `features` key | Effect when disabled |
| --- | --- |
| `synthetic_diagnostics` | Removes the callable synthetic instrument and its tool schema; execution is rejected |
| `training_statistics` | Disables the training final-statistics inspection action |
| `logical_trace_diagnostics` | Disables training logical-request trace slices |
| `controller_trace_diagnostics` | Disables training DRAM-controller trace slices |
| `request_extremes` | Disables training paired-extreme inspection and initial extreme slices |
| `per_workload_feedback` | Omits automatic per-workload score tables, including comparison score tables |
| `comparison_feedback` | Omits comparison scores from agent prompts; comparisons are still evaluated and reported |
| `evolution_history` | Omits previous-iteration history and the automatic archive score summary |
| `oracle_source` | Removes GenericDDR, its shared controller base and the FRFCFS scheduler from readable files |
| `comparison_source` | Removes the four comparison implementations and their initialization helper from readable files |

These are distinct access channels, not guarantees that overlapping information
cannot be derived. For example, disabling score tables still permits per-workload
trace inspection if enabled; disabling comparison scores does not hide their
source. `evolution_history=false` still supplies the current parent and required
scores, and preserves conversation/repair feedback within an iteration.
Training trace switches do not disable the separate synthetic instrument's
statistics or slices. To test aggregate-only feedback, explicitly disable all
diagnostic channels as well as per-workload tables.

`search.parent_selection` supports `pareto_round_robin` (default) or `incumbent`.
The former rotates through the sorted nondominated archive; the latter always
uses the incumbent. Both retain exactly the same two-objective promotion rule:
neither application error may increase and at least one must decrease.

`limits` controls model turns, diagnostic calls and draft repairs per evaluated
iteration, plus the number of initially supplied extreme rows. Defaults remain
48 turns, 192 inspection calls, 12 drafts and 3 initial rows. The last two turns
remain reserved for finalization. A synthetic action counts as an inspection;
its case count has an additional bound. Provider/context limits, retry guards
and overall wall-time guards remain operative even if these limits are raised.

Atomicity, bounded causal state, immutable lifecycle/instrumentation, exact
application request pairing, compliance review, held-out isolation, source-bound
builds, usage accounting and optimized builds are mandatory. They are not
feature ablations. All evaluations share the existing twelve-CPU lease pool.

## Agent-callable synthetic experiments

An enabled agent can return an inspection action such as:

```json
{
  "status": "inspect",
  "requests": [{
    "tool": "synthetic_diagnostics",
    "cases": [{"mlp": 16, "row_run": 16, "bank_spread": 4}],
    "limit": 8
  }]
}
```

This illustrates the syntax, not a prescribed diagnostic or modeling rule.
The runtime tool schema describes the available axes: stream composition and
shared regions; row-run length and bank spread/randomness; read dependence,
MLP, issue delay and jitter; writeback probability, destination and batching;
active/idle phases; and alternating parameter regimes. `mode_b` is a typed
axis-override object and `stream_params` a list of such objects. The adapter
constructs the generator's strings; agent-supplied scripts, paths and config
trees are not accepted. Organization-dependent bounds are validated explicitly.

`num_requests` means **reads per stream**, not total requests or CPU instructions.
Defaults are 50,000 reads per stream, seed 1, at most 200,000 reads per case,
4 cases per call, 128 distinct parent/parameter cases per run, and 120 CPU
seconds per simulator process. Trace responses default to 8 rows per slice and
are capped at 100 rows per slice. Each bound is configurable within the
validated resource envelope; the minimum population is 1,000 reads per stream.
These controlled microexperiments do not shorten the 20M-instruction application
training/test windows. Repeated identical cases do not consume another unique
case reservation; a different parent binary does.

Every case runs serially against the training oracle and **the supplied parent's
exact `candidate.so`**, built at `-O3` with its source/build hashes verified.
The trusted library excludes the checkout's Atomic implementation. No candidate
path, controller override, test workload or arbitrary program can be requested.
The native isolated runner denies the candidate access to inputs, other runs,
repository history, credentials and networking after loading its frontend.
The tool becomes inaccessible once selection freezes or held-out testing starts.

The generator starts cold and the simulation drains all admitted read and write
callbacks. The generic generator now counts write completions: previously it
could stop after admitting its final writes, while callbacks remained pending.
This observation/lifecycle correction does not change the address recipe or add
controller rules. The reference's write callback is its configured completion
convention (including synchronous coalescing), not a new physical-bus-drain model.

## Diagnostic metrics and matching

There is no CPU or LLC in this experiment. Frontend and memory clocks both tick
in **DRAM cycles**. Synthetic elapsed-cycle error is not application core-cycle
error and is never pooled with it.

For each stream, the generator's read address sequence is deterministic by read
index. Request admission times are unique because it makes at most one send
attempt per tick. The adapter orders reads by `(source, arrival)`, assigns a
per-stream ordinal, and pairs `(source, ordinal)` across sides. It requires
the exact expected read count for every stream and address equality for every
pair. It does not pair by completion order or infer matches from repeated
addresses. Missing reads, ambiguous order, mismatched addresses or callback
statistics invalidate the result; no favorable partial score is substituted.
This is generator-specific ordinal pairing, not frontend-provided stable IDs.

Let `d_i = candidate_latency_i - oracle_latency_i`, with each latency equal to
`departure - arrival`. Here `L` is the mean latency of **all oracle synthetic
reads in that case**, in DRAM cycles. The response includes:

- `request_mae_over_L = mean(abs(d_i)) / L`.
- `signed_drift_over_L = mean(d_i) / L`, and its absolute value.
- `paired_p99_over_L = percentile_99(abs(d_i)) / L`.
- Minimum and maximum signed errors in cycles; divide by `L` to normalize.
- `synthetic_elapsed_error_pct = 100 * (candidate_cycles - oracle_cycles) / oracle_cycles`.
- Read coverage/counts, frontend final statistics, first-read slices and the
  most-negative/most-positive paired slices with addresses and timestamps.

Arrival schedules are independent closed loops, not aligned events. Read/write
interaction affects latency, but only reads are paired. Callback-dependent
writeback timing—and under mode switching, write population—may differ between
sides; write counts are retained separately. Mode 0 is read-only, regardless of
`wfrac_pct`. Zero-latency coalesced oracle writes are valid, while read latencies
and the atomic candidate's committed latencies must be positive.

These measurements inform hypotheses only. The two promotion objectives remain
full-window SimpleO3 core-cycle MAE and logical-request MAE/L. ChampSim and gem5
remain post-freeze DDR5 transfer evaluations, never optimization feedback.

## Evidence, compression and failure handling

`diagnostics/synthetic_calls/` records each request, parent identity, start/end
time, result/error and cache use, in addition to the backend's model conversation.
`diagnostics/synthetic/<case-hash>/` retains resolved parameters, source/plugin,
runtime, generator, layout-helper and adapter hashes, native configuration,
simulation logs/statistics, metric receipts and per-side trace manifests.

Traces are gzip-compressed immediately after completion and verified against
their raw identities. Analysis reads gzip directly; no bulk decompression is
required. Cached results require matching manifests, report hashes and verified
archives. The campaign restoration manifest includes synthetic traces.
Failed attempts remain separate and ineligible; cache corruption is an
operational failure, not modeling feedback. No automatic deletion occurs.

## Running an ablation

The supplied
[`loop_no_synthetic_v1.json`](../tools/chia_loop/configs/loop_no_synthetic_v1.json)
differs from the default only in its experiment name and synthetic-tool switch:

```json
{
  "schema_version": 1,
  "name": "ablation_no_synthetic_v1",
  "features": {"synthetic_diagnostics": false}
}
```

Pass one of these profiles to `prepare_gemini.py`,
`python -m tools.chia_loop.codex_cli prepare`, or the Astra dependency queue.
Keep evaluation profiles, model/effort, iteration and usage guards, prompts,
reviewer, CPU allocation and other settings the same across independent fresh
runs. Start both from the same skeleton; do not initialize the ablated run with
a design or conversation trained using the removed feature. Repeat trials to
separate a feature effect from stochastic model behavior. Compare final errors,
learning curves, usage and diagnostic cost, not just the best observed run.

No feature-effect or accuracy improvement is claimed merely because this
instrument passes its integration tests. Preparing an ablation does not authorize
paid generation, sharing results across runs, committing or pushing anything.

## Offline/native validation

```sh
PYTHONPATH=python:tools:. python -m pytest -q \
  tests/unit_tests/test_chia_loop_config.py \
  tests/unit_tests/test_chia_synthetic_diagnostics.py

PYTHONPATH=python:tools:. python -m tools.chia_loop.validate_synthetic \
  --root eval_out/chia/UNIQUE_SYNTHETIC_VALIDATION_ID
```

The native fixture builds at `-O3` under a three-slot shared lease and exercises
eight generic parameter combinations (all write modes, locality/parallelism,
dependencies, stream sharing and phases). It also checks cached reuse and a
separate fixed-delay DSO fixture that must change measured latency by exactly
17 cycles, proving the selected plugin is used. Each stream has 50,000 reads;
the fixture runs 18 simulations and verifies 18 compressed traces, then confirms
that post-freeze calls are rejected. It makes no model calls and produces no
evolved controller or application accuracy claim.
