# CHIA framework: implementation and validation

Updated 2026-09-09. The active code is
[`tools/chia_loop/framework/`](../tools/chia_loop/framework/README.md);
the [design](chia_framework_design.md) defines its scientific contract.

## Implemented

- One campaign driver for Gemini, Astra and Fable, using CHIA native adapters,
  authenticated tools, Ray jobs and SQLite primitives.
- Continuous proposer sessions within an iteration; separate same-model,
  same-effort advisory review; model-written summaries across iterations.
- A restricted atomic-model API with a fixed-delay seed. The trusted wrapper
  owns admission, capacities, mapping, callbacks, clocks and observations.
- Full-window SimpleO3 DDR5 training, unchanged comparison models, generic
  synthetic diagnostics and open-loop replay. Promotion uses unrounded
  core-cycle and paired-request errors.
- Positive workspace/tool grants and private CLI profiles. Credentials remain
  in a trusted fixed-provider relay, not the agent's filesystem or history.
- Source-bound evaluations, compressed traces/native records, usage accounting,
  checkpointed phases and bounded retries. Codex capacity errors default to a
  one-hour cooldown; retry counts and prior usage survive recovery.
- Source-only ZIP64 export with checksums. External projects use source pins,
  patches and build recipes, not bundled executables or environments.

The [CHIA extension](../tools/chia_loop/framework/upstream/README.md) is pinned
by revision and source hashes. It supplies single-attempt session hooks,
unchanged native continuation, authenticated MCP configuration and raw evidence.
The Codex classifier uses native failure events rather than searching generated
answers for error codes. The legacy runners remain separate compatibility code.

## Validation evidence

Pre-commit regression checks passed: **262** common-framework/public-suite tests
(12 opt-in skips, one upstream Pydantic warning) and **200** backend, metric and
compatibility tests. Newly added Python files pass formatting and lint checks.

The existing optimized runtime was built from 130 translation units, all
checked for `-O3`. Its source inventory and build logs are retained locally.
The cleanup verified all 790 recorded build-source files against that inventory
without changing the live runtime. Standalone optimized checks passed 1,500
simulation-lifecycle parity cases, 1,000 batch-replay cases, and trusted
admission/callback checks.

A representative no-provider check completed oracle, seed and four comparison
evaluations on mcf at 20M instructions, with all 1,477,776 logical reads paired.
Scripted common-loop checks exercised two iterations, synthetic/open-loop
diagnostics, actual authenticated CHIA MCP transport, summaries, promotion,
checkpoint reuse and frozen milc evaluation. They qualify the harness, not
model optimization.

A source/data archive from the native MCP check was verified and extracted.
The source-only guest/export regression rebuilds a small guest from an extracted
archive. This does not establish offline reconstruction of an entire external
frontend installation.

The clean gem5 v25.1.0.0 build passed an O3/DDR5 installation check. The twelve
[new public guests](../tools/eval/gem5/PUBLIC_SUITE.md) built with `-O3` and ran
natively; all six GAP numerical verifiers passed. Full gem5 transfer
qualification remains incomplete.

Fresh paid Gemini and Astra searches have completed and promoted candidates.
The [dated results snapshot](chia_hackathon_review.md#current-common-framework-runs)
reports training results separately from earlier protocols and held-out tests.

## Known limitations and next work

1. Finish the ongoing search campaigns and evaluate their frozen selections on
   held-out SimpleO3. Do not present training scores as generalization results.
2. Resolve the reproducible Claude runtime permission failure. A monitored
   two-iteration retry stopped on repeated memory-tool errors before completing
   its first iteration. Live evidence showed a denied `/proc/self/stat` read,
   not physical-memory exhaustion; the precise trigger is unresolved.
3. Complete whole-program gem5 qualification. Several GAP cases reached the
   30-minute simulation guard; partial traces are not completed measurements.
   The v2 public guests are a new cohort, not reruns of the source-missing
   historical executables.
4. Avoid unnecessary full trace decompression on resume once an identical
   compressed checksum can be matched to a prior complete validation record.
   Current checks are local CPU work, not LLM calls.
5. Perform post-campaign review and final source/evidence export after results
   settle. Docker workers remain optional and deferred.

The local boundary restricts filesystem access and TCP ports; it is not an
IP-level network namespace. Native CLI startup/offline replay tests do not prove
all long-running runtime behavior, as the Claude failure demonstrates.
The common launcher has no dollar-cap implementation; current runs use explicit
iteration limits and retain API-equivalent usage estimates.
