# CHIA DDR5 workload expansion and frontend transfer

The optimization task is unchanged: synthesize one generic, immediate-response
atomic memory controller and minimize SimpleO3 core-cycle error and paired
request-latency error. The two metrics retain equal standing under strict
Pareto promotion. This revision broadens the workload population and adds
post-selection frontend transfer; it does not change DRAM standards, import
the independent feasibility design, or add frontend-specific tuning.

## Frozen evaluation profile

Both backend preparations accept `--evaluation-config PATH`. The default is
[`ddr5_frontend_transfer_v1.json`](../tools/chia_loop/configs/ddr5_frontend_transfer_v1.json).

| Stage | Population | Execution window | Agent feedback? |
| --- | --- | --- | --- |
| SimpleO3 training | 8 application families | 20M issued instructions/core, cold start, full drain | Yes |
| SimpleO3 final test | 8 different application families | Same full window | No |
| ChampSim transfer | 6 SPEC17 traces | 2M warmup + 20M measured instructions | No |
| gem5 transfer | 12 SE programs, one O3 core | Whole program to successful normal exit | No |

Training: `429.mcf`, `519.lbm`, `401.bzip2`, `436.cactusADM`,
`462.libquantum`, `483.xalancbmk`, `434.zeusmp`, `482.sphinx3`.

Final test: `433.milc`, `450.soplex`, `459.GemsFDTD`, `549.fotonik3d`,
`403.gcc`, `471.omnetpp`, `435.gromacs`, `444.namd`.

ChampSim: `603.bwaves_s-1080B`, `605.mcf_s-1554B`, `619.lbm_s-2676B`,
`623.xalancbmk_s-165B`, `649.fotonik3d_s-1176B`, `654.roms_s-1021B`.

gem5: `stream`, `gups`, `ptrchase`, `matmul`, `bfs`, `pr`, `pb_atax`,
`pb_mvt`, `pb_bicg`, `pb_gemm`, `pb_jacobi2d`, `pb_heat3d`.

Training adequacy requires at least 10,000 oracle DRAM-owner reads per case.
The profile was screened using oracle traffic, not evolved-model accuracy.
Low-memory-traffic programs remain in the test population: weak DRAM exposure
is a real workload property, not grounds for silently dropping a test result.

Preparation checks application-family disjointness, including SPEC versions
and trace intervals; duplicate content hashes; sufficient trace length without
wrapping; optimized runtime provenance; and external frontend/benchmark input
identities. Profiles are pinned before any model call. Each run, backend and
reasoning effort retains its own configuration, history and usage ledger.

Application transfer and frontend transfer are different claims. The ChampSim
mcf/lbm/xalancbmk cases share application families with training; fotonik3d
shares a SimpleO3 test family. Reports identify these groups separately from
previously unseen families. Existing operator exposure to old test results is
not erased by renaming or expanding a cohort; this is not a new pristine
study-level holdout.

## Controller and comparison contract

All cases use DDR5 `DDR5_16Gb_x8` / `DDR5_4800AN`, one channel, 64-entry
candidate read/write capacities, and disabled refresh. The oracle remains
GenericDDR with FRFCFS-RowHit, open rows, and 0.5/0.8 write-drain watermarks.
The published comparison implementations and shared parameters are unchanged:
FixedLat, MD1, Sniper-style channel-level WMG1, and MESS. MESS keeps its existing
checksum-bound DDR5 calibration; it is not recalibrated on transfer data.
Baseline controller traces now also retain frontend ID, sub-ID and admission
ordinal. This instrumentation-only extension enables stable request pairing
without changing baseline prediction or admission logic.

Transfer loads the exact selected `candidate.so` with the same trusted
`libramulator.so` used in training. Source, build, binary and immutable
selection hashes must agree. Neither the current working-tree Atomic source
nor another run's candidate can substitute for the selection. The unchanged
seed runs alongside the selection and all four comparisons.

ChampSim uses the installed bridge's `RAMULATOR_TICKS_PER_8=12` convention:
12 controller ticks per eight 625 ps memory-controller operations, or a
416.67 ps mean controller period. Preparation checks that generated frontend
clock and fingerprints both the input and generated ChampSim configuration.
gem5 uses a 3.2 GHz O3 CPU, private 32 KiB instruction/data L1 caches, 1 MiB L2,
and 3 GiB memory. These frontend settings are fixed, not optimized by agents.
Native controller/runtime builds require `-O3`; external executables and their
optimized build recipes are fingerprinted. Existing frontend binaries are not
represented as freshly rebuilt or cryptographically source-attested binaries.

## Metrics and completeness

SimpleO3 metrics and their formulas are unchanged; see the
[metric definitions](../tools/eval/README.md#metric-definitions).

For ChampSim, core error is signed ROI core-cycle deviation from the oracle.
For gem5, it is signed whole-program `simTicks` deviation at the same CPU
frequency; executed instruction counts must agree. The headline is the
unweighted mean absolute per-workload error within each frontend.

Transfer request traces describe **DRAM-controller** transactions, not all
logical LLC accesses. Let `d_i` be model minus oracle latency for paired read
`i`, and let `L` be that workload's mean latency over **all oracle controller
reads**, including unmatched reads. Every latency uses controller-cycle units:

- MAE/L = `mean(abs(d_i)) / L`.
- Signed drift/L = `mean(d_i) / L`; the macro diagnostic averages its absolute
  per-workload value.
- Paired P99/L = `percentile_99(abs(d_i)) / L`.
- Signed extremes/L = `min(d_i) / L` and `max(d_i) / L`.

Stable logical IDs pair reads; ChampSim additionally requires physical-address
agreement within those logical pairs and reports mismatches. A request headline
requires exact bidirectional coverage of all read populations with eligible
stable IDs. Otherwise the matched-subset error remains explicitly diagnostic,
with counts, coverage and eligibility shown; no partial subset is presented as
an exact full-population score. Empty or invalid matching fails closed.

Controller traces cover the simulation lifecycle, not a newly introduced
ROI-only request slice; ChampSim warmup may bypass DRAM. Reports record this
scope. SimpleO3 and external-frontend request values must not be pooled into
one headline because their populations and observation boundaries differ.

## Isolation, recovery and storage

Application-trace diagnostic tools accept only SimpleO3 training names/data.
The optional [synthetic diagnostic tool](chia_diagnostics_and_ablations.md)
accepts bounded generic generator parameters instead of workload files. It is
also training-only and cannot access transfer inputs or scores.
Transfer begins after scientific termination and a matching immutable freeze
receipt; it never changes the incumbent or feeds a subsequent proposal. A
transfer failure preserves the frozen selection and completed cases. Recovery
can finish evaluation but cannot restart training.

External simulators need their specific workload files while executing. Their
launch sandbox allows those files, runtime assets and a per-case output
directory, and denies unrelated runs, repository history, credentials and
network access. This is not the stronger SimpleO3 guarantee that input files
are inaccessible after candidate loading. The source/compliance contract also
forbids model I/O; no optimization agent receives transfer output.

Each completed case has a checksum-bearing manifest and verified gzip trace.
Interrupted outputs remain separate and ineligible. Resume reuses a case only
when its selection, plugin, runtime, input profile and trace identities match.
The root archive manifest includes transfer traces; matching reads gzip
directly. Per-frontend JSON, Markdown, CSV and SVG reports retain all models
and per-workload diagnostics. Auxiliary logs are compressed after completion.

## Use and validation

Pass `--evaluation-config` to `prepare_gemini.py` or
`python -m tools.chia_loop.codex_cli prepare`; the Astra dependency queue also
pins this argument. Preparing is not authorization to launch paid generation.
The expanded scientific protocol requires fresh campaign roots. Historical
roots and their frozen protocol snapshots remain unchanged.

The backend-free integration fixture runs the fixed seed and makes no LLM
calls:

```sh
PYTHONPATH=python:tools:. python -m tools.chia_loop.validate_evaluation \
  --root eval_out/chia/UNIQUE_VALIDATION_ID --workers 4 --transfer-smoke
```

The fixture covers every configured SimpleO3 workload and one full-length case
per external frontend. Omit `--transfer-smoke` to validate the full transfer
matrix. It is not a trained candidate or a scientific performance result.
All jobs participate in the shared twelve-CPU lease pool; a run's worker count
must fit its allocation. No generation, commit or push happens automatically.

## Validation on 2026-09-06 and remaining limitation

The completed fixed-seed fixture is
`eval_out/chia/ddr5_transfer_validation_v3_20260906`:

- 96 full-window SimpleO3 cases: 16 workloads × oracle, seed and four baselines.
- 14 full-length external cases: ChampSim bwaves and gem5 stream × oracle,
  seed, frozen fixture and four baselines. The full 6/12 external input matrix
  was also inventoried; it has not yet been evaluated with an evolved model.
- All 96 SimpleO3 cycle/statistic records and logical request-trace hashes
  match the pre-instrumentation results exactly.
- All gem5 cases executed 35,783,611 instructions and exited normally.
- 206 traces passed archive verification: 7.51 GB raw became 2.32 GB gzip
  (69.1% smaller). These are fixture storage totals, not campaign sample counts.

The fixture's ChampSim selected/oracle pair covers 69,877 of 71,597 reads
(97.60%). Its error is conditional on that matched subset, not an exact
headline. gem5 stream matches only 13 of approximately 3.93 million reads,
despite nearly all reads carrying IDs. The existing bridge propagates O3
dynamic sequence numbers, which include squashed instructions and are only
provisional identities across differently paced executions. Consequently,
**gem5 request accuracy is not interpretable as a whole-workload result**.
Core-cycle transfer remains measurable; the pipeline does not substitute
address-occurrence matching or report missing request accuracy as zero.

A robust cross-run gem5 identity/instrumentation design is a separate required
step before claiming closed-loop gem5 request accuracy. Raw subset diagnostics
remain in the CSV/JSON with counts and eligibility. When coverage is incomplete,
the figure displays pairing coverage instead of an apparently favorable request
error bar. These tests establish pipeline operation, not a trained controller's
accuracy or transfer success.
