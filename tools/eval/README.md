# Atomic-controller evaluation harness

This raw-first harness compares one immediate-response candidate with a
cycle-level Ramulator oracle and published immediate-response baselines. Runners
record per-core cycles, controller statistics, and request traces. Metrics are
derived only after all required artifacts have been validated.

The first CHIA proof of concept executes only SimpleO3 + DDR5. The same harness
contains forward plumbing for LPDDR5, LPDDR6, HBM4, ChampSim, and gem5; those
targets require their own full campaigns before any accuracy claim.

## Models

| Label | Meaning and origin |
|---|---|
| `oracle` | `GenericDDR`, FRFCFS-RowHit, open rows, no refresh, 64-entry read/write queues, and 0.5/0.8 write-drain watermarks. |
| `candidate` | One generic `Atomic` implementation. Its seed is a fixed positive delay with bounded read/write admission and deterministic completion; it has no bank, row, command-bus, scheduler, write-drain, or refresh model. |
| `fixedlat` | Fixed access latency plus an optional deterministic peak-bandwidth pipe, matching the abstraction used by gem5 `SimpleMemory` and, without the pipe, zsim `SimpleMemory`. |
| `md1` | zsim-style whole-memory M/D/1 formula using a smoothed aggregate request rate and deterministic burst service time. |
| `wmg1` | Sniper `QueueModelWindowedMG1` semantics at channel scope: windowed arrival/service statistics feed one aggregate M/G/1 waiting-time formula. It has no independently serialized bank or bus resources. |
| `mess` | MESS-style feedback onto read-ratio-specific bandwidth/latency curves. The checked-in DDR5 curve is calibrated against this harness's oracle and is checksum-bound as an external input. |

Every reported split includes all four comparison models. They provide context;
none is silently promoted to the oracle.

## Workload and run contracts

The proof-of-concept split is fixed in `tools/chia_loop/smoke.json`:

- Training: `429.mcf`, `519.lbm`
- Validation: `603.bwaves_s-1080B`, `654.roms_s-1021B`

The smoke ROI is 50,000 instructions per core. It tests orchestration and data
contracts only. Full SimpleO3 campaigns retain 20 million instructions for
single-core workloads and 10 million per core for mixes.

All evaluation processes must use a Release build containing an explicit
`-O3`. `config.py` records and checks the CMake cache and target flags, and all
runners accept at most 12 workers.

## Metric definitions

For core `c` of workload `w`, let `C_o` and `C_m` be oracle and model cycles:

```text
core signed error (%) = 100 * (C_m - C_o) / C_o
```

The workload cycle error is the mean absolute error across its cores. The
headline `cycle_macro_mae_pct` is the unweighted mean of those workload values.
The worst absolute individual-core error, makespan error, and signed mean are
also retained; the signed mean is diagnostic because positive and negative
errors can cancel.

Requests are paired by stable logical identity
`(source, frontend_id, frontend_sub_id)`. Let
`lat = departure_cycle - arrival_cycle`, and for paired request `i`:

```text
d_i = lat_model_i - lat_oracle_i
L_w = mean latency of every oracle read in workload w
Request MAE / L = mean_i(|d_i|) / L_w
signed drift / L = mean_i(d_i) / L_w
paired P99 / L = percentile_99_i(|d_i|) / L_w
```

`L` therefore means the workload's full-oracle mean read latency, measured in
the same trace clock domain. It is not the candidate latency parameter and is
recomputed per workload. The report also retains the most negative and most
positive `d_i`, both in cycles and divided by `L_w`. Positive error means the
model is late; negative error means it is early.

Closed-loop request metrics are valid only when stable-ID pairing has exact
bidirectional coverage: matched, oracle, model, and stable-eligible populations
must all be equal. Missing, partial, empty, non-finite, or non-positive inputs
fail the report; they are never converted to zero. Distribution-only p50,
p99, p99.9, Wasserstein-1, and tail-mass values are retained as diagnostics.

Core-cycle MAE and request MAE/L are co-equal loop objectives. Candidate
selection uses Pareto dominance; it does not hide a tradeoff in a weighted
scalar score. Signed drift, paired P99, extremes, coverage, runtime, and memory
remain visible guardrails and diagnostics.

## Files

| Path | Role |
|---|---|
| `config.py` | Standards, workloads, model map, calibrated inputs, and provenance rules. |
| `run_simpleo3.py` | Closed-loop raw runner; iteration labels preserve multiple candidates. |
| `postprocess.py` | Cycle, paired-request, extremes, distribution, and headline aggregation. |
| `matchlib.py` | Stable-ID request pairing and explicit legacy diagnostics. |
| `metrics.py` | Model-neutral checksum-bound request metric caching and validation. |
| `artifacts.py` | Transparent access to raw or verified per-file gzip traces. |
| `archive_results.py` | Verified, atomic compression/verification/restoration of completed traces. |
| `run_champsim.py`, `gem5/` | Future closed-loop frontend transfer plumbing. |
| `synth.py`, `replay_screen.py` | Synthetic and open-loop diagnostics; never promotion evidence. |

## Direct use

```sh
python tools/eval/run_simpleo3.py \
  --std DDR5 --workloads 429.mcf 519.lbm \
  --models oracle,candidate,fixedlat,md1,wmg1,mess \
  --workers 12 --insts-per-core 50000

python tools/eval/postprocess.py \
  --frontend simpleo3 --std DDR5 \
  --workloads 429.mcf 519.lbm
```

Completed traces can be compressed and still read transparently:

```sh
python tools/eval/archive_results.py compress eval_out/CAMPAIGN \
  --manifest eval_out/CAMPAIGN/archive_manifest.json
python tools/eval/archive_results.py verify \
  eval_out/CAMPAIGN/archive_manifest.json
python tools/eval/archive_results.py restore \
  eval_out/CAMPAIGN/archive_manifest.json
```

Compression is per file and deterministic. The raw file is removed only after
decompression reproduces its byte count and SHA-256 and the archive manifest is
atomically published.
