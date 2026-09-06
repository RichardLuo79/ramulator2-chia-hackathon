# Agent-evolved atomic DRAM simulation

Updated 2026-09-06. This is the review entry point for the private CHIA
hackathon repository. It summarizes the scientific contract, implementation,
and observed results; it contains no setup-conversation transcript.

## Research objective

Synthesize a fast, generic immediate-response DRAM controller through an
agent/evaluation feedback loop. At admission, the model commits an immutable
request completion time from the current request, resolved configuration, and
bounded causal state. It must not simulate a DRAM command scheduler, inspect
future requests, or specialize to workload identities. Approximation rules
and parameters require physical interpretations and stated limitations.

An independent feasibility study developed with Fable 5, GPT-5.6, and human
interventions obtained promising results. CHIA formalizes that general approach
as a reproducible agent-evolution workflow. The earlier implementation, rules,
results, and discussions are not available to these optimization agents.

The current training scope is SimpleO3 + DDR5, with post-selection DDR5
ChampSim and gem5 transfer. Other DRAM standards remain deferred. Each model
and effort is an independent run with its own history, selection, accounting,
and reports; comparisons do not merge scientific state or budgets.

## Evaluation

The reference is GenericDDR with FRFCFS-RowHit, open rows, no refresh,
DDR5_16Gb_x8 / DDR5_4800AN, and one channel/rank. Read/write buffers hold
64 requests each; write-drain low/high watermarks are 0.5/0.8. Hardware
organization and timing come from configuration, not candidate constants.

The [frozen cohort profile](../tools/chia_loop/configs/ddr5_frontend_transfer_v1.json)
defines eight SimpleO3 training and eight disjoint final-test families:

- Training: mcf, lbm, bzip2, cactusADM, libquantum, xalancbmk, zeusmp, sphinx3.
- Final test: milc, soplex, GemsFDTD, fotonik3d, gcc, omnetpp, gromacs, namd.

Every SimpleO3 case issues 20 million instructions per core from a cold
prefix, without trace wrap, and drains admitted requests. Input hashes,
application-family overlap, request coverage, and minimum oracle traffic
are checked. No final-test result enters agent feedback or promotion.
Researchers have used these benchmark families before: fresh agent context
is not a claim of a historically untouched benchmark.

Two objectives have equal standing; lower is better:

1. **Core-cycle MAE (%)**: for each core, compute
   `100 * abs(model_cycles - oracle_cycles) / oracle_cycles`; average cores
   within a workload, then average workloads equally.
2. **Request MAE / L**: pair logical reads by
   `(source, frontend_id, frontend_sub_id)`, verify address/type and exact
   bidirectional coverage, and let `d = model_latency - oracle_latency`.
   Latency is departure minus arrival in frontend cycles. Within each workload,
   `L` is the mean latency of all oracle logical reads, including LLC hits,
   merged misses, and DRAM owners. Compute `mean(abs(d)) / L`, then average
   workloads equally. This is a dimensionless ratio, not a percentage.

Diagnostics include `abs(mean(d))/L`, paired `percentile99(abs(d))/L`, signed
extremes, latency-distribution differences, final statistics, and trace slices.
The headline paired P99 is the worst workload value, not a pooled percentile.
Controller traces use DRAM cycles; logical traces use frontend cycles.
Instrumented wall times are recorded but are not controlled speed benchmarks.

FixedLat, zsim-style whole-memory M/D/1, Sniper channel-level windowed M/G/1,
and MESS-style bandwidth/latency curves remain comparison models. MESS uses
an oracle-calibrated, checksum-bound DDR5 surface and therefore has a different
calibration advantage. The under-review bank-level Anatomy model is excluded.

## Autonomous evolution and diagnostics

CHIA/Ray dispatches generation, optimized build, and evaluation. The checkout
retains the unchanged fixed-delay Atomic seed. Agents edit only marked model
regions: bounded state, helpers, model-specific parameters, initialization,
and admission-time prediction. The protected wrapper owns configuration
plumbing, mapping, clocks, capacities, callbacks, tracing, and statistics.

Deterministic checks and an isolated automated LLM compliance gate precede
scoring. Every compliance rule must pass. The reviewer receives one candidate
and its explanation plus a fixed rubric, not other campaigns or held-out
data. Gemini uses a Pro/HIGH reviewer; Astra and Fable use their respective
model at fixed xhigh for both proposing efforts. Reviewer calls and repairs
are recorded in the corresponding run. No human approval is required.

The rubric prohibits repair code and accuracy advice, but free-text review
reasons are not a formal guarantee of advice-free feedback. This is autonomous
evolution with an automated compliance gate, not an unaided single-model
experiment. Semantic review is not a correctness or memory-safety proof.

A valid challenger is promoted only if neither primary error increases and
at least one decreases. Incomparable candidates may remain parents in the
nondominated archive. Selection freezes before final testing; an infrastructure
fault cannot open testing or silently alter selection, data, or accounting.

The [default loop profile](../tools/chia_loop/configs/loop_default_v1.json)
enables training statistics, logical/controller trace inspection, paired
extremes, comparison feedback, and agent-callable generic synthetic experiments.
The synthetic tool reuses only the generic generator, not findings or
model-specific cases from independent exploration. Synthetic scores diagnose
behavior; they do not enter the application promotion objectives.
Feature switches support future ablations; no completed ablation is claimed.

After selection freezes, the same candidate is evaluated on six ChampSim
traces (2M warmup + 20M ROI) and twelve gem5 SE programs run to normal exit.
DDR5 and candidate parameters remain fixed. These are frontend-transfer tests,
not necessarily new application families. Transfer request metrics are at the
controller boundary and are not pooled with SimpleO3 logical-read metrics.
Incomplete pairing is diagnostic-only, not an exact-coverage headline result.

## Rich-DDR5 campaign snapshot

The following values were read from the four independent run records on
2026-09-06 around 22:55 UTC. Selection and eight-family SimpleO3 testing have
finished, but **all four full campaigns are still in frozen transfer evaluation**.
These are provisional report values pending completion and final evidence audit,
not a declaration that the whole transfer matrix has passed.

| Proposing backend | Evaluated designs | Selected | Train core MAE (%) | Train request MAE/L | Test core MAE (%) | Test request MAE/L |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Gemini 3.1 Pro Preview / HIGH | 20 | pro_002 | 35.7929 | 0.41799 | 30.6582 | 0.41224 |
| Gemini 3.8 Flash / HIGH | 18 | flash_005 | 8.4881 | 0.23803 | 5.2170 | 0.24595 |
| GPT-6 Astra / xhigh | 20 | astra_018 | 4.4867 | 0.16054 | 2.6681 | 0.19367 |
| GPT-6 Astra / max | 20 | astra_020 | 7.7972 | 0.14199 | 4.5301 | 0.14582 |

On this cohort, Flash's selected model improves both test averages over Pro.
Astra xhigh has the lowest core-cycle error; Astra max has the lowest
paired-request MAE. Neither Astra effort dominates the other. Request tails
remain substantial: worst-workload paired P99/L is 8.1850 for xhigh and
5.0533 for max; maximum positive signed d/L is 55.5068 and 51.3916.
These are one trial per configuration, with unequal search/usage budgets and
provider-specific reviewers, not a general ranking of base-model capability.

| Run | Generation attempts, including reviews/retries | Known usage estimate (USD) | Unknown-cost attempts |
| --- | ---: | ---: | ---: |
| Gemini Pro | 60 | 15.53 | 0 |
| Gemini Flash | 470 | 40.64 | 1 |
| Astra xhigh | 131 | 145.99 | 5 |
| Astra max | 127 | 145.78 | 4 |

These are the current calls' known API-equivalent estimates under the recorded
tariffs, not invoices or complete totals when usage is unknown. Unknown calls
retain conservative reservations. Gemini has USD 100 guards per run: Flash
stopped before its nineteenth evaluated design because conservative charges
were USD 98.62, despite lower known usage. Astra uses an iteration guard,
not a dollar cap. Native-subscription quotas remain external constraints.

Two independent Fable 5.1 campaigns, xhigh and max, are authorized for at most
20 evaluated iterations each with no USD guard. At this snapshot both are
queued for CPU slots and have made no live model calls. A separate authorized
Flash continuation is queued; it is not included in the results above.

Earlier two-training/four-test Gemini results, audited sources, and plots remain
in the [five-design comparison](results/gemini_20260905/summary.json) and
[extended-trial report](chia_extended_trials_20260905.md). Those results use
a different cohort/protocol and should not be interpreted as a matched
before/after comparison with the table above.

## Reproduction, evidence, and publication

- [Runner and evaluation entry points](../tools/chia_loop/README.md).
- [DDR5 cohorts, metrics, and frontend transfer](chia_ddr5_frontend_transfer.md).
- [Synthetic diagnostics and configurable ablations](chia_diagnostics_and_ablations.md).
- [Unattended recovery and compliance](chia_unattended_runs.md).
- [Isolated Astra backend](chia_codex_cli.md) and [isolated Fable backend](chia_claude_cli.md).

All evaluations use `-O3`; current campaigns request three slots each from
one twelve-CPU pool. Prompts, requests, provider-exposed reasoning summaries,
observable actions, review feedback, drafts, source hashes, diagnostics,
metrics, and usage are retained locally. Hidden reasoning is not claimed.
Checksummed traces and finalized interactions are compressed; readers support
transparent decompression. Unknown usage and partial infrastructure outcomes
remain explicit rather than being presented as successful free calls.

The pre-publication offline regression suite passed **313 tests**, with one
opt-in full-window integration test skipped. This validates infrastructure;
it does not substitute for the outstanding full transfer runs and audit.

The review repository excludes raw conversations, operator handoffs, unrelated
DRAM-timing audits, credentials, licensed traces, and large evaluation stores.
Compact summaries do not replace the local audit evidence. No prior
feasibility controller is an active build target or agent input.

Next work is to finish and audit the transfer matrices, compare independent
Fable runs, conduct controlled feature/rule ablations and repetitions, and
measure speed under controlled conditions. Cross-standard transfer is deferred.
Scientific documentation and reproducibility take priority over placement.
