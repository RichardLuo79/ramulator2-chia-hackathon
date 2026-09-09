# Agent-evolved atomic DRAM simulation

Updated 2026-09-09. This is the review entry point for the private CHIA
hackathon repository. It summarizes the scientific contract, implementation
and measured results without setup transcripts or raw interaction logs.

## Research objective

Synthesize a fast, generic immediate-response DRAM model through an
agent/evaluation feedback loop. At admission, the model commits a request
completion time from the current request, resolved configuration and bounded
causal state. It must not run a command scheduler, inspect future requests or
specialize to workload identities. Rules and parameters need physical
interpretations and stated limitations.

An independent feasibility study developed with Fable 5, GPT-5.6 and human
interventions obtained promising results. CHIA formalizes that general approach
as a reproducible agent-evolution workflow. The earlier implementation, rules,
results and discussions are unavailable to optimization agents.

Training currently uses SimpleO3 + DDR5. Held-out SimpleO3 and DDR5
ChampSim/gem5 transfer follow selection; other standards are deferred.
Each model/effort has its own campaign, history, selection and usage records.
Comparisons do not merge state or feed another campaign's results to an agent.

## Evaluation and metrics

The reference is GenericDDR with FRFCFS-RowHit, open rows, no refresh,
DDR5_16Gb_x8 / DDR5_4800AN, and one channel/rank. Read/write buffers hold
64 requests each; write-drain low/high watermarks are 0.5/0.8. Organization and
timing come from configuration, not candidate constants.

The [v2 cohort](../tools/chia_loop/configs/ddr5_public_transfer_v2.json) fixes:

- Training: mcf, lbm, bzip2, cactusADM, libquantum, xalancbmk, zeusmp, sphinx3.
- Held-out SimpleO3: milc, soplex, GemsFDTD, fotonik3d, gcc, omnetpp, gromacs, namd.
- Transfer: six ChampSim traces and twelve source-built public gem5 guests.

Every SimpleO3 case issues 20M instructions/core from a cold prefix, without
trace wrap, and drains admitted requests. Application-family overlap, input
identity, request coverage and minimum oracle traffic are checked. Researchers
have used these benchmark families before: fresh agent context is not a claim
of a historically untouched benchmark.

Two objectives have equal standing; lower is better:

1. **Core-cycle MAE (%)**: `100 * abs(model_cycles - oracle_cycles) / oracle_cycles`
   per core, averaged within each workload and then equally across workloads.
2. **Request MAE / L**: pair logical reads by
   `(source, frontend_id, frontend_sub_id)`, verify address equality and exact
   bidirectional coverage, and let `d = model_latency - oracle_latency`.
   Latency is departure minus arrival in frontend cycles. Within each workload,
   **L is the mean latency of all oracle logical reads**, including LLC hits,
   merged misses and DRAM owners. Compute `mean(abs(d)) / L`, then average
   workloads equally. Tables below use this dimensionless ratio.

Diagnostics retain `abs(mean(d))/L`, paired `percentile99(abs(d))/L`,
positive/negative extremes, statistics, trace slices, coverage and runtime.
The worst-workload paired P99 is not a pooled percentile. Controller traces
use DRAM cycles; logical traces use frontend cycles. Instrumented wall times
are not controlled simulator-speed benchmarks.

Comparisons include FixedLatency, zsim-style whole-memory M/D/1, published
Sniper channel-level WMG1 and MeSS-style curves. MeSS uses an oracle-calibrated,
checksum-bound DDR5 surface, giving it a different calibration advantage.
The unpublished bank-level Anatomy model is excluded.

## Common autonomous workflow

All backends use the same [campaign framework](../tools/chia_loop/framework/README.md).
Agents edit the restricted model API implementation and model parameters. The
trusted wrapper owns admission, mapping, capacities, clocks, callbacks, tracing
and statistics. The repository seed remains a fixed delay, not a selected
agent design.

An iteration consists of exploration, an optional independent critique,
revision in the same proposer session, final checks/training, and reflection.
The model writes a summary of changes, reasoning, results, failed ideas and
next questions; the next iteration receives its own history. Native
continuation and provider-exposed events are preserved. This does not claim
access to hidden reasoning that a provider does not expose.

When `semantic_llm_check` is enabled, the same model and effort gives one
independent advisory critique. It sees neither hidden data nor prior designs.
The critique can inform one revision phase, but does not veto promotion or
create an unlimited repair loop. Mechanical checks remain mandatory.
A final candidate is promoted only if neither unrounded training objective
worsens and at least one improves.

Generic synthetic patterns and open-loop replay are selectable training
diagnostics, not promotion datasets. No findings or model-specific diagnostic
cases are imported from independent non-CHIA exploration. Held-out and transfer
results cannot enter prompts, diagnostic responses or promotion.

ChampSim transfer uses 2M warmup + 20M measured instructions. Gem5 runs complete
programs to exit. DDR5 and selected parameters remain fixed. Transfer request
metrics use the controller boundary and are reported separately from SimpleO3
logical-read metrics; incomplete pairing withholds a whole-population request
headline. The new public gem5 cohort is not yet fully qualified.

## Current common-framework runs

The following is an interim snapshot on 2026-09-09 around 14:05 UTC, taken from
committed iteration records. These are **training results only**, not final
test scores or a matched-budget ranking of model capability.

| Proposer | Completed iterations | Selected candidate prefix | Core MAE (%) | Request MAE/L |
| --- | ---: | --- | ---: | ---: |
| Gemini 3.8 Flash / high | 12 / 20 | `27c1843cda62` | 3.6454 | 0.13634 |
| GPT-6 Astra / xhigh | 11 / 20 | `fd698e604dda1` | 1.9817 | 0.14721 |
| GPT-6 Astra / max | 2 / 20 | `22ebda3e0008` | 5.5191 | 0.16315 |
| Fable 5.1 / xhigh retry | 0 / 2 | No completed candidate | — | — |

Flash has lower request error than xhigh at this snapshot; xhigh has lower core
error. Max has completed far fewer iterations following infrastructure
interruptions. No effort-level conclusion follows from these unequal interim
runs. Gemini Pro and Fable max are not active arms.

The three main runs allow 20 iterations, no dollar cap and three evaluation
CPUs each. The separate Fable retry used the same settings with two iterations.
The common launcher imposes no output-token or diagnostic-turn cap; native
limits and operational timeouts still apply. All four use enabled diagnostics
and same-model/effort advisory review.

The original Fable search and the short retry encountered repeated local
`Failed to get memory usage` tool errors. The retry's monitor stopped it before
the first iteration completed. A separate live diagnostic recorded an
`EACCES` read of `/proc/self/stat`, a 253 MiB memory high-water mark and more
than 82 GiB available memory, with no OOM events. The exact permission trigger
remains unresolved; these failures are not evidence of poor model accuracy.

Astra recovery preserves completed work and usage. Native capacity failures
use a bounded cooldown; stream failures are classified from native events
rather than generated answer text. Operational interventions are retained in
local evidence, not represented as uninterrupted runs.

## Earlier protocol: reference snapshot

These previously reported 2026-09-06 values used the older harness and reviewer
policy. They are retained as historical evidence, not pooled with the common
framework or presented as a controlled before/after comparison.

| Proposer | Evaluated designs | Train core (%) | Train request MAE/L | Test core (%) | Test request MAE/L |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini 3.1 Pro Preview / HIGH | 20 | 35.7929 | 0.41799 | 30.6582 | 0.41224 |
| Gemini 3.8 Flash / HIGH | 18 | 8.4881 | 0.23803 | 5.2170 | 0.24595 |
| GPT-6 Astra / xhigh | 20 | 4.4867 | 0.16054 | 2.6681 | 0.19367 |
| GPT-6 Astra / max | 20 | 7.7972 | 0.14199 | 4.5301 | 0.14582 |

That snapshot had completed SimpleO3 selection/testing but not the full transfer
matrix. It used one trial per configuration, unequal search/usage budgets and
provider-specific reviewers. Large request tails remained. Historical gem5
executables are a different cohort from the new source-built public suite.

Earlier two-training/four-test results and plots remain in the
[five-design Gemini comparison](results/gemini_20260905/summary.json) and
[extended-trial report](chia_extended_trials_20260905.md).

## Reproduction, evidence and publication

Use the [operator guide](../tools/chia_loop/framework/README.md) and
[pinned CHIA extension](../tools/chia_loop/framework/upstream/README.md).
The [implementation status](chia_framework_implementation.md) lists validation
evidence and open work, including gem5 qualification and the Claude failure.

Evaluations use verified `-O3` builds, with at most twelve CPU cores combined.
Prompts, exposed native events, tool actions, reviews, sources, summaries,
metrics and usage are retained per campaign. Logs and traces are compressed
and checksummed. Unknown usage is not zero; subscription API-equivalent
estimates are not invoices or quota measurements.

Artifacts are source-first: external frontends use pinned revisions, patches
and rebuild recipes, not bundled binaries. Detailed setup discussions, operator
recovery notes, credentials, licensed traces and raw interactions are not in
this repository. Post-campaign review and Docker deployment remain deferred.
