# Agent-evolved atomic DRAM simulation: project status

Updated 2026-09-05. This document is the concise review entry point for the
private hackathon repository. It contains no setup-conversation transcript.

## Research objective

Synthesize a fast, generic immediate-response DRAM controller through an
agent/evaluation feedback loop. At admission, the model commits an immutable
request completion time using the current request, resolved configuration, and
bounded causal state. It must not reproduce the cycle-level command scheduler,
inspect future requests, or specialize to workload identities. Every proposed
approximation must have a technical explanation.

An earlier independent feasibility study, developed with Fable 5, GPT-5.6, and
human interventions, obtained promising atomic-controller results. This work
attempts to formalize that approach in the existing CHIA evolution framework.
That implementation, its rules, and its results are not available to the CHIA
optimization agents. Consequently, failure of a particular backend or trial
does not by itself disprove the broader approach.

The initial scope is SimpleO3 + DDR5. Evaluation/instrumentation infrastructure
for LPDDR5/LPDDR6, HBM4, ChampSim, and gem5 is retained, but cross-frontend and
cross-standard transfer studies are deferred. Documentation and reproducible
scientific evidence take priority over competition placement.

## Evaluation and metrics

The reference is Ramulator GenericDDR with FRFCFS-RowHit scheduling, open rows,
no refresh, DDR5_16Gb_x8 / DDR5_4800AN, and one channel. Read/write buffers each
hold 64 requests; write-drain low/high watermarks are 0.5/0.8. These settings
are supplied as configuration rather than embedded in candidate models.

Training families are mcf and lbm. Final testing uses milc, soplex, GemsFDTD,
and fotonik3d, after both backend selections are frozen. Each run issues
20 million instructions/core from cold initialization and drains outstanding
work completely. There is no separate unscored warmup. Inputs must not wrap;
hashes and family identities are checked across the split. The meta-reviewers
have seen these test families in earlier work; fresh optimization agents have
not. These results are not a pristine unseen benchmark for adaptive campaigns.

Two objectives have equal standing; lower is better:

1. **Core-cycle MAE (%)**: for each core, take
   `abs(100 * (model_cycles - oracle_cycles) / oracle_cycles)`. Average across
   cores in a workload, then equally across workloads.
2. **Request MAE / L**: pair logical reads by stable
   `(source, frontend_id, frontend_sub_id)` and verify address/type identity and
   exact bidirectional coverage. For each pair,
   `d = model_latency - oracle_latency`, where latency is departure minus
   arrival in frontend cycles. For each workload, `L` is the mean latency of
   **all oracle logical reads**, including LLC hits, merged misses, and DRAM
   miss owners. The workload score is `mean(abs(d))/L`; average workloads
   equally for the headline.

Diagnostics retain `abs(mean(d))/L` (absolute signed drift),
`percentile99(abs(d))/L` (paired P99 error), signed minima/maxima, distribution
errors, final frontend/controller statistics, and instrumented simulation wall
time. The headline P99 is the worst workload P99; it is not a pooled latency
percentile. Positive signed error means predicted latency is too large.

Closed-loop scores control evolution. Logical traces measure the frontend LLC
boundary; separate controller traces use DRAM cycles and explain memory-system
behavior. They must not be mixed without clock conversion. Synthetic and
open-loop tools are retained for diagnosis, but are not promotion evidence
and are not yet wired into the real Gemini tool adapter.

The four immediate-response comparators are FixedLat (fixed delay and a
bandwidth pipe), zsim-style whole-memory M/D/1, Sniper's channel-level windowed
M/G/1, and MESS-style bandwidth/latency curves. The MESS DDR5 surface is
oracle-calibrated and checksum-bound; it has a different calibration advantage
from an untrained seed. The under-review bank-level queueing model is absent.

## Current implementation

CHIA dispatches proposal, optimized build, and evaluation nodes using Ray.
Candidate and reference simulations share a pinned interleave. Preflight
compares normal and isolated execution for identical traces, core cycles, and
integer statistics; it also tests actual loaded-candidate file/network denial.

The repository's Atomic controller remains a fixed-delay seed. Agents may edit
bounded model state, helpers, parameter defaults/ranges, initialization, and
admission-time prediction. The trusted wrapper owns configuration plumbing,
mapping, capacities, clocks, callbacks, tracing, and statistics. A scoped
`model_param` interface supports new approximation parameters, while resolved
DRAM/controller behavior remains read-only.

Candidates submit complete editable-region bodies. The runner assembles them
without changing the protected wrapper, builds with `-O3`, and obtains a
source-specific compliance review from the orchestrating coding agent. This
review enforces the scientific contract without supplying modeling repairs or
tuning hints. Only the proposing backend repairs a rejected draft. Native C++
isolation and lexical checks are defenses, not a memory-safety proof.

A valid challenger replaces the incumbent only if neither objective worsens
and at least one improves. Non-dominated tradeoffs may remain in a Pareto
archive without promotion. There is no weighted objective, invented accuracy
threshold, or test-based selection. Human intervention is recorded and allowed
in principle; initial backend trials receive no human modeling insights.

## Gemini results to date

Earlier trials used Gemini 3.1 Pro Preview and **Gemini 3.5 Flash**, not 3.8.
An initial short-window run produced one promoted Pro design, but full-window
reassessment did not support that training improvement: Pro's training core
MAE was 95.36% and request MAE/L was 1.5312, versus 68.15% and 0.6291 for the
seed. Short-window accuracy is therefore excluded as success evidence.

A subsequent fresh 20M-instruction run raised both generation ceilings to
65,536 tokens. All 41 returned generations ended normally, with no truncation.
Neither backend produced a valid evaluated candidate under that submission
protocol: Pro completed five proposals, Flash three, but formatting/interface
errors, inspection limits, and a boundedness violation prevented acceptance.
Both selections remained the seed. Known-usage standard-rate estimates were
approximately USD 1.98 (Pro) and USD 2.70 (Flash), excluding cache discounts;
one Flash HTTP-error attempt had unknown usage in that historical ledger.
These are not billing invoices or a general ranking of model capabilities.

The established matched full-window comparison is:

| Model | Training core MAE (%) | Training request MAE/L | Test core MAE (%) | Test request MAE/L |
| --- | ---: | ---: | ---: | ---: |
| Fixed-delay seed | 68.15 | 0.6291 | 57.57 | 0.5580 |
| FixedLat | 45.96 | 0.4176 | 49.54 | 0.4804 |
| MD1 | 51.72 | 0.4468 | 53.70 | 0.5115 |
| Sniper WMG1 | 48.80 | 0.4420 | 53.50 | 0.5130 |
| MESS | 16.20 | 0.3126 | 9.13 | 0.3700 |

Protocol v4 compares Gemini 3.1 Pro Preview against **Gemini 3.8 Flash** from
the same clean seed, with HIGH thinking, 65,536 output tokens, and USD 50 per
arm. It permits five **evaluated designs**, with draft repair inside each
iteration; raw draft/call counts are reported separately. Generous runaway
guards allow 48 model turns, 192 inspections, and 12 drafts per iteration.
Input context is token-counted, and model-specific conservative spending
reservations replace the earlier uniform overly conservative tariff.

The v4 campaign has not yet completed at this checkpoint. Updated results,
selected source identities, and numerical plots will be added after both
selections are frozen and the integrity audit passes. Changes to both the
backend and protocol mean this is not a controlled single-variable comparison
against historical Flash 3.5 runs.

## Reproduction, artifacts, and next steps

See the [runner guide](../tools/chia_loop/README.md) for setup, execution, review,
and analysis commands, and the [evaluation guide](../tools/eval/README.md) for
measurement contracts. Execution snapshots, prompts, source lineage, all model
interactions, rejection reasons, budgets, and compressed raw traces are
retained locally for audit. Compression verifies byte count and SHA-256 before
removing the raw copy; readers support gzip without bulk decompression.
The latest completed archival audit reduced raw trace storage by about 69%.

This review repository excludes raw interaction transcripts, detailed pipeline
setup discussions, unrelated DRAM-timing review notes, credentials, and
licensed traces. Compact results do not replace the retained full audit data.

Immediate next steps are to complete the v4 end-to-end comparison, inspect
accuracy and per-request failure modes, and assess whether the richer workflow
enables useful evolution. Follow-on work includes repeated independent trials,
rule ablations, controlled speed measurements, additional model backends from
clean starts if needed, broader diagnostic tools, and eventually transfer.

Provider configuration is checked against Google's
[Gemini 3.8 Flash specification](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash),
[Gemini 3.1 Pro specification](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-pro),
and [pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing).
HIGH is a relative dynamic effort setting, not equal reasoning-token use.
