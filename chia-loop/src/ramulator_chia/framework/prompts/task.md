# Atomic DRAM modeling task

Improve a generic, immediate-response DRAM latency model. Seek lower core-cycle
and per-request latency errors against the cycle-level oracle. Explain the
physical behavior represented by each state variable, approximation and rule.
Do not fit answers to workload names, known addresses or evaluation results.

Your files are `draft/model.cpp` and `draft/parameters.json`. The C++20 interface
is in `references/api.h`. Implement initialization and one prediction per
admitted request. Return an absolute future DRAM cycle. The controller, not your
model, owns admission, callbacks and immutable predicted departures. There is
no model tick or completion callback. Do not implement a hidden cycle-level
scheduler, read future requests, access oracle state or change instrumentation.
Model-owned numeric parameters and their defaults/ranges are yours to define.

Use training measurements and the available tools to diagnose failure modes.
The four comparisons are FixedLatency, M/D/1, published channel-level WMG1 and
MeSS. Synthetic patterns and fixed-arrival open-loop replay, when enabled, are
diagnostic instruments rather than promotion datasets. Inspect statistics and
traces when an aggregate score does not explain the behavior.

The two promotion objectives are equally weighted workload means of:

- Absolute per-core cycle percentage error (averaged within each workload).
- Paired request latency MAE divided by that workload's full oracle mean
  latency, L. Each observation's population and clock domain are explicit.

Promotion requires neither objective to worsen and at least one to improve,
using valid complete training measurements. Missing data is not zero error.
Held-out testing and frontend-transfer results are unavailable during search.

The task settings identify the primary frontend and request population. For
ChampSim, inputs use one 2M-instruction warmup followed by a 20M-instruction ROI.
Warmup primes the core and caches; DRAM timing starts at the ROI. Training input
instructions are fully accessible through `inspect_input` pages, and controller
observations through `inspect_training`. Input memory addresses are virtual;
controller traces use physical addresses. The fixed virtual-memory seed does
not make page assignment identical across independently timed runs.

ChampSim request error therefore uses recorded reads with both the same stable
request ID and the same physical address. L is the mean latency of all recorded
oracle reads, not just matched reads. Coverage is reported with training
measurements; the paired error does not describe unmatched requests. Synthetic
diagnostics still run the generic controlled DRAM frontend. ChampSim open-loop
diagnostics replay recorded completions and may omit requests outstanding at
the ROI boundary; they are not complete-admission replays.

When a validation cohort is configured, the harness evaluates the final draft
once per iteration, after editing closes. You receive only two errors for each
anonymous validation workload and their equal-workload means. There is no
validation diagnostic tool or access to validation inputs. If
`validation_non_worsening` is enabled, promotion additionally requires neither
validation mean to worsen relative to the incumbent. Individual validation
scores are feedback, not separate promotion gates. Validation is search
feedback, not an unbiased final test.

You can make multiple diagnostic drafts, but submit one final model per
iteration. An optional independent semantic critique is advisory. Continue
in the same proposer session when revising or reflecting. Native continuation
is preserved; you do not need to restate your entire conversation.

After receiving the final evaluation, write `notes/summary.md`. Describe what
changed, why, the measured results, failed ideas and remaining questions. It
will be available, with this campaign's prior history, in the next iteration.
Keep the summary factual; distinguish measurements from hypotheses.
