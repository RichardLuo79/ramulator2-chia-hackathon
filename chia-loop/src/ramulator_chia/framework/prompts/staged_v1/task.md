# Immediate-response DRAM modeling

Develop a generic immediate-response DRAM model using the supplied interface.
Predict once at request admission; issued predictions are immutable. Do not
implement an internal command scheduler, inspect future requests or oracle
internals, or specialize to workload names, addresses, or evaluation identities.

Explain the physical meaning of model state, parameters, and approximations.
Use named training inputs, statistics, request observations, generic synthetic
patterns, and replay diagnostics to investigate weaknesses. Include both
streaming and dependent-random behavior in your diagnostic reasoning.

Core-cycle accuracy and request MAE/L are the primary objectives. Also
investigate individual-core failures, signed bias, tails, and recurring extreme
errors. Good averages can conceal serious weaknesses. Do not optimize a maximum
without considering its frequency and sample population.

Validation provides anonymized numerical feedback only. Do not identify
validation workloads or seek their inputs. Test data and previous campaigns
are unavailable.

An independent reviewer selects between your frozen candidate and the
incumbent. Small regressions can be acceptable when supported by meaningful
improvements elsewhere. There is no strict non-worsening rule.

You may develop, debug, and evaluate training drafts before submission. Submit
one final candidate per round. Once frozen, do not edit it or request further
validation. Preserve learning through the required end-of-round summary.

## Measurements and instruments

For each matched read, let `d = predicted latency - oracle latency`, in DRAM
cycles. Negative errors underestimate latency. `L` is the mean latency of all
recorded oracle reads admitted in the measured windows, including unpaired
reads. Request MAE/L is `mean(abs(d))/L`; signed drift/L is `mean(d)/L`.
P99 and P99.9 are percentiles of paired absolute errors, not differences between
the models' latency percentiles. Unmatched errors are unknown, not zero.

Core error is `100 * (model cycles - oracle cycles) / oracle cycles`; first
average absolute errors equally over cores, then equally over cases. Keep
one-, four-, and eight-core groups separate. Request headlines also weight
cases equally rather than pooling their reads.

ChampSim warms each core/cache for 2M instructions, then measures 20M per core.
The DRAM controller starts cold after frontend warmup. Physical placement is
fixed across models. Background replay keeps multicore contention active but
does not enter the measured request population. There is no final drain.

FixedLat, M/D/1, channel-level WMG1 and MESS are the comparison baselines.
Synthetic patterns use a controlled DRAM frontend, not ChampSim instructions.
The existing replay tool uses recorded oracle completions and can omit requests
outstanding at termination; do not describe it as complete-admission causal
evidence without verifying that separately. Poll all diagnostic jobs to completion.
