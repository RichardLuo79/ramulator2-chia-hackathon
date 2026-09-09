# CHIA: evolving immediate-response DRAM models

The active implementation is [`framework/`](framework/README.md). Gemini,
Astra and Fable use one campaign loop, the same DRAM tools and the same
scientific selection policy. Provider adapters supply native sessions; they
do not own a separate evolution loop.

See the [research overview and results](../../doc/chia_hackathon_review.md),
[design](../../doc/chia_framework_design.md) and
[implementation status](../../doc/chia_framework_implementation.md).

## Current workflow

Each campaign starts from the fixed-delay seed and has its own source,
configuration, sessions, candidates, summaries and usage records. An iteration
contains exploration, optional independent advisory review, revision in the
same proposer session, final training evaluation and a model-written summary.
The next iteration receives its own campaign history, not other campaigns or
the independent feasibility design.

The proposing agent can inspect and edit the model API implementation, run
training evaluations, and use enabled synthetic/open-loop diagnostics. A
positive file/tool allowlist excludes repository history, credentials,
held-out inputs and other experiments. Native CLIs run under the existing
Landlock/seccomp boundary; this is not Docker-grade network isolation.

Promotion uses two unrounded training objectives: core-cycle percentage error
and paired request latency MAE / oracle mean latency. A mechanically valid
candidate must improve at least one without worsening the other. Optional
semantic review gives one same-model, same-effort critique; it is not a veto
or an unbounded repair loop. Held-out scores never drive search.

## Evaluation and evidence

The [DDR5 profile](configs/ddr5_public_transfer_v2.json) fixes eight SimpleO3
training families, eight disjoint test families, six ChampSim traces and twelve
source-built gem5 guests. SimpleO3 uses 20M instructions/core and full drain;
ChampSim uses 2M warmup plus 20M measured instructions. Generic synthetic
patterns and open-loop replay are training diagnostics, not promotion datasets.

The reference is the cycle-level controller. Comparisons include FixedLatency,
M/D/1, published channel-level WMG1 and MeSS. The unpublished bank-level model
is not part of this branch. New gem5 guests still need complete whole-program
qualification before transfer results can support claims.

Every build is checked for `-O3`. Concurrent evaluations must total at most
twelve CPU cores. Large observations and native logs are compressed and
checksummed. Usage estimates distinguish API-equivalent cost from subscription
billing and preserve unknown usage rather than treating it as zero.

Final artifacts contain source, inputs, configuration, candidate history,
observations, logs and results. External frontends use pinned source revisions,
patches and rebuild recipes; installed binaries and environments are not
bundled. Docker packaging is deferred.

## Entry points

- [Build, offline qualification and paid launch](framework/README.md).
- [Pinned CHIA session extension](framework/upstream/README.md).
- [Public gem5 guest suite](../eval/gem5/PUBLIC_SUITE.md).
- [Clean gem5 build](../../resources/gem5_wrappers/BUILD.md).

Paid inference requires explicit authorization and `--allow-paid`. The common
launcher currently uses iteration/resource guards and usage accounting, not
a dollar spending cap. Use an immutable checkout or captured source tree for
each running campaign; do not update its code or configuration in place.

## Historical runners

The older `prepare_gemini.py`, `gemini_loop.py`, `codex_cli/` and
`claude_cli/` workflows remain for interpreting frozen historical experiments.
Some credential, boundary and evaluation utilities are shared with the common
framework. Their old prompt-cache, blocking-review, turn-limit and budget
settings are not the common launcher's configuration.

Historical documentation is retained in the
[Astra guide](../../doc/chia_codex_cli.md),
[Fable guide](../../doc/chia_claude_cli.md),
[cache notes](../../doc/chia_prompt_caching.md) and
[unattended-run guide](../../doc/chia_unattended_runs.md).
New campaigns should use the common framework.

Setup conversations, operator recovery notes, unrelated DRAM timing audits,
credentials, licensed traces and raw interaction logs stay outside the
published repository. A concise dated research snapshot is included instead.
