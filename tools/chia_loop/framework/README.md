# CHIA atomic-DRAM research loop

This package implements the
[common workflow](../../../doc/chia_framework_design.md). The campaign core
runs with native agents or scripted responses and real DRAM evaluation through
authenticated CHIA tools. See the
[research snapshot](../../../doc/chia_hackathon_review.md) and
[validation status](../../../doc/chia_framework_implementation.md).

CHIA owns native sessions, tool transport, scheduling and database primitives.
Our layer supplies DRAM tools, scientific selection, positive agent visibility
and campaign evidence. Gemini, Codex and Claude share the same workflow.

## Build and qualify locally

Use a separate environment with the [pinned CHIA extension](upstream/README.md)
and this repository's requirements. Run commands from the repository root, with
that CHIA checkout, `.` and `python` on `PYTHONPATH`. No command below needs a
provider login. Set `RAMULATOR_TRACES` to the trace collection if it is not at
`/home/dev/traces`; raw and gzip SimpleO3 inputs are supported.

Build a new local evaluator:

```sh
python -m tools.chia_loop.framework.build \
  --repo . --output /path/to/new-runtime --workers 6
```

This preserves source, commands and logs, and checks every translation unit for
`-O3`. It does not overwrite an existing runtime. Count these build workers
against the combined twelve-CPU allowance; do not start another large build or
evaluation when that allowance is already occupied.

The smallest representative no-provider check uses mcf for training and milc
for held-out evaluation, both at 20M instructions. It exercises diagnostics,
advisory review, summary handoff and checkpoint reuse:

```sh
python tests/utils/check_chia_clean_campaign.py \
  --runtime /path/to/new-runtime --output /path/to/new-check \
  --workers 6 --iterations 2 --mcp
```

The proposer changes only a comment. A successful check qualifies the pipeline,
not agent reasoning or a better DRAM model. `qualification.json` is written only
after search, frozen evaluation and the checkpoint-reuse assertions complete.
Partial measurement receipts do not make an unfinished campaign successful.

For the full cohort, supply
`--evaluation tools/chia_loop/configs/ddr5_public_transfer_v2.json` and
`--gem5-programs /path/to/public-guests/suite.json`. Follow the
[public guest build instructions](../../eval/gem5/PUBLIC_SUITE.md) and
[clean gem5 recipe](../../../resources/gem5_wrappers/BUILD.md). Set `GEM5_BIN`
explicitly to that new build. `CHAMPSIM_DIR`, `CHAMPSIM_BIN` and
`CHAMPSIM_TRACES` select the attested ChampSim installation and input collection.

For configurable checks, use `--campaign-config /path/to/campaign.json` instead
of the individual evaluation/worker/iteration/MCP flags. The file follows
`config.CampaignConfig`; it must specify backend `fixture`, scenario
`comment-only` or `comment-only-mcp`, and native evaluation. All experiment
switches and resource limits are taken from that file without overrides. The
command rejects real-provider configurations before preparing any inputs.

## Paid search

Use an immutable checkout or captured source tree for each active campaign.
The complete JSON configuration follows `config.CampaignConfig`: `experiment`
contains the evaluation profile, diagnostic switches and `semantic_llm_check`;
`backend` specifies the exact model/effort; `run` holds iteration, timeout,
retry and evaluation-resource limits. The evaluation object can be taken from
`tools/chia_loop/configs/ddr5_public_transfer_v2.json`.

```sh
python -m tools.chia_loop.framework.launch \
  --allow-paid --config /path/to/campaign.json \
  --runtime /path/to/runtime --output /path/to/new-campaign \
  --tariff /path/to/model-tariff.json \
  --gem5-programs /path/to/public-guests/suite.json
```

Set `GEM5_BIN` and the ChampSim/trace paths as above. Vertex uses existing Google
ADC. The CLI transport reads the selected local subscription credential in its
trusted relay; it does not copy the operator's login or conversation into the
agent profile. Credential paths are explicit fields on `NativeTransport`.

This runs search only. Add `--evaluate` to evaluate the frozen selection
afterward; full transfer qualification is still incomplete. Artifacts and a
strong-model final review are separate from search. The current common launcher
has iteration/resource guards and usage accounting, but no dollar spending cap
or harness output-token cap. Native/provider limits still apply.

Codex capacity failures default to a 3,600-second cooldown, configurable with
`--capacity-cooldown-seconds`. Retries require saved native continuation and
remain bounded by `run.maximum_attempts`; restarting does not reset usage or
the recorded retry deadline. Authentication errors and ambiguous interrupted
operations do not silently become fresh calls.

## Native sessions and evidence

`agent.NativeSessions` is the common session factory for `dram.DramResearch`.
It constructs role-private native sessions, authenticated phase tools and
`NativeAgent` evidence capture. It requires an explicit host-supplied transport;
it has no unrestricted subprocess or inherited-login fallback. The host must
enforce the workspace's changing read/write grants for every native CLI call.
`transport.NativeTransport` supplies the existing Landlock/seccomp boundary and
a fixed-provider relay for native CLIs, or the tool-only Vertex SDK transport.
Its local port restriction is not an IP-level network namespace. Docker remains
deferred. The installed CLI startup and denied-file/port checks pass.

`run.model_timeout_seconds` is forwarded to the CHIA adapter and recorded.
CHIA's CLI timeout covers a native process call; the Vertex setting is an HTTP
request timeout, not a total autonomous-session deadline. These are operational
limits, not evidence of equal wall-time budgets. Tool access is revoked by
closing its phase endpoint; a long diagnostic does not silently exhaust a
separate authentication timer.

The campaign stores configurations/checkpoints in `campaign.sqlite`, immutable
candidates and summaries in their named directories, and compressed exposed
events/native continuation under `events/` and `native-evidence/`. Never resume
by deleting an unresolved attempt or resetting usage. Resume requires the same
source, settings, inputs and private native-session paths.

## Source-only artifacts

Add `--archive-output /path/to/artifacts` to the scripted campaign command to
seal its source and evidence after frozen evaluation. Export does not start
missing jobs. The resulting path, SHA-256 and size are recorded in
`qualification.json`. The exporter and argument checks pass offline tests;
the combined command still needs a fresh native qualification.

Transfer campaigns also require `--frontend-recipes /path/to/frontends.json`.
This file records each enabled frontend's source pin and rebuild instructions:

```json
{
  "gem5": {
    "metadata": {
      "upstream": {"url": "https://github.com/gem5/gem5.git", "revision": "<40-character commit>"},
      "working_directory": "rebuild/gem5",
      "build_commands": [["scons", "build/X86/gem5.opt", "-j6", "--verbose"]]
    },
    "files": [{"source": "build.log", "member": "external/gem5/build.log"}]
  }
}
```

This illustrates the format, not a complete build recipe: record checkout,
patch/wrapper installation and the required build environment as well. File
paths are relative to the recipe file; working directories and commands refer
to the extracted artifact. Commands are retained as data, never executed by
export. Enabled frontend names must match the recipe keys. Source-built guest
recipes are taken from `--gem5-programs`; source/patch files and logs are included,
not executables, shared libraries or installed environments. Docker is deferred.

Inspect an existing archive without launching any execution backend:

```sh
python -m tools.chia_loop.framework.inspect verify /path/to/campaign-HASH.zip
python -m tools.chia_loop.framework.inspect extract /path/to/campaign-HASH.zip \
  --output /path/to/new-extraction
```

See the [implementation status](../../../doc/chia_framework_implementation.md)
for completed checks and remaining limitations, including the unresolved Claude
runtime permission failure. Detailed operator records remain local.
