# CHIA campaign harness

`ramulator_chia` supplies DRAM experiments, metrics, selection and the restricted
agent workspace. The pinned upstream `chia` package supplies native sessions,
transport, scheduling and persistence. They are separate packages.

Use the entry points in the root `REPRODUCING.md`. Source staging intentionally
retains some internal `tools/chia_loop` build keys so qualified native recipes
remain unchanged. Those keys are not dependencies on another repository.

## Scientific workflow

The staged configurations run ten single-core rounds, then five rounds adding
four- and eight-core mixes. Each round develops on named training inputs,
freezes one candidate, evaluates training and anonymous validation, and asks an
independent same-model/same-effort reviewer to promote or retain. The reviewer
may accept justified trade-offs but cannot waive mechanical integrity checks.
There is no revision after final validation/review within the round. Test
evidence stays hidden until selection freezes.

Prompts are in `prompts/staged_v1/`. Opus stops after ten single-core rounds.
Stage endpoints and the rounds introducing their models are recorded separately.

## Diagnostics, recovery and isolation

Optional synthetic and open-loop tools have a 1,800-second simulator deadline,
starting after worker admission. Timeouts preserve evidence and return a failed
diagnostic without an accuracy score. Full evaluations retain their configured
limits, including no deadline. Agent waiting does not consume a whole-session
timeout.

Native events, committed tool effects, summaries, decisions and usage receipts
are persisted by the existing recovery path. Unknown or ambiguous failures
remain stopped. Public log exports are redacted evidence, not resumable private
native-session databases.

Fresh campaigns receive an allowlisted workspace. Published `results/`, private
credentials, other campaigns and test evidence are not agent-visible. Do not
disable Landlock/seccomp checks to accommodate an incompatible kernel.

`scripts/run-campaign` requires explicit `--allow-paid`, configuration, runtime
and tariff. Setup, tests, frozen evaluation and plotting never invoke a provider.
Bundled tariffs are historical accounting inputs, not current price quotes.
See `upstream/README.md` for the pinned CHIA changes.
