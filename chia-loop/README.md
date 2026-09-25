# Campaign and evaluation package

`ramulator_chia` owns execution, metrics, visibility, sessions and evidence.
Upstream `chia` remains a separate vendored package; it is not shadowed by the
artifact package name. `native/` contains the model API runner and common seed.
The setup command uses editable installations. For a non-editable package
installation, explicitly set `RAMULATOR_CHIA_ROOT` to this source checkout so
native build recipes and external assets can be located; they are not bundled
as simulator binaries in the Python distribution.

`configs/` supplies fresh-campaign templates with repository-relative or explicit
`${DATA_ROOT}`, `${WORK_ROOT}` and `${ARTIFACT_ROOT}` substitutions. Scientific
inputs are pinned. Credentials, provider projects and cloud lifecycle are not
bundled. The five reviewed stage prompts remain separate files under
`src/ramulator_chia/framework/prompts/staged_v1/`.

Builds, caches and output directories are outside agent-visible storage.
The published results directory is never granted to the agent. Tests use
scripted providers and cannot make paid calls by default. Production inference
requires the launcher's explicit `--allow-paid` opt-in.
