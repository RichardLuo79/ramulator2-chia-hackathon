# Campaign and evaluation package

`ramulator_chia` runs the CHIA campaigns and the evaluations: simulation, metrics,
the agents' workspace, LLM sessions and evidence. The upstream `chia` package is
vendored in `vendor/chia`, and `native/` holds the model API runner and the
common seed model. `scripts/setup` installs everything in editable mode; for a
regular install, set `RAMULATOR_CHIA_ROOT` to this checkout so the native build
recipes can be found.

`configs/` holds one campaign template per LLM. `scripts/run-campaign` fills in
`${DATA_ROOT}`, `${WORK_ROOT}`, `${ARTIFACT_ROOT}` and, for Gemini,
`${VERTEX_PROJECT}`. `configs/compaction/` holds the DeepSeek and Gemini
context-compaction settings and `configs/tariffs/` the token prices. The agent
prompts are in `src/ramulator_chia/framework/prompts/staged_v1/`.

Builds, caches, outputs and `results/` stay outside the agents' workspace. A
search makes paid LLM calls only with `--allow-paid`.
