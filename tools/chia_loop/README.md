# CHIA loop: generic Atomic controller

`atomic_loop.py` is a native CHIA graph for the first proof-of-concept run. It
uses `@ChiaFunction` nodes for the agent, optimized build, baseline matrix,
candidate evaluations, Pareto selection, validation, and trace archival. The
dummy backend replaces only the LLM/coding agent; every build, simulator run,
comparator, metric, and integrity gate is real.

The smoke run deliberately adjusts only the skeleton's constant latency. After
the first training result, the dummy agent subtracts the median signed request
residual from that intercept, bounded to 25% of its current value. This follows
the only causal degree of freedom in the seed and avoids inventing bank or bus
behavior before an agent proposes and tests such structure.

Selection treats core-cycle macro MAE and request macro MAE/L as co-equal.
A challenger replaces the incumbent only if it is no worse on both and better
on at least one. Incomparable smoke candidates retain the earlier incumbent.
Validation is evaluated once, after selection, and cannot feed the agent.

## Reproducible local setup

CHIA recommends Python 3.10.19 for parity with its containers. The package
metadata supports Python 3.10 and newer.

```sh
python3.10 -m venv /tmp/ramulator-chia
/tmp/ramulator-chia/bin/pip install -r tools/chia_loop/requirements.txt
PYTHONPATH=$PWD/python:$PWD/tools:$PWD \
  /tmp/ramulator-chia/bin/python tools/chia_loop/atomic_loop.py \
  --config tools/chia_loop/smoke.json --run-id smoke-local
```

No API key is required for `backend: dummy`. A later real-agent backend will
need credentials appropriate to the selected CHIA model adapter.

The runner refuses a tracked-dirty worktree, requires the local
`atomic-chia-loop` branch, never invokes a push, builds with explicit `-O3`,
and caps build/evaluation parallelism at 12.

## Audit and storage

Each run writes `eval_out/chia/<run-id>/`:

- `run_manifest.json`: scope, versions, human decisions, agent mode, results,
  selection, validation, and final status;
- `audit.jsonl`: ordered actions and whether a human intervened in-run;
- `interactions/`: complete structured dummy-agent requests and responses;
- `profiles/`: CHIA's native dispatch/completion/dependency JSONL trace;
- `reports/`: baseline, per-iteration training, selection, and held-out
  validation reports;
- `evaluation/`: raw manifests and request traces;
- `artifacts/`: verified trace and auxiliary gzip manifests.

Every tracked file except the explicitly agent-owned Atomic implementation is
hashed before the loop and checked after each build/evaluation. Any evaluator,
metric, frontend, comparison model, or orchestration mutation fails the run.

Completed `*.chN` traces are gzip-compressed individually, verified, and read
transparently by the evaluator. Large command logs, transcripts, and CHIA
profiles are also compressed above the configured threshold. Restore them with:

```sh
python tools/eval/archive_results.py restore \
  eval_out/chia/RUN/artifacts/trace_archive_manifest.json
python tools/chia_loop/artifacts.py restore \
  eval_out/chia/RUN/artifacts/aux_archive_manifest.json
```

CHIA execution caches, when enabled in a later distributed campaign, are
disposable bounded caches. They are not result archives and do not replace the
checksum-bearing manifests above.
