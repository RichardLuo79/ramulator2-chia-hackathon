# Source-built gem5 transfer inputs

This replaces the twelve old executables whose sources were unavailable. It is
a new cohort, not a rerun of those programs. The profile is
`tools/chia_loop/configs/ddr5_public_transfer_v2.json`; SimpleO3 training/testing
and ChampSim transfer are unchanged. **Oracle qualification is still pending.**
The clean gem5 v25.1.0.0 installation check passed. All twelve guests built and
ran natively, including successful GAP numerical verification. Their full
common-loop gem5 qualification is incomplete; native execution alone is not acceptance.
Several GAP cases have reached the current 30-minute simulation guard, including
the BFS oracle. Their partial traces show advancing simulation, but do not prove
normal completion. The whole-program inputs remain unchanged. Longer-timeout
qualification is deferred while paid search runs; these failures are not scores.

## Sources and workload choices

Exact archives, SHA-256 checksums, compilation flags and arguments are data in
[public_suite.json](public_suite.json). No kernel source patch is applied.

| Public source | Selected programs | Memory behavior |
| --- | --- | --- |
| [PolyBench/C 4.2.1 beta](https://sourceforge.net/projects/polybench/files/) | ATAX, BiCG, MVT, GEMVER, GESUMMV, TRISOLV | Row/column traversal, matrix-vector reuse, streaming updates and dependent triangular solve |
| [GAP Benchmark Suite](https://github.com/sbeamer/gapbs/tree/2972aeb2703165bafd921222f4ed7196f542d3a8) | BFS, PageRank, SSSP, betweenness centrality, connected components, triangle counting | Irregular graph traversal, frontier/queue traffic and indirect accesses |

PolyBench uses the published `LARGE_DATASET`: its principal matrix is about
32 MB (13.52 MB per matrix for GESUMMV), well beyond the configured 1 MiB L2.
Output-vector dumps prevent dead-code elimination and preserve computed results.
Cache-flush/timing instrumentation is disabled; it is not the workload we want
to measure. Arithmetic uses the compiler's normal rules, not `-ffast-math`.

GAP uses its serial implementation and built-in uniform graph generator with
2^17 vertices, average input degree 16 and upstream seed 27491095. CC and TC
symmetrize the graph; the other kernels use directed graphs. One complete trial
runs with the algorithm's upstream defaults (including PageRank's 20-iteration
maximum). This is not a published GAP benchmark score or its full input suite.

All programs run to normal exit. Measurement includes allocation, initialization,
graph generation and result printing—not just the central kernel. Before this
cohort is reported as a completed transfer study, the oracle check must report executed
instructions, DRAM traffic and runtime for every program. Do not shorten windows
or select inputs using a candidate's errors. Transfer remains hidden until the
selected model is frozen; incomplete request pairing still withholds its request
headline.

## Build and use

Download the two URLs in `public_suite.json` under their declared archive names.
The builder verifies the complete archive checksums before extracting anything.
It then uses the system GCC/G++ with `-O3`, static linking and one compilation job.
The source archives and build are not obtained from an earlier experiment.

```sh
python -m tools.eval.gem5.build_suite \
  --archives /path/to/downloads --output /path/to/new-guest-build
```

The output contains `sources/`, `bin/`, compiler logs and `suite.json`, which
records source/program hashes and actual commands. Use a fresh output directory;
failed build evidence is retained. The builder does not download software, call
a model or run a simulation.

Build gem5 using [the pinned v25.1 recipe](../../../resources/gem5_wrappers/BUILD.md).
For the no-provider common-loop qualification:

```sh
GEM5_BIN=/path/to/gem5/build/X86/gem5.opt \
python tests/utils/check_chia_clean_campaign.py \
  --runtime /path/to/ramulator-runtime --output /path/to/new-check \
  --workers 6 --iterations 1 --mcp \
  --evaluation tools/chia_loop/configs/ddr5_public_transfer_v2.json \
  --gem5-programs /path/to/new-guest-build/suite.json
```

Use the [pinned CHIA environment](../../chia_loop/framework/upstream/README.md).
This scripted check qualifies the harness; it does not optimize a DRAM model.
Add `--archive-output /path/to/artifacts` and
`--frontend-recipes /path/to/frontends.json` to seal a completed check. See the
[operator guide](../../chia_loop/framework/README.md#source-only-artifacts) for
the frontend recipe format. The exporter uses `build_suite.source_recipes` for
the selected guests automatically. Only source files and build commands are
exported; installed executables stay local. External dependencies
may be fetched again when rebuilding; environment bundling is not supported.
The source-export regression extracts the archive to a new directory and runs
its recorded build commands. Frontend build-provenance receipts are included
separately, so an archived source recipe stays tied to the build used for scoring.
