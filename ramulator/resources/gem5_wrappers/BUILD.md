# gem5 O3CPU transfer build

Upstream: <https://github.com/gem5/gem5.git>, tag `v25.1.0.0`, commit
`7a2b0e413d06c5ce7097104abef3b1d9eaabca91`.

Use a new gem5 source directory and this artifact's wrapper snapshot. The paper
protocol uses **polling**, with `event_driven=False` and `clock_edge_first=False`.
Experimental event-driven hooks remain in the snapshot but must not be enabled
for transfer reproduction.

From the artifact root, after building the native runtime:

```sh
git clone --branch v25.1.0.0 --single-branch \
  https://github.com/gem5/gem5.git .work/gem5-source
git -C .work/gem5-source rev-parse HEAD
cp -R ramulator/resources/gem5_wrappers .work/gem5-source/src/mem/ramulator2
```

Verify the commit. From `.work/gem5-source`, supply **absolute paths** and build:

```sh
env RAMULATOR2_SOURCE=/absolute/artifact/.work/runtime/runtime-source \
    RAMULATOR2_LIBDIR=/absolute/artifact/.work/runtime/runtime \
    scons build/X86/gem5.opt -j8 --ignore-style --verbose
```

The source directory must contain `src/`; the library directory must contain
`libramulator.so` from that source. Retain the verbose log and verify `-O3`.
Use build prerequisites documented at the pinned gem5 revision. No binaries or
dependency bundle are release payloads.

The driver is `ramulator_chia/eval/gem5/roi_study.py`, with explicit paths to the
Ramulator Python package, memory configuration and guest inputs. Use O3CPU,
2M in-ROI warmup, then up to 2B additional committed instructions or ROI end.
Use an eight-hour process watchdog; partial attempts remain unscored. Select
the active O3 CPU's statistics, not the inactive CPU's zero-cycle counters.

SE recipes are in `chia-loop/src/ramulator_chia/eval/gem5/PUBLIC_SUITE.md`.
FS guest images and checkpoints require acquisition/redistribution review;
their identities remain in the input provenance. Do not edit checkpoint PCs.
