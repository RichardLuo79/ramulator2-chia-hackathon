# Rebuild gem5 for the CHIA DDR5 experiments

Ramulator 2.1's [integration guide](https://github.com/CMU-SAFARI/ramulator2#6-gem5-integration)
documents gem5 v25.1 support. Use this source pin for new experiments:

- Upstream: `https://github.com/gem5/gem5.git`
- Tag: `v25.1.0.0`
- Commit: `7a2b0e413d06c5ce7097104abef3b1d9eaabca91`

Start in a new directory. Do not reuse the old patched gem5 v24 build tree.

```sh
git clone --depth 1 --branch v25.1.0.0 --single-branch \
  https://github.com/gem5/gem5.git gem5
git -C gem5 rev-parse HEAD
cp -R /path/to/ramulator2/resources/gem5_wrappers gem5/src/mem/ramulator2
cd gem5
env RAMULATOR2_SOURCE=/path/to/ramulator2 \
    RAMULATOR2_LIBDIR=/path/to/ramulator2 \
    scons build/X86/gem5.opt -j6 --ignore-style --verbose
```

The source path supplies `src/`; the library path supplies `libramulator.so`.
They must come from the same recorded Ramulator build. Separate paths allow the
CHIA build's frozen source tree and its output directory to be used directly.
gem5's `opt` target uses `-O3`; retain the verbose build log. Reserve at most
12 cores across this build and any concurrent evaluations.

Use the wrapper sources from the campaign's source snapshot. Relative to the
public wrapper, this CHIA branch also records request identities and binds
completion callbacks to individual accepted requests. Those changes are part of
the source evidence. No patches to gem5's core or prefetcher are required by this
recipe. Do not transfer old local core patches implicitly.

A source-built installation check can use the program shipped at the same pin:

```sh
cc -O3 -static tests/test-progs/hello/src/hello.c -o /path/to/build/hello
```

Run it with Ramulator's `examples/gem5_se_ramulator_hello_world.py`, supplying
the Python package path and the newly built guest path. Hello-world checks the
integration only; it is not a representative DRAM workload or a replacement for
the scientific transfer suite.

The old `gem5bench` executables are separate inputs; rebuilding gem5 does not
recover their sources. The user approved a new, source-built transfer cohort;
see [the public guest suite](../../tools/eval/gem5/PUBLIC_SUITE.md). Keep the old
results separate. Archive the source pins, wrapper source, build commands and
logs—not gem5, shared libraries, or guest executables.
