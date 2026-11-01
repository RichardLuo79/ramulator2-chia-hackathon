# Building gem5 with Ramulator

Fig. 7 uses gem5 v25.1 (tag `v25.1.0.0`, commit
`7a2b0e413d06c5ce7097104abef3b1d9eaabca91`) with the Ramulator wrapper in this
directory. Build the native runtime first (`scripts/setup --component runtime`),
then, from the artifact root:

```sh
git clone --branch v25.1.0.0 --single-branch \
  https://github.com/gem5/gem5.git .work/gem5-source
git -C .work/gem5-source rev-parse HEAD     # prints 7a2b0e413d06…
cp -R ramulator/resources/gem5_wrappers .work/gem5-source/src/mem/ramulator2
```

From `.work/gem5-source`, build with absolute paths to the runtime:

```sh
env RAMULATOR2_SOURCE=/absolute/path/to/artifact/.work/runtime/runtime-source \
    RAMULATOR2_LIBDIR=/absolute/path/to/artifact/.work/runtime/runtime \
    scons build/X86/gem5.opt -j8 --ignore-style --verbose
```

`RAMULATOR2_SOURCE` must contain `src/`, and `RAMULATOR2_LIBDIR` the
`libramulator.so` built from it. gem5's own build prerequisites are listed in its
documentation. Check in the verbose log that the build uses `-O3`.

Run the workloads with `chia-loop/src/ramulator_chia/eval/gem5/roi_study.py` as
described in the root `REPRODUCING.md`, in polling mode (`event_driven=False`,
`clock_edge_first=False`). Read the statistics of the active O3 CPU; the
switched-out CPU reports zero cycles. The guest programs, images and checkpoints
are described in `chia-loop/src/ramulator_chia/eval/gem5/PUBLIC_SUITE.md`.

The recorded results are in `results/extensions/` (`transfer.json`,
`opus-gem5.json`, `opus-gemini-per-case.csv`); Fig. 7 plots `results/paper/gem5.csv`.
