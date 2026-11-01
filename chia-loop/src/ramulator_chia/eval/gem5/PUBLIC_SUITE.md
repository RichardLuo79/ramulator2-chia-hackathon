# gem5 inputs (Fig. 7)

Fig. 7 uses six PolyBench/C kernels in syscall-emulation (SE) mode (ATAX, BiCG,
MVT, GEMVER, GESUMMV, TRISOLV) and nine workloads in full-system (FS) mode: NPB
IS/CG/MG, GAPBS BFS/PageRank/SSSP and PARSEC Canneal/Streamcluster/Black-Scholes.
Each run uses the O3CPU model with 2M warmup instructions inside the region of
interest (ROI), then measures up to 2B instructions or until the ROI ends.

Build gem5 first, as described in `ramulator/resources/gem5_wrappers/BUILD.md`.

## SE: PolyBench kernels

Download [polybench-c-4.2.1-beta.tar.gz](https://downloads.sourceforge.net/project/polybench/polybench-c-4.2.1-beta.tar.gz)
and build the kernels with the ROI markers used in the paper:

```sh
.venv-campaign/bin/python -m ramulator_chia.eval.gem5.build_roi_suite \
  --archive /path/to/polybench-c-4.2.1-beta.tar.gz \
  --gem5-source .work/gem5-source \
  --output .work/polybench-roi
```

The builder checks the archive and the patched sources against `roi_guests.json`
and records the commands and the hashes of the binaries it builds. Run the
binaries only inside gem5.

## FS: images and checkpoints

Download these [gem5 Resources](https://resources.gem5.org) into one directory,
for example with gem5's `obtain_resource`:

- kernels `x86-linux-kernel-6.8.0-52-generic-1.0.0` (NPB, GAPBS) and
  `x86-linux-kernel-4.19.83-1.0.0` (PARSEC);
- disk images `x86-ubuntu-24.04-npb-img-5.0.0`, `x86-ubuntu-24.04-gapbs-img-1.0.0`
  and `x86-parsec-1.0.0`.

Then create one checkpoint per suite, just before the workload starts (this
needs KVM):

```sh
build/X86/gem5.opt --outdir=.work/fs-npb \
  /path/to/artifact/chia-loop/src/ramulator_chia/eval/gem5/fs_board.py prepare \
  --suite npb --cores 1 --resources /path/to/resources
```

The checkpoint is written to `<outdir>/checkpoint`. Pass it to `roi_study.py`
with `--checkpoint` and `--resources`. The resource versions and checkpoint
identities of the paper's runs are in `results/extensions/transfer.json`.

`public_suite.json` and `build_suite.py` build the whole-program guests used by
the test fixtures.
