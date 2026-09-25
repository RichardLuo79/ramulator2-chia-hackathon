# Source-built gem5 ROI inputs

The transfer study uses six SE PolyBench/C kernels and nine FS workloads. SE
uses ATAX, BiCG, MVT, GEMVER, GESUMMV and TRISOLV. FS uses NPB IS/CG/MG, GAPBS
BFS/PageRank/SSSP and PARSEC Canneal/Streamcluster/Black-Scholes.

The final protocol is O3CPU, 2M in-ROI warmup, then up to 2B additional committed
instructions or original ROI end. Initialization is not silently counted as
kernel work. Short ROIs are not replayed. Instruction-capped and naturally
completed intervals are reported separately.

## SE build

`roi_guests.json` pins the PolyBench/C 4.2.1-beta archive, original/patched kernel
hashes, flags and original guest identities. `polybench-roi.patch` reconstructs
the exact recorded ROI markers. The builder verifies source bytes before and
after patching; it does not download inputs, call a provider or run a simulation.

```sh
.venv-campaign/bin/python -m ramulator_chia.eval.gem5.build_roi_suite \
  --archive /path/to/polybench-c-4.2.1-beta.tar.gz \
  --gem5-source /path/to/pinned-gem5-source \
  --output .work/polybench-roi
```

Run from the artifact root. Source URL/SHA256 are in `roi_guests.json`. Use the
gem5 pin in `ramulator/resources/gem5_wrappers/BUILD.md`. Output records actual
commands and new binary hashes; cross-toolchain byte identity is not assumed.
Do not run m5-instrumented binaries as ordinary native host applications.

`public_suite.json` and `build_suite.py` retain an earlier source-acquisition
recipe used by harness fixtures. It measures whole programs and is **not** the
final ROI protocol. Do not combine its timings with the final transfer results.

## FS inputs

Kernel/image/checkpoint identities are recorded under `results/provenance/`.
Images and checkpoints are not redistributed here. Until acquisition and
licensing are cleared, FS remains an explicit reproduction blocker, not an
implicit private-bucket download. Do not substitute inputs or unsafe checkpoints
to fill missing results.
