# Inputs

The ChampSim simulations need two kinds of input that are too large to keep in
this repository:

- **Workload traces:** the first 23M instructions (2M warmup, 20M measured, plus
  read-ahead) of 52 traces from the
  [DPC-4 trace distribution](https://github.com/CMU-SAFARI/DPC4#workload-traces).
- **Placement maps:** one per workload or mix (100 in total), assigning physical
  pages before the simulation starts.

Build ChampSim for 1, 4 and 8 cores (see [REPRODUCING.md](../REPRODUCING.md)),
then run:

```sh
.venv-campaign/bin/python scripts/fetch-data --download
```

This downloads the trace prefixes from `https://traces.rbera.com/dpc4/`, rebuilds
the placement maps with ChampSim, and checks every file against
`manifests/inputs.json`. The files go to `.data/` (change it with `--data-root`),
together with `prepared-inputs.json`, which records what was checked. With
`--workers 4` it took about five minutes on an 8-core machine. It needs about
9 GB of disk: 4 GB for the inputs and 4.5 GB of raw downloads in
`.data/.downloads`, which you can delete afterwards. If the download is
interrupted, run the same command again; it resumes. Use the DPC-4 traces under
the terms of their distribution.

To check the inputs later:

```sh
.venv-campaign/bin/python scripts/fetch-data
```

`manifests/input-sources.json` lists, for each trace, the upstream file, the
number of bytes read from it and the content hash of the prefix.
`manifests/inputs.json` lists every input file with its hash.

The speed measurements (Fig. 5) generate their own inputs with
`scripts/setup --component speed-inputs`. The gem5 inputs (Fig. 7) are described
in `chia-loop/src/ramulator_chia/eval/gem5/PUBLIC_SUITE.md`.
