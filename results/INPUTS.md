# Inputs and evidence acquisition

ChampSim inputs come directly from the official
[DPC4 trace collection](https://github.com/CMU-SAFARI/DPC4#workload-traces).
The upstream [download manifest](https://traces.rbera.com/dpc4/manifest.txt)
lists object names, sizes, and ETags. Our
[source manifest](manifests/input-sources.json) records the exact 52 objects,
download ranges, prefix lengths, and decoded SHA256 hashes. No mirror is needed.
Observe upstream's trace-use terms; no redistribution permission is implied.

After building the runtime and the ChampSim core counts required by your
configuration, run from the repository root:

```sh
.venv-campaign/bin/python scripts/fetch-data --download \
  --config chia-loop/configs/astra.json --data-root .data --workers 2
```

Use `opus.json` for the single-core-only configuration. Download concurrency
defaults to two (maximum four). `--disk-reserve-gib 32` leaves that much space
free; increase the reserve if other jobs share the filesystem. Downloads use
bounded `curl` retries and HTTP ranges. Rerun the identical command after an
interruption. Verified fragments stay in `.data/.downloads`; incomplete output
files are never accepted as prepared inputs. An ETag/range mismatch, short
prefix, corrupt stream, wrong hash, or inadequate disk space stops preparation.

Each trace is exactly **23 million 64-byte instruction records**: 2M warmup,
20M measurement, and read-ahead allowance. The decoded hash verifies those exact
bytes, and the completed local gzip stream is read back through its CRC. This
does **not** verify the unused remainder or CRC of the full upstream object.
An ETag is an object-version check, not necessarily a cryptographic digest.
The historical download ranges total about 4.4 GiB and recompressed prefixes
about 3.8 GiB; these are preparation estimates, not promised transfer sizes.

The existing instruction-page scanner and ChampSim allocator regenerate the
100 placement maps for the 10+5 cohort (52 for the single-core cohort). Every
map must match its frozen decoded SHA256. Maps cover data/instruction pages and
page-table keys in distinct address spaces. They preassign physical frames;
they do not prewarm caches or remove first-touch costs. Core ordering is fixed
by the configuration. Map preparation runs the allocator, not a simulation.

`.data/prepared-inputs.json` records the newly compressed file hashes and their
frozen decoded identities. The launcher resolves these storage hashes into a
fresh configuration. Historical hashes in `manifests/inputs.json` stay intact;
new compression never authorizes reusing an incompatible measurement cache.

To verify without downloading or preparing anything:

```sh
.venv-campaign/bin/python scripts/fetch-data --config chia-loop/configs/astra.json
```

Use matching `--data-root` and `--work-root` options for preparation, evaluation,
and campaign launch when storing inputs/builds elsewhere. Keep prepared inputs
immutable while a campaign uses them. If upstream objects change, investigate
and create a separately versioned dataset; do not edit hashes to bypass checks.

For gem5, use the exact recorded resource/kernel/image versions and ROI
checkpoint identities in `extensions/transfer.json` and `paper/paper-v1/paper-extensions.json`.
Build the pinned gem5 version and source-built guest programs using the
integration recipes. Guest images and checkpoints need separate permission;
their original private acquisition locations are not public download URLs.
Do not regenerate an approximate checkpoint and claim identity with a published
measurement. The standalone synthetic suites generate their own fixed-seed
inputs and do not require workload traces.

Missing or mismatched input files fail verification. There is no implicit cloud
fetch, private-repository dependency, trace substitution or cache fallback.
