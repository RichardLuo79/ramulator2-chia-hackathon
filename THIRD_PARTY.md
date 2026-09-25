# Source provenance and notices

This is a source snapshot, not a claim that all files have one license. Preserve
the notices in every vendored component. The root MIT text is the retained
Ramulator notice; it does not override third-party terms.

| Component | Immutable source | Included material / terms |
|---|---|---|
| Ramulator | `72427a1bba3771564c4fb0e494ba02242fd1eaa7` | Source with integration patches; MIT notice in `ramulator/LICENSE` |
| CHIA | `16c35e92aaaf9511c6453bf94cd5cf589698f4e3` | Patched source under `chia-loop/vendor/chia`; BSD-3-Clause; session-hook provenance retained |
| ChampSim | `51588e1d6f97875fe8de1a3621d28668bff83fcf` | Apache-2.0 source snapshot plus consolidated integration/placement/contention patch |
| gem5 | `7a2b0e413d06c5ce7097104abef3b1d9eaabca91` (`v25.1.0.0`) | Integration files/build recipe; upstream per-file BSD-style notices; no simulator binaries |
| MeSS standalone | `7feba12ce4af081116592fb5f045895a3b86de34` | `ramulator/ext/mess`; upstream prediction code, BSD-3-Clause notice and narrow loader patch |
| Sniper WMG1 | `56505e42fd98bca863fac181e769bd3c98d2bb33` | `ramulator/ext/sniper_wmg1`; computational routines, adapter and original notices |
| zsim M/D/1 | `f524c05647a152f44f85ad445de3dcf9e4610bfd` | Local port; zsim reference is GPLv2 |

MeSS's standalone revision is not identical to the publication-era integrated
implementation. WMG1 preserves its clock/window quantization adapter. M/D/1 is
a documented port, not unchanged upstream source. FixedLat is the bandwidth-
queued local baseline.

`results/provenance/source-extraction.json` records source-file identities.
`results/provenance/runtime-profiles.json` records the runtime variants used
for accuracy and standalone speed measurements.

The included fmt and yaml-cpp trees retain their own notices. Sniper's bundled
license text contains multiple notices, which are preserved with the extracted
routines. Benchmark inputs and guest images are supplied separately.
