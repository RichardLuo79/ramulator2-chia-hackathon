# Preserved frozen models

`astra_xhigh_iter12_20260915` preserves the exact Astra xhigh iteration-12
selection and its pre-rerun ChampSim evidence. Its model-source SHA256 is
`f21ab420f204a14fe72fc64f913c3ba2d7eedd3e55499dd07bbb2d678ef3d210`.

The compact archive contains source, parameters, selection and build provenance,
per-workload results, and checksums for the separately retained raw evidence.
It contains no simulator binaries, credentials, or operator conversations.
The expanded copy is included for review without unpacking the archive.

These results are historical: single-core used first-touch placement; multicore
used fixed placement but parked each finished core. The latter protocol is
superseded by continuous background execution. Neither dataset is pooled into
the new timing-independent, continuous-contention study.

Large raw records remain locally and on the persistent Milan disk. Their exact
identity and verification records are included in this snapshot. The original
16 GiB study archive is not stored as a Git object.
