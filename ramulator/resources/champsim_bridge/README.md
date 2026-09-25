# ChampSim bridge

The snapshot under `ramulator/integrations/champsim/source` includes the bridge
and continuous-contention/placement patches. The adjacent provenance records
its immutable upstream revision. Build with `scripts/setup --component champsim
--cores 1|4|8`.

The bridge selects Ramulator through `RAMULATOR_CONFIG`. Configuration expansion
and input staging use the native evaluator. `RAMULATOR_LIBRARY_DIR` is explicit
at build time; no private library directory is a default.

## Measurement semantics

- Each core runs 2M warmup instructions, then waits at a warmup barrier.
  CPU/cache warmup bypasses Ramulator; its controller starts measurement cold.
- Each core snapshots statistics after its own 20M-instruction window and keeps
  running as background traffic until the final core finishes. Trace replay
  rewinds input, not CPU/cache/controller state or instruction identities.
- Phase records and admission ordinals identify measured-window reads.
  Background traffic creates contention but is excluded from scores. Eligible
  reads may depart after their source core's measurement endpoint.
- Timing-independent maps load before page-table walker construction.
  Preassignment preserves first-touch penalties and does not warm simulation
  state. Missing maps, capacity exhaustion and aliases are errors.
- Pairing rejects physical-address mismatch. Closed-loop cache behavior can
  still reduce coverage. No extra drain is added to improve coverage.

The native sources define clock conversion, write acknowledgements, and callback
ordering. Changes require new runtime and measurement identities.
