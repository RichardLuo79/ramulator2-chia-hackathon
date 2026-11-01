# ChampSim bridge

`ramulator/integrations/champsim/source` is ChampSim with the Ramulator bridge and
the continuous-contention and placement patches; the adjacent provenance file
records the upstream revision. Build it with
`scripts/setup --component champsim --cores 1|4|8`.

The bridge reads the Ramulator configuration from `RAMULATOR_CONFIG`; the library
directory is set at build time with `RAMULATOR_LIBRARY_DIR`.

## How a run is measured

- Each core runs 2M warmup instructions and waits at a barrier until all cores
  finish warmup. Warmup bypasses Ramulator, so the controller starts measuring
  from an empty state.
- Each core records its statistics after its own 20M measured instructions, then
  keeps running as background traffic until the last core finishes. A trace that
  ends starts again from its beginning; CPU, cache and controller state carry on.
- Reads admitted in a core's measured window are scored, even if they complete
  after the window ends. Background traffic only adds contention.
- The placement map is loaded before the page-table walker starts, so every page
  lands on a fixed physical frame. First-touch costs remain.
- A model's reads are paired with the oracle's only when their physical
  addresses match.
