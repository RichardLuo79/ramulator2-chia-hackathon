# Ramulator 2 sources

The Ramulator 2 sources used by the artifact: the simulator, its Python
configuration API, dependencies with their notices, and the ChampSim and gem5
integrations.

Build through `scripts/setup --component runtime` from the artifact root. The
native runner and common model interface live in `chia-loop/native/`; source
staging binds them to these sources and records the compiler and source hashes.
Provenance is in `patches/`, `ext/*/UPSTREAM.json` and `integrations/champsim/`;
each component keeps its own license notice.

Standalone speed options and physical-byte synthetic addresses are opt-in. The
gem5 results use polling mode.
