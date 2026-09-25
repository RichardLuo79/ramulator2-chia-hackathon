# Ramulator source and integrations

This directory contains the selected Ramulator sources, Python configuration
API, dependencies with notices, and frontend integrations.

Build through `scripts/setup --component runtime` from the artifact root. The
native runner and common model interface live in `chia-loop/native/`; source
staging binds them to this snapshot and records the compiler and source hashes.
See `patches/`, `ext/*/UPSTREAM.json`, `integrations/champsim/`, and the root
`THIRD_PARTY.md` for provenance and redistribution limitations.

Standalone speed options and physical-byte synthetic addresses are opt-in.
The gem5 transfer protocol uses polling; experimental event-driven hooks are
not part of its reported results.
