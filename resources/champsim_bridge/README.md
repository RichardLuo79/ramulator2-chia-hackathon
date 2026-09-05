# ChampSim bridge

Runs Ramulator2 (cycle-level oracle or candidate controller) as ChampSim's
memory backend, selected at runtime:

    RAMULATOR_CONFIG=<expanded-config.json> bin/champsim ...

## Setup

Against a ChampSim checkout (tested at `4adb621`):

1. Copy `ramulator_bridge.h` and `ramulator_bridge_iface.h` to `inc/`, and
   `ramulator_bridge.cc` to `src/`.
2. In `src/dram_controller.cc`, guard `MEMORY_CONTROLLER::operate()`: include
   `ramulator_bridge_iface.h` under `#ifdef CHAMPSIM_RAMULATOR`; at the start of
   `operate()`, return `warmup_cycle(queues)` during warmup or `cycle(queues)`
   during the measured region whenever `ramulator_bridge::active()` is true.
3. Add `-I<ramulator2>/src -DCHAMPSIM_RAMULATOR` to `absolute.options`, add
   `-L<ramulator2> -lramulator -Wl,-rpath,<ramulator2>` to `LDLIBS`, and compile
   `ramulator_bridge.o` with `-std=gnu++20` (the rest of ChampSim may remain
   C++17).
4. Generate configs with the Python module from the exact worktree under test.
   Presets resolve Python-side; the C++ bridge consumes expanded configs:

       PYTHONPATH=/path/to/ramulator2/python python3 \
         resources/champsim_bridge/gen_configs.py /tmp/ramcfg \
         --standards DDR5

5. Rebuild every Ramulator worktree, then run a clean pre/post matrix.
   `cs_gentest.py` hashes the executable, configs, traces, outputs, and every
   resolved `libramulator.so`; rejects dirty tracked source trees and nonempty
   output directories; and writes raw output plus `summary.json`. The shared
   library is an untracked build product, so the runner records its hash but
   cannot independently prove that it came from the adjacent revision; the
   clean rebuild is a required attestation.

       python3 resources/champsim_bridge/cs_gentest.py \
         --champsim /path/to/champsim/bin/champsim \
         --trace-dir /path/to/dpc4 \
         --oracle-config /tmp/ramcfg/oracle_DDR5.json \
         --candidate-config /tmp/ramcfg/candidate_DDR5.json \
         --oracle-library-dir /path/to/postfix-ramulator2 \
         --candidate pre_audit=/path/to/base-ramulator2 \
         --candidate post_fix=/path/to/postfix-ramulator2 \
         --output-dir /tmp/champsim-transfer \
         --ticks-per-8 12

The checked-in runner evaluates the six configured DPC-4 traces as a transfer
matrix. It is future plumbing and is not part of the initial SimpleO3 smoke run.

## Clock and measurement semantics

Clock calibration matters. The tested ChampSim build calls
`MEMORY_CONTROLLER::operate()` every 625 ps, while DDR5-4800's configured tCK
is approximately 416 ps. `RAMULATOR_TICKS_PER_8=12` advances 1.5 Ramulator
ticks per ChampSim memory-controller step and is the calibrated DDR5 setting.
The historical default of 8 advances one tick per step; it is internally fair
between oracle and candidate but stretches DDR5 timing by about 1.5x.

The bridge intentionally bypasses Ramulator during ChampSim warmup, so the
memory backend starts the measured ROI cold. This behavior is identical for
both controllers and must be disclosed with results. The bridge does not emit
paired-request latency traces; its transfer metric is closed-loop core cycles.
