# MESS calibration inputs

`mess_DDR5.txt` is a bandwidth/latency surface fitted to the no-refresh DDR5
cycle-level reference used by this evaluator. Each row is
`read_percentage bandwidth_GBps latency_ns`. The retained source artifact has
SHA-256 `3b2b8f7664173f185249b8a447b0bdebd37e0be7d0a9ac77eca275389cfa3417`.

The file is an input to the MESS comparison model, not a learned candidate rule.
It must be recalibrated—and its checksum changed—when the oracle, DDR timing,
reference queue policy, or latency-throughput calibration method changes.
