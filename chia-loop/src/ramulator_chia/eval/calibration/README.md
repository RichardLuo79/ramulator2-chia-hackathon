# Mess calibration curve

`mess_DDR5.txt` is the bandwidth–latency curve used by the Mess baseline, fitted
to the cycle-level DDR5 oracle with refresh disabled. Each row is
`read_percentage bandwidth_GBps latency_ns`. Its SHA-256 is
`3b2b8f7664173f185249b8a447b0bdebd37e0be7d0a9ac77eca275389cfa3417`.

If you change the oracle, the DDR5 timings, the controller's queue policy or the
calibration method, fit a new curve and update the checksum.
