"""Read-only full-window oracle traffic coverage, independent of candidate scores."""
import json
import pathlib

import numpy as np
import pandas as pd

from tools.eval import artifacts as A


def traffic_population(directory, minimum):
    directory = pathlib.Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    cycles = manifest["per_core_cycles"][0]
    counts = np.zeros((10, 3), dtype=np.int64)
    sums = np.zeros(3, dtype=np.int64)
    # Time bins describe traffic; they are not independent/scored ROIs.
    with pd.read_csv(A.resolve(directory / "trace.csv.ch0"),
            usecols=["type", "arrive", "depart", "llc_path"], chunksize=250_000) as chunks:
        for chunk in chunks:
            reads = chunk[chunk.type == 0]
            paths = reads.llc_path.to_numpy()
            bins = np.minimum(9, reads.arrive.to_numpy() * 10 // cycles)
            np.add.at(counts, (bins, paths), 1)
            np.add.at(sums, paths, (reads.depart - reads.arrive).to_numpy())
    total = counts.sum(axis=0)
    if total.sum() == 0 or cycles <= 0:
        raise ValueError("empty oracle traffic population")
    return {"workload": manifest["workload"], "logical_reads": int(total.sum()),
        "hits": int(total[0]), "merges": int(total[1]), "owners": int(total[2]),
        "L": float(sums.sum() / total.sum()),
        "owner_mean_latency": float(sums[2] / total[2]) if total[2] else None,
        "minimum_owners": minimum, "owner_count_pass": bool(total[2] >= minimum),
        "read_counts_by_oracle_time_decile": counts.tolist(),
        "sustained_traffic_pass": bool((counts[1:, 2] >= minimum // 100).all()),
        "dram_reads": manifest["controller_stats"]["num_read_reqs"],
        "dram_writes": manifest["controller_stats"]["num_write_reqs"],
        "cycles": cycles, "simulation_wall_s": manifest["wall_s"]}
