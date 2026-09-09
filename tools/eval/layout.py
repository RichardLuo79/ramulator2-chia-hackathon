"""Resolve the address recipe shared by Ramulator's generic trace generators."""


def extract_dram_layout(dram):
    """Return native generator parameters from the Python DRAM specification.

    This is the existing test generator's mapping: bank groups cycle fastest,
    followed by pseudo-channels when the standard has them. It is not a model
    policy, a measured workload or a hand-written table of DRAM presets.
    """
    cls = type(dram)
    levels = list(cls.levels)
    organization, _ = dram.resolve()
    counts = [organization.get(name.lower(), 1) for name in levels]
    row, column = levels.index("Row"), levels.index("Column")
    banks = list(range(1, row))
    for level in ("BankGroup", "PseudoChannel"):
        if level in levels:
            position = levels.index(level)
            banks.remove(position)
            banks.append(position)
    bank_counts = [counts[position] for position in banks]
    total = 1
    for count in bank_counts:
        total *= count
    return {
        "addr_vec_size": len(levels),
        "bank_positions": banks,
        "bank_counts": bank_counts,
        "total_bank_units": total,
        "row_pos": row,
        "col_pos": column,
        "num_rows": counts[row],
        "num_cols": counts[column],
        "internal_prefetch_size": cls.internal_prefetch_size,
        "num_cls": counts[column] // cls.internal_prefetch_size,
    }
