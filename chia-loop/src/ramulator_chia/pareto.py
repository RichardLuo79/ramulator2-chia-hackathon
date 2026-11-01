"""The established two-objective rule, independent of any provider runner."""


def objectives(summary):
    aggregate = summary["aggregate"]
    return (aggregate["cycle_macro_mae_pct"], aggregate["request_macro_mae_over_L"])


def dominates(left, right):
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def pareto(candidates):
    return sorted(
        key
        for key, value in candidates.items()
        if not any(
            dominates(objectives(other["metrics"]), objectives(value["metrics"]))
            for other in candidates.values()
        )
    )
