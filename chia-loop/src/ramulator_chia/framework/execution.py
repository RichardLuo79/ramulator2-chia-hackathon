"""Small host-lifetime controls, separate from scientific campaign settings."""

from datetime import datetime, timezone
import time


class HostDeadlineReached(RuntimeError):
    """Leave the next round untouched so another host can resume it."""


def round_admission(state, deadline, *, now=None):
    """Conservatively use completed round durations, including provider waiting.

    Six hours is the initial allowance; thereafter use at least 1.5 times the
    longest completed round. This is admission control, never an evaluation
    timeout or permission to shorten a measurement.
    """
    clock = time.time if now is None else now

    def admit(number, stage):
        # Resuming a partially executed round is not starting a new round.
        if state._query("SELECT 1 FROM attempts WHERE step LIKE ? LIMIT 1",
                        (f"iteration:{number}:%",)):
            return
        rows = state._query(
            "SELECT step,started_at,finished_at FROM attempts "
            "WHERE step LIKE 'iteration:%' AND status='complete'"
        )
        durations = {}
        for row in rows:
            iteration = int(row["step"].split(":")[1])
            if state.get(f"iteration:{iteration}") is None:
                continue
            start, finish = (datetime.fromisoformat(row[k].replace("Z", "+00:00")).timestamp()
                             for k in ("started_at", "finished_at"))
            low, high = durations.get(iteration, (start, finish))
            durations[iteration] = min(low, start), max(high, finish)
        estimate = max([6 * 3600, *(1.5 * (b - a) for a, b in durations.values())])
        if clock() + estimate >= deadline:
            end = datetime.fromtimestamp(deadline, timezone.utc).isoformat()
            raise HostDeadlineReached(
                f"round {number} deferred: estimated {estimate / 3600:.1f} hours "
                f"would cross host work deadline {end}"
            )

    return admit
