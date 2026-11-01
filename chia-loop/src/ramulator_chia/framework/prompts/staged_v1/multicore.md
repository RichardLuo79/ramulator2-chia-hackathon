# Multicore stage

This is the multicore stage, rounds 11–15. Continue from the selected single-core
model and this campaign's accumulated knowledge.

Improve four- and eight-core behavior while retaining useful single-core
accuracy. Inspect contention, read/write interactions, individual-core errors,
and request tails. Completed cores continue generating background traffic;
only their measured-window requests enter scored errors.

Treat one-, four-, and eight-core groups separately. Explain any trade-off
rather than hiding it in a pooled mean. Use only named training cases for
detailed diagnosis. End with one frozen candidate.
