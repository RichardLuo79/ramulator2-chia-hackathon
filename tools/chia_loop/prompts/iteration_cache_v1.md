Use the frozen system contract. Propose one coherent change to the supplied
parent, informed by this run's training feedback. The following fields
are supplied by the runner; all optimization results and history belong to
this individual model run. Published comparison models are fixed baselines.

## Fixed policy and available actions

{{promotion_and_guardrail_configuration}}

{{allowed_tools_and_readable_file_manifest}}

## Fixed training configuration and comparisons

{{training_configuration_and_trace_clock_contract}}

{{published_comparison_training_metrics}}

## Current iteration

Evaluated design iteration {{iteration_number}} of {{max_proposal_iterations}}.
Draft repairs and inspections stay within this iteration until one design is
successfully evaluated or a budget/safety limit stops this run.

## Parent and incumbent

Parent ID: {{parent_id}}
Parent source SHA-256: {{parent_source_sha256}}
Incumbent ID: {{incumbent_id}}

{{parent_source}}

{{parent_and_incumbent_training_scores}}

## Training evidence

{{per_workload_training_metrics_and_coverage}}

{{training_diagnostic_slices}}

## Evolution history

{{accepted_and_rejected_proposals_with_reasons}}

{{last_build_or_evaluation_diagnostics}}

{{this_run_pareto_archive}}

{{remaining_model_call_diagnostic_and_token_limits}}

Return the JSON proposal defined by the system contract. Do not use or request
held-out information. No human modeling hint is supplied for this trial.
