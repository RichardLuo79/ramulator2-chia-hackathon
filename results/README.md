# Results and models

Start with the [paper snapshot](paper/paper-v1/) to reproduce the current paper,
or [models/](models/) to inspect the frozen implementations. The
[figure inventory](paper/paper-v1/manifest.json) maps Figures 1–8 to their data,
captions, plotting functions, and checksums.

- [paper/paper-v1/](paper/paper-v1/): complete compact inputs for the paper
  notebook, reference CSVs, and configuration/mechanisms tables.
- [foundation/](foundation/): single-core measurements, search histories, and token usage.
- [extensions/](extensions/): multicore, organization/queue transfer, and gem5 records.
- [diagnostics/](diagnostics/): standalone speed measurements; Lat–Tp points
  are embedded in `foundation/campaigns.json`.
- [models/](models/): exact frozen source, parameters and selection identities.
- [provenance/](provenance/): source/runtime identities, campaign records, and accounting methods.
- [manifests/](manifests/): source/input/archive/object identities and redaction links.
- [figures/](figures/) and [tables/](tables/): additional analysis exports.
  The current paper figures and their CSV source tables are in
  [analyses/figures/](../analyses/figures/).

The paper snapshot includes all four single-core campaigns, Opus 5.5 gem5
accuracy, Gemini 3.8 Flash multicore results, and both standalone-speed batches.
Gemini 3.8 Flash gem5 and Opus 5.5 multicore results are not included.

Campaign feedback and the paper's baseline comparisons have separate measurement
identities. Use the paper snapshot for the plotted comparisons, and campaign
records for the feedback used during search.

Large observations and campaign logs are indexed by
[manifests/archives.json](manifests/archives.json). Transfer and synthetic records
are listed in [manifests/supplemental-evidence.json](manifests/supplemental-evidence.json).
The compact data needed for all paper figures is included in Git; raw archives
are supplied separately and have no configured download URL.
