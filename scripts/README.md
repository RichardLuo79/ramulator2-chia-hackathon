# Reproduction entry points

The six executable wrappers dispatch to `ramulator_chia.commands` through a
small repository bootstrap. Run their `--help` from
the artifact root; full examples are in `REPRODUCING.md`.

- `setup`: install one environment or build one component.
- `fetch-data`: download the selected configuration's DPC4 prefixes, generate
  placement maps, or verify existing inputs. Downloads require `--download`;
  the default is verification only. Archive extraction is explicit.
- `reproduce-figures`: reproduce Figures 1–8 from compact local evidence, offline.
- `evaluate`: run one frozen ChampSim case or synthetic point without an LLM.
- `run-campaign`: local `--check`, or fresh/resumed procedure with explicit configuration and paid
  opt-in; published evidence is not exposed to the agent.
- `verify-artifact`: compare public files with the sealed release inventory.

`qualify_quickstart.py` checks a relocated copy with old paths denied;
`qualify_artifact.py` checks representative full-window measurements.
`build_frontend.py` is a compatibility import for the package's build helper. No script here
provisions cloud resources, installs a credential, uploads results or pushes Git.
