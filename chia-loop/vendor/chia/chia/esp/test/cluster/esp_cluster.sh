#!/usr/bin/env bash
#
# Bring the local ESP test cluster up/down using YOUR currently-active
# environment — no hardcoded venv. Activate any conda/venv that has ray +
# chia, then run this from anywhere:
#
#   conda activate myenv        # or: source /path/to/venv/bin/activate
#   ./esp_cluster.sh up         # also: up --dry-run
#   ./esp_cluster.sh down
#
# chia launches the head node via a fresh SSH shell that does NOT inherit your
# interactive env, so we derive an activation command from the active env and
# pass it through CHIA_HEAD_ENV_SETUP (the YAML's head_env_commands expands it).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# .../chia/esp/test/cluster -> repo root is 4 levels up
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
# Default cluster config; point ESP_CLUSTER_YAML at another one in this
# directory (e.g. esp_xcelium.yaml) to target it instead.
YAML="${ESP_CLUSTER_YAML:-$HERE/esp_local.yaml}"

# Site-specific values (hostnames, CAD mounts — private, gitignored; see
# esp_site_env.example.sh).
[ -f "$HERE/esp_site_env.sh" ] && source "$HERE/esp_site_env.sh"

# chia expands ${VAR} in YAML values but not mapping KEYS (used by
# auth.overrides for tunneled hosts) — pre-render just the host var into a
# gitignored copy.
if [ -n "${CHIA_ESP_SIM_HOST:-}" ] && grep -q 'CHIA_ESP_SIM_HOST' "$YAML"; then
    RENDERED="$HERE/.esp_rendered.yaml"
    envsubst '${CHIA_ESP_SIM_HOST}' < "$YAML" > "$RENDERED"
    YAML="$RENDERED"
fi

if [ "$#" -lt 1 ]; then
    echo "usage: $0 up|down [extra chia args]" >&2
    exit 1
fi

# Derive a head-node activation command from the active environment.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    export CHIA_HEAD_ENV_SETUP="source ${VIRTUAL_ENV}/bin/activate"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    export CHIA_HEAD_ENV_SETUP="source $(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_DEFAULT_ENV:-base}"
else
    echo "ERROR: no active venv/conda env detected." >&2
    echo "Activate an environment with ray + chia first, then re-run." >&2
    exit 1
fi

cd "$REPO_ROOT"

# Preflight (from the repo root, where `chia` resolves): confirm THIS env can
# import what the head/driver need.
if ! python -c "import ray, chia.cli.main" >/dev/null 2>&1; then
    echo "ERROR: the active env can't import ray + chia (from $REPO_ROOT)." >&2
    echo "Use an env with ray (2.54.0) + chia installed/importable." >&2
    exit 1
fi

echo "[esp-cluster] head env activation: ${CHIA_HEAD_ENV_SETUP}"
exec python -m chia.cli.main "$@" "$YAML"
