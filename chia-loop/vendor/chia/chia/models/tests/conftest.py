"""Shared fixtures for the model-backend tests.

The ``live_remote`` marker + ``remote_prompt`` fixture back the
``test_live_remote_*`` permission tests. Where the ``@live`` tests run a
backend's ``prompt`` in-process, the ``live_remote`` tests dispatch it onto a
real Ray worker via ``.chia_remote`` so they exercise the permission flag /
config block on the worker (e.g. inside the chia-claude-code / chia-opencode
container) exactly as a production run would.

Gating (two layers, so the same suite runs against whatever cluster is up):
  * ``live_remote`` tests are skipped unless ``CHIA_LIVE_CLUSTER=1``.
  * ``remote_prompt`` additionally skips a test when the connected cluster does
    not advertise that backend's creds resource (e.g. ``claude_creds``).

Bring up a worker for the backend(s) under test (see
``chia/models/tests/cluster/``), then::

    CHIA_LIVE_CLUSTER=1 RAY_ADDRESS=auto \
        pytest chia/models/tests/ -k live_remote -v

``RAY_ADDRESS`` defaults to ``auto`` (a head on this machine). Workers run the
image's ``chia``; this checkout is shipped via ``runtime_env`` py_modules so
they have ``chia.models`` + the current ``ChiaFunction`` trampoline. Override
the shipped path with ``CHIA_SHIP_PY_MODULES`` (``""`` to skip shipping when the
cluster already deploys this code).
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_remote: dispatches prompt onto a live Ray cluster (needs CHIA_LIVE_CLUSTER=1)",
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CHIA_LIVE_CLUSTER") == "1":
        return
    skip = pytest.mark.skip(
        reason="set CHIA_LIVE_CLUSTER=1 (and bring up a cluster) to run live_remote tests"
    )
    for item in items:
        if "live_remote" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def ray_cluster():
    """Connect to the cluster with THIS checkout's ``chia`` shipped to workers.
    """
    import ray

    import chia

    default_pkg = next(p for p in chia.__path__ if os.path.isdir(p))
    ship = os.environ.get("CHIA_SHIP_PY_MODULES", default_pkg)
    runtime_env = {"py_modules": [ship]} if ship else None
    if ray.is_initialized():
        ray.shutdown()
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), runtime_env=runtime_env)
    try:
        yield
    finally:
        ray.shutdown()


@pytest.fixture
def remote_prompt(ray_cluster):
    """Return ``dispatch(llm, message, resource, tools=None) -> QueryResult``.

    Dispatches ``llm.prompt`` onto a worker advertising ``resource`` (via
    ``.options(resources=...).chia_remote``) and resolves the result. Skips the
    test when the cluster doesn't advertise ``resource``, so a test only runs
    where its backend's worker is actually up.
    """
    import ray

    from chia.base.ChiaFunction import get

    def _present(resource):
        return any(
            n.get("Alive") and n.get("Resources", {}).get(resource, 0) >= 0.01
            for n in ray.nodes()
        )

    def dispatch(llm, message, resource, tools=None):
        if not _present(resource):
            pytest.skip(f"cluster does not advertise the {resource!r} resource")
        ref = llm.prompt.options(resources={resource: 0.01}).chia_remote(
            llm, message, tools or []
        )
        return get(ref)

    return dispatch
