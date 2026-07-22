"""Shared pytest config: the fast/slow split.

`pytest tests/` is the DEV LOOP: it skips tests marked `slow` (the
end-to-end solver/search runs, ~5 min of scipy optimization) and
finishes in ~20 s. The skips are reported in the summary so the gap is
never silent.

`pytest tests/ --full` is the GATE: runs everything. Run it (ideally
with `-n auto`, pytest-xdist) before every commit/push.
"""

import os

import pytest

# Under pytest-xdist, N workers x N BLAS threads thrash the cores and
# erase the whole speedup (measured: -n 4 was as slow as serial). Pin
# each worker's numpy/scipy to one thread; serial runs keep full BLAS.
# Must happen at conftest import, before any test module pulls in numpy.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_var, "1")


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", default=False,
        help="also run the slow end-to-end solver tests (the pre-push gate)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--full"):
        return
    skip = pytest.mark.skip(
        reason="slow solver test — run the full gate with --full before pushing"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
