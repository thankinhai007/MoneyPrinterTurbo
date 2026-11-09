import pytest


@pytest.fixture(autouse=True)
def _disable_qc_gate_by_default():
    """Pipeline tests use placeholder media; the QC gate has its own tests.

    Tests that exercise the gate opt in with ``VideoParams(qc_enabled=True)``,
    which takes precedence over this config default.
    """
    from app.config import config

    previous = dict(config.qc)
    config.qc["enabled"] = False
    try:
        yield
    finally:
        config.qc.clear()
        config.qc.update(previous)
