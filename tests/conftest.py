import pytest


@pytest.fixture(autouse=True)
def no_unattended_workers_in_tests(monkeypatch):
    monkeypatch.setenv("PS_DISABLE_WORKER_KICK", "1")
