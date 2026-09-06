"""Global offline-safety guard for the test suite."""

import os
import socket

import pytest

# Tests exercise observability through fakes. Never export from a developer's
# local Logfire credentials while importing the application under test.
os.environ["YT_CRAWL_DISABLE_TELEMETRY"] = "1"


@pytest.fixture(autouse=True)
def block_real_network(monkeypatch):
    """Fail every test that accidentally attempts a real socket connection."""

    def blocked(*_args, **_kwargs):
        raise AssertionError("network access is forbidden in offline tests")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
