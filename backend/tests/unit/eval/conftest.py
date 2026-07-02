"""Pure-unit-test scope for eval logic (collector, etc.).

These tests exercise in-memory logic only and must not depend on a database.
The repo-wide ``setup_database`` fixture (tests/conftest.py) is session-scoped
and autouse; here we override it with a no-op so a polluted/unavailable shared
test DB can never block these pure unit tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_database():  # noqa: D401 — overrides the root DB-creating fixture
    yield
