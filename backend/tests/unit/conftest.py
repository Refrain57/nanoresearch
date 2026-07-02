"""Override session fixtures for pure unit tests (no DB / Redis required)."""
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """No-op: unit tests are pure and do not need a database."""
    pass
