"""Tests for the /cases/audit endpoint and underlying repo query."""
import uuid
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from tests.conftest import make_factory


# ---------------------------------------------------------------------------
# App fixture — minimal FastAPI app with only the eval router mounted
# ---------------------------------------------------------------------------

class _FakeLoop:
    model = "fake-model"
    _last_usage: dict = {}

    async def process_direct(self, content, *, on_stream, on_progress, **kwargs):
        await on_stream("ok")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    import nanoresearch.storage.database as db_mod
    monkeypatch.setattr(db_mod, "_AsyncSessionLocal", make_factory())
    from nanoresearch.server.main import create_app
    return create_app(channel_loop=_FakeLoop(), session_factory=make_factory())


@pytest.fixture
def auth_headers(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    from nanoresearch.auth.jwt import create_token
    return {"Authorization": f"Bearer {create_token('testuser')}"}


@pytest.fixture
async def api_client(app, auth_headers):
    """Async HTTPX client backed by the FastAPI test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=auth_headers) as client:
        yield client


# ---------------------------------------------------------------------------
# Seeded data fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
async def seeded_cases(monkeypatch):
    """Insert 3 test cases (2 with real origin, 1 with NULL origin / legacy_pre_b4)."""
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    import nanoresearch.storage.database as db_mod
    factory = make_factory()
    monkeypatch.setattr(db_mod, "_AsyncSessionLocal", factory)

    from nanoresearch.storage.models import AgentTestCase

    cases = [
        AgentTestCase(
            dataset_type="golden",
            name="case_a",
            user_input="query A",
            target_dimension="tool_schema_correctness",
            added_at=_NOW,
            added_by="test",
            coverage_tags=[],
        ),
        AgentTestCase(
            dataset_type="golden",
            name="case_b",
            user_input="query B",
            target_dimension="tool_schema_correctness",
            added_at=_NOW,
            added_by="test",
            coverage_tags=[],
        ),
        AgentTestCase(
            dataset_type="regression",
            name="case_c",
            user_input="query C",
            origin_badcase_id=None,
            target_dimension="legacy_pre_b4",
            added_at=_NOW,
            added_by="backfill",
            coverage_tags=[],
        ),
    ]

    async with factory() as session:
        for c in cases:
            session.add(c)
        await session.commit()
        for c in cases:
            await session.refresh(c)

    yield cases

    # Cleanup: delete the seeded rows so tests are isolated
    async with factory() as session:
        for c in cases:
            from sqlalchemy import delete
            await session.execute(delete(AgentTestCase).where(AgentTestCase.id == c.id))
        await session.commit()


@pytest.fixture
async def seeded_cases_with_orphans(monkeypatch):
    """Insert cases where some have NULL origin_badcase_id (orphans)."""
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret" * 6)
    import nanoresearch.storage.database as db_mod
    factory = make_factory()
    monkeypatch.setattr(db_mod, "_AsyncSessionLocal", factory)

    from nanoresearch.storage.models import AgentTestCase

    orphan_id = uuid.uuid4()
    cases = [
        AgentTestCase(
            id=orphan_id,
            dataset_type="golden",
            name="orphan_case",
            user_input="orphan query",
            origin_badcase_id=None,
            target_dimension="legacy_pre_b4",
            added_at=_NOW,
            added_by="backfill",
            coverage_tags=[],
        ),
        AgentTestCase(
            dataset_type="golden",
            name="linked_case",
            user_input="linked query",
            origin_badcase_id=None,  # also NULL — orphan
            target_dimension="faithfulness_score",
            added_at=_NOW,
            added_by="test",
            coverage_tags=[],
        ),
    ]

    async with factory() as session:
        for c in cases:
            session.add(c)
        await session.commit()
        for c in cases:
            await session.refresh(c)

    yield cases

    async with factory() as session:
        for c in cases:
            from sqlalchemy import delete
            await session.execute(delete(AgentTestCase).where(AgentTestCase.id == c.id))
        await session.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_audit_endpoint_returns_total_and_histogram(api_client: AsyncClient, seeded_cases):
    resp = await api_client.get("/api/eval/agent/cases/audit")
    assert resp.status_code == 200
    body = resp.json()
    assert "total" in body
    assert "by_dimension" in body
    assert "orphaned" in body
    assert body["total"] == len(seeded_cases)
    assert isinstance(body["by_dimension"], dict)
    assert isinstance(body["orphaned"], list)


@pytest.mark.asyncio
async def test_audit_lists_orphan_cases(api_client: AsyncClient, seeded_cases_with_orphans):
    resp = await api_client.get("/api/eval/agent/cases/audit")
    body = resp.json()
    orphan_ids = {o["id"] for o in body["orphaned"]}
    expected_orphans = {str(c.id) for c in seeded_cases_with_orphans if c.origin_badcase_id is None}
    assert orphan_ids == expected_orphans


@pytest.mark.asyncio
async def test_audit_histogram_matches_real_distribution(api_client: AsyncClient, seeded_cases):
    resp = await api_client.get("/api/eval/agent/cases/audit")
    body = resp.json()
    expected = {}
    for c in seeded_cases:
        expected[c.target_dimension] = expected.get(c.target_dimension, 0) + 1
    assert body["by_dimension"] == expected
