"""P2.2: consolidation emits atomic events + backfills memory_facts.derived_from (C4)."""
import asyncio
import pytest

from nanoresearch.agent.memory import MemoryStore
from nanoresearch.storage.repositories.memory_facts_repo import MemoryFactsRepository
from tests.conftest import make_factory, pg_conn


def run(coro):
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.fixture(autouse=True)
def clean_memory_facts():
    conn = pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE memory_facts RESTART IDENTITY CASCADE")
    finally:
        conn.close()


class _FakeKS:
    def __init__(self):
        self.events = []
        self.summaries = []

    def write_events_sync(self, events, uid=None):
        self.events.append((uid, events))
        return [f"ev_{i}" for i in range(len(events))]

    def write_user_memory_sync(self, memories, uid=None):
        self.summaries.append((uid, memories))
        return (len(memories), 0)

    def write_conv_summary_sync(self, text, uid=None, conversation_id=None,
                                turn_start=0, turn_end=0, topic=""):
        self.summaries.append({"text": text, "conversation_id": conversation_id, "turn_end": turn_end})
        return "cs_1"


class _TC:
    def __init__(self, args):
        self.arguments = args


class _Resp:
    def __init__(self, args):
        self.finish_reason = "tool_calls"
        self.content = ""
        self.has_tool_calls = True
        self.tool_calls = [_TC(args)]


class _Prov:
    def __init__(self, args):
        self._args = args

    async def chat_with_retry(self, **kw):
        return _Resp(self._args)


def test_consolidate_writes_events_and_backfills_derived_from(tmp_path):
    async def _():
        factory = make_factory()
        ks = _FakeKS()
        store = MemoryStore(tmp_path, knowledge_search=ks, session_factory=factory)
        args = {
            "history_entry": "[2026-07-01 10:00] user compared A vs B",
            "memory_update": "# User Memory\n\n## FACTS\n- 偏好 Python\n",
            "events": [{"topic": "3DGS", "action": "compared", "result": "A wins"}],
        }
        ok = await store.consolidate(
            [{"role": "user", "content": "x"}], _Prov(args), "m", uid="u1", conversation_id="c1",
        )
        assert ok
        # events written with conversation_id + time
        assert ks.events and ks.events[0][0] == "u1"
        ev = ks.events[0][1][0]
        assert ev["conversation_id"] == "c1" and ev["topic"] == "3DGS" and ev["time"]
        # profile fact carries derived_from = the event ids just written (C4)
        facts = await MemoryFactsRepository(factory).list_active("u1")
        assert any(f.text == "偏好 Python" and f.derived_from == ["ev_0"] for f in facts)
    run(_())


def test_consolidate_no_events_key_is_safe(tmp_path):
    async def _():
        factory = make_factory()
        ks = _FakeKS()
        store = MemoryStore(tmp_path, knowledge_search=ks, session_factory=factory)
        args = {"history_entry": "[t] x", "memory_update": "# User Memory\n\n## FACTS\n- a\n"}
        ok = await store.consolidate([{"role": "user", "content": "x"}], _Prov(args), "m",
                                     uid="u1", conversation_id="c1")
        assert ok and ks.events == []  # no events key → no event write
        facts = await MemoryFactsRepository(factory).list_active("u1")
        assert any(f.text == "a" and f.derived_from == [] for f in facts)
    run(_())
