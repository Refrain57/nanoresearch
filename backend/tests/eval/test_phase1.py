"""Phase 1 verification tests — P1-V2 through P1-V8.

Covers:
  P1-V2  Classifier outputs layer+target_kind for rule-based cases
  P1-V3  PersonaObject.read() returns correct content
  P1-V4  PersonaObject.apply() writes version registry + updates agent
  P1-V5  PersonaObject.rollback() restores old content
  P1-V6  ToolDescriptionObject.generate_candidates() returns candidates
  P1-V7  generate_proposals(ToolDescriptionObject) persists unscored, no crash
  P1-V8  context_trace appears in classifier LLM prompt
"""

from __future__ import annotations

import uuid
import pytest

from nanoresearch.eval.badcase_classifier import (
    BadcaseClassifier,
    ClassifyResult,
    FIXABLE_LAYERS,
    DIAGNOSIS_ONLY_LAYERS,
    _rule_based_root_cause,
    _build_classify_prompt,
)
from nanoresearch.eval.tunable import PersonaObject, ToolDescriptionObject, OptimizationCandidate


# ---------------------------------------------------------------------------
# Fake snapshot factory
# ---------------------------------------------------------------------------

def _snap(**kwargs):
    defaults = dict(
        user_input="test",
        tool_call_chain=[],
        final_response="ok",
        run_status="success",
        scores=None,
        total_input_tokens=100,
        judge_metadata=None,
        badcase_category=None,
        context_trace=None,
    )
    defaults.update(kwargs)
    return type("Snap", (), defaults)()


# ---------------------------------------------------------------------------
# P1-V2: Classifier rule layer outputs structured pointer
# ---------------------------------------------------------------------------

class TestClassifierRuleLayer:
    def test_tool_error_maps_to_tool_layer(self):
        snap = _snap(tool_call_chain=[{"tool": "web_search", "error": True}])
        result = _rule_based_root_cause(snap)
        assert result is not None
        assert result.layer == "Tool"
        assert result.target_kind == "tool_impl"
        assert result.target_id == "web_search"
        assert result.root_cause_auto == "tool"

    def test_failed_no_tools_maps_to_context_system_prompt(self):
        snap = _snap(run_status="failed", tool_call_chain=[])
        result = _rule_based_root_cause(snap)
        assert result is not None
        assert result.layer == "Context"
        assert result.target_kind == "system_prompt"

    def test_recall_ok_completion_low_maps_to_system_prompt(self):
        snap = _snap(scores={"contextual_recall": 0.8, "task_completion": 0.3})
        result = _rule_based_root_cause(snap)
        assert result is not None
        assert result.layer == "Context"
        assert result.target_kind == "system_prompt"

    def test_token_limit_maps_to_layer_none(self):
        snap = _snap(total_input_tokens=120_000)  # > 85% of 128k
        result = _rule_based_root_cause(snap)
        assert result is not None
        assert result.layer is None          # cannot determine layer without more info
        assert result.target_kind is None
        assert result.root_cause_auto == "model"

    def test_empty_tools_no_rule_fires(self):
        """All-empty tool outputs must NOT fire a rule — new-user safety."""
        snap = _snap(
            tool_call_chain=[{"output": ""}, {"output": []}],
            context_trace={"memory_fragment_ids": [], "history_actual_chars": 0},
        )
        result = _rule_based_root_cause(snap)
        assert result is None  # must fall through to LLM

    def test_classify_result_has_phase1_fields(self):
        snap = _snap(tool_call_chain=[{"error": True}])
        result = _rule_based_root_cause(snap)
        assert hasattr(result, "layer")
        assert hasattr(result, "target_kind")
        assert hasattr(result, "target_id")

    def test_fixable_layers_constant(self):
        assert "Context" in FIXABLE_LAYERS
        assert "Tool" in FIXABLE_LAYERS
        assert "Memory" not in FIXABLE_LAYERS
        assert "Recovery" not in FIXABLE_LAYERS

    def test_diagnosis_only_layers_constant(self):
        assert "Memory" in DIAGNOSIS_ONLY_LAYERS
        assert "Recovery" in DIAGNOSIS_ONLY_LAYERS
        assert "Context" not in DIAGNOSIS_ONLY_LAYERS


# ---------------------------------------------------------------------------
# P1-V8: context_trace appears in classifier LLM prompt
# ---------------------------------------------------------------------------

class TestClassifierPrompt:
    def test_context_trace_in_prompt(self):
        snap = _snap(
            user_input="hello",
            context_trace={
                "history_query": "what did I say yesterday",
                "memory_fragment_ids": ["um_abc", "um_def"],
                "memory_actual_chars": 512,
                "history_actual_chars": 1024,
                "memory_budget_tokens": 2000,
                "knowledge_budget_tokens": 3000,
                "persona_active": True,
                "skill_names": ["research"],
            },
        )
        prompt = _build_classify_prompt(snap)
        assert "history_query" in prompt
        assert "what did I say yesterday" in prompt
        assert "memory_fragment_ids" in prompt
        assert "2 个" in prompt  # 2 fragment ids
        assert "memory_actual_chars" in prompt
        assert "persona_active" in prompt

    def test_empty_context_trace_no_section(self):
        snap = _snap(context_trace=None)
        prompt = _build_classify_prompt(snap)
        assert "记忆检索" not in prompt

    def test_empty_fragments_shows_warning_hint(self):
        snap = _snap(context_trace={
            "history_query": "some query",
            "memory_fragment_ids": [],
            "memory_actual_chars": 0,
            "history_actual_chars": 0,
        })
        prompt = _build_classify_prompt(snap)
        # Must hint that empty could be new-user, not necessarily retrieval failure
        assert "新用户" in prompt or "可能" in prompt

    def test_no_history_query_shows_not_initiated(self):
        snap = _snap(context_trace={"history_query": None, "memory_fragment_ids": []})
        prompt = _build_classify_prompt(snap)
        assert "未发起检索" in prompt


# ---------------------------------------------------------------------------
# Mock repos for PersonaObject / ToolDescriptionObject
# ---------------------------------------------------------------------------

class _MockAgent:
    def __init__(self, agent_id, persona="initial persona", tools_config=None):
        self.id = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
        self.persona = persona
        self.tools_config = tools_config or []


class _MockAgentRepo:
    def __init__(self, agent: _MockAgent):
        self._agent = agent
        self.updates = []

    async def get_by_id(self, agent_id):
        return self._agent

    async def update(self, agent_id, **fields):
        self.updates.append(fields)
        for k, v in fields.items():
            setattr(self._agent, k, v)
        return self._agent


class _MockEvalRepo:
    def __init__(self):
        self._versions: dict[uuid.UUID, object] = {}
        self._active: dict[tuple, uuid.UUID] = {}  # (kind, target_id) -> version_id

    async def create_tunable_version(self, kind, target_id, content, created_by="system"):
        vid = uuid.uuid4()
        v = type("V", (), {"id": vid, "kind": kind, "target_id": target_id,
                            "content": content, "active": True})()
        # deactivate old
        old_key = (kind, target_id)
        self._active[old_key] = vid
        self._versions[vid] = v
        return vid

    async def get_current_tunable_version(self, kind, target_id):
        vid = self._active.get((kind, target_id))
        if vid is None:
            return None
        return self._versions.get(vid)

    async def get_tunable_version_by_id(self, version_id):
        return self._versions.get(version_id)

    async def create_optimization_proposal(self, category, proposals, created_by):
        return type("P", (), {"id": uuid.uuid4(), "category": category,
                               "proposals": proposals})()

    async def get_latest_snapshot_with_recordings(self, user_input):
        return None


class _MockProvider:
    def get_default_model(self):
        return "test-model"

    async def chat_with_retry(self, messages, system=None, model=None, **kw):
        return type("R", (), {"content": '[{"prompt": "improved persona", "rationale": "better"}]'})()


# ---------------------------------------------------------------------------
# P1-V3 + V4 + V5: PersonaObject read / apply / rollback
# ---------------------------------------------------------------------------

class TestPersonaObject:
    def _make(self, persona="initial persona"):
        agent_id = str(uuid.uuid4())
        agent = _MockAgent(agent_id, persona=persona)
        agent_repo = _MockAgentRepo(agent)
        eval_repo = _MockEvalRepo()
        obj = PersonaObject(
            agent_id=agent_id,
            agent_repo=agent_repo,
            eval_repo=eval_repo,
            provider=_MockProvider(),
        )
        return obj, agent_repo, eval_repo

    @pytest.mark.asyncio
    async def test_read_returns_persona(self):
        obj, _, _ = self._make("hello world persona")
        content = await obj.read()
        assert content == "hello world persona"

    @pytest.mark.asyncio
    async def test_read_empty_agent_returns_empty(self):
        obj, agent_repo, _ = self._make()
        agent_repo._agent.persona = None
        content = await obj.read()
        assert content == ""

    @pytest.mark.asyncio
    async def test_apply_writes_version_and_updates_agent(self):
        obj, agent_repo, eval_repo = self._make("old persona")
        version_id = await obj.apply("new persona text")

        # version registry has the new entry
        v = await eval_repo.get_tunable_version_by_id(uuid.UUID(version_id))
        assert v is not None
        assert v.content == "new persona text"

        # agent.persona was updated
        assert agent_repo._agent.persona == "new persona text"

        # read() reflects the update
        content = await obj.read()
        assert content == "new persona text"

    @pytest.mark.asyncio
    async def test_get_current_version_returns_none_before_apply(self):
        obj, _, _ = self._make()
        vid = await obj.get_current_version()
        assert vid is None

    @pytest.mark.asyncio
    async def test_get_current_version_after_apply(self):
        obj, _, _ = self._make()
        version_id = await obj.apply("some content")
        current = await obj.get_current_version()
        assert current == version_id

    @pytest.mark.asyncio
    async def test_rollback_restores_old_content(self):
        obj, agent_repo, _ = self._make("v1 persona")

        v1_id = await obj.apply("v1 persona")
        await obj.apply("v2 persona")
        assert agent_repo._agent.persona == "v2 persona"

        await obj.rollback(v1_id)
        # After rollback, agent should have v1 content
        assert agent_repo._agent.persona == "v1 persona"
        assert await obj.read() == "v1 persona"

    @pytest.mark.asyncio
    async def test_rollback_invalid_version_raises(self):
        obj, _, _ = self._make()
        with pytest.raises(ValueError, match="not found"):
            await obj.rollback(str(uuid.uuid4()))

    @pytest.mark.asyncio
    async def test_generate_candidates_returns_list(self):
        obj, _, _ = self._make()
        candidates = await obj.generate_candidates([])
        assert isinstance(candidates, list)
        # MockProvider returns one candidate
        assert len(candidates) >= 1
        assert candidates[0].prompt == "improved persona"

    def test_kind_is_system_prompt(self):
        obj, _, _ = self._make()
        assert obj.kind == "system_prompt"

    def test_target_id_is_agent_id(self):
        agent_id = str(uuid.uuid4())
        agent = _MockAgent(agent_id)
        obj = PersonaObject(
            agent_id=agent_id,
            agent_repo=_MockAgentRepo(agent),
            eval_repo=_MockEvalRepo(),
            provider=_MockProvider(),
        )
        assert obj.target_id == agent_id


# ---------------------------------------------------------------------------
# P1-V6: ToolDescriptionObject.generate_candidates
# ---------------------------------------------------------------------------

class TestToolDescriptionObject:
    def _make(self, tool_name="web_search", description="Search the web"):
        agent_id = str(uuid.uuid4())
        agent = _MockAgent(agent_id, tools_config=[
            {"name": tool_name, "description": description}
        ])
        agent_repo = _MockAgentRepo(agent)
        eval_repo = _MockEvalRepo()
        obj = ToolDescriptionObject(
            tool_name=tool_name,
            agent_id=agent_id,
            agent_repo=agent_repo,
            eval_repo=eval_repo,
            provider=_MockProvider(),
        )
        return obj, agent_repo, eval_repo

    @pytest.mark.asyncio
    async def test_read_returns_description(self):
        obj, _, _ = self._make(description="original description")
        content = await obj.read()
        assert content == "original description"

    @pytest.mark.asyncio
    async def test_read_missing_tool_returns_empty(self):
        obj, _, _ = self._make()
        # Change tool name to something not in tools_config
        obj._tool_name = "nonexistent_tool"
        content = await obj.read()
        assert content == ""

    @pytest.mark.asyncio
    async def test_generate_candidates_returns_list(self):
        obj, _, _ = self._make()
        candidates = await obj.generate_candidates([])
        assert isinstance(candidates, list)
        assert len(candidates) >= 1

    @pytest.mark.asyncio
    async def test_apply_updates_tools_config(self):
        obj, agent_repo, eval_repo = self._make(description="old desc")
        version_id = await obj.apply("new description")

        # version registry updated
        v = await eval_repo.get_tunable_version_by_id(uuid.UUID(version_id))
        assert v.content == "new description"

        # tools_config updated
        updated_tools = agent_repo._agent.tools_config
        tool = next(t for t in updated_tools if t["name"] == "web_search")
        assert tool["description"] == "new description"

    @pytest.mark.asyncio
    async def test_rollback_restores_description(self):
        obj, agent_repo, _ = self._make(description="v1 desc")
        v1_id = await obj.apply("v1 desc")
        await obj.apply("v2 desc")

        await obj.rollback(v1_id)
        tool = next(t for t in agent_repo._agent.tools_config if t["name"] == "web_search")
        assert tool["description"] == "v1 desc"

    def test_kind_is_tool_description(self):
        obj, _, _ = self._make()
        assert obj.kind == "tool_description"

    def test_target_id_is_tool_name(self):
        obj, _, _ = self._make(tool_name="my_tool")
        assert obj.target_id == "my_tool"
