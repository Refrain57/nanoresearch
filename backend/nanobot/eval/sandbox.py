"""Sandboxed tool registry for deterministic eval replay."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry


_MAX_RESULT_CHARS = 10000
_MAX_ENTRIES = 200


def _normalize_params(params: Any) -> Any:
    """Recursively sort dict keys so replay keys are stable regardless of insertion order."""
    if isinstance(params, dict):
        return {k: _normalize_params(v) for k, v in sorted(params.items())}
    if isinstance(params, list):
        return [_normalize_params(i) for i in params]
    return params


class SandboxReplayError(Exception):
    """Raised when a tool call has no recorded result in replay mode."""


class SandboxedToolRegistry:
    """Wraps a ToolRegistry with passthrough / record / replay / side_effect_only modes.

    - passthrough: calls through to the real registry, no recording
    - record: calls through and saves results keyed by (name, normalized_params);
              results are truncated to _MAX_RESULT_CHARS, capped at _MAX_ENTRIES
    - replay: returns recorded results without calling real tools;
              raises SandboxReplayError if a key is missing
    - side_effect_only: hybrid mode for Tool Description evaluation.
              Recording-first policy: if the recording key matches, return the recorded
              result regardless of tool type — a match means the candidate description
              produced the same call as baseline (same tool + same params), which is a
              meaningful signal that this invocation was unaffected by the description change.
              On recording miss:
                - query tools (side_effect=False): passthrough to live call
                - side-effect tools (side_effect=True): intercepted, appended to
                  audit_log, SandboxReplayError raised
                - unknown tools (not in registry): treated as side-effect (conservative)

    description_overrides: optional {tool_name: new_description} mapping applied in
    get_definitions() so the model sees the candidate description during evaluation.
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        mode: Literal["passthrough", "record", "replay", "side_effect_only"],
        recorded: dict[str, Any] | None = None,
        description_overrides: dict[str, str] | None = None,
    ) -> None:
        self._registry = registry
        self._mode = mode
        self._recorded: dict[str, Any] = dict(recorded) if recorded else {}
        self._dropped_count = 0
        self._description_overrides: dict[str, str] = dict(description_overrides) if description_overrides else {}
        self._audit_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # ToolRegistry interface — forward non-execution calls directly
    # ------------------------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        return self._registry.tool_names

    def get_definitions(self) -> list[dict]:
        defs = self._registry.get_definitions()
        if not self._description_overrides:
            return defs
        result = []
        for d in defs:
            fn = d.get("function", {})
            tool_name = fn.get("name", "")
            if tool_name in self._description_overrides:
                d = {**d, "function": {**fn, "description": self._description_overrides[tool_name]}}
            result.append(d)
        return result

    def register(self, tool: Any) -> None:
        self._registry.register(tool)

    # ------------------------------------------------------------------
    # Sandboxed execution
    # ------------------------------------------------------------------

    async def execute(self, name: str, params: dict) -> Any:
        key = f"{name}:{json.dumps(_normalize_params(params), ensure_ascii=False)}"

        if self._mode == "replay":
            if key not in self._recorded:
                raise SandboxReplayError(
                    f"No recorded result for tool '{name}' with params: {key}"
                )
            return self._recorded[key]

        if self._mode == "side_effect_only":
            return await self._execute_side_effect_only(name, params, key)

        result = await self._registry.execute(name, params)

        if self._mode == "record":
            if len(self._recorded) >= _MAX_ENTRIES:
                self._dropped_count += 1
                if self._dropped_count <= 3:
                    logger.warning(
                        "SandboxedToolRegistry: max entries ({}) reached, "
                        "dropping recording for '{}' ({} dropped so far)",
                        _MAX_ENTRIES, name, self._dropped_count,
                    )
            else:
                self._recorded[key] = self._truncate_result(result)

        return result

    async def _execute_side_effect_only(self, name: str, params: dict, key: str) -> Any:
        # Recording-first: if the baseline run produced this exact call, return its result.
        # A cache hit signals that the description change did NOT alter this invocation —
        # the recorded result is the real baseline outcome, not fabricated data.
        if key in self._recorded:
            return self._recorded[key]

        # Recording miss: route by side_effect classification.
        tool = self._registry.get(name) if hasattr(self._registry, "get") else None
        is_side_effect = tool.side_effect if tool is not None else True  # unknown → conservative

        if is_side_effect:
            entry = {
                "tool": name,
                "params": params,
                "key": key,
                "action": "intercepted",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            self._audit_log.append(entry)
            logger.warning(
                "SandboxedToolRegistry [side_effect_only]: intercepted side-effect tool '{}' "
                "(no recording) — call blocked, logged to audit_log",
                name,
            )
            raise SandboxReplayError(
                f"Side-effect tool '{name}' called with no recording in side_effect_only mode. "
                f"This call was intercepted to prevent real writes during evaluation."
            )

        # Query tool with no recording → passthrough to live call.
        logger.debug(
            "SandboxedToolRegistry [side_effect_only]: query tool '{}' cache miss → passthrough",
            name,
        )
        return await self._registry.execute(name, params)

    # ------------------------------------------------------------------
    # Recording access
    # ------------------------------------------------------------------

    @property
    def recordings(self) -> dict[str, Any]:
        """Return a copy of all recorded tool results."""
        return dict(self._recorded)

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        """Intercepted side-effect calls in side_effect_only mode (read-only copy)."""
        return list(self._audit_log)

    def export_recordings(self) -> str:
        """Serialize recordings to JSON string for storage."""
        try:
            return json.dumps(self._recorded, ensure_ascii=False, default=str)
        except (TypeError, ValueError, OverflowError) as e:
            logger.warning("SandboxedToolRegistry: export_recordings failed: {}", e)
            return "{}"

    @classmethod
    def from_recordings_json(
        cls,
        registry: "ToolRegistry",
        recordings_json: str,
    ) -> "SandboxedToolRegistry":
        """Reconstruct a replay-mode sandbox from a stored JSON string."""
        try:
            recorded = json.loads(recordings_json)
        except json.JSONDecodeError as e:
            raise SandboxReplayError(
                f"Invalid recording JSON: {e}"
            ) from e
        return cls(registry=registry, mode="replay", recorded=recorded)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_result(result: Any) -> Any:
        if isinstance(result, str) and len(result) > _MAX_RESULT_CHARS:
            return result[:_MAX_RESULT_CHARS] + "...(truncated)"
        return result
