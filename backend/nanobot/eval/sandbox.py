"""Sandboxed tool registry for deterministic eval replay."""

from __future__ import annotations

import json
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
    """Wraps a ToolRegistry with passthrough / record / replay modes.

    - passthrough: calls through to the real registry, no recording
    - record: calls through and saves results keyed by (name, normalized_params);
              results are truncated to _MAX_RESULT_CHARS, capped at _MAX_ENTRIES
    - replay: returns recorded results without calling real tools;
              raises SandboxReplayError if a key is missing
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        mode: Literal["passthrough", "record", "replay"],
        recorded: dict[str, Any] | None = None,
    ) -> None:
        self._registry = registry
        self._mode = mode
        self._recorded: dict[str, Any] = dict(recorded) if recorded else {}
        self._dropped_count = 0

    # ------------------------------------------------------------------
    # ToolRegistry interface — forward non-execution calls directly
    # ------------------------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        return self._registry.tool_names

    def get_definitions(self) -> list[dict]:
        return self._registry.get_definitions()

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

    # ------------------------------------------------------------------
    # Recording access
    # ------------------------------------------------------------------

    @property
    def recordings(self) -> dict[str, Any]:
        """Return a copy of all recorded tool results."""
        return dict(self._recorded)

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
