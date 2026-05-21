"""Tool registry for dynamic tool management."""

import re
from typing import Any

from nanobot.agent.tools.base import Tool


def _diagnose_error(stderr: str, tool_name: str) -> str:
    """根据错误类型返回具体建议。"""
    # 导入缺失
    if "ModuleNotFoundError" in stderr or "No module named" in stderr:
        match = re.search(r"No module named '(\w+)'", stderr)
        module = match.group(1) if match else None
        if module:
            return f"\n\n[Suggestion: Run 'pip install {module}' first, then retry.]"
        return "\n\n[Suggestion: Install the missing Python package using pip, then retry.]"

    # 语法错误
    if "SyntaxError" in stderr:
        return "\n\n[Suggestion: Write the code to a .py file first, then use exec to run it.]"

    # 权限错误
    if "PermissionError" in stderr or "Permission denied" in stderr:
        return "\n\n[Suggestion: Permission denied. Try a different file path or ask the user for help.]"

    # 文件未找到
    if "FileNotFoundError" in stderr or "No such file" in stderr:
        return "\n\n[Suggestion: File not found. Check the path with list_dir first.]"

    # 超时
    if "timeout" in stderr.lower() or "TimedOut" in stderr:
        return "\n\n[Suggestion: Command timed out. Try splitting the task into smaller steps.]"

    # ImportError (非 ModuleNotFoundError)
    if "ImportError" in stderr:
        return "\n\n[Suggestion: Import error. Check if the module is installed with 'pip list'.]"

    # 通用提示（不重复给出 generic hint）
    return ""


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        """Execute a tool by name with given parameters."""

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        try:
            # Attempt to cast parameters to match schema types
            params = tool.cast_params(params)

            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)

            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                suggestion = _diagnose_error(result, name)
                if suggestion:
                    return result + suggestion
            return result
        except Exception as e:
            suggestion = _diagnose_error(str(e), name)
            return f"Error executing {name}: {str(e)}" + suggestion

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
