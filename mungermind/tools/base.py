"""工具系统基础：参数声明、工具基类与注册表。

``ToolRegistry`` 同时支持两种接入方式：

- :meth:`ToolRegistry.register_tool`：注册实现了 :class:`Tool` 的对象，可声明参数并
  生成 OpenAI function-calling schema；
- :meth:`ToolRegistry.register_function`：直接包装一个 ``str -> str`` 函数，便于复用
  已有工具代码。
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.exceptions import ToolExecutionError, ToolNotFoundError

logger = logging.getLogger(__name__)


class ToolParameter(BaseModel):
    """工具参数定义。"""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None


class Tool(ABC):
    """工具基类。"""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具并返回字符串结果。"""

    def get_parameters(self) -> List[ToolParameter]:
        """声明参数列表，默认无参数。"""
        return []

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function-calling schema。"""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in self.get_parameters():
            prop: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.default is not None:
                prop["description"] = f"{param.description} (默认: {param.default})".strip()
            if param.type == "array":
                prop["items"] = {"type": "string"}
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ToolRegistry:
    """工具注册表：注册、发现与执行。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._functions: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ 注册
    def register_tool(self, tool: Tool) -> None:
        """注册 ``Tool`` 对象。"""
        if not isinstance(tool, Tool):
            raise TypeError(f"期望 Tool 实例，收到 {type(tool).__name__}")
        if tool.name in self._tools or tool.name in self._functions:
            logger.warning("工具 '%s' 已存在，将被覆盖。", tool.name)
        self._tools[tool.name] = tool
        logger.debug("工具 '%s' 已注册。", tool.name)

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable[[str], str],
    ) -> None:
        """把 ``str -> str`` 函数注册为工具。"""
        if not name:
            raise ValueError("工具名称不能为空")
        if not callable(func):
            raise TypeError("func 必须是可调用对象")
        if name in self._tools or name in self._functions:
            logger.warning("工具 '%s' 已存在，将被覆盖。", name)
        self._functions[name] = {"description": description, "func": func}
        logger.debug("工具 '%s' 已注册。", name)

    def unregister(self, name: str) -> bool:
        """移除工具，返回是否存在。"""
        removed_tool = self._tools.pop(name, None) is not None
        removed_function = self._functions.pop(name, None) is not None
        return removed_tool or removed_function

    # ------------------------------------------------------------------ 发现
    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable[[str], str]]:
        info = self._functions.get(name)
        return info["func"] if info else None

    def list_tools(self) -> List[str]:
        return [*self._tools, *self._functions]

    def get_tools_description(self) -> str:
        """生成用于提示词的工具描述。"""
        descriptions = [
            f"- {tool.name}: {tool.description}" for tool in self._tools.values()
        ]
        descriptions.extend(
            f"- {name}: {info['description']}"
            for name, info in self._functions.items()
        )
        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def get_openai_schemas(self) -> List[Dict[str, Any]]:
        """生成 function-calling 工具列表；函数工具统一使用 ``input`` 字符串参数。"""
        schemas = [tool.to_openai_schema() for tool in self._tools.values()]
        for name, info in self._functions.items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info["description"],
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "传递给该工具的输入文本",
                                }
                            },
                            "required": ["input"],
                        },
                    },
                }
            )
        return schemas

    # ------------------------------------------------------------------ 执行
    def execute_tool(self, name: str, tool_input: Any) -> str:
        """执行工具。

        ``tool_input`` 可以是字符串（按参数定义自动映射）或参数字典。
        """
        if name in self._tools:
            tool = self._tools[name]
            params = tool_input if isinstance(tool_input, dict) else self._coerce_input(
                tool, tool_input
            )
            return self._run_tool(tool, params)
        if name in self._functions:
            return self._run_function(name, tool_input)
        raise ToolNotFoundError(
            f"未找到名为 '{name}' 的工具。可用工具: {', '.join(self.list_tools()) or '无'}"
        )

    def execute_tool_params(self, name: str, params: Dict[str, Any]) -> str:
        """执行工具并显式传入参数字典（供 function calling 使用）。"""
        if name in self._tools:
            return self._run_tool(self._tools[name], dict(params))
        if name in self._functions:
            return self._run_function(name, params)
        raise ToolNotFoundError(
            f"未找到名为 '{name}' 的工具。可用工具: {', '.join(self.list_tools()) or '无'}"
        )

    def _run_tool(self, tool: Tool, params: Dict[str, Any]) -> str:
        if not isinstance(params, dict):
            raise ToolExecutionError(
                f"工具 '{tool.name}' 的参数必须是字典，收到 {type(params).__name__}"
            )
        try:
            return str(tool.run(params))
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"工具 '{tool.name}' 执行失败: {exc}") from exc

    def _run_function(self, name: str, tool_input: Any) -> str:
        info = self._functions[name]
        func = info["func"]
        try:
            if isinstance(tool_input, dict):
                if "input" in tool_input:
                    return str(func(tool_input["input"]))
                if len(tool_input) == 1:
                    return str(func(next(iter(tool_input.values()))))
                return str(func(json.dumps(tool_input, ensure_ascii=False)))
            return str(func("" if tool_input is None else tool_input))
        except Exception as exc:
            raise ToolExecutionError(f"工具 '{name}' 执行失败: {exc}") from exc

    @staticmethod
    def _coerce_input(tool: Tool, tool_input: Any) -> Dict[str, Any]:
        if tool_input is None:
            text = ""
        elif isinstance(tool_input, str):
            text = tool_input.strip()
        else:
            raise ToolExecutionError(
                f"工具 '{tool.name}' 的参数必须是字符串或字典，收到 {type(tool_input).__name__}"
            )

        if text.startswith("{"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                return data

        params = tool.get_parameters()
        if len(params) == 1:
            return {params[0].name: _strip_quotes(text)}
        if not params:
            return {}
        if "=" in text:
            parsed: Dict[str, Any] = {}
            for pair in text.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    parsed[key.strip()] = _strip_quotes(value.strip())
            if parsed:
                return parsed
        raise ToolExecutionError(
            f"工具 '{tool.name}' 需要参数 {[p.name for p in params]}，"
            "请使用 JSON 对象或 key=value 形式传入。"
        )


def _strip_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text
