"""FunctionCallAgent：基于 OpenAI 原生 function calling 的工具型 Agent。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..core.agent import Agent
from ..core.config import Config
from ..core.exceptions import MungerMindError, ToolExecutionError
from ..core.llm import LLM
from ..tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class FunctionCallAgent(Agent):
    """使用模型原生工具调用能力完成多步任务。"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_tool_iterations: int = 10,
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_tool_iterations = max(1, max_tool_iterations)

    def run(self, input_text: str, **kwargs: Any) -> str:
        messages = self.build_messages(input_text)
        schemas = self._build_tool_schemas()

        if not schemas:
            final_answer = self.llm.invoke(messages, **kwargs)
            self.record_exchange(input_text, final_answer)
            return final_answer

        for step in range(1, self.max_tool_iterations + 1):
            response = self.llm.create_completion(
                messages, tools=schemas, tool_choice="auto", **kwargs
            )
            choices = getattr(response, "choices", None)
            if not choices:
                break

            message = choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                final_answer = self._extract_message_content(message)
                self.record_exchange(input_text, final_answer)
                return final_answer

            logger.debug("第 %d 轮检测到 %d 个工具调用", step, len(tool_calls))
            messages.append(self._assistant_tool_call_message(message, tool_calls))

            for call in tool_calls:
                messages.append(self._execute_tool_call(call))

        # 达到迭代上限后，关闭工具再请求一次，让模型总结已有结果。
        final_answer = self.llm.invoke(messages, **kwargs)
        self.record_exchange(input_text, final_answer)
        return final_answer

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        return self.tool_registry.get_openai_schemas()

    def _execute_tool_call(self, call: Any) -> Dict[str, Any]:
        function = getattr(call, "function", None)
        name = getattr(function, "name", "")
        raw_arguments = getattr(function, "arguments", "") or "{}"

        try:
            arguments = self._parse_function_call_arguments(raw_arguments)
            arguments = self._convert_parameter_types(name, arguments)
            result = self.tool_registry.execute_tool_params(name, arguments)
        except MungerMindError as exc:
            result = f"❌ {exc}"

        return {
            "role": "tool",
            "tool_call_id": getattr(call, "id", ""),
            "content": result,
        }

    @staticmethod
    def _assistant_tool_call_message(message: Any, tool_calls: List[Any]) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [
                {
                    "id": getattr(call, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(getattr(call, "function", None), "name", ""),
                        "arguments": getattr(
                            getattr(call, "function", None), "arguments", "{}"
                        ),
                    },
                }
                for call in tool_calls
            ],
        }

    @staticmethod
    def _extract_message_content(message: Any) -> str:
        if isinstance(message, dict):
            return message.get("content") or ""
        return getattr(message, "content", None) or ""

    @staticmethod
    def _parse_function_call_arguments(raw_arguments: str) -> Dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"工具参数不是合法 JSON: {exc}") from exc
        if not isinstance(arguments, dict):
            raise ToolExecutionError("工具参数必须是 JSON 对象")
        return arguments

    def _convert_parameter_types(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        tool = self.tool_registry.get_tool(tool_name)
        if tool is None:
            return arguments

        types = {param.name: param.type for param in tool.get_parameters()}
        converted: Dict[str, Any] = {}
        for key, value in arguments.items():
            target = types.get(key)
            if isinstance(value, str) and target in {"integer", "number", "boolean"}:
                try:
                    if target == "integer":
                        value = int(value)
                    elif target == "number":
                        value = float(value)
                    else:
                        value = value.strip().lower() in {"true", "1", "yes"}
                except ValueError:
                    pass
            converted[key] = value
        return converted
