"""SimpleAgent：基础对话范式，可选文本式工具调用。

工具调用采用 ``[TOOL_CALL:name:parameters]`` 文本协议，兼容不支持原生
function calling 的模型。
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Optional

from ..core.agent import Agent
from ..core.config import Config
from ..core.exceptions import MungerMindError
from ..core.llm import LLM
from ..core.message import Message
from ..tools.base import ToolRegistry

TOOL_CALL_PATTERN = re.compile(r"\[TOOL_CALL:\s*([^:\]]+?)\s*:\s*(.*?)\]")


class SimpleAgent(Agent):
    """对话型 Agent，支持多轮历史与可选工具调用。"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tool_calling: bool = True,
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

    def run(
        self,
        input_text: str,
        max_tool_iterations: int = 3,
        **kwargs: Any,
    ) -> str:
        """处理一次用户输入并返回最终回答。"""
        messages = self.build_messages(
            input_text, system_prompt=self._get_enhanced_system_prompt()
        )

        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.record_exchange(input_text, response)
            return response

        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        """在系统提示词后追加工具描述和调用格式。"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"
        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        return (
            base_prompt
            + "\n\n## 可用工具\n你可以使用以下工具来帮助回答问题：\n"
            + tools_description
            + "\n\n## 工具调用格式\n当需要使用工具时，请使用以下格式：\n"
            + "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
            + "例如：`[TOOL_CALL:calculator:(25 + 15) * 3]`\n\n"
            + "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"
        )

    def _run_with_tools(
        self,
        messages: list[dict[str, Any]],
        input_text: str,
        max_tool_iterations: int,
        **kwargs: Any,
    ) -> str:
        final_response = ""

        for _ in range(max(1, max_tool_iterations)):
            response = self.llm.invoke(messages, **kwargs)
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                final_response = response
                break

            results = []
            clean_response = response
            for call in tool_calls:
                results.append(
                    self._execute_tool_call(call["tool_name"], call["parameters"])
                )
                clean_response = clean_response.replace(call["original"], "")

            messages.append({"role": "assistant", "content": clean_response.strip()})
            messages.append(
                {
                    "role": "user",
                    "content": "工具执行结果：\n"
                    + "\n\n".join(results)
                    + "\n\n请基于这些结果给出完整的回答。",
                }
            )

        if not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        self.record_exchange(input_text, final_response)
        return final_response

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict[str, str]]:
        """解析文本中的 ``[TOOL_CALL:name:parameters]`` 标记。"""
        calls = []
        for match in TOOL_CALL_PATTERN.finditer(text):
            tool_name, parameters = match.groups()
            calls.append(
                {
                    "tool_name": tool_name.strip(),
                    "parameters": parameters.strip(),
                    "original": match.group(0),
                }
            )
        return calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"
        try:
            result = self.tool_registry.execute_tool(tool_name, parameters)
        except MungerMindError as exc:
            return f"❌ {exc}"
        return f"🔧 工具 {tool_name} 执行结果：\n{result}"

    def stream_run(self, input_text: str, **kwargs: Any) -> Iterator[str]:
        """流式输出版本，同时写入对话历史。"""
        messages = self.build_messages(input_text)
        collected: list[str] = []
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            collected.append(chunk)
            yield chunk
        response = "".join(collected)
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=response, role="assistant"))

    def add_tool(self, tool: Any) -> None:
        """动态添加工具对象。"""
        if self.tool_registry is None:
            self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool(tool)
        self.enable_tool_calling = True

    def has_tools(self) -> bool:
        return bool(self.enable_tool_calling and self.tool_registry and self.tool_registry.list_tools())

    def remove_tool(self, tool_name: str) -> bool:
        return bool(self.tool_registry and self.tool_registry.unregister(tool_name))

    def list_tools(self) -> list[str]:
        return self.tool_registry.list_tools() if self.tool_registry else []
