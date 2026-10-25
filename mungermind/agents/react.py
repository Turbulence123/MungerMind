"""ReActAgent：Thought → Action → Observation 循环。"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Tuple

from ..core.agent import Agent
from ..core.config import Config
from ..core.exceptions import AgentError, MungerMindError
from ..core.llm import LLM
from ..tools.base import ToolRegistry

logger = logging.getLogger(__name__)

DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循：工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动：
"""

_THOUGHT_RE = re.compile(
    r"Thought\s*[:：]\s*(.*?)(?=\n\s*Action\s*[:：]|$)",
    re.DOTALL | re.IGNORECASE,
)
_ACTION_RE = re.compile(r"Action\s*[:：]\s*(.*)", re.DOTALL | re.IGNORECASE)
_ACTION_CALL_RE = re.compile(r"([A-Za-z_][\w.\-]*)\s*\[(.*)\]\s*$", re.DOTALL)


class ReActAgent(Agent):
    """推理与行动结合的 Agent。"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max(1, max_steps)
        self.current_history: list[str] = []
        self.prompt_template = custom_prompt or DEFAULT_REACT_PROMPT

    def run(self, input_text: str, **kwargs: Any) -> str:
        """运行 ReAct 循环，返回最终答案。"""
        self.current_history = []

        for step in range(1, self.max_steps + 1):
            logger.debug("%s 第 %d/%d 步", self.name, step, self.max_steps)
            prompt = self._build_prompt(input_text)
            response_text = self.llm.invoke(
                [{"role": "user", "content": prompt}], **kwargs
            )

            thought, action = self._parse_output(response_text)
            if thought:
                logger.debug("Thought: %s", thought)

            if not action:
                # 模型没有给出 Action，视为直接回答。
                final_answer = response_text.strip()
                self.record_exchange(input_text, final_answer)
                return final_answer

            if action.lower().startswith("finish"):
                final_answer = self._parse_action_input(action)
                self.record_exchange(input_text, final_answer)
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name:
                self.current_history.append(
                    f"Observation: 无法解析 Action '{action}'，请使用 tool[input] 或 Finish[答案] 格式。"
                )
                continue

            try:
                observation = self.tool_registry.execute_tool(tool_name, tool_input)
            except MungerMindError as exc:
                observation = f"❌ {exc}"

            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.record_exchange(input_text, final_answer)
        return final_answer

    def _build_prompt(self, question: str) -> str:
        try:
            return self.prompt_template.format(
                tools=self.tool_registry.get_tools_description(),
                question=question,
                history="\n".join(self.current_history),
            )
        except (KeyError, IndexError) as exc:
            raise AgentError(
                "ReAct 提示词模板格式错误，请转义字面量花括号或保留 "
                "{tools}/{question}/{history} 占位符。"
            ) from exc

    @staticmethod
    def _parse_output(text: str) -> Tuple[Optional[str], Optional[str]]:
        """从模型输出中提取 Thought 与 Action。"""
        thought_match = _THOUGHT_RE.search(text)
        action_match = _ACTION_RE.search(text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip().strip("`").strip() if action_match else None
        return thought, action

    @staticmethod
    def _parse_action(action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """解析 ``tool[input]``，失败返回 ``(None, None)``。"""
        match = _ACTION_CALL_RE.match(action_text.strip().strip("`").strip())
        if not match:
            return None, None
        return match.group(1), _strip_quotes(match.group(2).strip())

    @staticmethod
    def _parse_action_input(action_text: str) -> str:
        """解析 ``Finish[答案]`` / ``tool[输入]`` 中的方括号内容。"""
        match = _ACTION_CALL_RE.match(action_text.strip().strip("`").strip())
        return _strip_quotes(match.group(2).strip()) if match else action_text.strip()


def _strip_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text
