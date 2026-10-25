"""ReflectionAgent：生成 → 反思 → 改进 的迭代范式。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..core.agent import Agent
from ..core.config import Config
from ..core.llm import LLM

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS: Dict[str, str] = {
    "initial": """
请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
""",
}

_DONE_MARKERS = ("无需改进", "no need for improvement", "no improvement needed")


class ReflectionMemory:
    """记录执行与反思轨迹的短期记忆。"""

    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []

    def add_record(self, record_type: str, content: str) -> None:
        self.records.append({"type": record_type, "content": content})

    def get_last_execution(self) -> str:
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return ""

    def get_trajectory(self) -> str:
        parts = []
        for record in self.records:
            if record["type"] == "execution":
                parts.append(f"--- 上一轮尝试 ---\n{record['content']}")
            elif record["type"] == "reflection":
                parts.append(f"--- 评审反馈 ---\n{record['content']}")
        return "\n\n".join(parts)


class ReflectionAgent(Agent):
    """通用于文本生成、分析与创作场景的反思型 Agent。"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max(0, max_iterations)
        self.prompts = dict(DEFAULT_PROMPTS)
        if custom_prompts:
            unknown = set(custom_prompts) - set(DEFAULT_PROMPTS)
            if unknown:
                raise ValueError(
                    f"未知的提示词键: {', '.join(sorted(unknown))}；"
                    f"可选值: {', '.join(DEFAULT_PROMPTS)}"
                )
            self.prompts.update(custom_prompts)
        self.memory = ReflectionMemory()

    def run(self, input_text: str, **kwargs: Any) -> str:
        """执行初始尝试，并最多迭代 ``max_iterations`` 轮反思与优化。"""
        self.memory = ReflectionMemory()

        initial = self._ask(self.prompts["initial"].format(task=input_text), **kwargs)
        self.memory.add_record("execution", initial)

        for index in range(1, self.max_iterations + 1):
            last_attempt = self.memory.get_last_execution()
            feedback = self._ask(
                self.prompts["reflect"].format(
                    task=input_text, content=last_attempt
                ),
                **kwargs,
            )
            self.memory.add_record("reflection", feedback)

            if self._is_satisfied(feedback):
                logger.debug("第 %d 轮反思认为无需改进，提前结束。", index)
                break

            refined = self._ask(
                self.prompts["refine"].format(
                    task=input_text,
                    last_attempt=last_attempt,
                    feedback=feedback,
                ),
                **kwargs,
            )
            self.memory.add_record("execution", refined)

        final_answer = self.memory.get_last_execution()
        self.record_exchange(input_text, final_answer)
        return final_answer

    def _ask(self, prompt: str, **kwargs: Any) -> str:
        return self.llm.invoke([{"role": "user", "content": prompt}], **kwargs)

    @staticmethod
    def _is_satisfied(feedback: str) -> bool:
        lowered = feedback.lower()
        return any(marker in lowered for marker in _DONE_MARKERS)
