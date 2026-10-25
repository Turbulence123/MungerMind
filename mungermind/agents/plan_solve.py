"""PlanAndSolveAgent：先规划、再逐步执行。"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any, Dict, Optional

from ..core.agent import Agent
from ..core.config import Config
from ..core.exceptions import MungerMindError, PlanParseError
from ..core.llm import LLM

logger = logging.getLogger(__name__)

DEFAULT_PLANNER_PROMPT = """你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划：
```python
["步骤1", "步骤2", "步骤3"]
```
"""

DEFAULT_EXECUTOR_PROMPT = """你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答：
"""

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class Planner:
    """把问题分解为步骤列表。"""

    def __init__(self, llm: LLM, prompt_template: str = DEFAULT_PLANNER_PROMPT) -> None:
        self.llm = llm
        self.prompt_template = prompt_template

    def plan(self, question: str, **kwargs: Any) -> list[str]:
        prompt = self.prompt_template.format(question=question)
        response = self.llm.invoke([{"role": "user", "content": prompt}], **kwargs)
        return parse_plan(response)


class Executor:
    """按计划逐步执行并汇总结果。"""

    def __init__(self, llm: LLM, prompt_template: str = DEFAULT_EXECUTOR_PROMPT) -> None:
        self.llm = llm
        self.prompt_template = prompt_template

    def execute(self, question: str, plan: list[str], **kwargs: Any) -> str:
        history = ""
        final_answer = ""

        for index, step in enumerate(plan, start=1):
            logger.debug("执行步骤 %d/%d: %s", index, len(plan), step)
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history or "无",
                current_step=step,
            )
            result = self.llm.invoke(
                [{"role": "user", "content": prompt}], **kwargs
            )
            history += f"步骤 {index}: {step}\n结果: {result}\n\n"
            final_answer = result

        return final_answer


class PlanAndSolveAgent(Agent):
    """Plan-and-Solve 范式 Agent。"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        prompts = {
            "planner": DEFAULT_PLANNER_PROMPT,
            "executor": DEFAULT_EXECUTOR_PROMPT,
        }
        if custom_prompts:
            unknown = set(custom_prompts) - set(prompts)
            if unknown:
                raise ValueError(
                    f"未知的提示词键: {', '.join(sorted(unknown))}；"
                    f"可选值: {', '.join(prompts)}"
                )
            prompts.update(custom_prompts)
        self.planner = Planner(llm, prompts["planner"])
        self.executor = Executor(llm, prompts["executor"])

    def run(self, input_text: str, **kwargs: Any) -> str:
        """生成计划并执行，解析失败时返回错误说明而不是抛出异常。"""
        try:
            plan = self.planner.plan(input_text, **kwargs)
        except MungerMindError as exc:
            final_answer = f"❌ 无法生成有效的行动计划: {exc}"
            self.record_exchange(input_text, final_answer)
            return final_answer

        final_answer = self.executor.execute(input_text, plan, **kwargs)
        self.record_exchange(input_text, final_answer)
        return final_answer


def parse_plan(text: str) -> list[str]:
    """从模型输出中解析步骤列表。

    支持 ```python ...``` 代码块、裸列表以及列表前后夹杂说明文字的情况。
    """
    candidate = text.strip()
    fence = _CODE_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    start, end = candidate.find("["), candidate.rfind("]")
    if start == -1 or end <= start:
        raise PlanParseError(f"未找到 Python 列表：{text.strip()[:200]}")

    try:
        plan = ast.literal_eval(candidate[start : end + 1])
    except (ValueError, SyntaxError) as exc:
        raise PlanParseError(f"计划不是合法的 Python 列表：{exc}") from exc

    if not isinstance(plan, (list, tuple)):
        raise PlanParseError("计划必须是一个列表")

    steps = [str(step).strip() for step in plan if str(step).strip()]
    if not steps:
        raise PlanParseError("计划为空")
    return steps
