"""把现有 Agent 适配成 :class:`ReasoningMode`。

本轮只有 ReverseReAct 是结构化模式；其余范式等报告需要时再适配。
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Optional

from ..agents.reverse_react import ReverseReActAgent, ReverseReActResult
from ..core.task import Task
from .mode import FindingRef, ModeResult, ReasoningMode


class ReverseReActMode(ReasoningMode):
    """ReverseReAct 的结构化结果 → :class:`ModeResult`。"""

    name = "reverse_react"

    def __init__(self, agent: ReverseReActAgent) -> None:
        self.agent = agent
        self.last_result: Optional[ModeResult] = None

    def run(self, task: Task, question: str) -> ModeResult:
        result = self.agent.run_with_result(question, context=task.context_text())
        self.last_result = to_mode_result(result)
        return self.last_result

    def run_stream(
        self,
        task: Task,
        question: str,
        *,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        self.last_result = None
        for chunk in self.agent.run_stream(
            question, on_progress=on_progress, context=task.context_text()
        ):
            yield chunk
        if self.agent.last_result is not None:
            self.last_result = to_mode_result(self.agent.last_result)


def to_mode_result(result: ReverseReActResult) -> ModeResult:
    """无损转换：通用字段摊平，完整轮次保留在 ``details``。"""
    cycles = list(result.cycles)
    last = cycles[-1] if cycles else None
    findings = [
        FindingRef(
            statement=finding.statement,
            status=finding.status.value,
            source=finding.strategy.value,
        )
        for cycle in cycles
        for finding in cycle.findings
    ]
    evidence_refs: List[str] = []
    for cycle in cycles:
        for ref in cycle.evidence_refs:
            if ref not in evidence_refs:
                evidence_refs.append(ref)
    return ModeResult(
        mode=ReverseReActMode.name,
        answer=result.final_answer,
        stop_reason=result.stop_reason,
        findings=findings,
        evidence_refs=evidence_refs,
        unresolved_issues=list(last.unresolved_issues) if last else [],
        next_actions=list(last.proposed_actions) if last else [],
        externally_verified=result.externally_verified,
        details={"cycles": [cycle.model_dump(mode="json") for cycle in cycles]},
    )
