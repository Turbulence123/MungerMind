"""把一次模式运行装配成规格 2.3 的八段决策报告。"""

from __future__ import annotations

from ..core.report import DecisionReport
from ..core.task import Task
from .mode import ModeResult


def build_report(
    task: Task, result: ModeResult, question: str, *, turn: int = 1
) -> DecisionReport:
    """把通用模式结果映射到八段报告；没有依据的段落保持为空。"""
    open_findings = [finding for finding in result.findings if finding.status != "supported"]

    evidence_lines = ["外部验证：无（本次未调用外部证据工具）"]
    if result.evidence_refs:
        evidence_lines.append("引用的证据：" + "、".join(result.evidence_refs))
    if open_findings:
        evidence_lines.append("尚未核实的主张：")
        evidence_lines.extend(f"- {finding.statement}" for finding in open_findings)
    elif not result.findings:
        evidence_lines.append("本模式没有产出结构化发现。")

    if not result.findings:
        counter_arguments = ""
    elif open_findings:
        counter_arguments = "\n".join(
            f"- [{finding.status}] {finding.statement}" for finding in open_findings
        )
    else:
        counter_arguments = "本次分析没有留下未支持或未核实的主张。"

    if result.next_actions:
        next_steps = "\n".join(f"- {action}" for action in result.next_actions)
        if result.unresolved_issues:
            next_steps += "\n待决：" + "；".join(result.unresolved_issues)
    elif result.unresolved_issues:
        next_steps = "待决：" + "；".join(result.unresolved_issues)
    else:
        next_steps = "本次分析没有提出新的验证行动。"

    return DecisionReport(
        task_id=task.id,
        question=question,
        turn=turn,
        mode=result.mode,
        problem=(
            f"问题：{question}\n"
            "目标与约束：未单独整理（当前版本直接以用户问题为分析目标）。"
        ),
        evidence="\n".join(evidence_lines),
        counter_arguments=counter_arguments,
        recommendation=result.answer,
        next_steps=next_steps,
        stop_reason=result.stop_reason,
        externally_verified=result.externally_verified,
    )
