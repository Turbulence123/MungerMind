"""决策报告（规格 2.3）：固定八段，未生成的段落明确标注。

报告只装配当前模块真正产出的内容；没有能力的段落留空，由 :meth:`DecisionReport.to_dict`
与 :meth:`render_markdown` 统一填成「未生成」，不编造证据、专家或方案。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from pydantic import BaseModel, ConfigDict

NOT_GENERATED = "（未生成：当前版本尚未接入该能力）"

#: 八段的字段名与中文标题，顺序即报告顺序（2.3）。
SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("problem", "问题、目标与约束"),
    ("criteria_and_gaps", "决策标准与关键缺失信息"),
    ("options", "候选方案与取舍"),
    ("experts_and_models", "学科专家与思维模型"),
    ("evidence", "事实、来源与待验证事项"),
    ("counter_arguments", "反方观点、失败条件与分歧"),
    ("recommendation", "推荐方案与成立条件"),
    ("next_steps", "下一步行动与待决事项"),
)


class DecisionReport(BaseModel):
    """一次任务运行的用户可见报告。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    question: str
    turn: int = 1
    mode: str
    problem: str = ""
    criteria_and_gaps: str = ""
    options: str = ""
    experts_and_models: str = ""
    evidence: str = ""
    counter_arguments: str = ""
    recommendation: str = ""
    next_steps: str = ""
    stop_reason: str = ""
    externally_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """序列化；空段统一填「未生成」，避免下游把空字符串当成结论。"""
        data = self.model_dump(mode="json")
        for name, _ in SECTIONS:
            if not str(data.get(name, "")).strip():
                data[name] = NOT_GENERATED
        return data

    def render_markdown(self) -> str:
        """渲染为可直接阅读的 Markdown 报告。"""
        data = self.to_dict()
        lines = [
            f"# 决策报告 · {self.question}",
            "",
            f"- 任务：{self.task_id}（第 {self.turn} 轮）",
            f"- 模式：{self.mode}",
            f"- 外部验证：{'有' if self.externally_verified else '无'}",
            f"- 停止原因：{self.stop_reason or '未记录'}",
            "",
        ]
        for index, (name, title) in enumerate(SECTIONS, start=1):
            lines.extend([f"## {index}. {title}", "", str(data[name]), ""])
        return "\n".join(lines).rstrip() + "\n"
