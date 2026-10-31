"""Task / ModeResult / DecisionReport / CLI 多轮的离线测试。

只覆盖契约的核心：状态机、模式结果映射、八段报告、任务内多轮与落盘。
新模块在用例内部导入，让每个任务能独立红-绿。
"""

from __future__ import annotations

import io
import json
import logging
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Sequence
from unittest.mock import patch

logging.disable(logging.CRITICAL)

SECTION_NAMES = (
    "problem",
    "criteria_and_gaps",
    "options",
    "experts_and_models",
    "evidence",
    "counter_arguments",
    "recommendation",
    "next_steps",
)


class ScriptedLLM:
    """按顺序返回预设文本的假模型，记录每次调用。"""

    provider = "scripted"
    model = "scripted"

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: List[List[Dict[str, Any]]] = []
        self.stream_calls: List[List[Dict[str, Any]]] = []

    def _next(self) -> str:
        if not self._responses:
            raise AssertionError("模型调用次数超出预设")
        return self._responses.pop(0)

    def invoke(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append(messages)
        return self._next()

    def stream_invoke(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        self.stream_calls.append(messages)
        yield self._next()


def analysis_json(actions: Sequence[str]) -> str:
    return json.dumps(
        {
            "selected_strategies": ["A"],
            "selection_reason": "目标清晰但缺少必要条件",
            "findings": [
                {
                    "strategy": "A",
                    "statement": "需要先验证真实需求",
                    "relevance": "决定第一步",
                }
            ],
            "verification_questions": ["需求是否真实？"],
            "proposed_actions": list(actions),
            "evidence_refs": [],
            "decision_impact": "先验证再决定",
            "unresolved_issues": ["成本未知"],
            "continue_recommended": True,
            "stop_reason": None,
            "current_position": "倾向先验证",
        },
        ensure_ascii=False,
    )


def verification_json(position: str = "先验证再决定") -> str:
    return json.dumps(
        {
            "finding_updates": [{"index": 1, "status": "supported", "note": "推理确认"}],
            "updated_position": position,
            "decision_impact": "先验证再决定",
            "unresolved_issues": ["成本未知"],
            "continue_recommended": True,
            "stop_reason": None,
        },
        ensure_ascii=False,
    )


def run_script(answer: str) -> List[str]:
    """一次两轮闭环 + 汇总，共 5 次模型调用。"""
    return [
        analysis_json(("列出前置条件",)),
        verification_json(),
        analysis_json(("估算成本",)),
        verification_json(),
        answer,
    ]


class TaskStateTest(unittest.TestCase):
    def test_task_rejects_illegal_transitions(self) -> None:
        from mungermind.core.exceptions import TaskStateError
        from mungermind.core.task import Task, TaskState

        task = Task("要不要换工作？")

        with self.assertRaises(TaskStateError):
            task.finish()

        task.start()
        task.wait_for_user()
        task.start()
        task.cancel()
        self.assertEqual(task.state, TaskState.CANCELLED)

        with self.assertRaises(TaskStateError):
            task.finish()


class ModeResultTest(unittest.TestCase):
    def test_reverse_react_result_maps_to_mode_result(self) -> None:
        from mungermind.agents.reverse_react import ReverseReActAgent
        from mungermind.core.task import Task
        from mungermind.reasoning.adapters import ReverseReActMode

        llm = ScriptedLLM(run_script("先验证真实需求。"))
        agent = ReverseReActAgent(name="ReverseReAct", llm=llm)
        mode = ReverseReActMode(agent)
        task = Task("要不要换工作？")
        task.start()

        result = mode.run(task, "要不要换工作？")

        self.assertEqual(result.mode, "reverse_react")
        self.assertEqual(result.answer, "先验证真实需求。")
        self.assertTrue(result.stop_reason)
        self.assertEqual(
            [finding.status for finding in result.findings],
            ["supported", "supported"],
        )
        self.assertEqual(result.next_actions, ["估算成本"])
        self.assertFalse(result.externally_verified)
        self.assertIn("cycles", result.details)


class ReportTest(unittest.TestCase):
    def test_report_marks_sections_it_cannot_generate(self) -> None:
        from mungermind.core.report import NOT_GENERATED
        from mungermind.core.task import Task
        from mungermind.reasoning.mode import FindingRef, ModeResult
        from mungermind.reasoning.report import build_report

        task = Task("要不要换工作？")
        result = ModeResult(
            mode="reverse_react",
            answer="先验证真实需求。",
            stop_reason="默认 2 轮逆向分析已完成",
            findings=[FindingRef(statement="需求未验证", status="unverified", source="A")],
            unresolved_issues=["成本未知"],
            next_actions=["做一次访谈"],
        )

        report = build_report(task, result, "要不要换工作？", turn=1)
        data = report.to_dict()

        for name in SECTION_NAMES:
            self.assertIn(name, data)
        self.assertEqual(data["options"], NOT_GENERATED)
        self.assertEqual(data["recommendation"], "先验证真实需求。")
        self.assertIn("需求未验证", data["evidence"])
        self.assertIn("做一次访谈", data["next_steps"])
        json.dumps(data, ensure_ascii=False)
        self.assertIn("未生成", report.render_markdown())


class CliMultiTurnTest(unittest.TestCase):
    def test_two_turns_share_context_and_report_is_written(self) -> None:
        from mungermind.__main__ import main

        llm = ScriptedLLM(
            run_script("第一轮：先验证 X。") + run_script("第二轮：成本可控。")
        )
        report_path = Path(__file__).resolve().parents[1] / "_report_tmp.json"
        try:
            out, err = io.StringIO(), io.StringIO()
            with patch("sys.stdin", io.StringIO("第一问？\n第二问？\n")), redirect_stdout(
                out
            ), redirect_stderr(err):
                code = main(["--report", str(report_path)], llm_factory=lambda: llm)
            data = json.loads(report_path.read_text(encoding="utf-8"))
        finally:
            report_path.unlink(missing_ok=True)

        self.assertEqual(code, 0)
        second_analysis = llm.calls[4][-1]["content"]
        self.assertIn("上一轮问题", second_analysis)
        self.assertIn("第一轮：先验证 X。", second_analysis)
        self.assertEqual(data["recommendation"], "第二轮：成本可控。")
        self.assertIn("收到问题：第二问？", err.getvalue())


if __name__ == "__main__":
    unittest.main()
