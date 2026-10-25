"""ReverseReActAgent 的离线测试：用脚本化假模型验证协议、校验与循环预算。

不需要网络与 API Key，直接运行::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import json
import logging
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import patch

from mungermind.agents.reverse_react import (
    FindingStatus,
    ReverseReActAgent,
    Strategy,
)
from mungermind.core.exceptions import ReverseParseError

# 测试刻意走到「丢弃证据 / 修复输出」等告警分支，静音以免污染断言输出。
logging.disable(logging.CRITICAL)


class ScriptedLLM:
    """按顺序返回预设文本的假模型，并记录每次调用的消息列表。"""

    provider = "fake"
    model = "scripted"

    #: 流式调用每次产出的字符数；取小值以便断言真的分块到达。
    chunk_size = 3

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses: List[str] = list(responses)
        self.calls: List[List[Dict[str, Any]]] = []
        self.stream_calls: List[List[Dict[str, Any]]] = []

    def _next(self) -> str:
        if not self._responses:
            raise AssertionError("模型调用次数超出预设，请检查循环预算")
        return self._responses.pop(0)

    def invoke(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append(messages)
        return self._next()

    def stream_invoke(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        self.stream_calls.append(messages)
        text = self._next()
        for start in range(0, len(text), self.chunk_size):
            yield text[start : start + self.chunk_size]


def analysis_json(
    *,
    strategies: Sequence[str] = ("A", "B", "C"),
    findings: Optional[Sequence[Dict[str, Any]]] = None,
    actions: Sequence[str] = ("列出 X 的前置条件",),
    refs: Sequence[str] = (),
    unresolved: Sequence[str] = ("X 的获取成本未知",),
    continue_recommended: bool = True,
    stop_reason: Optional[str] = None,
    position: str = "倾向方案甲",
    selection_reason: str = "目标清晰但缺少必要条件，同时存在待检验的判断。",
) -> str:
    if findings is None:
        findings = [
            {"strategy": "A", "statement": "必须先具备 X", "relevance": "决定第一步"},
            {"strategy": "B", "statement": "若 Y 成立则结论被推翻", "relevance": "影响结论"},
        ]
    return json.dumps(
        {
            "selected_strategies": list(strategies),
            "selection_reason": selection_reason,
            "findings": list(findings),
            "verification_questions": ["X 是否已经具备？"],
            "proposed_actions": list(actions),
            "evidence_refs": list(refs),
            "decision_impact": "可能改变第一步的顺序",
            "unresolved_issues": list(unresolved),
            "continue_recommended": continue_recommended,
            "stop_reason": stop_reason,
            "current_position": position,
        },
        ensure_ascii=False,
    )


def verification_json(
    *,
    updates: Optional[Sequence[Dict[str, Any]]] = None,
    position: str = "倾向方案甲，但需先验证 X",
    unresolved: Sequence[str] = ("X 的获取成本未知",),
    continue_recommended: bool = True,
    stop_reason: Optional[str] = None,
) -> str:
    if updates is None:
        updates = [{"index": 1, "status": "supported", "note": "已由推理确认"}]
    return json.dumps(
        {
            "finding_updates": list(updates),
            "updated_position": position,
            "decision_impact": "第一步改为先验证 X",
            "unresolved_issues": list(unresolved),
            "continue_recommended": continue_recommended,
            "stop_reason": stop_reason,
        },
        ensure_ascii=False,
    )


def two_cycle_script(answer: str = "最终答案") -> ScriptedLLM:
    """两轮默认闭环 + 一次汇总，共 5 次模型调用。"""
    return ScriptedLLM(
        [
            analysis_json(actions=("列出 X 的前置条件",)),
            verification_json(updates=[{"index": 1, "status": "supported"}]),
            analysis_json(actions=("估算 X 的获取成本",)),
            verification_json(
                updates=[{"index": 2, "status": "challenged"}],
                position="暂缓方案甲，先做验证",
            ),
            answer,
        ]
    )


def make_agent(llm: ScriptedLLM, **kwargs: Any) -> ReverseReActAgent:
    return ReverseReActAgent(name="ReverseReAct", llm=llm, **kwargs)


class ReturnValueTest(unittest.TestCase):
    def test_run_returns_synthesis_answer(self) -> None:
        llm = two_cycle_script("先验证 X，再决定是否采用方案甲。")
        agent = make_agent(llm)

        self.assertEqual(agent.run("我该不该选方案甲？"), "先验证 X，再决定是否采用方案甲。")

    def test_run_keeps_structured_result_for_audit(self) -> None:
        llm = two_cycle_script()
        agent = make_agent(llm)

        agent.run("我该不该选方案甲？")

        result = agent.last_result
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.question, "我该不该选方案甲？")
        self.assertEqual(len(result.cycles), 2)
        self.assertEqual(result.final_answer, "最终答案")
        self.assertFalse(result.externally_verified)
        self.assertEqual(agent.get_history()[-2].role, "user")


class ProtocolTest(unittest.TestCase):
    def test_cycle_result_carries_all_required_fields(self) -> None:
        llm = two_cycle_script()
        agent = make_agent(llm)

        result = agent.run_with_result("我该不该选方案甲？")

        cycle = result.cycles[0]
        self.assertEqual(cycle.cycle_number, 1)
        self.assertEqual(cycle.selected_strategies, [Strategy.A, Strategy.B, Strategy.C])
        self.assertTrue(cycle.selection_reason)
        self.assertEqual(cycle.findings[0].statement, "必须先具备 X")
        self.assertEqual(cycle.findings[0].status, FindingStatus.SUPPORTED)
        self.assertEqual(cycle.verification_questions, ["X 是否已经具备？"])
        self.assertEqual(cycle.proposed_actions, ["列出 X 的前置条件"])
        self.assertEqual(cycle.evidence_refs, [])
        self.assertTrue(cycle.decision_impact)
        self.assertEqual(cycle.unresolved_issues, ["X 的获取成本未知"])
        self.assertTrue(cycle.continue_recommended)
        self.assertIsNone(cycle.stop_reason)

    def test_provisional_position_updates_between_cycles(self) -> None:
        llm = two_cycle_script()
        agent = make_agent(llm)

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(result.cycles[0].provisional_position, "倾向方案甲，但需先验证 X")
        self.assertEqual(result.cycles[1].provisional_position, "暂缓方案甲，先做验证")

    def test_cycle_numbers_are_sequential(self) -> None:
        llm = two_cycle_script()
        agent = make_agent(llm)

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual([c.cycle_number for c in result.cycles], [1, 2])


class ValidationTest(unittest.TestCase):
    def test_invalid_strategy_name_is_rejected(self) -> None:
        llm = ScriptedLLM(
            [
                analysis_json(strategies=("A", "D")),
                analysis_json(strategies=("A", "D")),
            ]
        )
        agent = make_agent(llm)

        with self.assertRaises(ReverseParseError):
            agent.run_with_result("我该不该选方案甲？")

    def test_run_reports_parse_failure_instead_of_raising(self) -> None:
        llm = ScriptedLLM(["不是 JSON", "仍然不是 JSON"])
        agent = make_agent(llm)

        answer = agent.run("我该不该选方案甲？")

        self.assertIn("❌", answer)

    def test_malformed_json_is_repaired_once(self) -> None:
        llm = ScriptedLLM(
            [
                "先说两句废话，然后给出结果：\n```json\n"
                + analysis_json()
                + "\n```\n以上。",
                verification_json(),
                analysis_json(actions=("估算 X 的获取成本",)),
                verification_json(),
                "最终答案",
            ]
        )
        agent = make_agent(llm)

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(len(result.cycles), 2)
        self.assertEqual(len(llm.calls), 5)

    def test_unknown_evidence_refs_are_dropped(self) -> None:
        llm = ScriptedLLM(
            [
                analysis_json(refs=("e1", "幽灵来源")),
                verification_json(),
                analysis_json(actions=("估算 X 的获取成本",), refs=("e2",)),
                verification_json(),
                "最终答案",
            ]
        )
        agent = make_agent(llm, evidence_ids=("e1",))

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(result.cycles[0].evidence_refs, ["e1"])
        self.assertEqual(result.cycles[1].evidence_refs, [])
        self.assertEqual(sorted(result.dropped_evidence_refs), ["e2", "幽灵来源"])

    def test_action_budget_truncates_proposals(self) -> None:
        llm = ScriptedLLM(
            [
                analysis_json(actions=("a1", "a2", "a3", "a4")),
                verification_json(),
                analysis_json(actions=("b1", "b2", "b3", "b4")),
                verification_json(),
                "最终答案",
            ]
        )
        agent = make_agent(llm, max_actions_per_cycle=2)

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(result.cycles[0].proposed_actions, ["a1", "a2"])

    def test_finding_update_with_unknown_index_is_ignored(self) -> None:
        llm = ScriptedLLM(
            [
                analysis_json(),
                verification_json(
                    updates=[
                        {"index": 1, "status": "challenged"},
                        {"index": 99, "status": "supported"},
                    ]
                ),
                analysis_json(actions=("估算 X 的获取成本",)),
                verification_json(),
                "最终答案",
            ]
        )
        agent = make_agent(llm)

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(result.cycles[0].findings[0].status, FindingStatus.CHALLENGED)
        self.assertEqual(result.cycles[0].findings[1].status, FindingStatus.UNVERIFIED)


class PinnedStrategyTest(unittest.TestCase):
    def test_pinned_strategies_are_not_overridden_by_model(self) -> None:
        llm = ScriptedLLM(
            [
                analysis_json(strategies=("A", "B", "C")),
                verification_json(),
                analysis_json(strategies=("A", "B", "C"), actions=("估算 X",)),
                verification_json(),
                "最终答案",
            ]
        )
        agent = make_agent(llm, strategies=("B",))

        result = agent.run_with_result("我该不该选方案甲？")

        cycle = result.cycles[0]
        self.assertEqual(cycle.selected_strategies, [Strategy.B])
        self.assertEqual([f.strategy for f in cycle.findings], [Strategy.B])
        self.assertIn("用户", cycle.selection_reason)


class StopRuleTest(unittest.TestCase):
    def test_stops_when_model_recommends_stop(self) -> None:
        llm = ScriptedLLM(
            [
                analysis_json(continue_recommended=False, stop_reason="继续分析的预期价值不足"),
                verification_json(continue_recommended=False, stop_reason="继续分析的预期价值不足"),
                "最终答案",
            ]
        )
        agent = make_agent(llm)

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(len(result.cycles), 1)
        self.assertEqual(result.stop_reason, "继续分析的预期价值不足")
        self.assertEqual(len(llm.calls), 3)

    def test_stops_when_no_new_verification_action_appears(self) -> None:
        llm = ScriptedLLM(
            [
                analysis_json(actions=("列出 X 的前置条件",)),
                verification_json(),
                analysis_json(actions=("列出 X 的前置条件",)),
                verification_json(),
                "最终答案",
            ]
        )
        agent = make_agent(llm, max_cycles=5)

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(len(result.cycles), 2)
        self.assertIn("新验证行动", result.stop_reason)

    def test_hard_cap_limits_cycles(self) -> None:
        llm = ScriptedLLM(
            [
                analysis_json(actions=("a1",)),
                verification_json(),
                analysis_json(actions=("a2",)),
                verification_json(),
                "最终答案",
            ]
        )
        agent = make_agent(llm, default_cycles=1, max_cycles=2, allow_extension=True)

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(len(result.cycles), 2)
        self.assertIn("上限", result.stop_reason)
        self.assertEqual(len(llm.calls), 5)

    def test_default_run_stops_after_two_cycles_despite_open_issues(self) -> None:
        llm = two_cycle_script()
        agent = make_agent(llm)

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(len(result.cycles), 2)
        self.assertIn("默认", result.stop_reason)
        self.assertEqual(len(llm.calls), 5)

    def test_extension_runs_extra_cycle_when_enabled(self) -> None:
        llm = ScriptedLLM(
            [
                analysis_json(actions=("a1",)),
                verification_json(),
                analysis_json(actions=("a2",)),
                verification_json(),
                analysis_json(actions=("a3",)),
                verification_json(),
                "最终答案",
            ]
        )
        agent = make_agent(llm, default_cycles=2, max_cycles=3, allow_extension=True)

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(len(result.cycles), 3)
        self.assertIn("上限", result.stop_reason)

    def test_extension_requires_unresolved_issues(self) -> None:
        llm = ScriptedLLM(
            [
                analysis_json(actions=("a1",)),
                verification_json(unresolved=()),
                analysis_json(actions=("a2",), unresolved=()),
                verification_json(unresolved=()),
                "最终答案",
            ]
        )
        agent = make_agent(llm, default_cycles=2, max_cycles=4, allow_extension=True)

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(len(result.cycles), 2)
        self.assertIn("默认", result.stop_reason)

    def test_cancel_blocks_further_model_calls(self) -> None:
        llm = two_cycle_script()
        agent = make_agent(llm)
        agent.cancel()

        result = agent.run_with_result("我该不该选方案甲？")

        self.assertEqual(llm.calls, [])
        self.assertEqual(result.cycles, [])
        self.assertIn("取消", result.stop_reason)

    def test_default_cycles_is_two(self) -> None:
        llm = two_cycle_script()
        agent = make_agent(llm)

        self.assertEqual(agent.default_cycles, 2)


class StreamingTest(unittest.TestCase):
    """最终答案改为流式输出：分块到达、走 stream_invoke、结束后仍可审计。"""

    def test_run_stream_yields_answer_in_multiple_chunks(self) -> None:
        llm = two_cycle_script("先验证 X，再决定是否采用方案甲。")
        agent = make_agent(llm)

        chunks = list(agent.run_stream("我该不该选方案甲？"))

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), "先验证 X，再决定是否采用方案甲。")

    def test_synthesis_is_the_only_streamed_call(self) -> None:
        llm = two_cycle_script()
        agent = make_agent(llm)

        list(agent.run_stream("我该不该选方案甲？"))

        self.assertEqual(len(llm.stream_calls), 1)
        self.assertEqual(len(llm.calls), 4)  # 2 轮 ×（分析 + 复核）

    def test_run_stream_sets_last_result_and_history(self) -> None:
        llm = two_cycle_script("先验证 X。")
        agent = make_agent(llm)

        list(agent.run_stream("我该不该选方案甲？"))

        result = agent.last_result
        assert result is not None
        self.assertEqual(result.final_answer, "先验证 X。")
        self.assertEqual(len(result.cycles), 2)
        self.assertEqual(len(agent.get_history()), 2)

    def test_run_stream_reports_progress_per_cycle(self) -> None:
        events: List[str] = []
        agent = make_agent(two_cycle_script())

        list(agent.run_stream("我该不该选方案甲？", on_progress=events.append))

        self.assertEqual(len(events), 3)
        self.assertIn("第 1 轮", events[0])
        self.assertIn("第 2 轮", events[1])
        self.assertIn("汇总", events[2])


class ConfigurationTest(unittest.TestCase):
    def test_rejects_max_cycles_below_default(self) -> None:
        with self.assertRaises(ValueError):
            make_agent(two_cycle_script(), default_cycles=3, max_cycles=2)

    def test_rejects_unknown_prompt_key(self) -> None:
        with self.assertRaises(ValueError):
            make_agent(two_cycle_script(), prompts={"summarise": "..."})

    def test_accepts_strategy_string(self) -> None:
        agent = make_agent(two_cycle_script(), strategies="AC")

        self.assertEqual(agent.pinned_strategies, (Strategy.A, Strategy.C))


class CliTest(unittest.TestCase):
    def test_cli_prints_answer_for_question(self) -> None:
        from mungermind.__main__ import main

        llm = two_cycle_script("先验证 X。")
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = main(["我该不该选方案甲？"], llm_factory=lambda: llm)

        self.assertEqual(code, 0)
        self.assertIn("先验证 X。", out.getvalue())

    def test_cli_prints_audit_when_requested(self) -> None:
        from mungermind.__main__ import main

        llm = two_cycle_script("先验证 X。")
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = main(["--audit", "我该不该选方案甲？"], llm_factory=lambda: llm)

        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("第 1 轮", text)
        self.assertIn("目标逆向规划", text)


class InterruptingStdin(io.StringIO):
    """模拟用户在提问时按 Ctrl+C。"""

    def readline(self, *args: Any, **kwargs: Any) -> str:
        raise KeyboardInterrupt


class CliStreamingTest(unittest.TestCase):
    def test_cli_streams_the_answer_to_stdout(self) -> None:
        from mungermind.__main__ import main

        llm = two_cycle_script("先验证 X。")
        out = io.StringIO()
        with patch("sys.stdin", io.StringIO("")), redirect_stdout(out), redirect_stderr(
            io.StringIO()
        ):
            code = main(["我该不该选方案甲？"], llm_factory=lambda: llm)

        self.assertEqual(code, 0)
        self.assertEqual(len(llm.stream_calls), 1)
        self.assertIn("先验证 X。", out.getvalue())

    def test_cli_writes_progress_to_stderr(self) -> None:
        from mungermind.__main__ import main

        llm = two_cycle_script("先验证 X。")
        err = io.StringIO()
        with patch("sys.stdin", io.StringIO("")), redirect_stdout(
            io.StringIO()
        ), redirect_stderr(err):
            code = main(["我该不该选方案甲？"], llm_factory=lambda: llm)

        self.assertEqual(code, 0)
        self.assertIn("第 1 轮", err.getvalue())


class InteractiveTest(unittest.TestCase):
    """终端里不带参数启动：接收一个问题、显示它，然后正常退出。

    约定：stdout 只有答案本身，提示语与进度都走 stderr。
    """

    def _run(self, stdin: Any, llm: ScriptedLLM) -> tuple[int, str, str]:
        from mungermind.__main__ import main

        out, err = io.StringIO(), io.StringIO()
        with patch("sys.stdin", stdin), redirect_stdout(out), redirect_stderr(err):
            code = main([], llm_factory=lambda: llm)
        return code, out.getvalue(), err.getvalue()

    def test_receives_typed_question_and_displays_it(self) -> None:
        code, _, err = self._run(
            io.StringIO("我该不该选方案甲？\n"), two_cycle_script("先验证 X。")
        )

        self.assertEqual(code, 0)
        self.assertIn("收到问题：我该不该选方案甲？", err)

    def test_answers_the_received_question(self) -> None:
        code, out, _ = self._run(
            io.StringIO("我该不该选方案甲？\n"), two_cycle_script("先验证 X。")
        )

        self.assertEqual(code, 0)
        self.assertIn("先验证 X。", out)

    def test_empty_input_exits_normally_without_calling_model(self) -> None:
        llm = two_cycle_script()
        code, _, _ = self._run(io.StringIO("\n"), llm)

        self.assertEqual(code, 0)
        self.assertEqual(llm.calls, [])

    def test_eof_exits_normally(self) -> None:
        llm = two_cycle_script()
        code, _, _ = self._run(io.StringIO(""), llm)

        self.assertEqual(code, 0)
        self.assertEqual(llm.calls, [])

    def test_ctrl_c_exits_normally(self) -> None:
        llm = two_cycle_script()
        code, _, _ = self._run(InterruptingStdin(), llm)

        self.assertEqual(code, 0)
        self.assertEqual(llm.calls, [])

    def test_quit_word_exits_without_calling_model(self) -> None:
        llm = two_cycle_script()
        code, _, _ = self._run(io.StringIO("quit\n"), llm)

        self.assertEqual(code, 0)
        self.assertEqual(llm.calls, [])

    def test_ctrl_c_while_thinking_exits_normally(self) -> None:
        from mungermind.__main__ import main

        class InterruptingLLM:
            provider = "fake"

            def invoke(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
                raise KeyboardInterrupt

        out, err = io.StringIO(), io.StringIO()
        with patch("sys.stdin", io.StringIO("")), redirect_stdout(out), redirect_stderr(err):
            code = main(["我该不该选方案甲？"], llm_factory=InterruptingLLM)

        self.assertEqual(code, 0)
        self.assertIn("已取消", err.getvalue())

    def test_question_argument_still_skips_the_prompt(self) -> None:
        from mungermind.__main__ import main

        llm = two_cycle_script("先验证 X。")
        out, err = io.StringIO(), io.StringIO()
        with patch("sys.stdin", io.StringIO("")), redirect_stdout(out), redirect_stderr(err):
            code = main(["我该不该选方案甲？"], llm_factory=lambda: llm)

        self.assertEqual(code, 0)
        self.assertNotIn("收到问题：", out.getvalue() + err.getvalue())


if __name__ == "__main__":
    unittest.main()
