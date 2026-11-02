"""Command-line entry point, invoked as ``python -m mungermind``.

最小闭环：在一个任务里用 ReverseReAct 逆向推理回答问题，并支持继续追问。

    python -m mungermind                         # 交互：可连续追问，空行退出
    python -m mungermind "要不要接受这份工作？"     # 单次运行
    python -m mungermind --report report.json "要不要接受这份工作？"
    python -m mungermind --audit "要不要接受这份工作？"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from .agents.reverse_react import ReverseReActAgent, ReverseReActResult
from .core.exceptions import MungerMindError
from .core.llm import LLM
from .core.report import DecisionReport
from .core.task import Task, TaskState
from .reasoning.adapters import ReverseReActMode
from .reasoning.report import build_report

LLMFactory = Callable[[], Any]

INTERACTIVE_BANNER = (
    "MungerMind · ReverseReAct 逆向推理\n"
    "输入问题后回车；同一任务内可以继续追问，直接回车、输入 exit 或按 Ctrl+C 退出。"
)
INTERACTIVE_PROMPT = "问题> "
CONTINUE_PROMPT = "继续> "
EXIT_WORDS = frozenset({"exit", "quit", "q", "退出"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mungermind",
        description="用 ReverseReAct 逆向推理引擎回答一个问题。",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="要分析的问题；省略时进入交互输入",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        metavar="ABC",
        help="启用的子策略组合：A 目标逆向规划、B 逆向假设验证、C 失败逆推分析（不指定时由模型按任务选择）",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="默认循环轮次（默认 2 轮）",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="硬性循环上限（默认 3 轮）",
    )
    parser.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="把最近一轮写成报告：.json 后缀写 JSON，其他后缀写 Markdown",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="额外打印每轮的结构化结果",
    )
    return parser


def _read_question(first: bool = True) -> Optional[str]:
    """交互式读取一个问题；空输入、退出词、EOF 与 Ctrl+C 都返回 ``None``。

    提示语一律走 stderr，保证 stdout 只有答案，可直接重定向到文件。
    """
    if first:
        print(INTERACTIVE_BANNER, file=sys.stderr, flush=True)
    print(INTERACTIVE_PROMPT if first else CONTINUE_PROMPT, end="", file=sys.stderr, flush=True)
    try:
        raw = input()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return None

    question = raw.strip()
    if not question or question.lower() in EXIT_WORDS:
        return None

    print(f"收到问题：{question}", file=sys.stderr, flush=True)
    return question


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    llm_factory: LLMFactory = LLM,
) -> int:
    """解析参数、在任务里运行逆向推理并打印答案；返回进程退出码。"""
    args = build_parser().parse_args(argv)
    interactive = not args.question
    question = " ".join(args.question).strip()
    if not question:
        question = _read_question()
        if question is None:
            print("未收到问题，已退出。", file=sys.stderr)
            return 0

    agent_kwargs: dict[str, Any] = {}
    if args.strategies is not None:
        agent_kwargs["strategies"] = args.strategies
    if args.cycles is not None:
        agent_kwargs["default_cycles"] = args.cycles
    if args.max_cycles is not None:
        agent_kwargs["max_cycles"] = args.max_cycles

    agent: Optional[ReverseReActAgent] = None
    mode: Optional[ReverseReActMode] = None
    task = Task(question)
    try:
        agent = ReverseReActAgent(name="ReverseReAct", llm=llm_factory(), **agent_kwargs)
        mode = ReverseReActMode(agent)

        while question is not None:
            task.start()
            for chunk in mode.run_stream(task, question, on_progress=_progress):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print()

            result = mode.last_result
            if result is None:
                if task.state is TaskState.RUNNING:
                    task.fail("流式输出没有产生完整结果")
                print("未取得完整结果，任务结束。", file=sys.stderr)
                return 1

            task.add_turn(question, result.answer)
            if args.report:
                _write_report(
                    args.report,
                    build_report(task, result, question, turn=len(task.turns)),
                )
            if args.audit and agent.last_result is not None:
                print()
                print(_format_audit(agent.last_result))

            if not interactive:
                task.finish()
                break

            task.wait_for_user()
            question = _read_question(first=False)

        if task.state is TaskState.WAITING_FOR_USER:
            task.finish()
    except KeyboardInterrupt:
        # 用户在等待模型时按 Ctrl+C：标记取消并正常退出，不抛栈。
        if agent is not None:
            agent.cancel()
        if task.state in (TaskState.RUNNING, TaskState.WAITING_FOR_USER):
            task.cancel()
        print("\n已取消，退出。", file=sys.stderr)
        return 0
    except (MungerMindError, ValueError, OSError) as exc:
        if task.state is TaskState.RUNNING:
            task.fail(str(exc))
        print(f"\n❌ {exc}", file=sys.stderr)
        return 1
    return 0


def _write_report(path: str, report: DecisionReport) -> None:
    """按后缀写 JSON 或 Markdown；写失败交给调用方处理。"""
    target = Path(path)
    if target.suffix.lower() == ".json":
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
    else:
        text = report.render_markdown()
    target.write_text(text, encoding="utf-8")
    print(f"报告已写入：{target}", file=sys.stderr)


def _progress(message: str) -> None:
    """阶段提示写到 stderr，保证 stdout 只有答案本身、可直接重定向。"""
    print(message, file=sys.stderr, flush=True)


def _format_audit(result: ReverseReActResult) -> str:
    """渲染审计视图：策略、发现、决策影响与停止原因（3.3）。"""
    lines: List[str] = ["— 审计视图 —"]
    for cycle in result.cycles:
        strategies = "、".join(f"{s.value} {s.label}" for s in cycle.selected_strategies)
        lines.append(f"第 {cycle.cycle_number} 轮 · 子策略：{strategies}")
        lines.append(f"  选择原因：{cycle.selection_reason}")
        for finding in cycle.findings:
            lines.append(
                f"  - [{finding.strategy.value}] {finding.statement}"
                f"（{finding.status.value}）"
            )
        if cycle.verification_questions:
            lines.append("  待验证：" + "；".join(cycle.verification_questions))
        if cycle.proposed_actions:
            lines.append("  建议行动：" + "；".join(cycle.proposed_actions))
        lines.append(f"  决策影响：{cycle.decision_impact}")
        if cycle.unresolved_issues:
            lines.append("  未解决：" + "；".join(cycle.unresolved_issues))
    lines.append(f"停止原因：{result.stop_reason}")
    if result.dropped_evidence_refs:
        lines.append("未采信的证据引用：" + "、".join(result.dropped_evidence_refs))
    if not result.externally_verified:
        lines.append("外部验证：无（本次未调用任何外部工具）")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
