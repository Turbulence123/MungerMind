"""Command-line entry point, invoked as ``python -m mungermind``.

最小闭环：把问题交给 ReverseReAct 逆向推理引擎，打印返回的答案。

    python -m mungermind                       # 交互输入：敲一个问题，回车
    python -m mungermind "要不要接受这份工作？"   # 直接把问题写在参数里
    python -m mungermind --strategies AB --cycles 1 "要不要接受这份工作？"
    python -m mungermind --audit "要不要接受这份工作？"
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, List, Optional, Sequence

from .agents.reverse_react import ReverseReActAgent, ReverseReActResult
from .core.exceptions import MungerMindError
from .core.llm import LLM

LLMFactory = Callable[[], Any]

INTERACTIVE_BANNER = (
    "MungerMind · ReverseReAct 逆向推理\n"
    "输入一个问题后回车；直接回车、输入 exit 或按 Ctrl+C 退出。"
)
INTERACTIVE_PROMPT = "问题> "
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
        "--audit",
        action="store_true",
        help="额外打印每轮的结构化结果",
    )
    return parser


def _read_question() -> Optional[str]:
    """交互式读取一个问题；空输入、退出词、EOF 与 Ctrl+C 都返回 ``None``。

    提示语一律走 stderr，保证 stdout 只有答案，可直接重定向到文件。
    """
    print(INTERACTIVE_BANNER, file=sys.stderr, flush=True)
    print(INTERACTIVE_PROMPT, end="", file=sys.stderr, flush=True)
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
    """解析参数、运行逆向推理并打印答案；返回进程退出码。"""
    args = build_parser().parse_args(argv)
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
    try:
        agent = ReverseReActAgent(name="ReverseReAct", llm=llm_factory(), **agent_kwargs)
        for chunk in agent.run_stream(question, on_progress=_progress):
            sys.stdout.write(chunk)
            sys.stdout.flush()
    except KeyboardInterrupt:
        # 用户在等待模型时按 Ctrl+C：标记取消并正常退出，不抛栈。
        if agent is not None:
            agent.cancel()
        print("\n已取消，退出。", file=sys.stderr)
        return 0
    except (MungerMindError, ValueError) as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        return 1
    print()

    if args.audit and agent.last_result is not None:
        print()
        print(_format_audit(agent.last_result))
    return 0


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
