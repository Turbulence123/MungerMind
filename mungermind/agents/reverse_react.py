"""ReverseReActAgent：MungerMind 的逆向推理引擎（产品规格第 5 章）。

定位（5.1）
    独立的逆向推理引擎，拥有自己的子策略接口、模式选择、验证任务生成、循环
    控制和结果协议。它由主 Agent 调用，不取代主 Agent。

三种子策略（5.2）
    A 目标逆向规划：从目标倒推必要条件、前置依赖与验证点。
    B 逆向假设验证：寻找可能推翻当前判断的证据与区分性观察。
    C 失败逆推分析：假设方案已经失败，回溯原因、早期信号与预防措施。

一次调用可以选择一种或多种子策略；调用方通过 ``strategies=`` 明确指定时，
模型的选择不会静默覆盖它（3.1）。

循环（5.4）
    默认最多两轮（验收清单「ReverseReAct 默认最多执行两轮完整循环」）。每轮 =
    逆向分析 → 生成验证行动 → 观察并更新判断的一次完整闭环。只有显式开启
    ``allow_extension``，且「模型建议继续 + 仍有未解决问题 + 出现新的验证行动」
    同时成立时才会追加循环，并永远受确定性硬上限 ``max_cycles`` 约束。

结构（5.5）
    每轮产出 :class:`ReverseCycleResult`，由 Pydantic 校验；无效模式名、无效
    证据引用与超预算行动在进入执行前被拒绝或剔除。

本版本只实现「发送问题 → 逆向思考 → 返回答案」的最小闭环：不接入工具、
权限网关与模式调度器。没有外部工具可用时，结果以 ``externally_verified``
与最终答案中的说明明确标记「未做外部验证」（5.3 第 6 步），绝不伪造引用。
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.agent import Agent
from ..core.config import Config
from ..core.exceptions import ConfigurationError, MungerMindError, ReverseParseError
from ..core.llm import LLM

logger = logging.getLogger(__name__)

# 5.4：默认循环两轮；硬性上限必须由确定性配置设置，模型不得自行扩展。
# 规格尚未确定正式硬上限，此处为原型期临时值，不得视为已确认的产品决策。
DEFAULT_CYCLE_COUNT = 2
DEFAULT_HARD_CYCLE_LIMIT = 3
DEFAULT_MAX_ACTIONS = 3

STOP_CANCELLED = "用户取消了任务"
STOP_HARD_LIMIT = "达到硬性循环上限 {limit} 轮"
STOP_DEFAULT_BUDGET = "默认 {cycles} 轮逆向分析已完成，且未满足追加循环的条件"
STOP_NO_ACTION = "本轮没有提出可执行的验证行动"
STOP_NO_NEW_ACTION = "没有出现有意义的新验证行动"
STOP_MODEL_ADVISED = "模型判断继续逆向分析的预期价值不足"
CANCELLED_ANSWER = "任务已取消，未开始逆向分析。"
FALLBACK_ANSWER = "逆向分析未形成可用的结论。"

# 流式运行时报告给调用方的阶段提示（对应 3.3 的简洁状态视图）。
PROGRESS_CYCLE = "第 {cycle} 轮 · 逆向分析与复核…"
PROGRESS_SYNTHESIS = "汇总答案…"


class Strategy(str, Enum):
    """5.2 节的三种子策略；A / B / C 是规格里的代号。"""

    A = "A"
    B = "B"
    C = "C"

    @property
    def label(self) -> str:
        return _STRATEGY_LABELS[self]


_STRATEGY_LABELS: Dict[Strategy, str] = {
    Strategy.A: "目标逆向规划",
    Strategy.B: "逆向假设验证",
    Strategy.C: "失败逆推分析",
}
_STRATEGY_ORDER: Tuple[Strategy, ...] = (Strategy.A, Strategy.B, Strategy.C)


class FindingStatus(str, Enum):
    """单条逆向发现在复核后的状态。"""

    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    CHALLENGED = "challenged"


class _Cancelled(Exception):
    """内部信号：任务已被取消，不得再发起模型调用（10.3）。"""


# --------------------------------------------------------------------------- #
# 5.5 结构化结果
# --------------------------------------------------------------------------- #


class ReverseFinding(BaseModel):
    """一条逆向发现及其与当前决策的相关性。"""

    model_config = ConfigDict(extra="forbid")

    strategy: Strategy
    statement: str = Field(min_length=1)
    relevance: str = ""
    status: FindingStatus = FindingStatus.UNVERIFIED
    note: str = ""


class ReverseCycleResult(BaseModel):
    """5.5 节要求的一轮结构化结果。"""

    model_config = ConfigDict(extra="forbid")

    cycle_number: int = Field(ge=1)
    selected_strategies: List[Strategy] = Field(min_length=1)
    selection_reason: str
    findings: List[ReverseFinding] = Field(default_factory=list)
    verification_questions: List[str] = Field(default_factory=list)
    proposed_actions: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    decision_impact: str
    unresolved_issues: List[str] = Field(default_factory=list)
    continue_recommended: bool
    stop_reason: Optional[str] = None

    # 规格之外的审计字段：逆向分析把「当前立场」推到哪一步。
    provisional_position: str = ""


class ReverseReActResult(BaseModel):
    """一次 ReverseReAct 调用的完整结果。"""

    model_config = ConfigDict(extra="forbid")

    question: str
    cycles: List[ReverseCycleResult] = Field(default_factory=list)
    final_answer: str
    stop_reason: str
    # 本引擎不调用外部工具，因此恒为 False；保留字段，接入工具后改为真实结果。
    externally_verified: bool = False
    dropped_evidence_refs: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可直接序列化的字典。"""
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# 模型原始输出的中间结构（面对不可信文本，允许多余键）
# --------------------------------------------------------------------------- #


class _AnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    selected_strategies: List[Strategy] = Field(default_factory=list)
    selection_reason: str = ""
    findings: List[ReverseFinding] = Field(default_factory=list)
    verification_questions: List[str] = Field(default_factory=list)
    proposed_actions: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    decision_impact: str = ""
    unresolved_issues: List[str] = Field(default_factory=list)
    continue_recommended: bool = True
    stop_reason: Optional[str] = None
    current_position: str = ""


class _FindingUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int
    status: FindingStatus = FindingStatus.UNVERIFIED
    note: str = ""


class _VerificationDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    finding_updates: List[_FindingUpdate] = Field(default_factory=list)
    updated_position: str = ""
    decision_impact: str = ""
    unresolved_issues: List[str] = Field(default_factory=list)
    continue_recommended: bool = True
    stop_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #

DEFAULT_SYSTEM_PROMPT = (
    "你是 MungerMind 的逆向推理引擎：从目标的必要条件、当前判断的相反假设和方案"
    "的失败情景出发，生成可验证的问题与行动，用新的判断更新结论。你不为了反对而"
    "反对，不编造证据，也不把没有核查过的推断说成事实。"
)

DEFAULT_ANALYSIS_PROMPT = """请对下面的问题做一轮逆向分析。

# 任务目标
{question}

# 你目前的立场
{position}

# 可用子策略
A 目标逆向规划：若要达到目标，哪些必要条件、前置步骤和依赖关系必须成立？
B 逆向假设验证：什么证据可能推翻当前判断？有哪些竞争解释，什么观察可以区分它们？
C 失败逆推分析：假设方案已经失败，可能的原因、早期信号和预防措施是什么？

# 要求
1. 只选择与本次判断真正相关的子策略，并给出简短的选择原因。
2. 每条发现都要说明它与当前决策的相关性。
3. 把发现转换成具体的验证问题；本环境没有外部工具，验证只能靠推理和向用户提问，不得声称已经取得外部资料。
4. proposed_actions 最多 {max_actions} 条，按价值从高到低排列；没有值得做的事就留空。
5. 只有当仍然存在重要未解决问题、且还有明确的新验证行动时，才把 continue_recommended 设为 true。
6. 只输出一个 JSON 对象，不要输出解释性文字。

# 输出结构
{schema}
"""

DEFAULT_VERIFICATION_PROMPT = """请复核第 {cycle_number} 轮逆向分析产生的发现，并更新判断。

# 任务目标
{question}

# 待复核的发现
{findings}

# 你目前的立场
{position}

# 要求
1. 逐条判定每条发现：supported 表示当前信息支持它，challenged 表示当前信息削弱它，unverified 表示信息不足。index 从 1 开始，对应上面发现的序号。
2. 给出更新后的立场：新信息是否改变方案？如果不足以改变，请明确说明。
3. 本环境没有外部工具，不要声称已经取得外部证据；无法确认的内容标为 unverified。
4. 只有仍然存在重要未解决问题、且还有明确的新验证行动时，才把 continue_recommended 设为 true。
5. 只输出一个 JSON 对象，不要输出解释性文字。

# 输出结构
{schema}
"""

DEFAULT_SYNTHESIS_PROMPT = """请把前面的逆向分析写成给用户的最终答案。

# 用户的问题
{question}

# 逆向分析后的立场
{position}

# 各轮逆向结果
{cycles}

# 停止逆向分析的原因
{stop_reason}

# 要求
1. 先直接回答问题，再说明依据；不要复述分析流程本身。
2. 写出可能推翻这个结论的条件、最可能的失败路径，以及仍未解决的问题。
3. 不要编造外部证据、引用或数据。{caveat}
4. 措辞克制，不用「绝对」「保证」这类无法验证的说法。

请直接输出答案正文：
"""

DEFAULT_REPAIR_PROMPT = """你上一次的输出不是合法的 JSON。请只输出一个 JSON 对象，键名与结构如下：

{schema}

上一次的输出：
{raw}
"""

DEFAULT_PROMPTS: Dict[str, str] = {
    "analysis": DEFAULT_ANALYSIS_PROMPT,
    "verification": DEFAULT_VERIFICATION_PROMPT,
    "synthesis": DEFAULT_SYNTHESIS_PROMPT,
    "repair": DEFAULT_REPAIR_PROMPT,
}

_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "selected_strategies": ["A", "B", "C"],
    "selection_reason": "为什么选择这些子策略",
    "findings": [
        {
            "strategy": "A",
            "statement": "一条逆向发现",
            "relevance": "它与当前决策的关系",
        }
    ],
    "verification_questions": ["需要验证的具体问题"],
    "proposed_actions": ["建议的验证行动，按价值排序"],
    "evidence_refs": ["若使用了已有证据，填其标识；没有就留空"],
    "decision_impact": "这些发现会怎样影响当前方案",
    "unresolved_issues": ["尚未解决的问题"],
    "continue_recommended": True,
    "stop_reason": None,
    "current_position": "你对该问题的当前立场",
}

_VERIFICATION_SCHEMA: Dict[str, Any] = {
    "finding_updates": [
        {"index": 1, "status": "supported", "note": "判定理由"},
    ],
    "updated_position": "复核后的立场",
    "decision_impact": "新信息对方案的影响",
    "unresolved_issues": ["尚未解决的问题"],
    "continue_recommended": True,
    "stop_reason": None,
}

_ANALYSIS_SCHEMA_TEXT = json.dumps(_ANALYSIS_SCHEMA, ensure_ascii=False, indent=2)
_VERIFICATION_SCHEMA_TEXT = json.dumps(_VERIFICATION_SCHEMA, ensure_ascii=False, indent=2)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> Dict[str, Any]:
    """从模型输出中取出 JSON 对象，容忍说明文字与代码块包裹。"""
    candidate = text.strip()
    fence = _CODE_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise ReverseParseError(f"未找到 JSON 对象：{text.strip()[:200]}")

    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ReverseParseError(f"JSON 解析失败：{exc}") from exc

    if not isinstance(payload, dict):
        raise ReverseParseError("模型输出必须是 JSON 对象")
    return payload


def _coerce_strategies(value: Optional[Iterable[Any] | str]) -> Tuple[Strategy, ...]:
    """把 ``"AC"``、``["A", "C"]`` 或 ``[Strategy.A]`` 统一成策略元组。"""
    if value is None:
        return ()

    if isinstance(value, str):
        tokens = [token for token in re.split(r"[\s,，、/|]+", value) if token]
        if len(tokens) == 1 and len(tokens[0]) > 1:
            compact = tokens[0].upper()
            if all(char in {"A", "B", "C"} for char in compact):
                tokens = list(compact)
    else:
        tokens = [str(token) for token in value]

    selected: List[Strategy] = []
    for token in tokens:
        try:
            strategy = Strategy(token.strip().upper())
        except ValueError as exc:
            raise ConfigurationError(
                f"不支持的子策略 {token!r}；可选值: A、B、C"
            ) from exc
        if strategy not in selected:
            selected.append(strategy)
    return tuple(sorted(selected, key=_STRATEGY_ORDER.index))


def _clean_items(items: Iterable[str]) -> List[str]:
    """去掉空白项与首尾空格，保持原顺序。"""
    cleaned: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _merge_findings(
    findings: Sequence[ReverseFinding], updates: Sequence[_FindingUpdate]
) -> List[ReverseFinding]:
    """按 1 起始的序号把复核结论合并进发现列表；越界序号被忽略。"""
    merged = list(findings)
    for update in updates:
        position = update.index - 1
        if not 0 <= position < len(merged):
            logger.warning("忽略越界的发现序号: %s（共 %d 条）", update.index, len(merged))
            continue
        merged[position] = merged[position].model_copy(
            update={"status": update.status, "note": update.note.strip()}
        )
    return merged


class ReverseReActAgent(Agent):
    """独立运行的 ReverseReAct 逆向推理引擎。"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        *,
        strategies: Optional[Iterable[Any] | str] = None,
        default_cycles: int = DEFAULT_CYCLE_COUNT,
        max_cycles: int = DEFAULT_HARD_CYCLE_LIMIT,
        allow_extension: bool = False,
        max_actions_per_cycle: int = DEFAULT_MAX_ACTIONS,
        evidence_ids: Optional[Iterable[str]] = None,
        prompts: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(
            name,
            llm,
            DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt,
            config,
        )
        if default_cycles < 1:
            raise ConfigurationError("default_cycles 至少为 1")
        if max_cycles < default_cycles:
            raise ConfigurationError(
                f"max_cycles({max_cycles}) 不得小于 default_cycles({default_cycles})"
            )
        if max_actions_per_cycle < 1:
            raise ConfigurationError("max_actions_per_cycle 至少为 1")

        self.prompts = dict(DEFAULT_PROMPTS)
        if prompts:
            unknown = set(prompts) - set(DEFAULT_PROMPTS)
            if unknown:
                raise ConfigurationError(
                    f"未知的提示词键: {', '.join(sorted(unknown))}；"
                    f"可选值: {', '.join(DEFAULT_PROMPTS)}"
                )
            self.prompts.update(prompts)

        self.pinned_strategies = _coerce_strategies(strategies)
        self.default_cycles = default_cycles
        self.max_cycles = max_cycles
        self.allow_extension = allow_extension
        self.max_actions_per_cycle = max_actions_per_cycle
        self.evidence_ids = frozenset(evidence_ids or ())

        self.last_result: Optional[ReverseReActResult] = None
        self._cancelled = False
        self._seen_actions: set[str] = set()
        self._dropped_evidence: List[str] = []

    # -- 运行控制（10.3：取消后不得启动新的模型或工具调用） ----------------- #

    def cancel(self) -> None:
        """标记任务已取消；此后不会再有新的模型调用，直到 :meth:`reset`。"""
        self._cancelled = True

    def reset(self) -> None:
        """清除取消标记，允许再次运行。"""
        self._cancelled = False

    # -- 入口 --------------------------------------------------------------- #

    def run(self, input_text: str, **kwargs: Any) -> str:
        """跑完逆向推理循环并返回最终答案字符串。"""
        try:
            result = self.run_with_result(input_text, **kwargs)
        except MungerMindError as exc:
            answer = f"❌ 逆向分析未能完成: {exc}"
            self.record_exchange(input_text, answer)
            return answer
        return result.final_answer

    def run_with_result(self, input_text: str, **kwargs: Any) -> ReverseReActResult:
        """同 :meth:`run`，但返回完整结构化结果，解析失败时抛出异常。"""
        result = self._execute(input_text, **kwargs)
        self.last_result = result
        self.record_exchange(input_text, result.final_answer)
        return result

    def run_stream(
        self,
        input_text: str,
        *,
        on_progress: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式版本：逐块产出最终答案，循环阶段通过 ``on_progress`` 报告进度。

        逐个消费产出的文本片段即可实时显示答案；迭代结束后
        :attr:`last_result` 与对话历史才写入，中途放弃迭代则不会留下记录。
        """
        cycles, position, stop_reason = self._run_cycles(
            input_text, on_progress=on_progress, **kwargs
        )

        pieces: List[str] = []
        if self._cancelled:
            answer = position.strip() or CANCELLED_ANSWER
            pieces.append(answer)
            yield answer
        else:
            if on_progress is not None:
                on_progress(PROGRESS_SYNTHESIS)
            try:
                for chunk in self._synthesise_stream(
                    input_text, position, cycles, stop_reason, **kwargs
                ):
                    pieces.append(chunk)
                    yield chunk
            except _Cancelled:
                pass
            if not pieces:
                fallback = position.strip() or FALLBACK_ANSWER
                pieces.append(fallback)
                yield fallback
            answer = "".join(pieces)

        self.last_result = self._build_result(input_text, cycles, answer, stop_reason)
        self.record_exchange(input_text, answer)

    # -- 主循环 ------------------------------------------------------------- #

    def _execute(self, question: str, **kwargs: Any) -> ReverseReActResult:
        cycles, position, stop_reason = self._run_cycles(question, **kwargs)
        if self._cancelled:
            final_answer = position.strip() or CANCELLED_ANSWER
        else:
            final_answer = self._synthesise(question, position, cycles, stop_reason, **kwargs)
        return self._build_result(question, cycles, final_answer, stop_reason)

    def _build_result(
        self,
        question: str,
        cycles: Sequence[ReverseCycleResult],
        final_answer: str,
        stop_reason: str,
    ) -> ReverseReActResult:
        return ReverseReActResult(
            question=question,
            cycles=list(cycles),
            final_answer=final_answer,
            stop_reason=stop_reason or STOP_DEFAULT_BUDGET.format(cycles=self.default_cycles),
            externally_verified=False,
            dropped_evidence_refs=sorted(set(self._dropped_evidence)),
        )

    def _run_cycles(
        self,
        question: str,
        *,
        on_progress: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Tuple[List[ReverseCycleResult], str, str]:
        """跑完逆向循环，返回 ``(各轮结果, 最新立场, 停止原因)``。"""
        self._seen_actions = set()
        self._dropped_evidence = []

        cycles: List[ReverseCycleResult] = []
        position = ""
        stop_reason = ""
        extend_requested = False
        cycle_number = 0

        while not stop_reason:
            if self._cancelled:
                stop_reason = STOP_CANCELLED
                break
            if cycle_number >= self.max_cycles:
                stop_reason = STOP_HARD_LIMIT.format(limit=self.max_cycles)
                break
            if cycle_number >= self.default_cycles and not extend_requested:
                stop_reason = STOP_DEFAULT_BUDGET.format(cycles=self.default_cycles)
                break

            cycle_number += 1
            if on_progress is not None:
                on_progress(PROGRESS_CYCLE.format(cycle=cycle_number))
            try:
                cycle = self._run_cycle(cycle_number, question, position, **kwargs)
            except _Cancelled:
                stop_reason = STOP_CANCELLED
                break

            cycles.append(cycle)
            position = cycle.provisional_position or position

            if not cycle.continue_recommended:
                stop_reason = (cycle.stop_reason or STOP_MODEL_ADVISED).strip()
                break
            if not cycle.proposed_actions:
                stop_reason = STOP_NO_ACTION
                break
            if not self._has_new_action(cycle):
                stop_reason = STOP_NO_NEW_ACTION
                break
            if cycle_number >= self.default_cycles:
                # 追加循环必须由配置开启，并且仍存在重要的未解决问题（5.4）。
                extend_requested = self.allow_extension and bool(cycle.unresolved_issues)

        return cycles, position, stop_reason

    def _run_cycle(
        self,
        cycle_number: int,
        question: str,
        position: str,
        **kwargs: Any,
    ) -> ReverseCycleResult:
        analysis: _AnalysisDraft = self._ask_json(
            self.prompts["analysis"].format(
                question=question,
                position=position or "（尚无立场：请先给出你对该问题的初步判断）",
                max_actions=self.max_actions_per_cycle,
                schema=_ANALYSIS_SCHEMA_TEXT,
            ),
            _AnalysisDraft,
            _ANALYSIS_SCHEMA_TEXT,
            **kwargs,
        )

        selected, selection_reason = self._resolve_strategies(analysis)
        findings = self._keep_selected(findings=analysis.findings, selected=selected)
        actions = _clean_items(analysis.proposed_actions)[: self.max_actions_per_cycle]
        evidence_refs = self._filter_evidence(analysis.evidence_refs)

        verification: _VerificationDraft = self._ask_json(
            self.prompts["verification"].format(
                cycle_number=cycle_number,
                question=question,
                findings=_render_findings(findings),
                position=_verification_position(position, analysis),
                schema=_VERIFICATION_SCHEMA_TEXT,
            ),
            _VerificationDraft,
            _VERIFICATION_SCHEMA_TEXT,
            **kwargs,
        )

        merged = _merge_findings(findings, verification.finding_updates)
        return ReverseCycleResult(
            cycle_number=cycle_number,
            selected_strategies=selected,
            selection_reason=selection_reason,
            findings=merged,
            verification_questions=_clean_items(analysis.verification_questions),
            proposed_actions=actions,
            evidence_refs=evidence_refs,
            decision_impact=(verification.decision_impact or analysis.decision_impact).strip(),
            unresolved_issues=_clean_items(verification.unresolved_issues)
            or _clean_items(analysis.unresolved_issues),
            # 分析与复核任一步判断「价值不足」即停止，避免无谓的追加循环。
            continue_recommended=bool(
                analysis.continue_recommended and verification.continue_recommended
            ),
            stop_reason=(verification.stop_reason or analysis.stop_reason) or None,
            provisional_position=(
                verification.updated_position or analysis.current_position
            ).strip(),
        )

    # -- 归约与校验 --------------------------------------------------------- #

    def _resolve_strategies(
        self, analysis: _AnalysisDraft
    ) -> Tuple[List[Strategy], str]:
        """确定本轮子策略；用户指定时不被模型静默覆盖（3.1）。"""
        requested = [s for s in _STRATEGY_ORDER if s in set(analysis.selected_strategies)]
        reason = analysis.selection_reason.strip()

        if not self.pinned_strategies:
            if not requested:
                raise ReverseParseError("模型没有选择任何有效的子策略")
            return requested, reason or "模型按任务特征选择了这些子策略"

        selected = list(self.pinned_strategies)
        note = "用户已指定子策略 " + "、".join(
            f"{s.value}（{s.label}）" for s in selected
        )
        ignored = [s for s in requested if s not in selected]
        if ignored:
            note += "；模型建议的 " + "、".join(
                f"{s.value}（{s.label}）" for s in ignored
            ) + " 未采纳，是否启用由用户决定"
        return selected, f"{reason}；{note}" if reason else note

    @staticmethod
    def _keep_selected(
        findings: Sequence[ReverseFinding], selected: Sequence[Strategy]
    ) -> List[ReverseFinding]:
        """丢弃不属于启用子策略的发现，保证结果与所选策略一致。"""
        allowed = set(selected)
        kept = [finding for finding in findings if finding.strategy in allowed]
        if len(kept) != len(findings):
            logger.warning(
                "丢弃 %d 条不属于启用子策略的发现", len(findings) - len(kept)
            )
        return kept

    def _filter_evidence(self, refs: Sequence[str]) -> List[str]:
        """只保留当前运行已登记的证据标识（5.5：无效证据引用不得进入执行）。"""
        kept: List[str] = []
        for ref in refs:
            text = ref.strip()
            if not text or text in kept:
                continue
            if text in self.evidence_ids:
                kept.append(text)
            else:
                self._dropped_evidence.append(text)
                logger.warning("丢弃无法核实的证据引用: %s", text)
        return kept

    def _has_new_action(self, cycle: ReverseCycleResult) -> bool:
        """判断本轮是否提出了此前未出现过的验证行动，并登记本轮行动。"""
        fresh = False
        for action in cycle.proposed_actions:
            key = " ".join(action.split()).lower()
            if key and key not in self._seen_actions:
                self._seen_actions.add(key)
                fresh = True
        return fresh

    # -- 模型调用 ----------------------------------------------------------- #

    def _ask(self, prompt: str, **kwargs: Any) -> str:
        if self._cancelled:
            raise _Cancelled
        return self.llm.invoke(
            self.build_messages(prompt, include_history=False), **kwargs
        )

    def _ask_json(
        self,
        prompt: str,
        model_type: Type[BaseModel],
        schema: str,
        **kwargs: Any,
    ) -> Any:
        """请求结构化输出；首次不合格时用修复提示重试一次。"""
        text = self._ask(prompt, **kwargs)
        try:
            return self._parse(text, model_type)
        except ReverseParseError as exc:
            logger.warning("%s 首次输出不合格，尝试修复一次：%s", self.name, exc)
            repaired = self._ask(
                self.prompts["repair"].format(schema=schema, raw=text[:2000]), **kwargs
            )
            return self._parse(repaired, model_type)

    @staticmethod
    def _parse(text: str, model_type: Type[BaseModel]) -> Any:
        payload = _extract_json(text)
        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            raise ReverseParseError(
                f"输出不符合规格 5.5 的结构约定: {exc}"
            ) from exc

    # -- 汇总 --------------------------------------------------------------- #

    def _synthesis_prompt(
        self,
        question: str,
        position: str,
        cycles: Sequence[ReverseCycleResult],
        stop_reason: str,
    ) -> str:
        caveat = (
            ""
            if any(cycle.evidence_refs for cycle in cycles)
            else "本环境没有外部工具，请在答案中说明哪些判断未经外部核验。"
        )
        return self.prompts["synthesis"].format(
            question=question,
            position=position or "（逆向分析未形成明确立场）",
            cycles="\n\n".join(
                f"[第 {cycle.cycle_number} 轮]\n{cycle.model_dump_json(indent=2)}"
                for cycle in cycles
            )
            or "（无）",
            stop_reason=stop_reason,
            caveat=caveat,
        )

    def _synthesise(
        self,
        question: str,
        position: str,
        cycles: Sequence[ReverseCycleResult],
        stop_reason: str,
        **kwargs: Any,
    ) -> str:
        answer = self._ask(
            self._synthesis_prompt(question, position, cycles, stop_reason), **kwargs
        ).strip()
        return answer or position.strip() or FALLBACK_ANSWER

    def _synthesise_stream(
        self,
        question: str,
        position: str,
        cycles: Sequence[ReverseCycleResult],
        stop_reason: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """逐块产出最终答案；取消后立即停止，不再消费模型输出。"""
        if self._cancelled:
            raise _Cancelled
        prompt = self._synthesis_prompt(question, position, cycles, stop_reason)
        for chunk in self.llm.stream_invoke(
            self.build_messages(prompt, include_history=False), **kwargs
        ):
            if self._cancelled:
                break
            if chunk:
                yield chunk


def _verification_position(position: str, analysis: _AnalysisDraft) -> str:
    """复核阶段看到的立场：优先用外部传入的立场，其次用本轮分析给出的立场。"""
    return position or analysis.current_position or "（尚未形成明确立场）"


def _render_findings(findings: Sequence[ReverseFinding]) -> str:
    """把发现渲染成带序号列表，供复核阶段按 index 引用。"""
    if not findings:
        return "（本轮没有产生逆向发现）"
    return "\n".join(
        f"{index}. [{finding.strategy.value} {finding.strategy.label}] "
        f"{finding.statement}（相关性：{finding.relevance or '未说明'}）"
        for index, finding in enumerate(findings, start=1)
    )
