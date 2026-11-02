"""推理模式的统一输出契约（规格 4.3）。

模式之间通过结构化数据交接，不依赖解析任意长的自然语言段落。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..core.task import Task


class FindingRef(BaseModel):
    """一条发现的轻量引用；完整结构留在 ``ModeResult.details``。"""

    model_config = ConfigDict(extra="forbid")

    statement: str
    status: str = "unverified"
    source: str = ""


class ModeResult(BaseModel):
    """一次模式运行交给编排层的结果。"""

    model_config = ConfigDict(extra="forbid")

    mode: str
    answer: str
    stop_reason: str = ""
    findings: List[FindingRef] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    unresolved_issues: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    externally_verified: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class ReasoningMode(ABC):
    """编排层只依赖这个接口；具体模式自己决定怎么调用模型与工具。"""

    name: str = ""
    last_result: Optional[ModeResult] = None

    @abstractmethod
    def run(self, task: Task, question: str) -> ModeResult:
        """跑完一次任务并返回结构化结果。"""

    @abstractmethod
    def run_stream(
        self,
        task: Task,
        question: str,
        *,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        """流式版本：逐块产出答案，结束后调用方可读取 ``last_result``。"""
