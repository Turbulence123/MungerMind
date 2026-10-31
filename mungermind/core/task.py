"""任务对象：状态机、取消与任务内上下文（规格 9.1）。

本切片只实现真正会用到的六个状态。``paused`` 等前端与恢复语义到位后再加，
``cleanup_pending`` 随隐私层的清理流程一起加。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .exceptions import TaskStateError


class TaskState(str, Enum):
    """任务生命周期状态。"""

    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_STATES = frozenset(
    {TaskState.COMPLETED, TaskState.CANCELLED, TaskState.FAILED}
)

_TRANSITIONS = {
    TaskState.CREATED: {TaskState.RUNNING, TaskState.CANCELLED},
    TaskState.RUNNING: {
        TaskState.WAITING_FOR_USER,
        TaskState.COMPLETED,
        TaskState.CANCELLED,
        TaskState.FAILED,
    },
    TaskState.WAITING_FOR_USER: {
        TaskState.RUNNING,
        TaskState.COMPLETED,
        TaskState.CANCELLED,
        TaskState.FAILED,
    },
}


@dataclass(frozen=True)
class Turn:
    """任务内的一问一答。"""

    question: str
    answer: str


class Task:
    """一次任务的状态、取消标记与任务内上下文。

    模式只读取任务（:meth:`context_text`），不修改状态；状态转移由编排层调用。
    """

    def __init__(self, question: str, *, task_id: Optional[str] = None) -> None:
        question = question.strip()
        if not question:
            raise ValueError("任务问题不能为空")
        self.id = task_id or uuid.uuid4().hex[:8]
        self.question = question
        self.state = TaskState.CREATED
        self.turns: List[Turn] = []
        self.failure: Optional[str] = None

    def _move(self, target: TaskState) -> None:
        if target not in _TRANSITIONS.get(self.state, set()):
            raise TaskStateError(
                f"任务状态不能从 {self.state.value} 变为 {target.value}"
            )
        self.state = target

    def start(self) -> None:
        self._move(TaskState.RUNNING)

    def wait_for_user(self) -> None:
        self._move(TaskState.WAITING_FOR_USER)

    def finish(self) -> None:
        self._move(TaskState.COMPLETED)

    def cancel(self) -> None:
        self._move(TaskState.CANCELLED)

    def fail(self, reason: str) -> None:
        self.failure = reason
        self._move(TaskState.FAILED)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))

    def context_text(self, max_chars: int = 600) -> str:
        """给下一轮模式的上一轮上下文；没有历史时返回空串。"""
        if not self.turns:
            return ""
        last = self.turns[-1]
        answer = last.answer.strip()
        if len(answer) > max_chars:
            answer = answer[:max_chars].rstrip() + "…"
        return f"上一轮问题：{last.question}\n上一轮结论：{answer}"
