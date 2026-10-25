"""Agent基类"""
from abc import ABC, abstractmethod
from typing import Any, Optional

from .config import Config
from .llm import LLM
from .message import Message


class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs: Any) -> str:
        """运行Agent"""
        raise NotImplementedError

    def add_message(self, message: Message) -> None:
        """添加消息到历史记录，超出 max_history_length 时丢弃最早的消息"""
        self._history.append(message)
        max_length = self.config.max_history_length
        if max_length > 0 and len(self._history) > max_length:
            del self._history[: len(self._history) - max_length]

    def clear_history(self) -> None:
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def build_messages(
        self,
        input_text: str,
        system_prompt: Optional[str] = None,
        include_history: bool = True,
    ) -> list[dict[str, Any]]:
        """构造 OpenAI 兼容的消息列表：系统提示词 + 历史 + 当前输入。"""
        prompt = self.system_prompt if system_prompt is None else system_prompt
        messages: list[dict[str, Any]] = []
        if prompt:
            messages.append({"role": "system", "content": prompt})
        if include_history:
            messages.extend(message.to_dict() for message in self._history)
        messages.append({"role": "user", "content": input_text})
        return messages

    def record_exchange(self, user_input: str, response: str) -> None:
        """把一轮问答写入历史记录。"""
        self.add_message(Message(content=user_input, role="user"))
        self.add_message(Message(content=response, role="assistant"))

    def __str__(self) -> str:
        provider = getattr(self.llm, "provider", "unknown")
        return f"Agent(name={self.name}, provider={provider})"
