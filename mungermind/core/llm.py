"""OpenAI-compatible client with automatic provider detection.

Every supported backend exposes the OpenAI chat-completions protocol, so the
call path never changes. Only three things differ per provider: the base URL,
the environment variable holding the API key, and the default model. They are
resolved in this order:

1. explicit ``provider`` argument (or ``LLM_PROVIDER``)
2. an explicitly configured endpoint: ``base_url`` argument or ``LLM_BASE_URL``
3. provider-specific environment variables (``MODELSCOPE_API_KEY``, ...)
4. the shape of the API key (``ms-`` prefix, Zhipu's ``id.secret`` format)
5. ``openai`` as the final fallback
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

from .exceptions import LLMConfigurationError, LLMRequestError

# 加载 .env 文件中的环境变量
load_dotenv()

logger = logging.getLogger(__name__)

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})
LOCAL_ONLY_PROVIDERS = frozenset({"ollama", "vllm", "local"})


@dataclass(frozen=True)
class ProviderSpec:
    """Static description of one OpenAI-compatible backend."""

    name: str
    api_key_envs: Tuple[str, ...] = ()
    base_url_envs: Tuple[str, ...] = ()
    model_envs: Tuple[str, ...] = ()
    default_base_url: Optional[str] = None
    default_model: Optional[str] = None
    host_markers: Tuple[str, ...] = ()
    local_ports: Tuple[str, ...] = ()
    key_prefixes: Tuple[str, ...] = ()
    key_pattern: Optional[str] = None
    requires_api_key: bool = True
    placeholder_api_key: Optional[str] = None


# Order matters for environment-variable detection: the first provider whose
# variables are present wins when no endpoint tells us otherwise.
_PROVIDER_SPECS: Tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="modelscope",
        api_key_envs=("MODELSCOPE_API_KEY",),
        base_url_envs=("MODELSCOPE_BASE_URL",),
        model_envs=("MODELSCOPE_MODEL",),
        default_base_url="https://api-inference.modelscope.cn/v1/",
        default_model="Qwen/Qwen3.8-27B",
        host_markers=("api-inference.modelscope.cn",),
        key_prefixes=("ms-",),
    ),
    ProviderSpec(
        name="openai",
        api_key_envs=("OPENAI_API_KEY",),
        base_url_envs=("OPENAI_BASE_URL",),
        model_envs=("OPENAI_MODEL",),
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-6.1-sol",
        host_markers=("api.openai.com",),
    ),
    ProviderSpec(
        name="zhipu",
        api_key_envs=("ZHIPU_API_KEY", "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY"),
        base_url_envs=("ZHIPU_BASE_URL", "BIGMODEL_BASE_URL"),
        model_envs=("ZHIPU_MODEL",),
        default_base_url="https://open.bigmodel.cn/api/paas/v4/",
        default_model="glm-4.7-flash",
        host_markers=("bigmodel.cn",),
        key_pattern=r"[0-9a-zA-Z]{16,}\.[0-9a-zA-Z]{16,}",
    ),
    ProviderSpec(
        name="deepseek",
        api_key_envs=("DEEPSEEK_API_KEY",),
        base_url_envs=("DEEPSEEK_BASE_URL",),
        model_envs=("DEEPSEEK_MODEL",),
        default_base_url="https://api.deepseek.com",
        default_model="deepseek-flash",
        host_markers=("api.deepseek.com",),
    ),
    ProviderSpec(
        name="dashscope",
        api_key_envs=("DASHSCOPE_API_KEY",),
        base_url_envs=("DASHSCOPE_BASE_URL",),
        model_envs=("DASHSCOPE_MODEL",),
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3.7-plus",
        host_markers=("dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "maas.aliyuncs.com"),
    ),
    ProviderSpec(
        name="moonshot",
        api_key_envs=("MOONSHOT_API_KEY",),
        base_url_envs=("MOONSHOT_BASE_URL",),
        model_envs=("MOONSHOT_MODEL",),
        default_base_url="https://api.moonshot.cn/v1",
        default_model="kimi-k3",
        host_markers=("api.moonshot.cn",),
    ),
    ProviderSpec(
        name="siliconflow",
        api_key_envs=("SILICONFLOW_API_KEY", "SiliconFlow_API_KEY"),
        base_url_envs=("SILICONFLOW_BASE_URL",),
        model_envs=("SILICONFLOW_MODEL",),
        default_base_url="https://api.siliconflow.cn/v1",
        default_model="Qwen/Qwen3.8-27B",
        host_markers=("api.siliconflow.cn",),
    ),
    ProviderSpec(
        name="openrouter",
        api_key_envs=("OPENROUTER_API_KEY",),
        base_url_envs=("OPENROUTER_BASE_URL",),
        default_base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-6.1-sol",
        host_markers=("openrouter.ai",),
        key_prefixes=("sk-or-",),
    ),
    ProviderSpec(
        name="ollama",
        api_key_envs=("OLLAMA_API_KEY",),
        base_url_envs=("OLLAMA_HOST", "OLLAMA_BASE_URL"),
        default_base_url="http://localhost:11434/v1",
        default_model="qwen3.5:4b",
        local_ports=(":11434",),
        requires_api_key=False,
        placeholder_api_key="ollama",
    ),
    ProviderSpec(
        name="vllm",
        api_key_envs=("VLLM_API_KEY",),
        base_url_envs=("VLLM_BASE_URL",),
        default_base_url="http://localhost:8000/v1",
        local_ports=(":8000",),
        requires_api_key=False,
        placeholder_api_key="vllm",
    ),
    ProviderSpec(
        name="local",
        placeholder_api_key="local",
    ),
    ProviderSpec(
        name="auto",
        placeholder_api_key="local",
    ),
)

PROVIDERS: Dict[str, ProviderSpec] = {spec.name: spec for spec in _PROVIDER_SPECS}


def _first_env(names: Tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _parse_base_url(base_url: str) -> Tuple[str, Optional[int], str]:
    candidate = base_url.strip()
    if "://" not in candidate:
        candidate = "http://" + candidate
    parsed = urlparse(candidate)
    try:
        port = parsed.port
    except ValueError:
        port = None
    return (parsed.hostname or "").lower(), port, candidate.lower()


def _is_local_endpoint(base_url: str) -> bool:
    host, _, _ = _parse_base_url(base_url)
    return host in LOCAL_HOSTS or host.endswith(".local")


def _host_matches(host: str, marker: str) -> bool:
    """Match a host exactly or as a subdomain of a provider domain.

    Plain substring matching would accept look-alike hosts such as
    ``api.openai.com.attacker.tld`` and leak the API key to them.
    """
    return host == marker or host.endswith("." + marker)


def _match_base_url(base_url: str) -> Optional[str]:
    """Map an endpoint to a provider name, or ``None`` for unknown gateways."""
    host, port, _ = _parse_base_url(base_url)
    for spec in _PROVIDER_SPECS:
        if any(_host_matches(host, marker) for marker in spec.host_markers):
            return spec.name
    if host in LOCAL_HOSTS or host.endswith(".local"):
        if port is not None:
            port_marker = f":{port}"
            for spec in _PROVIDER_SPECS:
                if port_marker in spec.local_ports:
                    return spec.name
        return "local"
    return None


def _validate_provider(name: str) -> str:
    if name not in PROVIDERS:
        supported = ", ".join(sorted(PROVIDERS))
        raise LLMConfigurationError(
            f"不支持的 provider: {name!r}。可选值: {supported}, auto"
        )
    return name


def detect_provider(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Infer the provider from explicit arguments, the endpoint and the environment."""
    explicit = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit and explicit != "auto":
        return _validate_provider(explicit)

    for candidate in (base_url, os.getenv("LLM_BASE_URL")):
        if candidate and candidate.strip():
            # An endpoint is the strongest signal of intent. Unknown gateways stay
            # as "auto" so the user-supplied endpoint is used unchanged.
            return _match_base_url(candidate) or "auto"

    for spec in _PROVIDER_SPECS:
        if _first_env(spec.base_url_envs) or _first_env(spec.api_key_envs):
            return spec.name

    actual_key = (api_key or _first_env(("LLM_API_KEY",)) or "").strip()
    if actual_key:
        for spec in _PROVIDER_SPECS:
            if spec.key_prefixes and actual_key.startswith(spec.key_prefixes):
                return spec.name
        for spec in _PROVIDER_SPECS:
            if spec.key_pattern and re.fullmatch(spec.key_pattern, actual_key):
                return spec.name

    return "openai"


def _normalize_provider_base_url(provider: str, value: str) -> str:
    value = value.strip()
    if provider in LOCAL_ONLY_PROVIDERS and "://" not in value:
        value = "http://" + value
    if provider == "ollama" and not value.rstrip("/").endswith("/v1"):
        value = value.rstrip("/") + "/v1"
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LLMConfigurationError(f"环境变量 {name} 必须是整数，当前值: {raw!r}") from exc


class LLM:
    """
    通用LLM客户端。
    用于调用任何兼容OpenAI接口的服务，支持阻塞与流式两种调用方式。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        apiKey: Optional[str] = None,
        baseUrl: Optional[str] = None,
        timeout: Optional[int] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载，
        并按 provider -> base_url -> 服务商环境变量 -> 密钥格式 的顺序自动识别服务商。
        """
        api_key_arg = api_key if api_key is not None else apiKey
        base_url_arg = base_url if base_url is not None else baseUrl
        requested = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()

        self.provider = detect_provider(
            api_key=api_key_arg, base_url=base_url_arg, provider=provider
        )
        if requested in ("", "auto"):
            logger.info("Auto-detected LLM provider: %s", self.provider)

        spec = PROVIDERS[self.provider]
        self.model = self._resolve_model(model, spec)
        self.api_key = self._resolve_api_key(api_key_arg, spec)
        self.base_url = self._resolve_base_url(base_url_arg, spec)
        self.timeout = timeout if timeout is not None else _env_int("LLM_TIMEOUT", 60)
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("TEMPERATURE", "0.7"))
        )
        max_tokens_env = os.getenv("MAX_TOKENS")
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else (int(max_tokens_env) if max_tokens_env else None)
        )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        self._client = self.client

    @staticmethod
    def _resolve_model(model: Optional[str], spec: ProviderSpec) -> str:
        resolved = (
            model
            or _first_env(spec.model_envs)
            or _first_env(("LLM_MODEL_ID",))
            or spec.default_model
        )
        if not resolved:
            raise LLMConfigurationError(
                f"provider={spec.name} 缺少模型ID，请传入 model 或设置 LLM_MODEL_ID。"
            )
        return resolved

    @staticmethod
    def _resolve_api_key(api_key: Optional[str], spec: ProviderSpec) -> str:
        resolved = api_key or _first_env(spec.api_key_envs)
        if not resolved and spec.requires_api_key:
            resolved = _first_env(("LLM_API_KEY",))
        if not resolved:
            resolved = spec.placeholder_api_key
        if not resolved:
            envs = " / ".join(spec.api_key_envs) or "LLM_API_KEY"
            raise LLMConfigurationError(
                f"provider={spec.name} 缺少API密钥，请传入 api_key 或设置 {envs}。"
            )
        return resolved

    @staticmethod
    def _resolve_base_url(base_url: Optional[str], spec: ProviderSpec) -> str:
        resolved = base_url
        if not resolved:
            provider_base = _first_env(spec.base_url_envs)
            if provider_base:
                resolved = _normalize_provider_base_url(spec.name, provider_base)

        if not resolved:
            generic = _first_env(("LLM_BASE_URL",))
            if generic:
                matched = _match_base_url(generic)
                # A generic LLM_BASE_URL must not leak into an explicitly chosen
                # cloud provider, but it is still valid for a local custom port.
                if (
                    matched == spec.name
                    or spec.name == "auto"
                    or (spec.name in LOCAL_ONLY_PROVIDERS and _is_local_endpoint(generic))
                ):
                    resolved = generic

        if not resolved:
            resolved = spec.default_base_url
        if not resolved:
            raise LLMConfigurationError(
                f"provider={spec.name} 缺少服务地址，请传入 base_url 或设置 LLM_BASE_URL。"
            )
        return resolved

    def __repr__(self) -> str:
        return f"LLM(provider={self.provider!r}, model={self.model!r}, base_url={self.base_url!r})"

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """调用底层 chat-completions 接口，失败时抛出 :class:`LLMRequestError`。"""
        payload: Dict[str, Any] = dict(kwargs)
        payload.setdefault("model", self.model)
        payload.setdefault("messages", messages)
        if temperature is not None:
            payload["temperature"] = temperature
        elif self.temperature is not None:
            payload.setdefault("temperature", self.temperature)
        if self.max_tokens is not None:
            payload.setdefault("max_tokens", self.max_tokens)
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        if stream:
            payload["stream"] = True

        try:
            return self.client.chat.completions.create(**payload)
        except Exception as exc:
            raise LLMRequestError(
                f"调用 {self.provider} 模型 {self.model} 失败: {exc}"
            ) from exc

    def invoke(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """阻塞式调用，返回完整文本响应。"""
        response = self.create_completion(messages, temperature=temperature, **kwargs)
        return self._extract_content(response)

    def stream_invoke(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式调用，逐段产出文本增量。"""
        response = self.create_completion(
            messages, temperature=temperature, stream=True, **kwargs
        )
        try:
            for chunk in response:
                if not getattr(chunk, "choices", None):
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as exc:
            raise LLMRequestError(f"读取流式响应失败: {exc}") from exc

    def think(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = 0.0,
    ) -> str:
        """兼容旧接口的别名，内部改用阻塞式 ``invoke``。"""
        return self.invoke(messages, temperature=temperature)

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        return getattr(message, "content", None) or ""


if __name__ == "__main__":
    try:
        client = LLM()
        print(repr(client))
        print(
            client.invoke(
                [{"role": "user", "content": "用一句话解释什么是复利。"}]
            )
        )
    except LLMConfigurationError as exc:
        print(f"配置错误: {exc}")
