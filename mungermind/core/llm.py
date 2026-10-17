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
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

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


def _match_base_url(base_url: str) -> Optional[str]:
    """Map an endpoint to a provider name, or ``None`` for unknown gateways."""
    host, port, lowered = _parse_base_url(base_url)
    for spec in _PROVIDER_SPECS:
        if any(marker in lowered for marker in spec.host_markers):
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
        raise ValueError(f"不支持的 provider: {name!r}。可选值: {supported}, auto")
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


def _safe_print(text: str, end: str = "\n", flush: bool = False) -> None:
    """Print without crashing on consoles whose encoding cannot represent text."""
    try:
        print(text, end=end, flush=flush)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe, end=end, flush=flush)


class LLM:
    """
    通用LLM客户端。
    用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
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
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

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
            raise ValueError(
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
            raise ValueError(
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
            raise ValueError(
                f"provider={spec.name} 缺少服务地址，请传入 base_url 或设置 LLM_BASE_URL。"
            )
        return resolved

    def __repr__(self) -> str:
        return f"LLM(provider={self.provider!r}, model={self.model!r}, base_url={self.base_url!r})"

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> Optional[str]:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        _safe_print(f"正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            _safe_print("大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                _safe_print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            _safe_print(f"调用LLM API时发生错误: {e}")
            return None

'''
# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = LLM()
        
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "决定我明天要不要早起"}
        ]
        
        print(f"--- 调用LLM（provider={llmClient.provider}） ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
'''

if __name__ == '__main__':
    llm = LLM(provider="ollama", model="deepseek-r1:1.5b")

    answer = llm.think([
        {"role": "user", "content": "用一句话解释什么是复利。"}
    ])
    print(answer)