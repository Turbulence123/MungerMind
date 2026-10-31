"""框架统一异常体系。

上层代码可以只捕获 :class:`MungerMindError`，也可以按需区分模型、工具与
Agent 运行期的错误。配置类异常同时继承 ``ValueError``，保持与既有调用方
的兼容性。
"""


class MungerMindError(Exception):
    """所有框架异常的基类。"""


class ConfigurationError(MungerMindError, ValueError):
    """配置缺失或取值非法。"""


class TaskStateError(MungerMindError):
    """任务状态不允许这次转移（规格 9.1）。"""


class LLMError(MungerMindError):
    """模型调用相关错误的基类。"""


class LLMConfigurationError(LLMError, ConfigurationError):
    """provider、模型、密钥或服务地址无法确定。"""


class LLMRequestError(LLMError):
    """调用模型接口失败（网络、鉴权、限流、服务端错误等）。"""


class ToolError(MungerMindError):
    """工具系统相关错误的基类。"""


class ToolNotFoundError(ToolError, LookupError):
    """请求的工具未注册。"""


class ToolExecutionError(ToolError):
    """工具执行失败或参数无法解析。"""


class AgentError(MungerMindError):
    """Agent 运行期错误的基类。"""


class PlanParseError(AgentError, ValueError):
    """Plan-and-Solve 规划结果无法解析为步骤列表。"""


class ReverseParseError(AgentError, ValueError):
    """ReverseReAct 的模型输出无法解析，或不符合规格 5.5 节的结构约定。"""
