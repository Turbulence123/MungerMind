"""内置工具。

``CalculatorTool`` 使用 AST 白名单求值，只允许算术运算和少量数学函数，
不会执行任意 Python 代码。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict, List

from ..core.exceptions import ToolExecutionError
from .base import Tool, ToolParameter, ToolRegistry

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

_ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_MAX_EXPRESSION_LENGTH = 500
_MAX_POWER_EXPONENT = 1000


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolExecutionError(f"不支持的常量: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ToolExecutionError(f"不支持的运算符: {type(node.op).__name__}")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER_EXPONENT:
            raise ToolExecutionError("指数过大，已拒绝计算")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ToolExecutionError(f"不支持的一元运算符: {type(node.op).__name__}")
        return op(_evaluate(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ToolExecutionError("只支持以名称调用的数学函数")
        func = _ALLOWED_FUNCTIONS.get(node.func.id)
        if func is None:
            raise ToolExecutionError(f"不支持的函数: {node.func.id}")
        return func(*(_evaluate(arg) for arg in node.args))
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise ToolExecutionError(f"未知标识符: {node.id}")
    raise ToolExecutionError(f"不支持的表达式节点: {type(node).__name__}")


class CalculatorTool(Tool):
    """安全的数学计算器。"""

    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description=(
                "数学计算工具，支持 + - * / // % **、括号，以及 "
                "sqrt/log/abs/round/min/max 等函数和 pi/e 常量。"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="要计算的数学表达式，例如 (25 + 15) * 3 - 8",
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        expression = parameters.get("expression") or parameters.get("input") or ""
        expression = str(expression).strip()
        if not expression:
            raise ToolExecutionError("计算表达式不能为空")
        if len(expression) > _MAX_EXPRESSION_LENGTH:
            raise ToolExecutionError("计算表达式过长")

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolExecutionError(f"表达式语法错误: {exc.msg}") from exc

        try:
            result = _evaluate(tree)
        except ZeroDivisionError as exc:
            raise ToolExecutionError("除数不能为零") from exc

        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)


def create_calculator_registry() -> ToolRegistry:
    """创建仅包含计算器的注册表。"""
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    return registry
