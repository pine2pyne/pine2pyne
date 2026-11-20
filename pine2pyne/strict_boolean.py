"""
Strict `and` / `or`: Pine v5 and earlier evaluate both operands.

The Pine v6 migration guide: "Pine v5 evaluates all bool expressions except for the `?:`
ternary operator strictly, meaning the second part of a conditional expression is always
evaluated, regardless of the value of the first argument"; v6 evaluates them lazily. Python's
`and` / `or` are lazy, so a stateful right operand (a `ta.*` call, a function with `var` or
history state) of a v5 script advanced its history only on the bars where it ran. Only v5 is
documented; v4 is measured strict like v5 on TradingView, v1-v3 are taken as strict too, and a
script without a version annotation as v6, the transpiler's target (as bool_is_never_na does).

The code generator emits `all((a, b, c))` / `any((a, b, c))` for a chain of one operator whose
operands after the first may have an effect: a tuple evaluates every element, left to right,
before `all` / `any` sees any. A chain whose later operands are pure (literals, names, history
reads, arithmetic, comparisons, `not`, `?:`, and calls of the builtins that keep no state:
`na`, `nz`, `math.*`, `str.*`, the strategy trade getters, collection getters) keeps the
Python `and` / `or`: skipping them changes nothing, and a getter run on an empty collection
(`strategy.opentrades.entry_price(0)` after a size check) must not start to run. A call of a
function the script declares itself is stateful: it may hold `var` state or call `ta.*`.
Every other call (`ta.*`, `request.*`, `fixnan`, a v4 `sma(...)`) is treated as stateful. This
module decides which.
"""
from typing import FrozenSet, List, Optional

from .ast_nodes import (BinaryOp, Expression, FunctionCall, Identifier, IndexAccess, Literal, MemberAccess,
                        MethodCall, NaLiteral, Script, TernaryOp, UnaryOp)
from .ast_walk import walk
from .pine_builtins import TA_VARIABLE_FUNCTIONS

BOOLEAN_OPERATORS = frozenset({'and', 'or'})
# Nodes that cannot advance a history or change a variable. A call, a bound name (`:=`) or any
# node not listed is taken to have an effect
PURE_NODE_TYPES = (Literal, NaLiteral, Identifier, MemberAccess, IndexAccess, BinaryOp, UnaryOp, TernaryOp)
# Builtin calls that read their arguments and keep no state, by exact name and by namespace
PURE_FUNCTIONS = frozenset({'na', 'nz', 'int', 'float', 'bool', 'array.size', 'array.get', 'array.includes',
                            'array.first', 'array.last', 'array.indexof', 'map.size', 'map.get', 'map.contains'})
PURE_NAMESPACES = ('math.', 'str.', 'string.', 'strategy.opentrades.', 'strategy.closedtrades.')
# `math.random` keeps its generator, `math.sum` its rolling window: both advance only on the bars that run them
STATEFUL_IN_PURE_NAMESPACES = frozenset({'math.random', 'math.sum'})
# The same getters called as methods: `values.size()`, `lookup.get(key)`
PURE_METHODS = frozenset({'size', 'get', 'includes', 'contains', 'first', 'last', 'indexof'})


def evaluates_and_or_strictly(script: Script) -> bool:
    """True for Pine v5 and earlier; a script without a version annotation is v6."""
    version = script.version.version if script.version else ''
    return version.isdigit() and int(version) < 6


def _dotted_name(node) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, Identifier):
        return node.name
    if isinstance(node, MemberAccess):
        return f'{_dotted_name(node.object)}.{node.member}'
    return ''


def _is_pure_call(node, user_function_names: FrozenSet[str]) -> bool:
    if isinstance(node, MethodCall):
        return node.method in PURE_METHODS and node.method not in user_function_names
    name = 'na' if isinstance(node.func, NaLiteral) else _dotted_name(node.func)
    if name in user_function_names:
        return False
    if name in PURE_FUNCTIONS:
        return True
    return name.startswith(PURE_NAMESPACES) and name not in STATEFUL_IN_PURE_NAMESPACES


def _is_pure(node, user_function_names: FrozenSet[str]) -> bool:
    if isinstance(node, (FunctionCall, MethodCall)):
        return _is_pure_call(node, user_function_names)
    if not isinstance(node, PURE_NODE_TYPES):
        return False
    # `ta.tr`, `ta.vwap`, ... are written without () but are calls with state
    return _dotted_name(node) not in TA_VARIABLE_FUNCTIONS


def may_have_effect(expression: Expression, user_function_names: FrozenSet[str]) -> bool:
    """Whether evaluating `expression` can advance a history or bind a name."""
    return any(not _is_pure(node, user_function_names) for node in walk(expression))


def _flatten(expression: Expression, operator: str, operands: List[Expression]) -> None:
    """The operands of a chain of one operator, left to right; a nested group of the same
    operator joins the chain, since strict evaluation is associative."""
    if isinstance(expression, BinaryOp) and expression.op == operator:
        _flatten(expression.left, operator, operands)
        _flatten(expression.right, operator, operands)
    else:
        operands.append(expression)


def strict_operands(expression: BinaryOp, user_function_names: FrozenSet[str]) -> Optional[List[Expression]]:
    """The operands to evaluate together, or None when the plain operator is enough."""
    if expression.op not in BOOLEAN_OPERATORS:
        return None
    operands: List[Expression] = []
    _flatten(expression, expression.op, operands)
    return operands if any(may_have_effect(operand, user_function_names) for operand in operands[1:]) else None
