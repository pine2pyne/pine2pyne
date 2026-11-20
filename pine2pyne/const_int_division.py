"""Pine v4 and v5 divide two const ints as ints: `(9 - 1) / 2` is 4 and `9 / 2` is 4.

pine2pyne prints such a division `cast_int(a / b)`. A const int, as it types one, is

* an int literal;
* a name declared from a const int and never reassigned (`:=`, `+=`), whether the declaration
  names the type or not: `x = 4`, `var x = 4`, `int x = 4`, `const int K = 4` (MEASURED on
  TradingView 2026-11-16, BINANCE:BTCUSDT 1D, //@version=5: `int declared = 7` divides as
  `declared / 2` = 3);
* a sign or a `+ - * / %` of const ints;
* a call of an int-returning builtin on constants, which the compiler folds before dividing
  (same measurement, //@version=5: `math.min(9, 5) / 2` and `math.max(3, 5) / 2` are 2,
  `math.abs(-7) / 2`, `math.round(7.4) / 2`, `math.floor(7.9) / 2` and `int(7.9) / 2` are 3,
  `str.length("abcde") / 2` is 2 and `str.length("abc") / 2` is 1).

An input, a loop counter, a tuple element and a builtin series (`bar_index`) are not const ints
here. The same measurement shows that an input really is not one — `input.int(7) / 2` is 3.5 —
and neither is a series — `array.size(arr) / 2` is 1.5 on an array of three.

A parameter is not one either, and this is where TradingView cannot be followed by a rule over
names: the compiler specialises a function per call site, so the same `half(v) => v / 2` divides
as an int or as a float depending on the argument written at the call. Measured in one script
(//@version=5): `half(3 + 4)` is 3 while `half(seven)`, `seven = input.int(7)`, is 3.5. One
Python function cannot hold both, so a parameter stays a float division here.

Pine v6 divides every int pair as floats (same measurement, //@version=6: `7 / 2` is 3.5), so
nothing is cast there.
"""
from __future__ import annotations

from typing import Dict, List, Set

from .ast_nodes import (
    Assignment, BinaryOp, ForInLoop, ForLoop, FuncDecl, Identifier, Literal, MemberAccess,
    FunctionCall, Reassignment, Script, TupleDestructure, UnaryOp, VarDecl, VaripDecl,
)
from .ast_walk import walk

_INT_OPERATORS = frozenset({'+', '-', '*', '/', '%'})

# builtins that return an int from const int arguments, and so fold into a const int
_INT_OF_INTS = frozenset({'math.min', 'math.max', 'math.abs'})
# builtins that return an int from a single const number, fraction or not
_INT_OF_NUMBER = frozenset({'math.round', 'math.floor', 'math.ceil', 'int'})
# builtins that return an int from a single const string
_INT_OF_STRING = frozenset({'str.length'})


def _callee(expression) -> str:
    """The written name of the called function, `math.max` or `int`, or '' if it is computed."""
    func = getattr(expression, 'func', None)
    if isinstance(func, str):
        return func
    if isinstance(func, MemberAccess) and isinstance(func.object, str):
        return f'{func.object}.{func.member}'
    return ''


def _is_const_number(expression, names: Set[str]) -> bool:
    """Whether `expression` is a constant number: a const int, or a float built the same way."""
    if isinstance(expression, Literal):
        return expression.literal_type in ('int', 'float')
    if isinstance(expression, UnaryOp):
        return expression.op in ('-', '+') and _is_const_number(expression.operand, names)
    if isinstance(expression, BinaryOp):
        return (expression.op in _INT_OPERATORS and _is_const_number(expression.left, names)
                and _is_const_number(expression.right, names))
    return is_const_int(expression, names)


def _is_const_string(expression) -> bool:
    """Whether `expression` is a constant string: a literal or a `+` of literals."""
    if isinstance(expression, Literal):
        return expression.literal_type == 'string'
    if isinstance(expression, BinaryOp) and expression.op == '+':
        return _is_const_string(expression.left) and _is_const_string(expression.right)
    return False


def _folds_to_const_int(call: FunctionCall, names: Set[str]) -> bool:
    """Whether the compiler folds `call` into a const int before the division."""
    if call.kwargs:
        return False
    name = _callee(call)
    if name in _INT_OF_INTS:
        return bool(call.args) and all(is_const_int(argument, names) for argument in call.args)
    if len(call.args) != 1:
        return False
    if name in _INT_OF_NUMBER:
        return _is_const_number(call.args[0], names)
    return name in _INT_OF_STRING and _is_const_string(call.args[0])


def is_const_int(expression, names: Set[str]) -> bool:
    """Whether `expression` is a const int, `names` being the const int names."""
    if isinstance(expression, Literal):
        return expression.literal_type == 'int'
    if isinstance(expression, Identifier):
        return expression.name in names
    if isinstance(expression, UnaryOp):
        return expression.op in ('-', '+') and is_const_int(expression.operand, names)
    if isinstance(expression, BinaryOp):
        return (expression.op in _INT_OPERATORS and is_const_int(expression.left, names)
                and is_const_int(expression.right, names))
    if isinstance(expression, FunctionCall):
        return _folds_to_const_int(expression, names)
    return False


def const_int_names(script: Script) -> Set[str]:
    """The names of `script` that hold a const int (see the module docstring)."""
    declared: Dict[str, List] = {}
    excluded: Set[str] = set()
    for node in walk(script):
        if isinstance(node, Assignment) and isinstance(node.target, str):
            if node.type_hint not in (None, '', 'int'):
                excluded.add(node.target)
            else:
                declared.setdefault(node.target, []).append(node.value)
        elif isinstance(node, (VarDecl, VaripDecl)):
            if node.type_hint not in (None, '', 'int'):
                excluded.add(node.name)
            else:
                declared.setdefault(node.name, []).append(node.value)
        elif isinstance(node, Reassignment):
            excluded.add(node.target)
        elif isinstance(node, TupleDestructure):
            excluded.update(node.names)
        elif isinstance(node, ForLoop):
            excluded.add(node.var)
        elif isinstance(node, ForInLoop):
            excluded.update(node.vars)
        elif isinstance(node, FuncDecl):
            excluded.update(parameter.name for parameter in node.params)
    candidates = {name: values for name, values in declared.items() if name not in excluded}
    names: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, values in candidates.items():
            if name not in names and all(is_const_int(value, names) for value in values):
                names.add(name)
                changed = True
    return names
