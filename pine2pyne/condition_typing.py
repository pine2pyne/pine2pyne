"""Which values are read as numbers where Pine wants a condition.

A number in a condition is cast: ``if x`` is ``if cast_bool(x):``. pine2pyne
decides it from what it knows of the value's type; the rules below say what
that is.

A *name* (or its history, or a literal) is typed from its declaration, at any
distance: a number literal, a built-in price or time series (``close``,
``bar_index``), arithmetic over numbers, ``time()``, ``nz()``, ``math.*``, the
``ta.*`` calls that do not give a bool, a number input, a cast. An arm that is na
does not count (``h = c ? close : na`` is a number), except in a typed declaration
read outside an ``if``/``while`` header: there ``float x = c ? close : na`` is not,
while the header reads the declared type (and ``float x = na`` is a number
everywhere). A name holding a user function's result is a number except in an
``if``/``while`` header; one holding an imported library function's result is a number
there too. Such a name is cast in every condition but under ``not`` (where only a pivot is).
A name holding a result that has no type is typed once it is reassigned another value
(``m := m * 2``, ``m := close``, but not ``m := f()``), outside an ``if``/``while`` header.

An *expression* written in the condition is cast only when it is ``ta.change(x)``
of a number, ``ta.dev(...)``, ``time(...)``/``time_close(...)`` or a pivot, or, as the condition of
a ternary that does not choose colours, arithmetic over numbers. ``nz()``,
``math.*``, ``array.size()`` and user function calls written there are not.

A case of a switch without a subject is read apart: its ``and``/``or`` operands as an if
header's, the whole case cast only where it is a name declared without a type from a value
holding ``c ? x : na``, and only in a switch statement, not in a switch whose value is used.

Pine v6 turns no number into a bool, so pine2pyne casts nothing in its place (an
explicit ``bool(x)`` is still ``cast_bool(x)``).

The condition of a ternary that is the value of an assignment (or an arm of one) is not
cast when an arm reads a name holding the result of a user function that is given no
type: a function of several lines, or of one line that is a call other than ``iff``.
"""
from __future__ import annotations

import dataclasses

from typing import Dict, Optional

from .ast_nodes import (
    Assignment, BinaryOp, ExpressionStatement, FuncDecl, FunctionCall, Identifier, IfExpression, IfStatement, IndexAccess, InputDecl,
    Literal,
    MethodCall, NaLiteral, Reassignment, Script, SwitchExpression, TernaryOp, UnaryOp, VarDecl, VaripDecl,
)
from .ast_walk import walk
from .bool_builtins import BOOL_BUILTINS

#: the built-in series that are numbers
NUMERIC_SERIES = frozenset({
    'open', 'high', 'low', 'close', 'volume', 'hl2', 'hlc3', 'ohlc4', 'hlcc4', 'bar_index', 'last_bar_index',
    'time', 'time_close', 'timenow', 'time_tradingday', 'last_bar_time', 'dayofweek', 'dayofmonth', 'hour',
    'minute', 'second', 'month', 'year', 'weekofyear', 'ta.tr', 'ta.vwap', 'ta.obv', 'ta.accdist', 'ta.iii',
    'ta.nvi', 'ta.pvi', 'ta.pvt', 'ta.wad', 'ta.wvad'})
#: calls whose result is a number
NUMERIC_CALLS = frozenset({'time', 'time_close', 'int', 'float', 'str.tonumber', 'input.int', 'abs', 'max', 'min',
                           'round', 'pow',
                           'input.float', 'input.source', 'input.price', 'input.time'})
#: strategy built-ins that are numbers in arithmetic (`strategy.position_avg_price * 0.99`); written alone as a
#: condition they are not cast
STRATEGY_NUMBERS = frozenset({
    'strategy.position_avg_price', 'strategy.position_size', 'strategy.equity', 'strategy.netprofit',
    'strategy.openprofit', 'strategy.grossprofit', 'strategy.grossloss', 'strategy.initial_capital',
    'strategy.opentrades', 'strategy.closedtrades', 'strategy.max_drawdown', 'strategy.max_runup',
    'strategy.eventrades'})
#: strategy built-ins that have no type (`strategy.position_size`, `strategy.closedtrades` are numbers)
_UNTYPED_STRATEGY_COUNTS = frozenset({'strategy.losstrades', 'strategy.wintrades'})
#: calls that give their first argument's type
PASS_THROUGH_CALLS = frozenset({'nz', 'fixnan'})
PIVOTS = frozenset({'ta.pivothigh', 'ta.pivotlow'})
#: calls cast where they are written as a condition; of the ta.* numbers only ta.dev
#: (`ta.sma`, `ta.atr`, `ta.stdev`, `ta.highest`, `ta.rsi` are not cast)
DIRECT_CASTS = frozenset({'time', 'time_close', 'ta.dev'}) | PIVOTS
ARITHMETIC = frozenset({'+', '-', '*', '/', '%'})
COMPARISONS = frozenset({'==', '!=', '<', '>', '<=', '>=', 'and', 'or'})
_DEPTH = 32
#: the namespaces of Pine's built-ins, which a library alias may shadow
_BUILTIN_NAMESPACES = frozenset({
    'ta', 'math', 'str', 'array', 'matrix', 'map', 'request', 'color', 'line', 'label', 'box', 'table', 'polyline',
    'linefill', 'strategy', 'input', 'chart', 'syminfo', 'timeframe', 'session', 'ticker', 'barstate', 'runtime',
    'log', 'plot', 'hline', 'display', 'format', 'location', 'position', 'shape', 'size', 'text', 'xloc', 'yloc',
    'extend', 'font', 'order', 'currency', 'dividends', 'earnings', 'splits', 'alert', 'barmerge', 'adjustment',
    'backadjustment', 'settlement_as_close', 'dayofweek', 'scale'})
#: a name declared without a type: a parameter or a tuple element
_UNTYPED = object()
#: a parameter declared without a type: beside it, an `and`/`or` operand is not cast
_UNTYPED_PARAMETER = object()


def _scope_key(function: FuncDecl) -> tuple:
    """A function's scope, the same for the copies later lowerings make of its declaration."""
    return function.name, function.line, function.column, len(function.params)


def _is_arithmetic(expression) -> bool:
    return (isinstance(expression, UnaryOp) and expression.op in ('-', '+')) or (
        isinstance(expression, BinaryOp) and expression.op in ARITHMETIC)


#: builtins whose result has a type of its own, whatever their arguments are
_TYPED_RESULT_CALLS = frozenset({'time', 'time_close', 'timestamp', 'year', 'month', 'weekofyear', 'dayofmonth',
                                 'dayofweek', 'hour', 'minute', 'second'})


def _first_arm_on_first_line(value, line: int) -> bool:
    """A ternary whose `?` and `:` stand on the declaration's first line (`x = c ? a > b :`, the else arm below):
    it is typed by that first arm."""
    return isinstance(value, TernaryOp) and value.question_line == line and value.colon_line == line


def _arithmetic_operands(expression):
    """The nodes of `expression` reached through arithmetic only, not inside a call's arguments."""
    stack = [expression]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, BinaryOp) and node.op not in COMPARISONS:
            stack.extend((node.left, node.right))
        elif isinstance(node, UnaryOp):
            stack.append(node.operand)


def _outside_typed_calls(expression):
    """The nodes of `expression`, but not those inside a call of a builtin whose result has a type of its own."""
    stack = [expression]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in _TYPED_RESULT_CALLS:
            continue
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            for field in dataclasses.fields(node):
                value = getattr(node, field.name)
                if isinstance(value, (list, tuple)):
                    stack.extend(item for item in value if dataclasses.is_dataclass(item))
                elif isinstance(value, dict):
                    stack.extend(item for item in value.values() if dataclasses.is_dataclass(item))
                elif dataclasses.is_dataclass(value):
                    stack.append(value)


class ConditionTyping:
    """The condition typing of one script (renamed, before the transformation)."""

    def __init__(self, script: Script) -> None:
        self._functions: Dict[str, FuncDecl] = {}
        for node in walk(script):
            if isinstance(node, FuncDecl):
                self._functions.setdefault(node.name, node)
        #: the declarations of the script's top level, and of each function (by id) on its own
        self._global_values: Dict[str, object] = {}
        self._function_values: Dict[tuple, Dict[str, object]] = {}
        self._function_parameters: Dict[tuple, set] = {}
        self._parameters: set = set()
        #: the aliases of the libraries the script imports, but one shadowing a built-in namespace (`as ta`)
        self._library_aliases = {imported.alias or imported.library for imported in getattr(script, 'imports', None) or []
                                 } - _BUILTIN_NAMESPACES
        version = getattr(getattr(script, 'version', None), 'version', '')
        self._version = int(version) if str(version).isdigit() else 6
        functions = [node for node in walk(script) if isinstance(node, FuncDecl)]
        inside = set()
        for function in functions:
            # a typed parameter is its type (`b = vx * 2` with `float vx` is a number)
            values: Dict[str, object] = {parameter.name: ((parameter.type_hint, NaLiteral())
                                                          if parameter.type_hint in ('int', 'float', 'bool')
                                                          else _UNTYPED_PARAMETER if not parameter.type_hint
                                                          else _UNTYPED)
                                         for parameter in function.params}
            for node in walk(function.body):
                inside.add(id(node))
                self._record(node, values)
            self._function_values[_scope_key(function)] = values
            self._function_parameters[_scope_key(function)] = {parameter.name for parameter in function.params}
        for node in walk(script):
            if id(node) not in inside and not isinstance(node, FuncDecl):
                self._record(node, self._global_values)
                self._record_typed_elements(node, self._global_values)
        #: Pine v4 `input(title = "T", defval = false)`: its first argument is read as the default, a string
        #: title here, so the input is no bool where it is compared (`a == 0` is not cast); `input(false, "T")` is
        self._title_first_inputs = {node.name for node in walk(script) if self._version <= 4
                                    and isinstance(node, InputDecl) and not node.args and node.kwargs
                                    and next(iter(node.kwargs)) != 'defval'}
        self._values: Dict[str, object] = self._global_values
        #: the names a tuple destructure declares (`[a, b] = f()`)
        self._tuple_elements = {name for node in walk(script) if isinstance(node, Assignment)
                                and hasattr(node.target, 'names') for name in node.target.names}
        #: the element of its user function's tail each tuple element is bound to (`[a, b] = f()`, `f` ending
        #: `[x, y]`)
        self._tuple_sources: Dict[str, tuple] = {}
        for node in walk(script):
            if isinstance(node, Assignment) and hasattr(node.target, 'names') and isinstance(node.value, FunctionCall) \
                    and isinstance(node.value.func, str) and node.value.func in self._functions:
                function = self._functions[node.value.func]
                tail = function.body[-1] if isinstance(function.body, list) and function.body else function.body
                elements = getattr(getattr(tail, 'expr', tail), 'elements', None) or []
                for name, element in zip(node.target.names, elements):
                    self._tuple_sources.setdefault(name, (function, element))
        self._in_function = False
        #: the values written into each block result (`__block_result__ := <value>`, an if/switch value lowered)
        self._block_values: Dict[str, list] = {}
        for node in walk(script):
            if isinstance(node, Reassignment) and isinstance(node.target, str) \
                    and node.target.startswith('__block_result'):
                self._block_values.setdefault(node.target, []).append(node.value)
        #: the names of each scope reassigned a value other than an untyped result (`m := m * 2`, `m := close`):
        #: they are typed from it, but in an if/while header
        self._global_retyped = {node.target for node in walk(script) if id(node) not in inside
                                and self._retypes(node)}
        self._function_retyped = {_scope_key(function): {node.target for node in walk(function.body)
                                                         if self._retypes(node)} for function in functions}
        self._retyped = self._global_retyped
        self._top_reassignments = [node for node in walk(script) if id(node) not in inside
                                   and isinstance(node, Reassignment) and isinstance(node.target, str)]
        #: the top-level names reassigned inside an `if` whose condition calls a user function, directly or through
        #: the names it reads (`if bull` with `bull = c and f(x)`): they have no type
        self._set_under_a_called_condition = set()
        for node in walk(script):
            if isinstance(node, IfStatement) and id(node) not in inside and self._calls_a_user_function(
                    node.condition, 4):
                for inner in walk(node.body) if isinstance(node.body, list) else []:
                    if isinstance(inner, Reassignment) and isinstance(inner.target, str):
                        self._set_under_a_called_condition.add(inner.target)
        #: the top-level names reassigned a value that reads a user function's result that has no type
        #: (`s := close > k ? 1 : -1` with `k = f(close)`): untyped, and their `ta.change` too, in a condition that
        #: is not a ternary's
        self._untyped_reassigned = {node.target for node in walk(script) if id(node) not in inside
                                    and isinstance(node, Reassignment) and isinstance(node.target, str)
                                    and self.reads_an_untyped_result(node.value)}
        #: names whose declaration cannot be typed by its first line: the value starts on a later line
        #: (`up =` and the value below), or a ternary continues there before its `:`
        self._multiline = {node.target if isinstance(node, Assignment) else node.name for node in walk(script)
                           if isinstance(node, (Assignment, VarDecl, VaripDecl))
                           and isinstance(getattr(node, 'target', ''), str) and node.line > 0
                           and (node.value_line > node.line
                                or (node.end_line > node.line and isinstance(node.value, TernaryOp)
                                    and not (_first_arm_on_first_line(node.value, node.line)
                                             and not self._derives_from_an_untyped_result(node.value.condition))))}

    def _record_typed_elements(self, node, values: Dict[str, object]) -> None:
        """`[a, b] = f()` with `f` ending in `[x, y]` where `x` is declared `float x = <value>` (or an int, a
        bool) in `f`: `a` is that type (an untyped element, or one declared `float x = na`,
        stays untyped)."""
        if not (isinstance(node, Assignment) and hasattr(node.target, 'names')
                and isinstance(node.value, FunctionCall) and isinstance(node.value.func, str)
                and node.value.func in self._functions):
            return
        function = self._functions[node.value.func]
        body = function.body
        tail = body[-1] if isinstance(body, list) and body else body
        elements = getattr(getattr(tail, 'expr', tail), 'elements', None) or []
        local = self._function_values.get(_scope_key(function), {})
        for name, element in zip(node.target.names, elements):
            declared = local.get(element.name) if isinstance(element, Identifier) else None
            if values.get(name) is _UNTYPED and isinstance(declared, tuple) and declared[0] in ('int', 'float', 'bool') \
                    and not isinstance(declared[1], NaLiteral):  # `float ob = na` leaves it untyped
                values[name] = (declared[0], NaLiteral())

    @staticmethod
    def _record(node, values: Dict[str, object]) -> None:
        if isinstance(node, Assignment) and isinstance(node.target, str):
            if node.target.startswith('__switch') and node.target in values:
                values[node.target] = _UNTYPED  # a switch subject name bound again by a later switch: its own each
                return
            values.setdefault(node.target, (node.type_hint, node.value))
        elif isinstance(node, Assignment) and hasattr(node.target, 'names'):
            builtin = isinstance(node.value, FunctionCall) and isinstance(node.value.func, str) \
                and node.value.func.startswith('ta.')
            for name in node.target.names:  # `[a, b] = f()`: untyped; the numbers of `ta.macd(...)` are numbers
                values.setdefault(name, ('float', NaLiteral()) if builtin else _UNTYPED)
        elif isinstance(node, (VarDecl, VaripDecl)):
            values.setdefault(node.name, (node.type_hint, node.value))
        elif isinstance(node, InputDecl):
            values.setdefault(node.name, FunctionCall(func=node.func, args=node.args, kwargs=node.kwargs))

    def _calls_a_user_function(self, expression, depth: int) -> bool:
        """`expression` calls a user function, or reads a top-level name declared from a value that does."""
        if depth <= 0 or expression is None:
            return False
        for node in walk(expression):
            if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in self._functions:
                return True
            if isinstance(node, Identifier):
                declared = self._global_values.get(node.name)
                if isinstance(declared, tuple) and declared[1] is not None \
                        and self._calls_a_user_function(declared[1], depth - 1):
                    return True
        return False

    def _retypes(self, node) -> bool:
        """`node` reassigns a name a value other than the result of a user function that has no type."""
        if not (isinstance(node, Reassignment) and isinstance(node.target, str)):
            return False
        value = node.value
        return not (isinstance(value, FunctionCall) and isinstance(value.func, str) and value.func in self._functions
                    and self._gives_no_result_type(value.func))

    def enter(self, function: Optional[FuncDecl]) -> None:
        """Type names as `function` sees them (its own declarations first); None: the top level."""
        self._retyped = self._function_retyped.get(_scope_key(function), set()) if function is not None \
            else self._global_retyped
        local = self._function_values.get(_scope_key(function)) if function is not None else None
        self._values = {**self._global_values, **local} if local is not None else self._global_values
        self._parameters = self._function_parameters.get(_scope_key(function), set()) if function is not None else set()
        self._in_function = local is not None
        self._locals = set(local or ()) | self._parameters

    # ------------------------------------------------------------ questions

    def is_number(self, expression, header: bool = False, colour_ternary: bool = False,
                  ternary: bool = False, argument: bool = False) -> bool:
        """Whether `expression` is cast, written as a condition: `header` an if/while
        condition, `ternary` the condition of a ternary (`colour_ternary`: one choosing colours),
        `argument` the condition argument of a builtin (`ta.valuewhen(c, ...)`)."""
        if self._version >= 6:
            return False  # Pine v6 turns no number into a bool, so none is cast there
        if argument and self._in_function and isinstance(expression, (Identifier, IndexAccess)) \
                and not self._declared_from_a_pivot(expression):
            # inside a function a name as a builtin's condition argument is not cast (`ta.valuewhen(v, ...)`), but
            # one declared from a pivot (`ph = ta.pivothigh(...)`); a call written there is (`ta.barssince(ta.change(x))`)
            return False
        self._argument = argument
        self._flow = not ternary
        self._as_a_condition = True
        try:
            return self._is_number(expression, header, colour_ternary, ternary)
        finally:
            self._argument = False
            self._flow = False
            self._as_a_condition = False

    _argument = False
    #: typing an expression read as a condition (is_number), not a value given to a name (is_number_value)
    _as_a_condition = False
    #: typing a condition that is not a ternary's (an if header, an `and`/`or` operand, an argument): a name
    #: reassigned an untyped value is untyped there
    _flow = False

    def is_cast_as_a_switch_case(self, expression) -> bool:
        """Pine v5: a name written as a whole case of a switch statement without a subject is cast where it
        is declared without a type from a value holding a ternary whose else arm is na (`h = c ? high : na`,
        `nz(c ? x : na)`, `c ? x : d ? y : na`); a typed name, one of another value (a number, a pivot, a user
        function's result), its history, a name in parentheses and an expression are not cast."""
        if self._version >= 6 or not isinstance(expression, Identifier) or getattr(expression, 'parenthesized', False):
            return False
        declared = self._values.get(expression.name)
        return isinstance(declared, tuple) and declared[0] is None and declared[1] is not None and any(
            isinstance(node, TernaryOp) and isinstance(node.false_expr, NaLiteral) for node in walk(declared[1]))

    def is_the_na_of_a_pivot_in_a_function(self, expression) -> bool:
        """In a function, a name declared from a value reading `na(ta.pivothigh(...))`: cast as a whole if header
        (`p = na(ta.pivothigh(...)) ? false : true`, `if p`), not as an `and` operand there."""
        return self._in_function and isinstance(expression, Identifier) \
            and self._declares_the_na_of_a_pivot(expression.name)

    def _declares_the_na_of_a_pivot(self, name: str) -> bool:
        """`name` is declared from a value that reads `na(ta.pivothigh(...))` / `na(ta.pivotlow(...))`."""
        declared = self._values.get(name)
        return isinstance(declared, tuple) and declared[1] is not None and any(
            isinstance(node, FunctionCall) and (node.func == 'na' or isinstance(node.func, NaLiteral)) and node.args
            and isinstance(node.args[0], FunctionCall) and node.args[0].func in PIVOTS
            for node in walk(declared[1]))

    def _declared_from_a_pivot(self, expression) -> bool:
        """A name (or its history) declared from `ta.pivothigh(...)` / `ta.pivotlow(...)`."""
        while isinstance(expression, IndexAccess):
            expression = expression.object
        declared = self._values.get(expression.name) if isinstance(expression, Identifier) else None
        return isinstance(declared, tuple) and isinstance(declared[1], FunctionCall) and declared[1].func in PIVOTS

    def _is_number(self, expression, header: bool, colour_ternary: bool, ternary: bool) -> bool:
        written = expression
        while isinstance(written, IndexAccess):
            written = written.object
        if isinstance(written, Identifier) and written.name in self._parameters:
            return False  # a parameter as a condition is never cast, typed or not (jordanfray library)
        if isinstance(written, (Identifier, Literal)):
            if isinstance(written, Identifier) and written.name in STRATEGY_NUMBERS:
                return False  # `strategy.position_avg_price ? 1 : 0` is not cast
            if ternary and written is expression and self._declared_from_na_ternary(written):
                return False  # `top = c ? h[2] : na`, then `top ? a : b`: not cast; `top and x` is
            return self._kind(expression, header, _DEPTH) == 'number'
        if isinstance(written, FunctionCall) and isinstance(written.func, str):
            if written.func in DIRECT_CASTS or (written.func == 'ta.change' and written.args):
                return self._kind(expression, header, _DEPTH) == 'number'
            # `array.get(a, i)` of a number array, written in the function form (not `a.get(i)`)
            return written.func == 'array.get' and bool(written.args) and self._is_number_array(written.args[0])
        if isinstance(written, (BinaryOp, UnaryOp)) and ternary and not colour_ternary:
            return self._kind(expression, header, _DEPTH) == 'number'
        if isinstance(written, TernaryOp) and not ternary:
            # a ternary of numbers written as an `and`/`or` operand or a header, an na arm too (`c ? up : na`)
            return self._kind(expression, header, _DEPTH) == 'number'
        return False

    _NUMBER_ARRAYS = frozenset({'array.new_float', 'array.new_int', 'array.new<float>', 'array.new<int>'})

    def _is_number_array(self, expression) -> bool:
        """A name declared as an array of numbers: `array.new_float(...)`, `array.new<int>(...)`, `float[] a`."""
        value = self._values.get(expression.name) if isinstance(expression, Identifier) else None
        if not isinstance(value, tuple):
            return False
        type_hint, value = value
        if str(type_hint or '').replace(' ', '') in ('float[]', 'int[]', 'array<float>', 'array<int>'):
            return True
        return isinstance(value, FunctionCall) and isinstance(value.func, str) \
            and value.func.replace(' ', '') in self._NUMBER_ARRAYS

    def _declared_from_na_ternary(self, name) -> bool:
        """A name declared without a type from a ternary with an na arm, whose other arm reads no name declared
        from a user function's result that has no type (`c ? m[1] : na` with `m = f()`, retyped or not); at the
        script's top level only one whose condition calls a user function (`ft = f(m) > 0 ? m[2] : na`)."""
        value = self._values.get(getattr(name, 'name', None))
        if not (isinstance(value, tuple) and value[0] is None and isinstance(value[1], TernaryOp)):
            return False
        if not self._in_function and not any(isinstance(node, FunctionCall) and isinstance(node.func, str)
                                             and node.func in self._functions for node in walk(value[1].condition)):
            return False  # `t = c ? close : na`, then `t ? 1 : 0`: cast
        arms = (value[1].true_expr, value[1].false_expr)
        return any(isinstance(arm, NaLiteral) for arm in arms) and not any(
            isinstance(node, Identifier) and self._declared_from_an_untyped_call(node.name)
            for arm in arms for node in walk(arm))

    def _declared_from_an_untyped_call(self, name: str) -> bool:
        """`name` is declared from a call of a user function whose result has no type."""
        value = self._values.get(name)
        value = value[1] if isinstance(value, tuple) else None
        return isinstance(value, FunctionCall) and isinstance(value.func, str) and value.func in self._functions \
            and self._gives_no_result_type(value.func)

    @staticmethod
    def _outside_builtin_calls(expression):
        """The nodes of `expression` down to its calls: a builtin's result has its own type (`ta.barssince(c)`)."""
        pending = [expression]
        while pending:
            node = pending.pop()
            yield node
            if isinstance(node, BinaryOp):
                pending.extend([node.left, node.right])
            elif isinstance(node, UnaryOp):
                pending.append(node.operand)
            elif isinstance(node, TernaryOp):
                pending.extend([node.condition, node.true_expr, node.false_expr])
            elif isinstance(node, IndexAccess):
                pending.append(node.object)

    def _condition_depends_on_an_untyped_result(self, condition) -> bool:
        """`condition` calls, or reads a name computed at any depth from, a user function's result that has no
        type, through names, comparisons, `and`/`or` and arithmetic but not through a builtin's call (`x and f(c)`,
        `f(c) > 0`; not `ta.barssince(f(c))`)."""
        for node in self._outside_builtin_calls(condition):
            if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in self._functions \
                    and self._gives_no_result_type(node.func):
                return True
            if isinstance(node, Identifier) and self._depends_on_an_untyped_result(
                    node.name, _DEPTH, through_comparisons=False):
                return True
        return False

    def _depends_on_an_untyped_result(self, name: str, depth: int, seen=None, through_comparisons: bool = True) -> bool:
        """`name`'s value calls, at any depth through the names it reads, a user function whose result has
        no type (`bearish = ta.barssince(s) < ...`, `s` from `FastMA = c ? ta.ema(f(x), n) : ...`); with
        `through_comparisons` false only through names, operators and ternaries, not a builtin's call."""
        seen = set() if seen is None else seen
        if depth <= 0 or name in seen:
            return False
        seen.add(name)
        value = self._values.get(name)
        if not isinstance(value, tuple) or value[1] is None:
            return False
        nodes = walk(value[1]) if through_comparisons else self._outside_builtin_calls(value[1])
        for node in nodes:
            if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in self._functions \
                    and self._gives_no_result_type(node.func):
                return True
            if isinstance(node, Identifier) and self._depends_on_an_untyped_result(
                    node.name, depth - 1, seen, through_comparisons):
                return True
        return False

    def _reads_an_untyped_element(self, expression, depth: int) -> bool:
        """`expression` reads, directly or through arithmetic and the names declared from it, an element of a tuple
        that has no type (`[a, b] = request.security(...)`, `[a, b] = g()`; not the numbers of `ta.macd`)."""
        if depth <= 0:
            return False
        for node in walk(expression):
            if not isinstance(node, Identifier):
                continue
            value = self._values.get(node.name)
            if value is _UNTYPED and node.name in self._tuple_elements:
                return True
            if isinstance(value, tuple) and value[0] is None and isinstance(value[1], (BinaryOp, UnaryOp, Identifier)) \
                    and not (isinstance(value[1], BinaryOp) and value[1].op in COMPARISONS) \
                    and self._reads_an_untyped_element(value[1], depth - 1):
                return True
        return False

    def reads_an_untyped_result(self, expression) -> bool:
        """Whether `expression` reads a name holding a user function's result that has no type, but not through
        a builtin whose result has a type of its own (`time(tf)` is an int whatever `tf` is)."""
        return any(isinstance(node, Identifier) and self._holds_an_untyped_result(node.name, _DEPTH)
                   for node in _outside_typed_calls(expression))

    def _holds_an_untyped_result(self, name: str, depth: int) -> bool:
        """`name` is declared (typed or not) from a call of a user function whose result has no
        type, from a name that is, or from arithmetic over one: a function of several lines, or of one line that
        is a call (`nz(a)`, `ta.sma(a, 3)`, but not `iff(...)`). One of one line that is arithmetic or a ternary
        is typed."""
        value = self._values.get(name)
        if depth <= 0 or not isinstance(value, tuple) or name in self._retyped:
            return False  # a name reassigned another value is typed from it (`m = f()`, `m := m * 2`)
        value = value[1]
        while isinstance(value, IndexAccess):
            value = value.object  # its history (`donchian(n)[k]`)
        if isinstance(value, Identifier):
            return self._holds_an_untyped_result(value.name, depth - 1)
        if isinstance(value, (BinaryOp, UnaryOp)) and not (isinstance(value, BinaryOp) and value.op in COMPARISONS):
            # arithmetic over such a name (`b = t + 1`) or over such a call (`b = f(x) * 2`); a builtin call over one
            # (`math.avg(t, k)`) is typed
            return any(isinstance(node, Identifier) and self._holds_an_untyped_result(node.name, depth - 1)
                       for node in walk(value)) or any(
                isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in self._functions
                and self._gives_no_result_type(node.func) for node in _arithmetic_operands(value))
        if not (isinstance(value, FunctionCall) and isinstance(value.func, str) and value.func in self._functions):
            return False
        return self._gives_no_result_type(value.func)

    def _gives_no_result_type(self, function: str) -> bool:
        """A user function whose result has no type: one whose parameters are all untyped, of several
        lines or of one line that is a call other than `iff` and a user function's. A typed parameter, or none at
        all, types the result (`f(float x) => ta.sma(x, 3)`, `k() => time(...)`)."""
        declaration = self._functions[function]
        body = declaration.body
        if not declaration.params or any(parameter.type_hint for parameter in declaration.params):
            # typed parameters (or none): the result has the type of the closing call, or arithmetic beside a typed
            # parameter; not of a name (`s` of `s = 0.0`, then a loop adding to it) or a user function's call
            last = body[-1].expr if isinstance(body, list) and body and isinstance(body[-1], ExpressionStatement) \
                else (None if isinstance(body, list) else body)
            if last is None or isinstance(last, (Identifier, IndexAccess)):
                return isinstance(body, list)
            if isinstance(last, FunctionCall):
                return isinstance(last.func, str) and last.func in self._functions
            # arithmetic closing several lines is typed beside a typed parameter only (`f() => s = close`, `s * 2`
            # gives no type)
            return isinstance(body, list) and not declaration.params
        return isinstance(body, list) or (isinstance(body, FunctionCall) and isinstance(body.func, str)
                                          and body.func != 'iff' and body.func not in self._functions)

    _locals: set = set()

    def is_a_hoisted_local_change(self, expression) -> bool:
        """Pine v4/v5, inside a function: `ta.change(x)` of a parameter or local of the function, as a later operand
        of an `and`/`or` in an if/while header, is computed first into a temporary that has no type
        (`change_1 = ta.change(x)`, `if x > 1 and change_1`), so it is not cast; of a global it is, and so is the
        temporary in a ternary's condition (`q and ta.change(q) ? 1 : 0`)."""
        return (self._version <= 5 and self._in_function and isinstance(expression, FunctionCall)
                and expression.func == 'ta.change' and bool(expression.args)
                and isinstance(expression.args[0], Identifier) and expression.args[0].name in self._locals)

    def has_a_type(self, expression, header: bool = False, colours: bool = False) -> bool:
        """False for a parameter, beside which no `and`/`or` operand is cast; in an if/while header also for
        a name declared without a type from a call that has no type (`t = timestamp(...)`, Pine v5
        `u = (time(tf, s))`): `if t and k` casts neither, `x = k and t` casts `k`."""
        if isinstance(expression, Identifier) and expression.name in self._parameters:
            return False
        if (header or colours) and isinstance(expression, Identifier) and expression.name in self._title_first_inputs:
            return False  # Pine v4 `input(title = "T", defval = false)`: read as a string, in a header or colour test
        declared = self._values.get(expression.name) if header and isinstance(expression, Identifier) else None
        return not (isinstance(declared, tuple) and declared[0] is None and isinstance(declared[1], FunctionCall)
                    and (declared[1].func == 'timestamp' or self._is_an_untyped_call(declared[1])))

    def _is_an_untyped_call(self, call: FunctionCall) -> bool:
        """Pine v5: a built-in call outside a namespace written in parentheses as a name's value
        (`u = (time(tf, s))`, `v = (nz(x))`) gives the name no type; one of a namespace keeps it
        (`(ta.sma(x, 3))`, `(math.max(a, b))`)."""
        return self._version == 5 and getattr(call, 'parenthesized', False) and isinstance(call.func, str) \
            and '.' not in call.func and call.func not in self._functions

    def is_led_by_a_cast(self, expression) -> bool:
        """An `and`/`or` whose first operand is cast there: a pivot call or a number name (`n and c`)."""
        if not (isinstance(expression, BinaryOp) and expression.op in ('and', 'or')):
            return False
        first = expression.left
        while isinstance(first, BinaryOp) and first.op == expression.op and not first.parenthesized:
            first = first.left
        if isinstance(first, FunctionCall) and isinstance(first.func, str) and first.func in PIVOTS:
            return True
        return isinstance(first, (Identifier, IndexAccess)) and self.is_number(first)

    def has_no_type_in_a_chain(self, expression, header: bool = False) -> bool:
        """An `and`/`or` operand beside which no operand of its chain is cast: a name (or its history) of the
        script's top level declared as a ternary whose condition reads a user function's result that has no
        type (`u = close > k ? 1 : 0` with `k = f(close)`). A name holding such a result itself (`b = f(close)`)
        does not stop the others' casts."""
        if header and not self._in_function and isinstance(expression, BinaryOp) and expression.op in COMPARISONS \
                and expression.op not in ('and', 'or') and any(
                    isinstance(operand, Identifier) and self._holds_only_untyped_results(operand.name)
                    for operand in (expression.left, expression.right)):
            return True  # in an if/while header a comparison of such a result (`mom > 0`, `mom = f(close)`)
        while isinstance(expression, IndexAccess):
            expression = expression.object
        declared = self._values.get(expression.name) if isinstance(expression, Identifier) else None
        return not self._in_function and isinstance(declared, tuple) and isinstance(declared[1], TernaryOp) \
            and self.reads_an_untyped_result(declared[1].condition)

    def _holds_only_untyped_results(self, name: str) -> bool:
        """`name` holds a user function's result that has no type, or is declared and reassigned only names
        holding one (`m = m1`, `m := m2`)."""
        if self._holds_an_untyped_result(name, _DEPTH):
            return True
        declared = self._values.get(name)
        if not isinstance(declared, tuple):
            return False
        values = [declared[1]] + [node.value for node in self._top_reassignments if node.target == name]
        return all(isinstance(value, Identifier) and self._holds_an_untyped_result(value.name, _DEPTH)
                   for value in values)

    def is_reassigned_a_tuple_value(self, expression) -> bool:
        """A top-level name reassigned a value computed (through declared names, arithmetic, comparisons) from an
        element of a user function's tuple (`p := c` with `c = d > 0`, `d = a - b`, `[a, b] = f()`): in a
        `plotshape` series it has no type, and no operand of its `and`/`or` chain is cast."""
        if self._in_function or not isinstance(expression, Identifier):
            return False
        return any(isinstance(node, Reassignment) and node.target == expression.name
                   and self._reads_a_tuple_element(node.value, 8) for node in self._top_reassignments)

    def _reads_a_tuple_element(self, expression, depth: int) -> bool:
        if depth <= 0:
            return False
        for node in walk(expression):
            if isinstance(node, Identifier):
                declared = self._global_values.get(node.name)
                if declared is _UNTYPED:
                    return True
                if isinstance(declared, tuple) and declared[1] is not None and not isinstance(declared[1], Identifier) \
                        and self._reads_a_tuple_element(declared[1], depth - 1):
                    return True
        return False

    def is_a_bool_nz(self, expression) -> bool:
        """`nz(x, false)`: a bool, whose number `x` is cast where the call is read as a condition."""
        return isinstance(expression, FunctionCall) and expression.func == 'nz' and len(expression.args) == 2 \
            and not expression.kwargs and (
                (isinstance(expression.args[1], Literal) and expression.args[1].literal_type == 'bool'
                 and not expression.args[1].value)
                or (isinstance(expression.args[1], Identifier) and expression.args[1].name == 'false'))

    def is_declared_from_a_direct_cast(self, expression) -> bool:
        """`time(...)` itself, or a name declared from a call cast wherever it is written as a condition: `time(...)`,
        `time_close(...)`, `ta.dev(...)`, a pivot (`s = time(tf, sess)`), or the change of a number
        (`s = ta.change(time('D'))`)."""
        if isinstance(expression, FunctionCall) and isinstance(expression.func, str) \
                and expression.func in ('time', 'time_close'):
            return True  # written there (`time(tf, sess) or x`)
        declared = self._values.get(expression.name) if isinstance(expression, Identifier) else None
        return isinstance(declared, tuple) and isinstance(declared[1], FunctionCall) \
            and not self._is_an_untyped_call(declared[1]) and (
            declared[1].func in DIRECT_CASTS
            or (declared[1].func == 'ta.change' and self._kind(declared[1], False, _DEPTH) == 'number'))

    def has_a_later_stateful_operand(self, chain: BinaryOp) -> bool:
        """Whether an operand of the `and`/`or` chain after its first is a call computed first (Pine v4/v5
        evaluate `and`/`or` strictly): a `ta.*` or `request.*` call, or a user function's."""
        operands, pending = [], [chain]
        while pending:
            node = pending.pop()
            if isinstance(node, BinaryOp) and node.op == chain.op and (node is chain or not node.parenthesized):
                pending.extend([node.right, node.left])
            else:
                operands.append(node)
        return any(isinstance(operand, FunctionCall) and isinstance(operand.func, str)
                   and (operand.func.startswith(('ta.', 'request.')) or operand.func in self._functions)
                   for operand in operands[1:])

    def is_declared_as_a_name(self, name: str) -> bool:
        """`name` is declared without a type as another name (`b = a`), which gives it no type of its own."""
        declared = self._values.get(name)
        return isinstance(declared, tuple) and declared[0] is None and isinstance(declared[1], Identifier) \
            and declared[1].name not in ('true', 'false')

    def is_number_value(self, expression) -> bool:
        """Whether `expression` is typed a number where it is assigned (`0`, `c ? 1 : 0`)."""
        return self._kind(expression, False, _DEPTH) == 'number'

    def is_bool(self, expression) -> bool:
        """Whether `expression` is a bool where it is compared (`flag == 1`, `flag[1] == 0`); the history of a
        name whose value reads a user function's result that has no type is not; nor is any in Pine v6,
        where nothing compared with a bool is cast."""
        if self._version >= 6:
            return False
        if isinstance(expression, Identifier) and expression.name in self._title_first_inputs:
            return False
        if isinstance(expression, IndexAccess):
            declared = self._values.get(expression.object.name) if isinstance(expression.object, Identifier) else None
            if not isinstance(declared, tuple) or (declared[1] is not None
                                                   and self._derives_from_an_untyped_result(declared[1])):
                return False  # `g = b and x > fast` with `fast` from a function whose result has no type: untyped
        if self.reads_an_untyped_value(expression):
            return False
        return self._kind(expression, False, _DEPTH) == 'bool'

    def reads_an_untyped_value(self, expression) -> bool:
        """Whether `expression`, or the value of the name it is, reads a user function's result that has
        no type (`a and f(x) < 5` with a one-line `f(x) => security(...)`): it has no type then."""
        declared = self._values.get(expression.name) if isinstance(expression, Identifier) else None
        return self.reads_an_untyped_result(expression) or (
            isinstance(declared, tuple) and self.reads_an_untyped_result(declared[1]))

    def is_plain_bool(self, expression, depth: int = _DEPTH) -> bool:
        """Whether `expression` is typed a bool where `plotshape` plots it: a bool literal or
        builtin, a comparison, a `not`, an `and`/`or` by its first operand, a ternary of bool arms, or
        a name declared from one (`bool x = na` too) or from the history of one (`b = a[1]`); a history read
        written in place, a user function's result, a value that starts below its name or a ternary that continues
        on the next line is not."""
        if depth <= 0 or expression is None:
            return False
        depth -= 1
        if isinstance(expression, Literal):
            return expression.literal_type == 'bool'
        if isinstance(expression, Identifier):
            if expression.name in ('true', 'false'):
                return True
            if expression.name in self._values and expression.name not in self._multiline:
                declared = self._values[expression.name]
                if declared is _UNTYPED and expression.name in self._tuple_sources:
                    return self._element_kind(expression.name, depth) == 'bool'
                if isinstance(declared, tuple) and declared[0] == 'bool' \
                        and not isinstance(declared[1], (NaLiteral, type(None))):
                    # `bool x = <value>` is plain by its value (`bool x = c ? false : f()` is not); `bool x = na` is
                    return not isinstance(declared[1], Identifier) and self.is_plain_bool(declared[1], depth)
                value, _ = self._declared_value(expression.name, False)
                if isinstance(value, IndexAccess) and isinstance(value.object, Identifier):
                    # a name declared as the history of a bool name (`b = a[1]`) is a bool; the history written
                    # in place is not
                    return self.is_plain_bool(value.object, depth) \
                        and not self._depends_on_an_untyped_result(value.object.name, _DEPTH)
                first = value
                while isinstance(first, BinaryOp) and first.op in ('and', 'or'):
                    first = first.left
                if first is not value and isinstance(first, IndexAccess) and isinstance(first.object, Identifier):
                    # so is one declared as an `and`/`or` led by such a history (`f = a[1] and b`), unless that
                    # name is computed, at any depth, from a user function's result that has no type
                    return self.is_plain_bool(first.object, depth) \
                        and not self._depends_on_an_untyped_result(first.object.name, _DEPTH)
                # a name declared as another name (`buy_plot = buy`) is not typed through it
                return not isinstance(value, Identifier) and self.is_plain_bool(value, depth)
            return False
        if isinstance(expression, UnaryOp):
            return expression.op == 'not'
        if isinstance(expression, BinaryOp):
            if expression.op in ('and', 'or'):
                # by its first operand, but not where an operand is a bool builtin over a user function's result
                # that has no type (`a > b and ta.cross(close, fh)`, `fh = f()`)
                # a first operand that is cast there makes the chain a bool (`ta.pivotlow(3, 3) and c`, `n and c`)
                first = expression.left
                first_is_bool = self.is_plain_bool(first, depth) or self.is_led_by_a_cast(expression)
                return first_is_bool and not any(
                    isinstance(operand, FunctionCall) and operand.func in BOOL_BUILTINS
                    and any(self._derives_from_an_untyped_result(argument) for argument in operand.args)
                    for operand in (expression.left, expression.right))
            # a comparison, but not one of a name holding a user function's result that has no type (`h > o`
            # with `h = f(close)`); a builtin's result over one is typed (`ta.valuewhen(c, high, 1) < x`)
            return expression.op in COMPARISONS and not any(
                isinstance(operand, Identifier) and self._holds_an_untyped_result(operand.name, _DEPTH)
                for operand in (expression.left, expression.right))
        if isinstance(expression, TernaryOp):
            arms = [arm for arm in (expression.true_expr, expression.false_expr) if not isinstance(arm, NaLiteral)]
            return bool(arms) and all(self.is_plain_bool(arm, depth) for arm in arms)
        if isinstance(expression, FunctionCall) and expression.func == 'input':  # `input(false, ...)`
            default = expression.args[0] if expression.args else expression.kwargs.get('defval')
            return isinstance(default, Literal) and default.literal_type == 'bool' or (
                isinstance(default, Identifier) and default.name in ('true', 'false'))
        return isinstance(expression, FunctionCall) and isinstance(expression.func, str) \
            and (expression.func in BOOL_BUILTINS or expression.func == 'input.bool') \
            and not any(self._derives_from_an_untyped_result(argument) for argument in expression.args)

    def is_a_typed_element(self, expression) -> bool:
        """A tuple element (`[a, b] = f()`) whose element of `f`'s tail has a type, a bool or a number: where a
        bool may be na `plotshape(c ? a : na)` plots it as a bool that keeps its na."""
        return isinstance(expression, Identifier) and self._values.get(expression.name) is _UNTYPED \
            and expression.name in self._tuple_sources and self._element_kind(expression.name, _DEPTH) is not None

    def _element_kind(self, name: str, depth: int) -> Optional[str]:
        """The kind of the element of its function's tail a tuple element is bound to: a local declared from a
        literal, a comparison or arithmetic (`a = false`, `h = close - open`), or such a value written there
        (`close > open`); none for one that reads an untyped parameter (`n > 2`)."""
        function, element = self._tuple_sources[name]
        untyped = {parameter.name for parameter in function.params if not parameter.type_hint}
        if any(isinstance(node, Identifier) and node.name in untyped for node in walk(element)):
            return None
        outer, in_function = self._values, self._in_function
        self._values = {**self._global_values, **self._function_values.get(_scope_key(function), {})}
        self._in_function = True
        try:
            return self._kind(element, False, depth)
        finally:
            self._values, self._in_function = outer, in_function

    def _derives_from_an_untyped_result(self, expression, depth: int = 8) -> bool:
        """Whether `expression` reads, through the names it is computed from (arithmetic, history reads, builtin
        calls), a user function's result that has no type (`maxC = max(a[2], b[2])`, `a = avg(x, y)`,
        `x = donchian(n)` with `donchian(len) => avg(...)`)."""
        if depth <= 0:
            return False
        for node in walk(expression):
            if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in self._functions \
                    and self._gives_no_result_type(node.func):
                return True  # `fh = c ? f(x) : na` with `f(x) => request.security(...)`
            if isinstance(node, Identifier) and node.name in self._values:
                if self._holds_an_untyped_result(node.name, _DEPTH):
                    return True
                declared = self._values[node.name]
                if isinstance(declared, tuple) and declared[1] is not None \
                        and self._derives_from_an_untyped_result(declared[1], depth - 1):
                    return True
        return False

    def is_pivot(self, expression) -> bool:
        """`ta.pivothigh(...)`/`ta.pivotlow(...)`, or a name declared as one."""
        for _ in range(_DEPTH):
            if isinstance(expression, Identifier) and expression.name in self._values:
                expression, _ = self._declared_value(expression.name, False)
                continue
            break
        return isinstance(expression, FunctionCall) and expression.func in PIVOTS

    # ------------------------------------------------------------ typing

    def _declared_value(self, name: str, header: bool):
        """The value `name` is typed from, and whether an arm that is na leaves it untyped."""
        value = self._values.get(name)
        if value is _UNTYPED or value is _UNTYPED_PARAMETER:
            return None, False
        if not isinstance(value, tuple):
            return value, False
        type_hint, value = value
        if type_hint == 'bool':  # a number given to a bool declaration is cast there (`bool b = ta.change(t)`)
            return Literal(value=True, literal_type='bool'), False
        if type_hint in ('int', 'float') and (header or isinstance(value, NaLiteral) or value is None):
            # an if/while header reads the declared type, as does every context for `float x = na`
            return Literal(value=0 if type_hint != 'bool' else True, literal_type=type_hint), False
        # elsewhere a typed declaration is typed from its value, where a na arm leaves it untyped
        return value, type_hint is not None

    def _kind(self, value, header: bool, depth: int, strict_na: bool = False) -> Optional[str]:
        if depth <= 0 or value is None:
            return None
        depth -= 1
        if isinstance(value, Literal):
            if value.literal_type in ('int', 'float'):
                return 'number'
            return 'bool' if value.literal_type == 'bool' else None
        if isinstance(value, NaLiteral):
            return None
        if isinstance(value, Identifier):
            if value.name in ('true', 'false'):
                return 'bool'
            if value.name in NUMERIC_SERIES or value.name in STRATEGY_NUMBERS:
                return 'number'
            if value.name in self._block_values:
                return self._block_kind(value.name, header, depth)
            if not self._in_function and value.name in self._set_under_a_called_condition:
                return None  # `os := 1` under `if bull`, `bull` calling a user function: no type
            if self._flow and not self._in_function and value.name in self._untyped_reassigned:
                return None  # `if s` with `s := close > k ? 1 : -1`, `k` untyped; `s ? 1 : 0` is cast
            if value.name in self._values:
                if not self._argument and not self._in_function and self._holds_an_untyped_result(value.name, _DEPTH):
                    # a global holding a user function's result that has no type (`m = f()` of a function of
                    # several lines), but as a builtin's condition argument
                    return None
                declared, strict = self._declared_value(value.name, header)
                if isinstance(declared, FunctionCall) and self._is_an_untyped_call(declared):
                    return None  # `u = (time(tf, s))`; `(time(tf, s))` written in a condition is cast
                if self._in_function and isinstance(declared, Identifier) and declared.name not in ('true', 'false') \
                        and declared.name not in NUMERIC_SERIES:
                    return None  # inside a function a local declared as another local (`b = a`) has no type
                return self._kind(declared, header, depth, strict)
            return None
        if isinstance(value, IndexAccess):
            if self._in_function and isinstance(value.object, Identifier) and value.object.name in self._values:
                declared, _ = self._declared_value(value.object.name, header)
                if isinstance(declared, Identifier) and declared.name not in ('true', 'false'):
                    return None  # inside a function the history of a local declared as another name (`h = high`,
                    # `h[2]`) has no type; `h` itself is a number
            return self._kind(value.object, header, depth)
        if isinstance(value, UnaryOp):
            if value.op == 'not':
                return 'bool'
            return self._kind(value.operand, header, depth)
        if isinstance(value, BinaryOp):
            if value.op in COMPARISONS:
                return 'bool'
            if value.op in ARITHMETIC:
                kinds = {self._kind(value.left, header, depth), self._kind(value.right, header, depth)}
                return 'number' if kinds == {'number'} else None
            return None
        if isinstance(value, TernaryOp):
            if not self._argument and not self._in_function and (
                    self.reads_an_untyped_result(value.condition) or (
                        not self._flow and self._condition_depends_on_an_untyped_result(value.condition))):
                # `x = close >= k ? 1 : 0` with `k = f()` untyped: x has no type either, but as a builtin's
                # condition argument; so where the condition reaches such a result at any depth
                # (`r = e or e[1] ? n : 0`, `e = c and d`, `d = x > y and f(c[1])`)
                return None
            if isinstance(value.true_expr, TernaryOp):
                return None  # `c ? d ? 1 : 0 : 0`, a ternary in the first arm, has no type; `c ? 0 : d ? 1 : 0` has
            arms = [value.true_expr, value.false_expr]
            if strict_na and any(isinstance(arm, NaLiteral) for arm in arms):
                return None  # `float x = c ? low : na` read outside an if/while header
            if self._version == 5 and any(isinstance(arm, NaLiteral) for arm in arms):
                # Pine v5: `c ? e : na` with `e` a tuple element (`[e, u] = f()`, a library's too) is a number
                arms = [Literal(value=0.0, literal_type='float') if isinstance(arm, Identifier)
                        and self._values.get(arm.name) is _UNTYPED and arm.name in self._tuple_elements else arm
                        for arm in arms]
            kinds = {self._kind(arm, header, depth, strict_na) for arm in arms if not isinstance(arm, NaLiteral)}
            return kinds.pop() if len(kinds) == 1 else None
        if isinstance(value, (IfExpression, SwitchExpression)):
            return None
        # `x.timeSess(...)` is a MethodCall, or a FunctionCall named `close.timeSess` on a dotted receiver
        method = value.method if isinstance(value, MethodCall) else (
            value.func.rpartition('.')[2] if isinstance(value, FunctionCall) and isinstance(value.func, str)
            and '.' in value.func else None)
        if method == 'timeSess' and self._as_a_condition and method in self._functions \
                and getattr(self._functions[method], 'is_method', False):
            # the result of a user method named `timeSess` is typed as a time, whatever its body: a number
            # where it is read as a condition (`a = "UTC".timeSess("0300-0400")`, `a and not a[1]`); given to a
            # bool (`active := a`) it is not cast. Any other method's result has no type
            return 'number'
        if isinstance(value, FunctionCall):
            return self._call_kind(value, header, depth)
        return None

    def _block_kind(self, name: str, header: bool, depth: int) -> Optional[str]:
        """An if/switch value: a number where every value it takes is `c ? <arithmetic> : na` of numbers
        (`p = if on` / `d ? close * 1.01 : na`); any other value (`close`, `close * 1.0`, `c ? close : na`) leaves it
        untyped."""
        values = self._block_values[name]
        for value in values:
            if not isinstance(value, TernaryOp):
                return None
            arms = [arm for arm in (value.true_expr, value.false_expr) if not isinstance(arm, NaLiteral)]
            if len(arms) != 1 or not _is_arithmetic(arms[0]) or self._kind(arms[0], header, depth) != 'number':
                return None
        return 'number' if values else None

    def _call_kind(self, call: FunctionCall, header: bool, depth: int) -> Optional[str]:
        function = call.func if isinstance(call.func, str) else None
        if function is None:
            return None
        if function == 'bool' and len(call.args) == 1:
            # `bool(x)` is typed as `x`: `p = bool(ta.pivothigh(3, 3))` is a number, cast where it is a condition
            return self._kind(call.args[0], header, depth)
        if function in BOOL_BUILTINS or function in ('input.bool', 'bool'):
            return 'bool'
        if function in ('request.security', 'security'):
            # the type of its expression (`request.security(s, tf, close)` is a number); a history read there
            # (`high[1]`) or a user function's result that has no type leaves it untyped
            expression = call.args[2] if len(call.args) > 2 else call.kwargs.get('expression')
            if expression is None or isinstance(expression, IndexAccess) \
                    or self._derives_from_an_untyped_result(expression):
                return None
            return self._kind(expression, header, depth)
        if function.split('.')[0] in self._library_aliases and '.' in function:
            # a script is typed without the libraries it imports: a library function's result is a number
            # here, cast in every condition but under `not`, an if/while header too
            return 'number'
        if function == 'iff' and len(call.args) == 3:  # Pine v4 `iff(c, a, b)` is the ternary `c ? a : b`
            return self._kind(TernaryOp(condition=call.args[0], true_expr=call.args[1], false_expr=call.args[2]),
                              header, depth)
        if function.startswith(('math.', 'ta.')) and not self._argument and not self._in_function \
                and any(self.reads_an_untyped_result(argument)
                        or (self._flow and not header and self._reads_an_untyped_element(argument, _DEPTH))
                        for argument in call.args):
            # `math.avg(t, k)` of untyped function results has no type either, nor, as an `and`/`or` operand,
            # `ta.change(s)` of an untyped tuple element (`[s, o] = request.security(..., f())`, `[s, o] = g()`)
            return None
        if function in NUMERIC_CALLS or function.startswith('math.'):
            return 'number'
        if self.is_a_bool_nz(call):
            return 'bool'
        if function in PASS_THROUGH_CALLS:
            return self._kind(call.args[0], header, depth) if call.args else None
        if function == 'ta.change' and call.args:
            if self._flow and not self._in_function and isinstance(call.args[0], Identifier) \
                    and call.args[0].name in self._untyped_reassigned:
                return None  # the change of a name reassigned an untyped value has no type
            changed = self._kind(call.args[0], header, depth)
            if changed == 'bool':
                return 'bool'  # the change of a bool is a bool
            if isinstance(call.args[0], Identifier) and call.args[0].name in _UNTYPED_STRATEGY_COUNTS \
                    and (self._in_function or not header):
                # the change of a built-in that has no type (`ta.change(strategy.losstrades)`) has none,
                # but in an if/while header of the script's own scope
                return None
        if function.startswith('ta.'):
            return 'number'
        if function == 'input':
            default = call.args[0] if call.args else call.kwargs.get('defval')
            return self._kind(default, header, depth)
        if function in self._functions:
            if header:
                return None  # a user function's result has no type in an if/while header
            if self._in_function:
                # inside a function no function's result type is read: a call is a number when an argument
                # is arithmetic with a negation (`f(x, -1.0)`, `f(v, -1.0 - m)`) or of numbers (`f(x, 0 - 1)`,
                # `f(r, 1.2 * e)` of a typed `e`), else untyped (`f(r, 1.2 * e)` of an untyped parameter `e`); as a
                # builtin's condition argument (`ta.barssince(d)`) it is untyped either way
                if self._argument:
                    return None
                arguments = list(call.args) + list(call.kwargs.values())
                return 'number' if any(_is_arithmetic(argument) and (
                    any(isinstance(node, UnaryOp) and node.op == '-' for node in _arithmetic_operands(argument))
                    or self._kind(argument, header, depth) == 'number') for argument in arguments) else None
            declaration = self._functions[function]
            body = declaration.body
            tail = body[-1] if isinstance(body, list) and body else body
            tail = getattr(tail, 'expr', tail)
            # the tail reads the callee's names: its typed parameters are their type (`v * 2`)
            outer = self._values
            self._values = {**self._global_values, **self._function_values.get(_scope_key(declaration), {}),
                            **{parameter.name: (parameter.type_hint, NaLiteral()) for parameter in declaration.params
                               if parameter.type_hint in ('int', 'float', 'bool')}}
            try:
                return self._kind(tail, header, depth)
            finally:
                self._values = outer
        return None
