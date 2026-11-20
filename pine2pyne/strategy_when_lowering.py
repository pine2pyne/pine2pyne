"""`when=` of a strategy call, lowered to an `if`.

    strategy.entry("l", strategy.long, when=c)      if c
                                              ->        strategy.entry("l", strategy.long)

Pine v1-v5 gave the order functions of `strategy.*` a `when` argument: the call acts only on the bars where it is
true. Pine v6 removed it, and the runtime's `strategy.entry` / `strategy.close` / ... take none, so a `when=` that
reached the generated code raised `TypeError: entry() got an unexpected keyword argument 'when'` at run. The
lowering runs at the end of parsing, so no later stage sees one.

Only a call that stands as a statement (the functions return nothing), and only in a script declaring //@version
1-5. A v6 script is left as written: TradingView refuses a `when` there, so no spelling is invented for it. A
positional `when` is lowered the same way, the arguments after it moving up one slot: the 9th of `strategy.entry`
and `strategy.order`, the 2nd of `strategy.close` and `strategy.cancel`, the 1st of `strategy.close_all` and
`strategy.cancel_all`; a positional `when` of `strategy.exit` is not. A string in that slot is the
`comment` of the later signatures (`strategy.close("L", "stop hit")`) and stays where it is.
"""
import dataclasses
import itertools

from .ast_nodes import (Assignment, BinaryOp, ExpressionStatement, FuncDecl, FunctionCall, Identifier, IfStatement,
                        IndexAccess, Literal, MethodCall, Script)
from .ast_walk import child_field_names, contains, is_node

STRATEGY_FUNCTIONS_WITH_WHEN = frozenset({
    'strategy.entry', 'strategy.order', 'strategy.exit', 'strategy.close', 'strategy.close_all',
    'strategy.cancel', 'strategy.cancel_all',
})


def lower_strategy_when_arguments(script: Script, pine_version: int) -> Script:
    """A copy of `script` with each `strategy.*(..., when=c)` statement turned into `if c` around the call
    without `when`. A script without one, or of another version, is returned unchanged."""
    if not 1 <= pine_version <= 5 or not contains(script, _is_call_with_when):
        return script
    return _WhenLowering(_stateless_functions(script)).lowered(script)


#: the slot of a positional `when` (Pine v1-v5 signatures)
POSITIONAL_WHEN = {'strategy.entry': 8, 'strategy.order': 8, 'strategy.close': 1, 'strategy.cancel': 1,
                   'strategy.close_all': 0, 'strategy.cancel_all': 0}


def _is_call_with_when(node) -> bool:
    # a callee that is not a name (`na(x)`: a NaLiteral node) is unhashable and is never a strategy function
    if not (isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in STRATEGY_FUNCTIONS_WITH_WHEN):
        return False
    if 'when' in node.kwargs:
        return True
    slot = POSITIONAL_WHEN.get(node.func)
    return slot is not None and len(node.args) > slot and not _is_text(node.args[slot])


def _is_text(node) -> bool:
    """A string by its syntax: a literal, a concatenation with one, or a `str.*` call."""
    if isinstance(node, Literal):
        return node.literal_type == 'string'
    if isinstance(node, BinaryOp) and node.op == '+':
        return _is_text(node.left) or _is_text(node.right)
    return isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func.startswith(('str.', 'tostring'))


def _with_keyword_when(call: FunctionCall) -> FunctionCall:
    """`call` with a positional `when` passed by keyword."""
    if 'when' in call.kwargs:
        return call
    slot = POSITIONAL_WHEN[call.func]
    return dataclasses.replace(call, args=call.args[:slot] + call.args[slot + 1:],
                               kwargs={**call.kwargs, 'when': call.args[slot]})


#: calls without state of their own: an argument made of them is computed inside the `if`
_PURE_NAMESPACES = ('str.', 'string.', 'math.', 'color.')
_PURE_LEGACY_NAMES = frozenset({'tostring', 'tonumber', 'abs', 'max', 'min', 'round', 'pow', 'sqrt', 'log',
                                'log10', 'exp', 'sign', 'floor', 'ceil', 'avg', 'nz', 'na', 'int', 'float',
                                'bool', 'string', 'color', 'round_to_mintick', 'timestamp'})


def _is_stateful_call(node, stateless_functions=frozenset()) -> bool:
    """A method call, or a call of a function with state of its own (`ta.*`, a user function that keeps any)."""
    if isinstance(node, MethodCall):
        return True
    if not isinstance(node, FunctionCall):
        return False
    if not isinstance(node.func, str):
        return True
    pure = node.func in _PURE_LEGACY_NAMES or node.func in stateless_functions or (
        node.func.startswith(_PURE_NAMESPACES) and node.func not in ('math.sum', 'math.random'))
    return not pure


def _is_stateful_or_history(node, stateless_functions=frozenset()) -> bool:
    """A stateful call, or a history index of anything but a name (`ta.sma(close, 3)[1]`): a name's past
    value (`x[1]`) reads the same inside an `if`."""
    if isinstance(node, IndexAccess):
        return not isinstance(node.object, Identifier)
    return _is_stateful_call(node, stateless_functions)


def _stateless_functions(script: Script) -> frozenset:
    """The user functions without state of their own: no `var`, no history index and no stateful call, through
    the functions they call. Called only on some bars, they give what they give on every bar."""
    from .ast_nodes import VarDecl, VaripDecl
    from .ast_walk import walk
    functions = {node.name: node for node in walk(script) if isinstance(node, FuncDecl)}
    stateless: set = set()
    changed = True
    while changed:
        changed = False
        for name, function in functions.items():
            if name in stateless:
                continue
            if not any(isinstance(node, (VarDecl, VaripDecl, IndexAccess)) or _is_stateful_call(node, stateless)
                       for node in walk(function.body)):
                stateless.add(name)
                changed = True
    return frozenset(stateless)


class _Statements(list):
    """The statements one statement is replaced by (a list item that expands to several)."""


class _WhenLowering:
    """Pine evaluates every argument of a `strategy.*(..., when=c)` call on every bar, whatever `c` is. An argument
    holding a call with state (`ta.lowest(low, 10)`, a user function that keeps any) or the history of an expression
    (`ta.sma(close, 3)[1]`) has state per bar, and inside an `if` it would update only on the bars where `c` holds.
    So such an argument is assigned to a temporary before the `if`, and the temporary is what the call inside the
    `if` receives. A name's history (`x[1]`) and a stateless user function stay inside."""

    def __init__(self, stateless_functions=frozenset()):
        self.temporary_numbers = itertools.count(1)
        self.stateless_functions = stateless_functions

    def lowered(self, node):
        children = {name: self._map(getattr(node, name)) for name in child_field_names(type(node))}
        rebuilt = dataclasses.replace(node, **children)
        if isinstance(rebuilt, ExpressionStatement) and _is_call_with_when(rebuilt.expr):
            return _Statements(self._if_around(rebuilt))
        if isinstance(rebuilt, FuncDecl) and _is_call_with_when(rebuilt.body):
            # `f(c) => strategy.close("l", when=c)`: the one-line body is the call, made a statement
            call = rebuilt.body
            statement = ExpressionStatement(expr=call, line=call.line, column=call.column)
            return dataclasses.replace(rebuilt, body=self._if_around(statement))
        return rebuilt

    def _map(self, value):
        if isinstance(value, list):
            items = []
            for item in value:
                mapped = self._map(item)
                items.extend(mapped) if isinstance(mapped, _Statements) else items.append(mapped)
            return items
        if isinstance(value, tuple):
            return tuple(self._map(item) for item in value)
        if isinstance(value, dict):
            return {key: self._map(item) for key, item in value.items()}
        return self.lowered(value) if is_node(value) else value

    def _if_around(self, statement: ExpressionStatement) -> list:
        call = _with_keyword_when(statement.expr)
        temporaries = []
        args = [self._every_bar(argument, 'argument', statement, temporaries) for argument in call.args]
        kwargs = {key: self._every_bar(value, key, statement, temporaries)
                  for key, value in call.kwargs.items() if key != 'when'}
        without_when = dataclasses.replace(call, args=args, kwargs=kwargs)
        guarded = IfStatement(condition=call.kwargs['when'], body=[dataclasses.replace(statement, expr=without_when)],
                              line=statement.line, column=statement.column)
        return [*temporaries, guarded]

    def _every_bar(self, argument, label, statement, temporaries):
        if not contains(argument, lambda node: _is_stateful_or_history(node, self.stateless_functions)):
            return argument
        name = f'_when_{label}_{next(self.temporary_numbers)}'
        temporaries.append(Assignment(target=name, value=argument, line=statement.line, column=statement.column))
        return Identifier(name=name, line=statement.line, column=statement.column)
