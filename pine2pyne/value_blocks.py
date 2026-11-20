"""Value blocks (an if or switch used as a value) lowered to statements.

A Pine ``if`` or ``switch`` used as a value (assigned, or ending a function body)
becomes statements around one temporary, numbered in source order across the
module (``__block_result__``, ``__block_result_1__``, ...)::

    ma = switch mode              __block_result__ = na
        "A" => ta.sma(close, 5)   __switch__ = mode
        "B" => ta.ema(close, 5)   if __switch__ == 'A':
                                      __block_result__ = ta.sma(close, 5)
                                  elif __switch__ == 'B':
                                      __block_result__ = ta.ema(close, 5)
                                  ma = __block_result__

A switch with a subject reads it once, into ``__switch__``; one nested ``n`` blocks
deep in its function reads it into ``__switch_<n>__``. A switch statement does too. A branch whose value is itself an ``if`` or a
``switch`` gets the next number and hands its value to the outer temporary. A
function whose body ends with an ``if`` or a ``switch`` returns the temporary.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

from .ast_nodes import (
    ArrayLiteral, Assignment, BinaryOp, BreakStatement, ExpressionStatement, ForInLoop, ForLoop, FuncDecl, FunctionCall, Identifier,
    IfExpression, IfStatement, Literal, NaLiteral, RawCode, Reassignment, Script, SwitchExpression, SwitchStatement,
    TupleDestructure, UnaryOp, VarDecl, VaripDecl, WhileLoop,
)
from .ast_walk import walk
from .block_value_lowering import yields_bool

_VALUE_NODES = (IfExpression, SwitchExpression)
_LOOPS = (ForLoop, ForInLoop, WhileLoop)
#: what may give a value: an if/switch (statement or expression) or a loop (its last pass's last statement)
_VALUE_GIVERS = (IfStatement, SwitchStatement, IfExpression, SwitchExpression) + _LOOPS


def _laid_out(statement, origin):
    """`statement` without a Pine line of its own printed as at the first line found in `origin`: the
    blank lines around it follow that line (`source_line`, read by the code generator only)."""
    if not statement.line:
        from .ast_walk import walk
        statement.source_line = next((found.line for found in walk(origin) if getattr(found, 'line', 0)), 0)
    return statement


def _is_bare_value(statement) -> bool:
    """A statement that only names a value (`x`, `na`, `false`) or computes one from names and literals without a
    call: it does nothing in a block."""
    if not isinstance(statement, ExpressionStatement):
        return False
    expr = statement.expr
    if isinstance(expr, FunctionCall) and expr.func in _CASTS and len(expr.args) == 1 and not expr.kwargs:
        expr = expr.args[0]  # `int(na)` too
    if isinstance(expr, (BinaryOp, UnaryOp)):
        # an expression of names and literals (`a and t <= end`) computes nothing either
        from .ast_nodes import IndexAccess, MemberAccess
        return all(isinstance(node, (Identifier, NaLiteral, Literal, BinaryOp, UnaryOp, IndexAccess, MemberAccess))
                   for node in walk(expr))
    return isinstance(expr, (Identifier, NaLiteral, Literal))


_CASTS = frozenset({'int', 'float', 'bool', 'string', 'color'})


def _without_bare_values(statements: list) -> list:
    """A loop's or a branch's statements without its last one when that is a bare value (the block's value, unused
    here); a bare value before it stays (`n := n + 1`, `n`, `x := true`). `pass` when nothing
    is left."""
    kept = statements[:-1] + [statement for statement in statements[-1:] if not _is_bare_value(statement)]
    if kept or not statements:
        return kept
    return [RawCode(code='pass', line=getattr(statements[0], 'line', 0))]


def _discarding_last_declaration(statements):
    """A statement block (an ``if``, ``switch`` or loop not used as a value) whose last statement
    declares a variable: only the declared value is kept, an expression statement, as a
    block's last statement is its (here unused) value. ``float x = na`` keeps ``na(float)``."""
    if not statements:
        return statements
    last = statements[-1]
    if isinstance(last, (VarDecl, VaripDecl)) or (isinstance(last, Assignment)
                                                  and isinstance(last.target, (str, TupleDestructure))):
        value = last.value
        if isinstance(value, _VALUE_NODES):
            return statements
        if isinstance(value, NaLiteral) and getattr(last, 'type_hint', None):
            value = FunctionCall(func=NaLiteral(), args=[Identifier(name=last.type_hint)],
                                 line=last.line, column=last.column)
        return statements[:-1] + [ExpressionStatement(expr=value, line=last.line, column=last.column)]
    return statements


def _statement_block(statements):
    if statements is None:
        return None
    return _discarding_last_declaration(_without_bare_values(statements))


#: user function name -> how many values its result has (0: not a tuple); set per script
_FUNCTION_WIDTHS: dict = {}
#: user function name -> the width of the `[a, b] = f(…)` that destructure it; set per script
_DESTRUCTURED_WIDTHS: dict = {}


def _tuple_width(node) -> int:
    """How many values an if/switch value yields: a tuple branch `[a, b]` gives 2, as does a call
    of a user function whose value is a 2-tuple; else 0."""
    if isinstance(node, ArrayLiteral):
        return len(node.elements)
    if isinstance(node, FunctionCall) and isinstance(node.func, str):
        return _FUNCTION_WIDTHS.get(node.func, 0)
    if isinstance(node, list):
        return _tuple_width(node[-1]) if node else 0
    if isinstance(node, ExpressionStatement):
        return _tuple_width(node.expr)
    if isinstance(node, IfExpression):
        return max(_tuple_width(node.true_expr), _tuple_width(node.false_expr) if node.false_expr is not None else 0)
    if isinstance(node, IfStatement):
        branches = [node.body] + [body for _, body in node.elseifs] + ([node.else_body] if node.else_body else [])
        return max(_tuple_width(branch) for branch in branches)
    if isinstance(node, (SwitchExpression, SwitchStatement)):
        branches = [body for _, body in node.cases] + ([node.default] if node.default is not None else [])
        return max((_tuple_width(branch) for branch in branches), default=0)
    if isinstance(node, _LOOPS):
        return _tuple_width(node.body)
    return 0


def _function_widths(script: Script) -> dict:
    """The tuple width of every user function's value, followed through calls of other functions."""
    functions = [node for node in list(script.declarations) + list(script.body) if isinstance(node, FuncDecl)]
    _FUNCTION_WIDTHS.clear()
    # a function whose value has no width of its own (its tail calls a library) is as wide as the
    # `[a, b] = f(…)` that destructure it: Pine gives a tuple function one width (not for an overload)
    _DESTRUCTURED_WIDTHS.clear()
    declared_once = {name for name in (function.name for function in functions)
                     if sum(function.name == name for function in functions) == 1}
    from .block_value_lowering import _destructured_calls
    for node in walk(script):
        if isinstance(node, Assignment) and isinstance(node.target, TupleDestructure):
            # the call itself, or each call an arm of an if/switch/ternary or a loop's body ends in
            for call in _destructured_calls(node.value):
                if call.func in declared_once:
                    _DESTRUCTURED_WIDTHS[call.func] = max(_DESTRUCTURED_WIDTHS.get(call.func, 0),
                                                          len(node.target.names))
    for _ in range(len(functions) + 1):
        changed = False
        for function in functions:
            width = _tuple_width(function.body if isinstance(function.body, list) else [function.body])
            if width and _FUNCTION_WIDTHS.get(function.name) != width:
                _FUNCTION_WIDTHS[function.name] = width
                changed = True
        if not changed:
            break
    return dict(_FUNCTION_WIDTHS)


class _Names:
    """``__<stem>__``, then ``__<stem>_1__``, ``__<stem>_2__``, ... per stem."""

    def __init__(self) -> None:
        self._counts: dict = {}

    def next(self, stem: str) -> str:
        count = self._counts.get(stem, 0)
        self._counts[stem] = count + 1
        return f'__{stem}__' if count == 0 else f'__{stem}_{count}__'


class _Lowering:
    def __init__(self, version: int = 6) -> None:
        self.version = version
        self.names = _Names()
        # how deep the statement being lowered is nested in blocks of its function (main() or a user
        # function): a switch subject is named by it, `__switch__` at 0, `__switch_<depth>__` below
        self.depth = 0
        # whether the value being lowered is of type bool: a value nested in its arms is too
        self.bool_value = False
        # Pine v5: the switches of the module counted in source order; one whose subject calls something
        # reads it first into `switch_var_<n>` (`<name>__<n>` for `name = switch ...`) and counts twice
        self.switch_count = 0
        # the name a switch value is assigned to, read by that switch only
        self.switch_target: Optional[str] = None

    def switch_subject(self) -> str:
        return '__switch__' if self.depth == 0 else f'__switch_{self.depth}__'

    # ------------------------------------------------------------ statements

    def nested(self, statements, lower):
        """`lower(statements)` one block deeper."""
        self.depth += 1
        try:
            return lower(statements)
        finally:
            self.depth -= 1

    def block(self, statements) -> list:
        if statements is None:
            return None
        lowered: List = []
        for statement in statements:
            lowered.extend(self.statement(statement))
        return lowered

    def count_mentions(self, node) -> None:
        """Counted with the switches: every other statement whose names or strings hold `switch`, in lower
        case (`switchCond = ...`, `x = "switch"`, not `mySwitch`): an `if` or a loop by its header."""
        if isinstance(node, IfStatement):
            headers = [node.condition] + [condition for condition, _ in node.elseifs]
        elif isinstance(node, (ForLoop, ForInLoop, WhileLoop)):
            headers = [dataclasses.replace(node, body=[])]
        elif isinstance(node, (SwitchStatement, FuncDecl)) or isinstance(getattr(node, 'value', None),
                                                                         _VALUE_NODES + _LOOPS):
            headers = []
        else:
            headers = [node]
        self.switch_count += sum(1 for header in headers if _mentions_switch(header))

    def statement(self, node) -> list:
        replace = dataclasses.replace
        self.count_mentions(node)
        if isinstance(node, Reassignment) and isinstance(node.value, BinaryOp) and node.value.compound \
                and isinstance(node.value.right, _VALUE_NODES):
            # `x /= switch ...`: the value is read into a block result, the operator kept (`x /= __block_result__`):
            # the block runs before `x` is read, also where the block reassigns `x` itself, as TradingView
            # computes it (`s += if c` with `s := 9.0` and a value of 2.0 inside gives 11, not 3)
            value = node.value.right
            result = self.names.next('block_result')
            is_bool = self.is_bool(value)
            initial = self._initial(result, _initial_width(value, _tuple_width(value)), node.line,
                                    self.untaken_is_false(value, is_bool))
            computing = self.value_of(value, result, is_bool)
            subject = computing[:1] if computing and getattr(computing[0], 'reads_switch_subject', False) else []
            combined = replace(node.value, right=Identifier(name=result), of_a_block_value=True)
            return subject + [initial] + computing[len(subject):] + [replace(node, value=combined)]
        if isinstance(node, (Assignment, Reassignment, VarDecl, VaripDecl)) \
                and isinstance(node.value, _VALUE_NODES + _LOOPS):
            target = getattr(node, 'target', None)
            width = len(target.names) if isinstance(target, TupleDestructure) else _tuple_width(node.value)
            result = self.names.next('block_result')
            value = node.value
            if isinstance(value, ForLoop):
                value = dataclasses.replace(value, is_assigned_value=True)
            is_bool = self.is_bool(value)
            initial = self._initial(result, _initial_width(value, width), node.line,
                                    self.untaken_is_false(value, is_bool))
            name = node.name if isinstance(node, (VarDecl, VaripDecl)) else target
            if isinstance(value, SwitchExpression) and isinstance(name, str):
                self.switch_target = name
            computing = self.value_of(value, result, is_bool)
            if isinstance(value, IfExpression):
                computing = [dataclasses.replace(statement, of_an_assigned_value=True)
                             if isinstance(statement, IfStatement) and statement.condition is value.condition
                             else statement for statement in computing]
            # a switch subject read into its own name comes first
            subject = computing[:1] if computing and getattr(computing[0], 'reads_switch_subject', False) else []
            computing = computing[len(subject):]
            if isinstance(node, (VarDecl, VaripDecl)) and isinstance(value, _LOOPS):
                # a loop given as the value of a var/varip declaration ends after its first iteration
                computing = [replace(statement, body=list(statement.body) + [BreakStatement(line=node.line)])
                             if isinstance(statement, _LOOPS) else statement for statement in computing]
            if isinstance(node, (VarDecl, VaripDecl)) and _changes_state(value):
                computing = self.first_time_only(node, computing)
            statements = subject + [initial] + computing
            lowered = replace(node, value=Identifier(name=result))
            # the value read the declared name (`m = switch m`): its outer binding, for the renamer
            lowered.reads_own_name = isinstance(name, str) and any(
                isinstance(found, Identifier) and found.name.split('.')[0] == name for found in walk(node.value))
            if isinstance(target, TupleDestructure) and isinstance(target.value, _VALUE_NODES + _LOOPS):
                # the destructure keeps its value too (parser): it reads the temporary as well
                lowered = replace(lowered, target=replace(target, value=Identifier(name=result)))
            statements.append(lowered)
            return statements
        if isinstance(node, SwitchStatement):
            return self.switch(node, None)
        if isinstance(node, IfStatement):
            return [replace(node, body=_statement_block(self.nested(node.body, self.block)),
                            elseifs=[(condition, _statement_block(self.nested(body, self.block)))
                                     for condition, body in node.elseifs],
                            else_body=_statement_block(self.nested(node.else_body, self.block))
                            if node.else_body is not None else None)]
        if isinstance(node, (ForLoop, ForInLoop, WhileLoop)):
            return [replace(node, body=_statement_block(self.nested(node.body, self.block)))]
        if isinstance(node, FuncDecl):
            return [self.function(node)]
        return [node]

    def first_time_only(self, node, computing: list) -> list:
        """`var v = if …` whose arms change state (`cnt := cnt + 1`, `a.push(x)`): Pine runs a var's initialiser
        once, the plain lowering on every bar. The arms run behind a flag the first time only; the var then
        keeps the value they gave (its initialiser is read once)."""
        flag = self.names.next(f'{node.name}_initialised')
        kind = type(node)
        declaration = kind(name=flag, value=Literal(value=False, literal_type='bool'), type_hint='bool',
                           line=node.line)
        first_time = IfStatement(condition=UnaryOp(op='not', operand=Identifier(name=flag)),
                                 body=[Reassignment(target=flag, value=Literal(value=True, literal_type='bool'))]
                                 + computing, elseifs=[], else_body=None, line=node.line)
        return [declaration, first_time]

    def function(self, node: FuncDecl) -> FuncDecl:
        saved, self.depth = self.depth, 0  # a function's body is the top of its own blocks
        try:
            lowered = self._function(node)
        finally:
            self.depth = saved
        original = node.body[-1] if isinstance(node.body, list) and node.body else node.body
        returns_if_value = isinstance(getattr(original, 'value', None), IfExpression) and isinstance(
            original, (Assignment, VarDecl, VaripDecl))
        if (self.version >= 6 or returns_if_value) and isinstance(lowered.body, list) and lowered.body:
            # Pine v6: a function ending with a declaration returns the declared value itself;
            # v5 keeps the declaration and returns its name, but for an
            # if value, whose block result it returns (not for a switch value)
            last = lowered.body[-1]
            declares = isinstance(last, (VarDecl, VaripDecl)) or (
                isinstance(last, Assignment) and isinstance(last.target, (str, TupleDestructure)))
            if declares and last.value is not None and not isinstance(last.value, _VALUE_NODES + _LOOPS):
                tail = ExpressionStatement(expr=last.value, line=last.line, column=last.column)
                lowered = dataclasses.replace(lowered, body=lowered.body[:-1] + [tail])
        return lowered

    def _function(self, node: FuncDecl) -> FuncDecl:
        body = node.body
        if isinstance(body, _VALUE_NODES):
            body = [ExpressionStatement(expr=body, line=node.line)]
        elif isinstance(body, (Assignment, Reassignment, VarDecl, VaripDecl)):
            body = [body]  # a one-line body that declares: `f(x) => float r = x + 1`
            node = dataclasses.replace(node, body=body)
        if not isinstance(body, list) or not body:
            return node
        last = body[-1]
        if isinstance(last, ExpressionStatement) and isinstance(last.expr, _VALUE_NODES):
            last = last.expr
        if not isinstance(last, _VALUE_GIVERS):
            return dataclasses.replace(node, body=self.block(body))
        lowered = self.block(body[:-1])
        result = self.names.next('block_result')
        is_bool = self.is_bool(last, node.value_is_bool)
        width = _tuple_width(last)
        if not width and _misses_an_arm(last):
            # an arm that may not run yields the start: a tuple of na as wide as the destructures (a plain `na`
            # is what `a, b = f()` cannot unpack); where an arm always runs the start is never read
            width = _DESTRUCTURED_WIDTHS.get(node.name, 0)
        initial = self._initial(result, _initial_width(last, width), getattr(body[-1], 'line', 0),
                                self.untaken_is_false(last, is_bool))
        lowered.extend(_subject_first([initial], self.value_of(last, result, is_bool)))
        lowered.append(ExpressionStatement(expr=Identifier(name=result)))  # the function's value
        return dataclasses.replace(node, body=lowered)

    # ------------------------------------------------------------ values

    @staticmethod
    def is_bool(node, given: bool = False) -> bool:
        """Whether the value of `node` is of type bool: its `value_is_bool` stamp (bool_value_typing), or
        `given`, the type of the value or function whose tail `node` is (Pine gives both one type)."""
        return given or bool(getattr(node, 'value_is_bool', False))

    def untaken_is_false(self, node, is_bool: bool) -> bool:
        """Whether the temporary of `node` starts as false, not na. Pine v6 removed na from bool: an
        if/switch of type bool gives false when none of its arms runs, and so does a loop of type bool
        that runs no pass, its tail a bool by its syntax (block_value_lowering.yields_bool) or a bool
        name. Where an arm
        always runs the start value is never read, and stays na; before v6 both
        are na."""
        if self.version < 6:
            return False
        if isinstance(node, _LOOPS):
            return is_bool or yields_bool(node)
        # an arm ending in `x := y` gives the typed bool no syntax of its own: it starts as na
        return is_bool and _misses_an_arm(node) and not _ends_in_reassignment(node)

    def value_of(self, node, result: str, is_bool: bool) -> list:
        """`value(node, result)`, the values nested in its arms typed as `node` is (`is_bool`)."""
        saved, self.bool_value = self.bool_value, is_bool
        try:
            return self.value(node, result)
        finally:
            self.bool_value = saved

    @staticmethod
    def _initial(result: str, width: int, line: int, untaken_is_false: bool = False) -> Assignment:
        if width:
            value = ArrayLiteral(elements=[NaLiteral() for _ in range(width)])
        elif untaken_is_false:
            value = Literal(value=False, literal_type='bool')
        else:
            value = NaLiteral()
        return Assignment(target=result, value=value, line=line)

    def value(self, node, result: str) -> list:
        """Statements that leave the value of an if/switch (expression or statement) in `result`."""
        if isinstance(node, (SwitchExpression, SwitchStatement)):
            return self.switch(node, result)
        if isinstance(node, _LOOPS):
            # the loop's value is its last pass's: each pass assigns its last statement's value
            return [dataclasses.replace(node, body=self.nested(node.body, lambda body: self.branch(body, result)))]
        def arm(body):
            return self.nested(body, lambda statements: self.branch(statements, result))
        if isinstance(node, IfExpression):
            # an `else if` chain is one if/elif with one block result; its blocks are numbered
            # in source order, the first arm's first (TFlab/AlertSenderLibrary)
            first = arm(node.true_expr)
            elseifs, rest = [], node.false_expr
            while isinstance(rest, IfExpression) and rest.chained:  # `else if`; an `if` inside an `else` block stays
                chained = rest
                elseifs.append((chained.condition, arm(chained.true_expr)))
                rest = chained.false_expr
            return [_laid_out(IfStatement(condition=node.condition, body=first, elseifs=elseifs,
                                          else_body=arm(rest) if rest is not None else None,
                                          line=node.line), node)]
        return [dataclasses.replace(
            node, body=arm(node.body),
            elseifs=[(condition, arm(body)) for condition, body in node.elseifs],
            else_body=arm(node.else_body) if node.else_body is not None else None)]

    def switch(self, node, result: Optional[str]) -> list:
        lowered: List = []
        conditions = [case for case, _ in node.cases]
        number, target, self.switch_target = self.switch_count, self.switch_target, None
        calls = self.version == 5 and node.expr is not None and _calls_something(node.expr)
        self.switch_count += 2 if calls else 1
        if node.expr is not None:
            subject = self.switch_subject()
            value = node.expr
            if calls:
                read = Assignment(target=f'{target}__{number}' if target else f'switch_var_{number}', value=value,
                                  line=node.line)
                read.reads_switch_subject = True  # it comes before the block result's initial value
                lowered.append(_laid_out(read, node))
                value = Identifier(name=read.target)
            lowered.append(_laid_out(Assignment(target=subject, value=value, line=node.line), node))
            conditions = [BinaryOp(left=Identifier(name=subject), op='==', right=case) for case in conditions]
        bodies = [body for _, body in node.cases]
        if result is None:
            arms = [_statement_block(self.nested(body, self.block)) for body in bodies]
            default = _statement_block(self.nested(node.default, self.block))
        else:
            arms = [self.nested(body, lambda statements: self.branch(statements, result)) for body in bodies]
            default = (self.nested(node.default, lambda statements: self.branch(statements, result))
                       if node.default is not None else None)
        if not arms:
            return lowered + (default or [])
        arms_of = '' if node.expr is not None else 'statement' if result is None else 'value'
        lowered.append(_laid_out(IfStatement(condition=conditions[0], body=arms[0],
                                             elseifs=list(zip(conditions[1:], arms[1:])), else_body=default,
                                             line=node.line, switch_arms=arms_of), node))
        return lowered

    def branch(self, body, result: str) -> list:
        """A branch's statements, its last one's value assigned to `result`."""
        if not isinstance(body, list):
            body = [ExpressionStatement(expr=body)]
        if not body:
            return body
        lowered = self.block(body[:-1])
        last = body[-1]
        nested = last.expr if isinstance(last, ExpressionStatement) and isinstance(last.expr, _VALUE_NODES) else last
        if isinstance(nested, _VALUE_GIVERS):
            inner = self.names.next('block_result')
            is_bool = self.is_bool(nested, self.bool_value)
            lowered.append(self._initial(inner, _initial_width(nested, _tuple_width(nested)), getattr(last, 'line', 0),
                                         self.untaken_is_false(nested, is_bool)))
            lowered.extend(self.value_of(nested, inner, is_bool))
            lowered.append(Reassignment(target=result, value=Identifier(name=inner)))
        elif isinstance(last, ExpressionStatement):
            lowered.append(Reassignment(target=result, value=last.expr, line=last.line))
        elif isinstance(last, Assignment) and isinstance(last.target, TupleDestructure) \
                and not isinstance(last.value, _VALUE_NODES + _LOOPS):
            # `[a, b] = g()` ends the arm: its value is the tuple, assigned whole
            lowered.append(Reassignment(target=result, value=last.value, line=last.line))
        elif self.version >= 6 and isinstance(last, Assignment) and isinstance(last.target, str) \
                and not isinstance(last.value, _VALUE_NODES + _LOOPS):
            # Pine v6 `x = close - 2` or `int x = 1` ends the arm: the local goes out of scope there, its value is
            # the arm's (v4 and v5 keep the local, `var` keeps it in v6 too)
            lowered.append(Reassignment(target=result, value=last.value, line=last.line))
        else:
            lowered.extend(self.statement(last))
            target = getattr(last, 'target', None)
            if isinstance(last, (VarDecl, VaripDecl)):
                # `var z = close` ends the arm: its value is z, kept from the bar it was first declared on
                # (not the initialiser itself, which would be the current bar's value)
                target = last.name
            if isinstance(target, str):
                lowered.append(Reassignment(target=result, value=Identifier(name=target)))
        return lowered


#: method or namespace functions that change the object they are given
_MUTATORS = frozenset({'push', 'unshift', 'insert', 'set', 'put', 'remove', 'pop', 'shift', 'clear', 'fill', 'sort',
                       'reverse', 'concat', 'delete', 'add_row', 'add_col', 'remove_row', 'remove_col'})


def _calls_something(expression) -> bool:
    """Whether `expression` calls a function or a method."""
    from .ast_walk import walk
    from .ast_nodes import MethodCall
    return any(isinstance(node, (FunctionCall, MethodCall)) for node in walk(expression))


def _subject_first(initial: list, computing: list) -> list:
    """The block result's initial value, then the statements computing it, a switch subject read into its
    own name first."""
    if computing and getattr(computing[0], 'reads_switch_subject', False):
        return computing[:1] + initial + computing[1:]
    return initial + computing


def _changes_state(value) -> bool:
    """Whether running `value` changes something outside it: a `:=` (or compound) assignment, a call that
    mutates its object (`a.push(x)`, `array.set(a, i, x)`), a drawing made or changed (`label.new`, `l.set_x`)."""
    from .ast_walk import walk
    for node in walk(value):
        if isinstance(node, Reassignment):
            return True
        if isinstance(node, FunctionCall) and isinstance(node.func, str):
            member = node.func.rpartition('.')[2]
            if member in _MUTATORS or member.startswith('set_') or (member == 'new' and node.func.partition('.')[0]
                                                                    in ('label', 'line', 'box', 'table', 'polyline',
                                                                        'linefill')):
                return True
    return False


def _ends_in_reassignment(node) -> bool:
    """Whether every arm of an if/switch value that has statements ends in a `:=`."""
    if isinstance(node, IfExpression):
        arms, rest = [], node
        while isinstance(rest, IfExpression):
            arms.append(rest.true_expr)
            rest = rest.false_expr if rest.chained or not isinstance(rest.false_expr, IfExpression) else None
            if not isinstance(rest, IfExpression):
                arms.append(rest)
    elif isinstance(node, IfStatement):
        arms = [node.body, *[body for _, body in node.elseifs], node.else_body]
    else:
        arms = [body for _, body in getattr(node, 'cases', [])] + [getattr(node, 'default', None)]
    tails = [arm[-1] for arm in arms if isinstance(arm, list) and arm]
    return bool(tails) and all(isinstance(tail, Reassignment) for tail in tails)


def _initial_width(node, width: int) -> int:
    """The tuple width the temporary of `node` starts with: a one-element tuple starts as a plain na where an
    arm always runs and the start is never read, as `(na,)` where none may run."""
    return 0 if width == 1 and not _misses_an_arm(node) else width


def _misses_an_arm(node) -> bool:
    """True when no arm of the if/switch `node` may run: an if (or `else if` chain) without else, a
    switch without a default. A nested value has a temporary of its own."""
    if isinstance(node, IfExpression):
        rest = node.false_expr
        while isinstance(rest, IfExpression) and rest.chained:
            rest = rest.false_expr
        return rest is None
    if isinstance(node, IfStatement):
        return node.else_body is None
    if isinstance(node, (SwitchExpression, SwitchStatement)):
        return node.default is None
    return False


def _mentions_switch(node) -> bool:
    """Whether a name or a string in `node` holds `switch`, in lower case."""
    for found in walk(node):
        if not dataclasses.is_dataclass(found):
            continue
        for field in dataclasses.fields(found):
            value = getattr(found, field.name)
            names = [value] if isinstance(value, str) else (
                [key for key in value if isinstance(key, str)] if isinstance(value, dict) else [])
            if any('switch' in name for name in names):
                return True
    return False


def lower_value_blocks(script: Script) -> Script:
    """`script` with every if/switch value lowered to statements around a temporary."""
    version_text = getattr(script.version, 'version', '') if script.version else ''
    lowering = _Lowering(int(version_text) if str(version_text).isdigit() else 6)
    _function_widths(script)
    items = sorted([(item.line, getattr(item, 'column', 0) or 0, 0, index, item)
                    for index, item in enumerate(script.declarations)]
                   + [(item.line, getattr(item, 'column', 0) or 0, 1, index, item)
                      for index, item in enumerate(script.body)],
                   key=lambda entry: entry[:4])
    declarations = list(script.declarations)
    lowered_vars: list = []
    body_parts = {index: [item] for index, item in enumerate(script.body)}
    for _, _, kind, index, item in items:
        if kind == 0:
            if isinstance(item, FuncDecl):
                declarations[index] = lowering.function(item)
            elif not isinstance(item, (VarDecl, VaripDecl)) or not isinstance(item.value, _VALUE_NODES + _LOOPS):
                lowering.count_mentions(item)
            elif isinstance(item, (VarDecl, VaripDecl)) and isinstance(item.value, _VALUE_NODES + _LOOPS):
                # the lowered statements run in main(): they move to the body, at the var's line
                parts = lowering.statement(item)
                for part in parts:
                    if not getattr(part, 'line', 0):
                        part.line = item.line
                declarations[index] = None
                lowered_vars.append((item.line, parts))
        else:
            parts = lowering.statement(item)
            for part in parts:  # the transformer orders the top level by Pine line
                if not getattr(part, 'line', 0):
                    part.line = item.line
            body_parts[index] = parts
    body = [statement for index in range(len(script.body)) for statement in body_parts[index]]
    for _, parts in lowered_vars:
        body.extend(parts)  # the transformer orders the top level by line
    declarations = [item for item in declarations if item is not None]
    return dataclasses.replace(script, declarations=declarations, body=body)
