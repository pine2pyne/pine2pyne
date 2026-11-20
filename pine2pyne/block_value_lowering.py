"""
Block-value lowering: an `if`/`switch` EXPRESSION whose branch is a block, and
a loop used as a value.

In Pine, a branch of an if/switch used as a value may hold several statements;
they run when the branch is taken, and its last expression is the value:

    y = if c
        z = close
        z * 2
    else
        0.0

A Python conditional expression cannot hold statements (the branch
was emitted as `None`). This pre-pass rewrites every such value into the idiom
Pine authors write by hand, which the transformer already emits correctly:

    y = na                 (keeps a declared type; untyped is Pine's typeless na)
    if c
        z = close
        y := z * 2
    else
        y := 0.0

A block-valued switch becomes an if / else-if chain over a temporary that holds
the subject, so the subject is evaluated once. `var`/`varip` targets run the
lowered statements once, behind a flag. A branch that is a single binding
(`k := close`) is a block too: its binding runs, and its value is k.

A loop used as a value (`s = for …`, `s = while …`, a branch or a function tail
that ends in a loop) yields what its body's tail gave on the last iteration
that evaluated it, na when no iteration ran (Pine docs, Loops). Each value is
written to a fresh local, so the body still reads the target's old value:

    s = for i = 0 to 2          s = na
        i * 10            ->    loop_value = na
                                for i = 0 to 2
                                    loop_value = i * 10
                                s := loop_value

An arm that does not run (an if without else) yields false for a bool value
and na otherwise; a tuple-returning one yields a tuple of na as wide as its
tuples. `s op= <block>` runs the block into a fresh local, then reads s:
Pine runs the block first, so an `s :=` inside it shows in the result.
`[a, b] = <if/switch/loop>` fills a fresh local that starts as a tuple of na,
destructured once. A block value anywhere else (an operand of an operator, a
call argument) is refused with its line, never emitted.
"""
import dataclasses
from typing import Callable, Dict, List, Optional

from .ast_nodes import (
    ArrayLiteral, Assignment, BinaryOp, BreakStatement, ContinueStatement,
    Expression, ExpressionStatement, ForInLoop, ForLoop, FuncDecl, FunctionCall,
    Identifier, IfExpression, IfStatement, Literal, NaLiteral, Reassignment,
    Script, SwitchExpression, SwitchStatement, TernaryOp, TupleDestructure, UnaryOp,
    VarDecl, VaripDecl, WhileLoop,
)
from .ast_walk import child_field_names, contains, walk
from .errors import UnsupportedFeatureError
from .scope_renamer import collect_names, fresh_name

VALUE_EXPRESSIONS = (IfExpression, SwitchExpression)
LOOPS = (ForLoop, ForInLoop, WhileLoop)
BINDINGS = (Assignment, Reassignment, VarDecl, VaripDecl)
# The nodes whose direct child is a value: a loop there is a loop used as a value
VALUE_HOLDERS = (Expression,) + BINDINGS
# Half the nodes of a script; they hold no node, so the pre-check skips them
CHILDLESS_NODE_TYPES = frozenset({Identifier, Literal})
# A branch ending in one of these has no value of its own to extract
VALUELESS_TAILS = (IfStatement, SwitchStatement) + LOOPS
LOOP_HEADER_FIELDS = ('from_val', 'to_val', 'step', 'iterable', 'condition')
# The parser reads `s += v` as `s := s + v`
COMPOUND_OPERATORS = ('+', '-', '*', '/', '%')
BOOL_OPERATORS = ('==', '!=', '<', '<=', '>', '>=', 'and', 'or')

Write = Callable[[Expression], object]  # the statement that takes a value


def lower_block_values(script: Script) -> Script:
    """Return a copy of `script` with every block value lowered to statements.
    A script without one is returned unchanged. An untaken bool arm is false
    only where `stamp_bool_values` stamped its value first."""
    if not _needs_lowering_somewhere(script):
        return script
    return _BlockValueLowering(collect_names(script), _UserFunctions.of(script),
                               bool_is_never_na(script)).script(script)


def _needs_lowering(node) -> bool:
    """A node other than a function that this pass rewrites or refuses: a block
    value, a loop as a value or an operand, a tuple target of an if/switch."""
    if isinstance(node, VALUE_EXPRESSIONS) and has_block_branch(node):
        return True
    if isinstance(node, VALUE_HOLDERS) and _has_loop_operand(node):
        return True
    return (isinstance(node, Assignment) and isinstance(node.target, TupleDestructure)
            and _has_untaken_arm(node.value))


def _needs_lowering_somewhere(script: Script) -> bool:
    """True when this pass would rewrite or refuse something in `script`: a
    node `_needs_lowering` names, a function whose tail reaches a loop, or a
    function whose tail if/switch has an arm that may not run and that yields
    tuples or is destructured, the tail `tail_is_lowered` lowers or refuses.
    Each function is checked on its own: overloads share a name, and the parser
    accepts a function in a block. Other such functions are left to the
    transformer, so they do not cost this pass a copy of the script. This runs
    on every node of every script, hence the dispatch on the node's type
    (transpile-time gate)."""
    untaken_tails = []
    for node in walk(script):
        if type(node) in CHILDLESS_NODE_TYPES:
            continue
        if isinstance(node, FuncDecl):
            if _tail_reaches_loop(node.body):
                return True
            tail = _untaken_tail(node)
            if tail is not None:
                untaken_tails.append((node.name, tail))
        elif _needs_lowering(node):
            return True
    if not untaken_tails:
        return False
    functions = _UserFunctions.of(script)
    return any(_tuple_width(tail, functions) or name in functions.destructured or name in functions.widths
               for name, tail in untaken_tails)


def refuse_tuple_tails_of_unknown_width(script: Script) -> None:
    """Refuse a function the script destructures whose tail if/switch has an arm that may not run and
    whose tuple width is unknown: the arm that does not run must yield a tuple of na as wide as the
    others, so the width must be known (give the tail an else that returns the tuple). Runs before the
    value blocks are lowered (value_blocks), which give a known width its tuple of na."""
    untaken_tails = [(node, tail) for node in walk(script) if isinstance(node, FuncDecl)
                     for tail in [_untaken_tail(node)] if tail is not None]
    if not untaken_tails:
        return
    functions = _UserFunctions.of(script)
    for node, tail in untaken_tails:
        if node.name in functions.destructured and not (_tuple_width(tail, functions)
                                                        or functions.widths.get(node.name)):
            raise UnsupportedFeatureError(
                f'the tuple function {node.name} has an if/switch tail without else, and its '
                'tuple width is unknown (give it an else that returns the tuple)', tail.line or node.line)


def _has_loop_operand(node) -> bool:
    """True when a loop is a direct child of `node`: a bound value, an operand.
    Runs on every expression of every script, so it reads the fields once and
    builds no lists (transpile-time gate)."""
    for name in child_field_names(type(node)):
        child = getattr(node, name)
        if isinstance(child, LOOPS) or (type(child) is list and any(isinstance(item, LOOPS) for item in child)):
            return True
    return False


def _branches(expression) -> list:
    if isinstance(expression, IfExpression):
        return [expression.true_expr, expression.false_expr]
    return [body for _, body in expression.cases] + [expression.default]


def _tail_expression(branch):
    tail = branch[-1] if isinstance(branch, list) and branch else branch
    return tail.expr if isinstance(tail, ExpressionStatement) else tail


def has_block_branch(expression) -> bool:
    """True when a branch of this if/switch expression, or of one in a branch
    tail, is a block: several statements, a sole binding, or a tail statement
    with no value. A default-only switch with a subject counts too: its
    subject runs as a statement before the default, so its side effect is
    kept."""
    if not isinstance(expression, VALUE_EXPRESSIONS):
        return False
    if isinstance(expression, SwitchExpression) and not expression.cases and expression.expr is not None:
        return True
    for branch in _branches(expression):
        if isinstance(branch, LOOPS):
            return True  # an inline switch case `0 => for …` keeps its loop bare
        if isinstance(branch, list) and branch and (len(branch) > 1
                                                    or isinstance(branch[-1], VALUELESS_TAILS + BINDINGS)):
            # a sole binding `k := close` is a block: the binding must run
            return True
        tail = _tail_expression(branch)
        if has_block_branch(tail):
            return True
        if isinstance(tail, BINDINGS) and has_block_branch(tail.value):
            # `z = if <block> …` as the branch: z's statements run, z is the value
            return True
    return False


def _is_block_value(node) -> bool:
    """A value that only statements can compute: a loop, or a block-valued if/switch."""
    return isinstance(node, LOOPS) or (isinstance(node, VALUE_EXPRESSIONS) and has_block_branch(node))


def _contains_block_value(value) -> bool:
    return contains(value, _is_block_value)


def _target_name(statement) -> Optional[str]:
    """The variable a declaration or assignment binds, when it is a plain name."""
    name = statement.name if isinstance(statement, (VarDecl, VaripDecl)) else getattr(statement, 'target', None)
    return name if isinstance(name, str) else None


def _is_compound_block(node) -> bool:
    """`s op= <block value>`, read by the parser as `s := s op <block value>`."""
    value = getattr(node, 'value', None)
    return (isinstance(node, Reassignment) and isinstance(value, BinaryOp) and value.op in COMPOUND_OPERATORS
            and isinstance(value.left, Identifier) and value.left.name == node.target
            and _is_block_value(value.right))


def _untaken_tail(node: FuncDecl):
    """The tail if/switch of a function body when an arm of it may not run, else None."""
    tail = node.body[-1] if isinstance(node.body, list) and node.body else None
    return tail if isinstance(tail, (IfStatement, SwitchStatement)) and _has_untaken_arm(tail) else None


def _tail_reaches_loop(body) -> bool:
    """True when a statement list's value comes from a loop: its tail is a
    loop, or a tail if/switch has an arm whose own tail reaches one."""
    if not isinstance(body, list) or not body:
        return False
    tail = body[-1]
    if isinstance(tail, LOOPS):
        return True
    if isinstance(tail, (IfStatement, SwitchStatement)):
        return any(_tail_reaches_loop(arm) for arm in _arms(tail))
    return False


# ---------------------------------------------------------------- arm values

def _arms(value) -> list:
    """The arms whose tails give `value` its value: the branches of an if or a
    switch (expression or statement), the body of a loop. A missing arm is None."""
    if isinstance(value, IfExpression):
        return [value.true_expr, value.false_expr]
    if isinstance(value, IfStatement):
        return [value.body] + [body for _, body in value.elseifs] + [value.else_body]
    if isinstance(value, (SwitchExpression, SwitchStatement)):
        return [body for _, body in value.cases] + [value.default]
    if isinstance(value, LOOPS):
        return [value.body]
    return []


def _value_tails(value):
    """Each plain value `value` can yield, through the tails of its arms."""
    arms = _arms(value)
    if not arms:
        yield value
    for arm in arms:
        if arm:
            yield from _value_tails(_tail_expression(arm))


def value_tails(value):
    """Each plain value an if/switch/loop value `value` can yield, through the tails of its arms."""
    return _value_tails(value)


def _has_untaken_arm(value) -> bool:
    """True when an arm of `value`, at any depth, may not run: an if without
    else, a switch without a default."""
    return any(not arm or _has_untaken_arm(_tail_expression(arm)) for arm in _arms(value))


def _is_bool_value(value) -> bool:
    if isinstance(value, Literal):
        return value.literal_type == 'bool'
    if isinstance(value, Identifier):
        return value.name in ('true', 'false')
    if isinstance(value, BinaryOp):
        return value.op in BOOL_OPERATORS
    return isinstance(value, UnaryOp) and value.op == 'not'


def yields_bool(value) -> bool:
    """True when an arm of `value` yields a bool by its syntax. Only the slot of
    a loop that runs no iteration reads it, next to the loop's bool type: Pine
    v6 gives false to a bool loop that runs no pass, a literal tail
    (`b = while c` + `true`) and a bool name tail (`flag`) alike, as it does to
    an untaken if/switch arm."""
    return any(_is_bool_value(tail) for tail in _value_tails(value))


def bool_is_never_na(script: Script) -> bool:
    """Pine v6 removed na from bool; before v6 an untaken bool arm is na. A
    script without a version annotation is taken as v6, the transpiler's target."""
    version = script.version.version if script.version else ''
    return not version.isdigit() or int(version) >= 6


# Pine v4 and v5 keep a third bool state, na; the runtime names exactly these two versions (`na_bool`)
THREE_STATE_BOOL_VERSIONS = ('4', '5')


def keeps_three_state_bool(script: Script) -> bool:
    """True for a Pine v4 or v5 script: its bool may be na (`bool b = na`, a bool history before
    warm-up), so its decorator sets the runtime's `na_bool` switch (MEASURED on
    TradingView 2026-10-06). A script with no version annotation is v6, as in `bool_is_never_na`."""
    return bool(script.version) and script.version.version in THREE_STATE_BOOL_VERSIONS


def _tuple_width(value, functions: '_UserFunctions') -> Optional[int]:
    """The width of the tuples `value` yields, from an arm that ends in `[x, y]`,
    in a tuple destructure, or in a call of a user function of known width."""
    for tail in _value_tails(value):
        if isinstance(tail, ArrayLiteral):
            return len(tail.elements)
        if isinstance(tail, Assignment) and isinstance(tail.target, TupleDestructure):
            return len(tail.target.names)
        if isinstance(tail, FunctionCall) and functions.width_of(tail):
            return functions.width_of(tail)
    return None


def _destructured_calls(value) -> list:
    """The calls whose tuple a `[a, b] = value` destructures: the value itself,
    either arm of a ternary, the tail of each arm of an if/switch or of a
    loop's body, or the expression `request.security` evaluates. A call nested in an argument
    yields a scalar and is not one of them."""
    if isinstance(value, TernaryOp):
        return _destructured_calls(value.true_expr) + _destructured_calls(value.false_expr)
    if isinstance(value, VALUE_EXPRESSIONS + LOOPS):
        return [call for tail in _value_tails(value) for call in _destructured_calls(tail)]
    if not isinstance(value, FunctionCall):
        return []
    if value.func == 'request.security':
        positional = value.args[2] if len(value.args) > 2 else None
        return _destructured_calls(value.kwargs.get('expression', positional))
    return [value]


@dataclasses.dataclass(frozen=True)
class _UserFunctions:
    """What the tuple fall-through needs to know of the script's functions."""
    names: frozenset = frozenset()
    methods: frozenset = frozenset()
    widths: Dict[str, int] = dataclasses.field(default_factory=dict)  # name -> tuple width
    destructured: frozenset = frozenset()  # the functions a `[a, b] = …` value calls

    @classmethod
    def of(cls, script: Script) -> '_UserFunctions':
        declared = [node for node in script.declarations if isinstance(node, FuncDecl)]
        known = cls(names=frozenset(function.name for function in declared),
                    methods=frozenset(function.name for function in declared if function.is_method))
        # a function's width read from the `[a, b] = f(…)` that destructure it: Pine gives every call of a
        # tuple function its one width, so a tail of unknown width (a library call) takes it from there
        destructured_widths: Dict[str, int] = {}
        for node in walk(script):
            if isinstance(node, Assignment) and isinstance(node.target, TupleDestructure):
                for name in filter(None, map(known.called, _destructured_calls(node.value))):
                    destructured_widths[name] = max(destructured_widths.get(name, 0), len(node.target.names))
        # Pine declares a function before its callers, so widths build in order
        for function in declared:
            width = _tuple_width(_tail_expression(function.body), known) or destructured_widths.get(function.name)
            if width:
                known = dataclasses.replace(known, widths={**known.widths, function.name: width})
        return dataclasses.replace(known, destructured=frozenset(destructured_widths))

    def called(self, call: FunctionCall) -> Optional[str]:
        """The user function a call runs: `f(…)`, or a method as `t.f(…)`."""
        if not isinstance(call.func, str):
            return None
        if call.func in self.names:
            return call.func
        _, dot, method = call.func.rpartition('.')
        return method if dot and method in self.methods else None

    def width_of(self, call: FunctionCall) -> Optional[int]:
        return self.widths.get(self.called(call))


@dataclasses.dataclass(frozen=True)
class _Untaken:
    """The value of an arm that does not run, and of a loop with no iteration:
    a tuple of na as wide as the value's tuples, false for a bool, else na.
    An arm is bool by the value's type, a loop's empty slot by its syntax."""
    tuple_width: Optional[int] = None
    is_bool: bool = False
    never_run_is_bool: bool = False

    def value(self, line: int) -> Expression:
        if self.tuple_width:
            return ArrayLiteral(elements=[NaLiteral(line=line) for _ in range(self.tuple_width)], line=line)
        if self.is_bool:
            return Literal(value=False, literal_type='bool', line=line)
        return NaLiteral(line=line)

    def never_run_value(self, line: int) -> Expression:
        """The value of a loop that runs no iteration."""
        return dataclasses.replace(self, is_bool=self.never_run_is_bool).value(line)


class _BlockValueLowering:
    def __init__(self, used_names, functions: _UserFunctions, bool_is_never_na: bool):
        self.used_names = used_names
        self.functions = functions
        self.bool_is_never_na = bool_is_never_na
        self.line = 0  # nearest enclosing source line, for refusals and new nodes

    def refuse(self, feature: str):
        raise UnsupportedFeatureError(feature, self.line)

    def refuse_nested(self, value):
        """A block value left inside an expression cannot be emitted."""
        if _contains_block_value(value):
            self.refuse('an if/switch expression with a multi-statement branch, a default-only switch with '
                        'a subject, or a loop, inside another expression (assign it to a variable first)')

    def untaken_of(self, value, value_is_bool: bool) -> _Untaken:
        """`value_is_bool` is the stamp of the if/switch/loop, or of the
        function, whose value `value` is (bool_value_typing)."""
        return _Untaken(tuple_width=_tuple_width(value, self.functions), is_bool=value_is_bool,
                        never_run_is_bool=self.bool_is_never_na and (value_is_bool or yields_bool(value)))

    # ---------------------------------------------------------------- values

    def value_into(self, expression, write: Write, untaken: Optional[_Untaken] = None) -> list:
        """Statements that compute `expression` and pass its value to `write`.
        Already in statement position, so every if/switch becomes statements:
        a tail if statement turned into an expression holds statements too.
        `untaken` is what an arm that does not run yields; one if/switch/loop
        has one type, so the outermost decides it for every nested arm."""
        if isinstance(expression, VALUE_EXPRESSIONS + LOOPS):
            untaken = untaken or self.untaken_of(expression, expression.value_is_bool)
        if isinstance(expression, LOOPS):
            return self.loop_into(expression, write, untaken)
        if isinstance(expression, IfExpression):
            self.refuse_nested(expression.condition)
            return [IfStatement(condition=expression.condition,
                                body=self.branch_into(expression.true_expr, write, untaken),
                                elseifs=[],
                                else_body=self.branch_into(expression.false_expr, write, untaken),
                                line=self.line)]
        if isinstance(expression, SwitchExpression):
            return self.switch_into(expression, write, untaken)
        self.refuse_nested(expression)
        return [write(expression)]

    def loop_into(self, loop, write: Write, untaken: _Untaken) -> list:
        """`slot = <untaken>`, the loop with each value its body's tail yields
        written to the slot, then `write(slot)`. The slot is fresh, so the body
        still reads the target's old value; an iteration that breaks or
        continues before its tail writes nothing, so the slot keeps the last
        value an iteration evaluated (Pine docs, Loops)."""
        self.refuse_nested([getattr(loop, name, None) for name in LOOP_HEADER_FIELDS])
        slot = fresh_name('loop_value', self.used_names)
        body = self.branch_into(loop.body, self.local_writer(slot), untaken)
        return [Assignment(target=slot, value=untaken.never_run_value(self.line), line=self.line),
                dataclasses.replace(loop, body=body, line=loop.line or self.line),
                write(Identifier(name=slot, line=self.line))]

    def switch_into(self, switch: SwitchExpression, write: Write, untaken: _Untaken) -> list:
        """An if / else-if chain; a subject is bound once, then compared per case."""
        self.refuse_nested([switch.expr] + [case for case, _ in switch.cases])
        statements = []
        if switch.expr is not None:
            subject = fresh_name('switch_subject', self.used_names)
            statements.append(Assignment(target=subject, value=switch.expr, line=self.line))

            def condition_of(case):
                return BinaryOp(left=Identifier(name=subject, line=self.line), op='==', right=case, line=self.line)
        else:
            def condition_of(case):
                return case
        if not switch.cases:
            return statements + self.branch_into(switch.default, write, untaken)
        (first_case, first_body), *other_cases = switch.cases
        return statements + [IfStatement(
            condition=condition_of(first_case),
            body=self.branch_into(first_body, write, untaken),
            elseifs=[(condition_of(case), self.branch_into(body, write, untaken)) for case, body in other_cases],
            else_body=self.branch_into(switch.default, write, untaken), line=self.line)]

    def branch_into(self, branch, write: Write, untaken: _Untaken) -> list:
        """Statements that run a branch and pass its value to `write`; a missing
        branch yields `untaken`."""
        if not branch:
            return [write(untaken.value(self.line))]
        if isinstance(branch, LOOPS):
            branch = [branch]
        if not isinstance(branch, list):
            return self.value_into(branch, write, untaken)
        *leading, tail = branch
        statements = self.statements(leading)
        if isinstance(tail, ExpressionStatement):
            return statements + self.value_into(tail.expr, write, untaken)
        if isinstance(tail, Expression):
            return statements + self.value_into(tail, write, untaken)
        if isinstance(tail, BINDINGS) and _target_name(tail):
            # The value of `z = ...` or `z := ...` is z
            return statements + self.statements([tail]) + [write(Identifier(name=_target_name(tail), line=self.line))]
        if isinstance(tail, Assignment) and isinstance(tail.target, TupleDestructure):
            # The value of `[a, b] = ...` is the tuple
            names = [Identifier(name=name, line=self.line) for name in tail.target.names]
            return statements + self.statements([tail]) + [write(ArrayLiteral(elements=names, line=self.line))]
        if isinstance(tail, VALUELESS_TAILS):
            return statements + self.value_into(self.as_value(tail), write, untaken)
        if isinstance(tail, (BreakStatement, ContinueStatement)):
            return statements + [tail]  # this iteration yields no value: the slot keeps the last one
        self.refuse(f'{type(tail).__name__} as the value of an if/switch expression branch')

    @classmethod
    def as_value(cls, statement):
        """A tail if/switch statement as the value expression it yields; a loop as itself."""
        if isinstance(statement, IfStatement):
            return cls.if_statement_as_value(statement)
        if isinstance(statement, SwitchStatement):
            return SwitchExpression(expr=statement.expr, cases=statement.cases, default=statement.default,
                                    line=statement.line)
        return statement

    @staticmethod
    def if_statement_as_value(statement: IfStatement) -> IfExpression:
        """`if a … else if b … else …` as nested if expressions, keeping every else-if."""
        else_branch = statement.else_body
        for condition, body in reversed(statement.elseifs):
            else_branch = IfExpression(condition=condition, true_expr=body, false_expr=else_branch,
                                       line=statement.line)
        return IfExpression(condition=statement.condition, true_expr=statement.body,
                            false_expr=else_branch, line=statement.line)

    def writer(self, target: str) -> Write:
        line = self.line
        return lambda value: Reassignment(target=target, value=value, line=line)

    def local_writer(self, name: str) -> Write:
        """`name = v` for a fresh local: `:=` would make it a Series in a function."""
        line = self.line
        return lambda value: Assignment(target=name, value=value, line=line)

    def yielder(self) -> Write:
        """A function tail's value as its last expression statement."""
        line = self.line
        return lambda value: ExpressionStatement(expr=value, line=line)

    # ---------------------------------------------------------------- statements

    def statements(self, statements) -> list:
        lowered = []
        for statement in statements:
            lowered.extend(self.statement(statement))
        return lowered

    def block(self, body):
        return None if body is None else self.statements(body)

    def statement(self, node) -> list:
        enclosing_line = self.line
        self.line = node.line or self.line
        try:
            return self.lowered_statement(node)
        finally:
            self.line = enclosing_line

    def lowered_statement(self, node) -> list:
        replace = dataclasses.replace
        value = getattr(node, 'value', None)
        if isinstance(node, Assignment) and isinstance(node.target, TupleDestructure) and (
                _is_block_value(value) or _has_untaken_arm(value)):
            return self.tuple_statement(node)
        if isinstance(node, BINDINGS) and _is_block_value(value):
            return self.block_value_statement(node)
        if _is_compound_block(node):
            return self.compound_statement(node)
        if isinstance(node, IfStatement):
            self.refuse_nested([node.condition] + [condition for condition, _ in node.elseifs])
            return [replace(node, body=self.block(node.body),
                            elseifs=[(condition, self.block(body)) for condition, body in node.elseifs],
                            else_body=self.block(node.else_body))]
        if isinstance(node, SwitchStatement):
            self.refuse_nested([node.expr] + [case for case, _ in node.cases])
            return [replace(node, cases=[(case, self.block(body)) for case, body in node.cases],
                            default=self.block(node.default))]
        if isinstance(node, LOOPS):
            self.refuse_nested([getattr(node, name, None) for name in LOOP_HEADER_FIELDS])
            return [replace(node, body=self.block(node.body))]
        if isinstance(node, FuncDecl):
            return [self.function(node)]
        self.refuse_nested(node)
        return [node]

    def block_value_statement(self, node) -> list:
        target = _target_name(node)
        if isinstance(node, Assignment):
            declaration = Assignment(target=target, value=NaLiteral(line=self.line),
                                     type_hint=node.type_hint, line=self.line)
            return [declaration] + self.value_into(node.value, self.writer(target))
        if isinstance(node, Reassignment):
            return self.value_into(node.value, self.writer(target))
        return self.run_once_declaration(node)

    def compound_statement(self, node: Reassignment) -> list:
        """`s op= <block>`: the block runs into a fresh local, then
        `s := s op <block value>` reads s, so an `s :=` inside the block shows
        in the result, as in Pine (the block runs before s is read)."""
        operation = node.value
        block_value = fresh_name('block_value', self.used_names)
        computing = self.value_into(operation.right, self.local_writer(block_value))
        return ([Assignment(target=block_value, value=NaLiteral(line=self.line), line=self.line)]
                + computing
                + [dataclasses.replace(node, value=dataclasses.replace(
                    operation, right=Identifier(name=block_value, line=self.line)))])

    def tuple_statement(self, node: Assignment) -> list:
        """`[a, b] = <if/switch/loop>`: the value goes to a fresh local that
        starts as a tuple of na as wide as the target, as does an arm that
        does not run; the target destructures it once."""
        names = list(node.target.names)
        untaken = _Untaken(tuple_width=len(names))
        slot = fresh_name('tuple_value', self.used_names)
        computing = self.value_into(node.value, self.local_writer(slot), untaken)
        destructure = Assignment(
            target=TupleDestructure(names=names, value=Identifier(name=slot, line=self.line), line=self.line),
            value=Identifier(name=slot, line=self.line), type_hint=node.type_hint, line=self.line)
        return [Assignment(target=slot, value=untaken.value(self.line), line=self.line)] + computing + [destructure]

    def run_once_declaration(self, node) -> list:
        """`var x = if …` -> `var x = na`, `var bool flag = false`, and
        `if not flag` { `flag := true`; <lowered value> }: the value runs on the
        first bar only, like a var initialiser."""
        kind = type(node)
        flag = fresh_name(f'{node.name}_initialised', self.used_names)
        declarations = [kind(name=node.name, value=NaLiteral(line=self.line), type_hint=node.type_hint,
                             line=self.line),
                        kind(name=flag, value=Literal(value=False, literal_type='bool', line=self.line),
                             type_hint='bool', line=self.line)]
        first_bar_only = IfStatement(
            condition=UnaryOp(op='not', operand=Identifier(name=flag, line=self.line), line=self.line),
            body=[Reassignment(target=flag, value=Literal(value=True, literal_type='bool', line=self.line),
                               line=self.line)] + self.value_into(node.value, self.writer(node.name)),
            elseifs=[], else_body=None, line=self.line)
        return declarations + [first_bar_only]

    def function(self, node: FuncDecl) -> FuncDecl:
        replace = dataclasses.replace
        if not isinstance(node.body, list):
            if not has_block_branch(node.body):
                self.refuse_nested(node.body)
                return node
            # `f() => if …`: the value into a fresh local, which the function yields
            result = fresh_name(f'{node.name}_result', self.used_names)
            body = self.statement(Assignment(target=result, value=node.body, line=self.line))
            return replace(node, body=body + [ExpressionStatement(expr=Identifier(name=result, line=self.line),
                                                                  line=self.line)])
        if node.body and self.tail_is_lowered(node):
            # every arm of the tail ends in the expression it yields, which the
            # transformer's tail walk returns
            *leading, tail = node.body
            lowered_leading = self.statements(leading)
            enclosing_line, self.line = self.line, tail.line or self.line
            try:
                value = self.as_value(tail)
                untaken = self.untaken_of(value, node.value_is_bool)
                if untaken.tuple_width is None and self.functions.widths.get(node.name):
                    # a tail of unknown width (a library call) is as wide as the function's destructures
                    untaken = dataclasses.replace(untaken, tuple_width=self.functions.widths[node.name])
                return replace(node, body=lowered_leading + self.value_into(value, self.yielder(), untaken))
            finally:
                self.line = enclosing_line
        return replace(node, body=self.body_with_its_value(node.body))

    def tail_is_lowered(self, node: FuncDecl) -> bool:
        """A tail the transformer cannot return by itself: one that reaches a
        loop, or a tuple-returning if/switch with an arm that may not run,
        which yields a tuple of na. A function the script
        destructures, whose tuple width is unknown, is refused."""
        if _tail_reaches_loop(node.body):
            return True
        tail = _untaken_tail(node)
        if tail is None:
            return False
        if _tuple_width(tail, self.functions) or self.functions.widths.get(node.name):
            return True
        if node.name in self.functions.destructured:
            self.line = tail.line or self.line
            self.refuse(f'the tuple function {node.name} has an if/switch tail without else, and its '
                        'tuple width is unknown (give it an else that returns the tuple)')
        return False

    def body_with_its_value(self, body: list) -> list:
        """Lower a function body whose value is its last statement. When that
        statement binds a block value, the lowered body ends in the if
        statement that writes it, so the bound name is appended as the value."""
        lowered = self.statements(body)
        last = body[-1] if body else None
        if isinstance(last, BINDINGS) and _target_name(last) and _is_block_value(last.value):
            lowered.append(ExpressionStatement(expr=Identifier(name=_target_name(last), line=self.line),
                                               line=self.line))
        return lowered

    # ---------------------------------------------------------------- script

    def script(self, script: Script) -> Script:
        declarations: List = []
        first_bar_initialisers: List = []
        for declaration in script.declarations:
            enclosing_line, self.line = self.line, declaration.line or self.line
            if isinstance(declaration, (VarDecl, VaripDecl)) and _is_block_value(declaration.value):
                *declared, first_bar_only = self.run_once_declaration(declaration)
                declarations.extend(declared)
                # Same line as the declaration: the transformer merges declarations
                # and body by line, so this runs right after it
                first_bar_initialisers.append(first_bar_only)
            elif isinstance(declaration, FuncDecl):
                declarations.append(self.function(declaration))
            else:
                self.refuse_nested(declaration)
                declarations.append(declaration)
            self.line = enclosing_line
        body = self.statements(script.body) + first_bar_initialisers
        return dataclasses.replace(script, declarations=declarations, body=body)
