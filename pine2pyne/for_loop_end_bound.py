"""A `for` loop whose end may change in its body reads it again before every pass.

MEASURED on TradingView (2026-10-05, `//@version=6`): `n = 10; for i = 0 to n:
count += 1; n -= 1` runs 6 times and leaves n at 4. The Pine docs (Loops) say the
direction is fixed when the loop starts and the end is checked on each pass;
`from` and `by` are evaluated once.

`for i in pine_range(from, to)` reads the end once. pine2pyne
emits the runtime's `pine_loop` for a loop of any
Pine version whose end is not provably the same on every pass; its counter is
named by the loop's block depth in its function, `__loop_<depth>__`. Pine v5 and
earlier read the end once: where the body changes what the end reads, or the end has state (a user function,
`ta.*`, a resizing call), such a loop keeps `pine_range`; any other v4/v5 end is re-read as a v6 one is, which
a pure end that the body leaves alone reads the same:

    __loop_0__ = pine_loop(from, by)      # `by` written out where pine_range writes it
    while __loop_0__.step(to):
        i = __loop_0__.value

A Pine v6 end that calls a function outside a namespace (`f()`, `nz(n)`, `na(n) ? 1 : 2`) is read once more on entry,
before `by` and the first check, as TradingView reads the header (from, to, by, then to before every pass):

    __loop_0__ = pine_loop(from, entry=True)
    while __loop_0__.step(to):
        if __loop_0__.entering:
            __loop_0__.enter(by)
            continue
        i = __loop_0__.value

An end is invariant when it is built from literals, operators, casts, names the
body does not assign (dotted names excepted: `timeframe.multiplier` is not), history
reads of those and `array.size(x)` / `matrix.rows(x)` / `matrix.columns(x)` /
`map.size(x)` of a container whose size the body does not change (`x` a name, not a
field: `array.size(s.h)` is read on every pass). Any other call
(`math.max`, `nz`, `na`, `str.length`, `ta.*`, a user function) makes it dynamic.
The body changes the size of `x` when it calls a resizing function on it (`push`,
`pop`, `insert`, `remove`, `shift`, `unshift`, `clear`, `concat`, `slice`, ...) or
passes it to a user function; `set`, `get`, `sort`, `reverse`, `fill` and `copy`
do not. Runs after the scope renamer, so a body local that shadows an outer name
has its own name and does not count as an assignment of the outer one.
"""
import dataclasses
from typing import AbstractSet, Set

from .ast_nodes import (
    Assignment, BinaryOp, ForInLoop, ForLoop, FuncDecl, FunctionCall, Identifier, IfStatement, IndexAccess, Literal,
    MethodCall, NaLiteral, Reassignment, Script, SwitchStatement, TernaryOp, TupleDestructure, UnaryOp, VarDecl,
    VaripDecl, WhileLoop,
)
from .ast_walk import walk
from .scope_renamer import collect_names, fresh_name

COUNTER_STEM = '__loop_{depth}__'
HELPER_NAME = 'pine_loop'
#: size functions an invariant end may call on a container the body does not resize
SIZE_FUNCTIONS = frozenset({'array.size', 'matrix.rows', 'matrix.columns', 'map.size'})
#: functions and methods that change the size of their first argument (or receiver)
RESIZING_FUNCTIONS = frozenset({
    'push', 'pop', 'insert', 'remove', 'shift', 'unshift', 'clear', 'concat', 'slice', 'put',
    'add_row', 'add_col', 'remove_row', 'remove_col', 'put_all',
    'row'})  # `m.row(r)` counts among them here
#: the functions that really change a size (`m.row(r)` reads a row, `array.slice(a, i, j)` makes a new array)
PINE_RESIZING_FUNCTIONS = RESIZING_FUNCTIONS - {'row', 'slice'}
_BLOCK_STATEMENTS = (ForLoop, ForInLoop, WhileLoop, IfStatement, SwitchStatement)


def mark_dynamic_end_bounds(script: Script) -> Script:
    """A copy of `script` in which every `for` loop whose end may change between
    passes carries its counter name; `script` itself when it has no such loop."""
    if not any(isinstance(node, ForLoop) for node in walk(script)):
        return script
    used_names = collect_names(script)
    function_names = {node.name for node in walk(script) if isinstance(node, FuncDecl)}
    version = getattr(script.version, 'version', '') if script.version else ''
    functions: dict = {}
    for node in walk(script):
        if isinstance(node, FuncDecl):
            functions.setdefault(node.name, []).append(node)
    marker = _EndBoundMarker(used_names, function_names, int(version) if str(version).isdigit() else 6, functions)
    return dataclasses.replace(script, declarations=[marker.mark(item, 0) for item in script.declarations],
                               body=[marker.mark(item, 0) for item in script.body])


class _EndBoundMarker:
    def __init__(self, used_names: Set[str], function_names: AbstractSet[str], version: int = 6,
                 functions: dict | None = None):
        self.function_names = function_names
        self.functions = functions or {}  # user function or method name -> its declarations
        self.version = version
        # a script name `pine_loop` keeps its spelling; the import takes an alias
        self.helper_name = fresh_name(HELPER_NAME, used_names) if HELPER_NAME in used_names else HELPER_NAME

    def mark(self, node, depth: int):
        """`node` rebuilt with every dynamic loop in it marked; `depth` is the block depth of
        `node` in its function (a function's body is depth 0 again)."""
        if isinstance(node, list):
            return [self.mark(item, depth) for item in node]
        if isinstance(node, FuncDecl):
            body = self.mark(node.body, 0) if isinstance(node.body, list) else node.body
            return dataclasses.replace(node, body=body)
        if isinstance(node, IfStatement):
            return dataclasses.replace(
                node, body=self.mark(node.body, depth + 1),
                elseifs=[(condition, self.mark(body, depth + 1)) for condition, body in node.elseifs],
                else_body=self.mark(node.else_body, depth + 1) if node.else_body is not None else None)
        if isinstance(node, SwitchStatement):
            return dataclasses.replace(
                node, cases=[(case, self.mark(body, depth + 1) if isinstance(body, list) else body)
                             for case, body in node.cases],
                default=self.mark(node.default, depth + 1) if isinstance(node.default, list) else node.default)
        if isinstance(node, (ForInLoop, WhileLoop)):
            return dataclasses.replace(node, body=self.mark(node.body, depth + 1))
        if isinstance(node, ForLoop):
            marks = {}
            assigned = _assigned_in(node.body) | {node.var}
            if not self.is_invariant(node.to_val, node.body, assigned) and not (
                    self.version < 6 and (self.changes_in_body(node.to_val, node.body, assigned)
                                          or self.has_state(node.to_val))):
                marks = {'counter_name': COUNTER_STEM.format(depth=depth), 'counter_helper': self.helper_name,
                         'counter_entry': self.version >= 6 and _calls_a_function(node.to_val)}
            return dataclasses.replace(node, body=self.mark(node.body, depth + 1), **marks)
        return node

    def is_invariant(self, expression, body, assigned: AbstractSet[str]) -> bool:
        def invariant(part) -> bool:
            if part is None or isinstance(part, (Literal, NaLiteral)):
                return True
            if isinstance(part, Identifier):
                return '.' not in part.name and part.name not in assigned
            if isinstance(part, BinaryOp):
                return invariant(part.left) and invariant(part.right)
            if isinstance(part, UnaryOp):
                return invariant(part.operand)
            if isinstance(part, TernaryOp):
                return invariant(part.condition) and invariant(part.true_expr) and invariant(part.false_expr)
            if isinstance(part, IndexAccess):
                return invariant(part.object) and invariant(part.index)
            if isinstance(part, FunctionCall) and isinstance(part.func, str):
                if part.func in CASTS:
                    return all(invariant(argument) for argument in part.args)
                if part.func in SIZE_FUNCTIONS and len(part.args) == 1 and isinstance(part.args[0], Identifier):
                    # the size of a field (`array.size(s.h)`) is read again on every pass, as `s.h.size()` is
                    return '.' not in part.args[0].name and part.args[0].name not in assigned \
                        and not self.resizes(body, part.args[0].name)
                receiver, _, method = part.func.rpartition('.')
                if (not part.args and method in SIZE_METHODS and receiver and '.' not in receiver
                        and receiver not in BUILTIN_HEADS):  # `x.size()`
                    return receiver not in assigned and not self.resizes(body, receiver)
                return False
            if isinstance(part, MethodCall) and isinstance(part.object, Identifier) and not part.args \
                    and part.method in SIZE_METHODS:
                return part.object.name not in assigned and not self.resizes(body, part.object.name)
            return False
        return invariant(expression)

    def changes_in_body(self, expression, body, assigned: AbstractSet[str]) -> bool:
        """Whether the body changes something `expression` reads: a name it assigns, a container it resizes.
        Pine v5 and earlier read the end once, so such a loop keeps `pine_range`."""
        names = {node.name.partition('.')[0] for node in walk(expression) if isinstance(node, Identifier)}
        names |= {node.func.partition('.')[0] for node in walk(expression)  # `a.size()`: its receiver
                  if isinstance(node, FunctionCall) and isinstance(node.func, str) and '.' in node.func
                  and node.func.partition('.')[0] not in BUILTIN_HEADS}
        # a container read through a field (`array.size(z.lines)`, `z.lines.size()`), resized the same way in the body
        paths = {node.name for node in walk(expression) if isinstance(node, Identifier) and '.' in node.name}
        paths |= {node.func.rpartition('.')[0] for node in walk(expression)
                  if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func.count('.') >= 2
                  and node.func.partition('.')[0] not in BUILTIN_HEADS}
        # a user function handed the container and resizing its parameter (`shrink(a)` pops it): read once, as Pine
        # does; one that only reads it (`array.slice(h, ...)`) leaves the end as it was
        return bool(names & assigned) or any(self.resizes(body, name, PINE_RESIZING_FUNCTIONS, False)
                                             or self.resized_by_a_function(body, name)
                                             for name in names | paths)

    def resized_by_a_function(self, body, name: str, depth: int = 0) -> bool:
        """Whether `body` hands the container `name` to a user function (an argument, or the receiver of a method)
        whose own body resizes the parameter it arrives as, followed through the functions that function calls."""
        if depth > 8:
            return False
        for node in walk(body):
            if not (isinstance(node, FunctionCall) and isinstance(node.func, str)):
                continue
            head, _, function = node.func.rpartition('.')
            if node.func in self.functions:
                arguments, declarations = list(node.args), self.functions[node.func]
            elif head == name and function in self.functions:
                arguments, declarations = [Identifier(name=name)] + list(node.args), self.functions[function]
            else:
                continue
            for index, argument in enumerate(arguments):
                if not (isinstance(argument, Identifier) and argument.name == name):
                    continue
                for declaration in declarations:
                    if index >= len(declaration.params) or not isinstance(declaration.body, list):
                        continue
                    parameter = declaration.params[index].name
                    if self.resizes(declaration.body, parameter, PINE_RESIZING_FUNCTIONS, False) \
                            or self.resized_by_a_function(declaration.body, parameter, depth + 1):
                        return True
        return False

    def has_state(self, expression) -> bool:
        """Whether reading `expression` changes or depends on state: it calls a user function or method, a `ta.*`,
        `request.*` or `strategy.*` function, or a function that resizes a container. Pine v5 and earlier read
        such an end once, where reading it on every pass would run it again."""
        stateful_heads = ('ta', 'request', 'strategy')
        for node in walk(expression):
            if isinstance(node, FunctionCall) and isinstance(node.func, str):
                head, _, function = node.func.rpartition('.')
                if node.func in self.function_names or function in self.function_names \
                        or head.partition('.')[0] in stateful_heads or function in PINE_RESIZING_FUNCTIONS:
                    return True
            elif isinstance(node, MethodCall) and (node.method in self.function_names
                                                   or node.method in PINE_RESIZING_FUNCTIONS):
                return True
        return False

    def resizes(self, body, name: str, resizing: AbstractSet[str] = RESIZING_FUNCTIONS,
                through_functions: bool = True) -> bool:
        """Whether `body` may change the size of the container `name` (by the list of resizing
        functions unless `resizing` says otherwise); a user function given it counts unless
        `through_functions` is false."""
        def names(arguments) -> bool:
            return any(isinstance(argument, Identifier) and argument.name == name for argument in arguments)
        for node in walk(body):
            if isinstance(node, FunctionCall) and isinstance(node.func, str):
                head, _, function = node.func.rpartition('.')
                if function in resizing and head in ('array', 'matrix', 'map') and names(node.args[:1]):
                    return True
                if function in resizing and head == name:  # `x.push(...)`
                    return True
                if through_functions and function in self.function_names and (head == name or names(node.args)):
                    return True
                if through_functions and node.func in self.function_names and names(node.args):
                    return True
            elif isinstance(node, MethodCall):
                if node.method in resizing and isinstance(node.object, Identifier) \
                        and node.object.name == name:
                    return True
                if through_functions and node.method in self.function_names and (names(node.args) or (
                        isinstance(node.object, Identifier) and node.object.name == name)):
                    return True
        return False


CASTS = frozenset({'int', 'float', 'bool', 'string', 'color'})
SIZE_METHODS = frozenset({'size', 'rows', 'columns'})
BUILTIN_HEADS = frozenset({'array', 'matrix', 'map', 'str', 'math', 'ta', 'request', 'timeframe', 'syminfo',
                           'strategy', 'line', 'label', 'box', 'table', 'polyline', 'linefill', 'color', 'chart'})


def _calls_a_function(expression) -> bool:
    """`expression` calls a function outside a namespace that is not a cast: a user function, `nz`, `na` (`f()`,
    `nz(n)`, `na(n) ? 1 : 2`); a namespaced builtin (`math.max`, `str.length`, `array.get`) does not count."""
    return any(isinstance(node, FunctionCall) and (isinstance(node.func, NaLiteral) or (
        isinstance(node.func, str) and '.' not in node.func and node.func not in CASTS)) for node in walk(expression))


def _assigned_in(body) -> Set[str]:
    """The head of every name a statement in `body` assigns or declares."""
    names: Set[str] = set()
    for node in walk(body):
        if isinstance(node, Assignment):
            target = node.target
            names.update(target.names if isinstance(target, TupleDestructure) else [target])
        elif isinstance(node, Reassignment):
            names.add(node.target)
        elif isinstance(node, (VarDecl, VaripDecl)):
            names.add(node.name)
        elif isinstance(node, ForLoop):
            names.add(node.var)
        elif isinstance(node, ForInLoop):
            names.update(node.vars)
    return {name.partition('.')[0] for name in names if isinstance(name, str)}
