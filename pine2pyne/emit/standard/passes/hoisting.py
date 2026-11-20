"""Temporaries hoisted in front of a statement (Pine v4/v5).

Pine v4 and v5 evaluate both operands of ``and``/``or``; Python evaluates the
second only when needed. pine2pyne keeps a stateful call (``ta.*``,
``request.*``, a user function) in the second or a later operand running on
every bar by computing it first, into ``<name>_<n>``; a time function given a
time (``month(time)``) counts as stateful, and a stateful call in the arguments
of one is computed before it::

    crossover_1 = ta.crossover(close, fastSma)
    long = fastAboveSlow and crossover_1

The number counts per name within one block. Pine v4 ``iff(c, a, b)`` is
``a if c else b``; an ``iff`` nested in its arguments is hoisted the same way
(``iff_1 = ...``), innermost first, after the calls.
"""
from __future__ import annotations

import ast
from typing import Callable

from .base import PassContext

_BLOCK_FIELDS = ('body', 'orelse', 'finalbody')
#: the time functions hoisted like a stateful call when they are given a time
TIME_FUNCTIONS = frozenset({'time', 'time_close', 'year', 'month', 'weekofyear', 'dayofmonth', 'dayofweek',
                            'hour', 'minute', 'second'})


#: the math functions that keep state from bar to bar (a rolling window)
STATEFUL_MATH = frozenset({'math.sum'})


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return f'{_dotted(node.value)}.{node.attr}'
    return node.id if isinstance(node, ast.Name) else ''


class _BlockNames:
    """``<name>_<n>`` numbering within one block, past the names the module binds itself (a script's own `cum_1`
    makes the temporary of `ta.cum(...)` `cum_2`)."""

    def __init__(self, taken: frozenset = frozenset()) -> None:
        self._counts: dict[str, int] = {}
        self._taken = taken
        self.own: set[str] = set()

    def next(self, stem: str) -> str:
        self._counts[stem] = self._counts.get(stem, 0) + 1
        while f'{stem}_{self._counts[stem]}' in self._taken:
            self._counts[stem] += 1
        name = f'{stem}_{self._counts[stem]}'
        self.own.add(name)
        return name


def _statement_expressions(statement: ast.stmt) -> list[tuple[ast.AST, str]]:
    """(owner, field) of the expressions a statement evaluates before its body runs;
    an `elif` condition belongs to its `if`."""
    if isinstance(statement, ast.If):
        found = [(statement, 'test')]
        orelse = statement.orelse
        while len(orelse) == 1 and isinstance(orelse[0], ast.If):
            found.append((orelse[0], 'test'))
            orelse = orelse[0].orelse
        return found
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Return)):
        return [(statement, 'value')] if getattr(statement, 'value', None) is not None else []
    if isinstance(statement, ast.For):
        return [(statement, 'iter')]
    return []


def _nested_blocks(statement: ast.stmt):
    """The blocks a statement holds, elif chains included."""
    for field in _BLOCK_FIELDS:
        block = getattr(statement, field, None)
        if isinstance(block, list) and block:
            if isinstance(statement, ast.If) and field == 'orelse' and len(block) == 1 and isinstance(block[0], ast.If):
                yield from _nested_blocks(block[0])
                yield block[0].body  # the elif's own body
                continue
            yield block


def _is_block_result_initial(statement: ast.stmt) -> bool:
    """`__block_result__ = na` (or `False`, or a tuple of na): the start of an if/switch value."""
    return isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id.startswith('__block_result')


def rewrite_blocks(tree: ast.Module, rewrite: Callable[[ast.stmt, _BlockNames], list[ast.stmt]],
                   context: PassContext | None = None) -> None:
    """Apply ``rewrite`` (statement -> statements to insert before it) to every block. A temporary of a
    nested block whose name an enclosing block of the same function holds too, before or after it,
    takes its block's id, ``iff_1__0000000b__``; a sibling block's does not count. A block nested in a
    function of main() counts main()'s own body as enclosing it; the function's body itself does not."""
    line_blocks = getattr(getattr(context.unit, 'transformed', None), 'line_blocks', {}) if context else {}
    origins = context.origins if context else {}

    def block_id(block: list[ast.stmt]) -> int:
        for statement in block:
            pine_line = origins.get(getattr(statement, 'lineno', 0), 0)
            if pine_line in line_blocks:
                return line_blocks[pine_line]
        return 0

    taken = frozenset(node.id for node in ast.walk(tree)
                      if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))

    def visit(block: list[ast.stmt], ancestors: list, blocks: list) -> None:
        names = _BlockNames(taken)
        index = 0
        while index < len(block):
            statement = block[index]
            inserted = rewrite(statement, names)
            # an `if` computing a block result: what its condition needs comes before the result's initial
            # value (`x_1 = f(a)`, `__block_result__ = na`, `if c and x_1:`)
            at = index - 1 if inserted and isinstance(statement, ast.If) and index > 0 \
                and _is_block_result_initial(block[index - 1]) else index
            for offset, new in enumerate(inserted):
                block.insert(at + offset, ast.copy_location(new, statement))
            index += len(inserted)
            if not isinstance(statement, (ast.FunctionDef, ast.ClassDef)):  # functions: visited on their own
                for nested in list(_nested_blocks(statement)):
                    visit(nested, ancestors + [names], blocks)
            index += 1
        blocks.append((block, names, ancestors))

    # the temporaries of main()'s own body: a block nested in one of its functions that holds one takes its id
    # too (`change_1__00000004__` in a function's `if`, `change_1` in main); the function's own body does not
    main_names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            blocks: list = []
            visit(node.body, [], blocks)
            for block, names, ancestors in blocks:
                held = set().union(main_names, *(outer.own for outer in ancestors)) if ancestors else set()
                shared = names.own & held
                number = block_id(block) if shared else 0
                if number:
                    _rename_in_block(block, {name: f'{name}__{number:08x}__' for name in shared})
            if node.name == 'main':
                main_names = set(blocks[-1][1].own)


def _rename_in_block(block: list[ast.stmt], renames: dict[str, str]) -> None:
    """The temporaries of ``block`` renamed where the block itself assigns and reads them."""
    def rename(node: ast.AST) -> None:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name) and inner.id in renames:
                inner.id = renames[inner.id]
    for statement in block:
        target = (statement.targets[0] if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                  else statement.target if isinstance(statement, ast.AnnAssign) else None)
        if isinstance(target, ast.Name) and target.id in renames:
            rename(statement)
            continue
        for owner, field in _statement_expressions(statement):
            rename(getattr(owner, field))


class StrictBooleanHoistingPass:
    """Pine v4/v5: stateful calls in a later operand of ``and``/``or`` are computed first."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        if context.pine_version >= 6:
            return
        user_functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

        def stateful(call: ast.Call) -> bool:
            dotted = _dotted(call.func)
            if dotted in TIME_FUNCTIONS:
                return bool(call.args or call.keywords)  # `month(time)`, `hour(time, tz)`
            # Pine v4's `sum(x, n)` is not hoisted, v5's `math.sum` is
            return (dotted.startswith(('ta.', 'request.')) or (dotted in STATEFUL_MATH and context.pine_version >= 5)
                    or dotted in user_functions) and dotted != 'main'

        def is_iff(node: ast.AST) -> bool:
            return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'iff'
                    and 'iff' not in user_functions and len(node.args) == 3 and not node.keywords)

        def has_stateful(node: ast.AST) -> bool:
            return any(isinstance(inner, ast.Call) and stateful(inner) for inner in ast.walk(node))

        def rewrite(statement: ast.stmt, names: _BlockNames) -> list[ast.stmt]:
            hoisted: list[ast.stmt] = []

            def hoist_in(node: ast.AST) -> ast.AST:
                """``node`` with its outermost stateful calls replaced by temporaries."""
                if isinstance(node, ast.Call) and stateful(node):
                    # a stateful call in its arguments first: `ema_1 = ta.ema(close, 5)`, `change_1 = ta.change(ema_1)`
                    for index, argument in enumerate(node.args):
                        node.args[index] = visit_operand_free(argument, hoist=True)
                    for keyword in node.keywords:
                        keyword.value = visit_operand_free(keyword.value, hoist=True)
                    name = names.next(_call_name(node))
                    hoisted.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=node))
                    return ast.Name(id=name, ctx=ast.Load())
                return visit_operand_free(node, hoist=True)

            def visit_operand_free(node: ast.AST, hoist: bool = False, in_arm: bool = False) -> ast.AST:
                """Walk ``node``; BoolOps inside it hoist from their later operands, but not in an arm of a
                ternary (`c ? (a and ta.rsi(x, 14) < 30) : true`), which runs only on the bars that take
                it."""
                if isinstance(node, ast.IfExp):
                    node.test = visit_operand_free(node.test, hoist, in_arm)
                    node.body = visit_operand_free(node.body, hoist, True)
                    node.orelse = visit_operand_free(node.orelse, hoist, True)
                    return node
                if hoist and isinstance(node, ast.Call) and _dotted(node.func) == 'inline_series' \
                        and len(node.args) == 2 and isinstance(node.args[0], ast.Call) and stateful(node.args[0]):
                    # `inline_series(ta.lowest(l, n), n)`: the call is hoisted as a series and read with history,
                    # `lowest_1: Series = ta.lowest(l, n)`, `lowest_1[n]` (MightyZinger/Chikou)
                    temporary = hoist_in(node.args[0])
                    hoisted[-1] = ast.AnnAssign(target=ast.Name(id=temporary.id, ctx=ast.Store()),
                                                annotation=ast.Name(id='Series', ctx=ast.Load()),
                                                value=hoisted[-1].value, simple=1)
                    return ast.Subscript(value=temporary, slice=visit_operand_free(node.args[1]), ctx=ast.Load())
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'iff'
                        and 'iff' not in user_functions and len(node.args) == 3 and not node.keywords):
                    # Pine v4 `iff(c, a, b)` is a function: its arguments run on every bar, so their stateful
                    # calls are computed first (`sma_1 = ta.sma(close, 5)`, `b = sma_1 if c else 0`)
                    node.args[0] = visit_operand_free(node.args[0], True)  # the condition's calls too
                    node.args[1:] = [hoist_in(argument) for argument in node.args[1:]]
                    return node
                if isinstance(node, ast.BoolOp) and not in_arm:
                    # in a later operand of an outer and/or, the first operand of this one runs on every bar too
                    node.values[0] = visit_operand_free(node.values[0], hoist)
                    later = list(range(1, len(node.values)))
                    if context.pine_version <= 4 and sum(is_iff(node.values[index]) for index in later) > 1 \
                            and not any(not is_iff(node.values[index]) and has_stateful(node.values[index])
                                        for index in later):
                        # Pine v4: the calls of the `iff`s of one chain are computed last `iff` first
                        # (`crossover_1` of the third operand before `crossunder_1` of the second)
                        later.reverse()
                    for index in later:
                        node.values[index] = hoist_in(node.values[index])
                    return node
                if hoist and isinstance(node, ast.Call) and stateful(node):
                    return hoist_in(node)
                for field, value in ast.iter_fields(node):
                    if isinstance(value, ast.AST) and not isinstance(value, (ast.expr_context, ast.operator)):
                        setattr(node, field, visit_operand_free(value, hoist, in_arm))
                    elif isinstance(value, list):
                        setattr(node, field, [visit_operand_free(item, hoist, in_arm) if isinstance(item, ast.AST)
                                              else item for item in value])
                return node

            for owner, field in _statement_expressions(statement):
                setattr(owner, field, visit_operand_free(getattr(owner, field)))
            return hoisted

        rewrite_blocks(tree, rewrite, context)


class LegacyIffPass:
    """Pine v4 ``iff(c, a, b)`` is ``a if c else b``; a nested ``iff`` is hoisted to ``iff_<n>``."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        if context.pine_version >= 5:
            return

        def is_iff(node: ast.AST) -> bool:
            return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'iff'
                    and len(node.args) == 3 and not node.keywords)

        def rewrite(statement: ast.stmt, names: _BlockNames) -> list[ast.stmt]:
            hoisted: list[ast.stmt] = []

            def ternary(call: ast.Call, nested: bool) -> ast.expr:
                condition, then, otherwise = (convert(argument, nested=True) for argument in call.args)
                expression = ast.IfExp(test=condition, body=then, orelse=otherwise)
                if not nested:
                    return expression
                name = names.next('iff')
                hoisted.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=expression))
                return ast.Name(id=name, ctx=ast.Load())

            def convert(node: ast.AST, nested: bool = False) -> ast.AST:
                if is_iff(node):
                    return ternary(node, nested)
                for field, value in ast.iter_fields(node):
                    if isinstance(value, ast.AST):
                        setattr(node, field, convert(value, nested))
                    elif isinstance(value, list):
                        setattr(node, field, [convert(item, nested) if isinstance(item, ast.AST) else item
                                              for item in value])
                return node

            for owner, field in _statement_expressions(statement):
                setattr(owner, field, convert(getattr(owner, field)))
            return hoisted

        rewrite_blocks(tree, rewrite, context)


_SERIES_WRAPPERS = ('Persistent[', 'PersistentSeries[', 'Series[', 'IBPersistent[', 'IBPersistentSeries[')


def _unwrapped(annotation: str | None) -> str | None:
    """`Persistent[T]` -> `T`: the type a declaration's value has."""
    while annotation:
        annotation = annotation.strip('\'"')
        wrapper = next((prefix for prefix in _SERIES_WRAPPERS if annotation.startswith(prefix)), None)
        if wrapper is None or not annotation.endswith(']'):
            return annotation
        annotation = annotation[len(wrapper):-1]
    return annotation


class FieldHistoryPass:
    """Pine v5: the history of a field of a user-defined type, ``t.x[1]``, reads the field's own series.
    pine2pyne computes it first, into ``field_<n>: Series[<field type>] = t.x``, and reads ``field_<n>[1]``;
    ``n`` counts across the module in source order, and the declaration goes right before the statement
    (an ``if`` with all its ``elif`` conditions, a ``for`` before its ``pine_loop``).
    Python has no history of an attribute: ``t.x[1]`` indexes the float."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        classes = {node.name: {field.target.id: ast.unparse(field.annotation) for field in node.body
                               if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)}
                   for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        if not classes:
            return
        module_names = {alias.asname or alias.name.split('.')[0] for node in tree.body
                        if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        module_names |= {target.id for node in tree.body if isinstance(node, ast.Assign)
                         for target in node.targets if isinstance(target, ast.Name)}
        module_names |= set(classes)
        counter = iter(range(1 << 30))

        def chain_of(node: ast.expr) -> list[str] | None:
            """`o.inner.x` -> ['o', 'inner', 'x'] when the root is a name of the script's own."""
            names = []
            while isinstance(node, ast.Attribute):
                names.append(node.attr)
                node = node.value
            if not names or not isinstance(node, ast.Name) or node.id in module_names:
                return None
            return [node.id] + names[::-1]

        def field_type(chain: list[str], types: dict[str, str]) -> str | None:
            owner = types.get(chain[0])
            if owner not in classes:  # an untyped object: the one class with that field
                owners = [name for name, fields in classes.items() if chain[1] in fields]
                owner = owners[0] if len(owners) == 1 else None
            annotation = None
            for attribute in chain[1:]:
                if owner not in classes or attribute not in classes[owner]:
                    return None
                annotation = classes[owner][attribute]
                owner = _unwrapped(annotation)
            return _unwrapped(annotation)

        def rewritten(node: ast.AST, inserted: list, types: dict[str, str]) -> ast.AST:
            for field, value in ast.iter_fields(node):
                if isinstance(value, ast.AST) and not isinstance(value, (ast.expr_context, ast.operator)):
                    setattr(node, field, rewritten(value, inserted, types))
                elif isinstance(value, list):
                    setattr(node, field, [rewritten(item, inserted, types) if isinstance(item, ast.AST) else item
                                          for item in value])
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                chain = chain_of(node.value)
                if chain is not None:
                    name = f'field_{next(counter)}'
                    found = field_type(chain, types)
                    annotation = ast.parse(f'Series[{found}]' if found else 'Series', mode='eval').body
                    inserted.append(ast.AnnAssign(target=ast.Name(id=name, ctx=ast.Store()), annotation=annotation,
                                                  value=node.value, simple=1))
                    node.value = ast.Name(id=name, ctx=ast.Load())
            return node

        def typed_names(statement: ast.stmt, types: dict[str, str]) -> None:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                types[statement.target.id] = _unwrapped(ast.unparse(statement.annotation))

        def visit(block: list[ast.stmt], types: dict[str, str]) -> None:
            index = 0
            while index < len(block):
                statement = block[index]
                if isinstance(statement, ast.FunctionDef):
                    inner = dict(types)
                    for argument in statement.args.args:
                        if argument.annotation is not None:
                            inner[argument.arg] = _unwrapped(ast.unparse(argument.annotation))
                    visit(statement.body, inner)
                    index += 1
                    continue
                if isinstance(statement, ast.ClassDef):
                    index += 1
                    continue
                inserted: list[ast.stmt] = []
                expressions = _statement_expressions(statement)
                if isinstance(statement, ast.While):
                    expressions = [(statement, 'test')]
                for owner, field in expressions:
                    setattr(owner, field, rewritten(getattr(owner, field), inserted, types))
                position = index
                if isinstance(statement, ast.While) and index and _is_loop_setup(block[index - 1]):
                    position = index - 1  # before `__loop_<n>__ = pine_loop(...)`
                for offset, new in enumerate(inserted):
                    block.insert(position + offset, ast.copy_location(new, statement))
                index += len(inserted)
                typed_names(statement, types)
                for nested in list(_nested_blocks(statement)):
                    visit(nested, types)
                index += 1

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                visit(node.body, {})


def _is_loop_setup(statement: ast.stmt) -> bool:
    """`__loop_<n>__ = pine_loop(...)`, the set-up of the `while` a Pine `for` becomes."""
    return (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name) and statement.value.func.id == 'pine_loop')
