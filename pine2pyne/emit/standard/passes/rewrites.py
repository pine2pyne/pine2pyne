"""The standard-layout rewrites, one class per difference.

Each docstring says what is written, and for which Pine version.
"""
from __future__ import annotations

import ast
import re

from .base import PassContext, function_scopes, main_function, own_nodes, statements_in_order

_HEX_COLOR = re.compile(r'#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})')


def _double_quoted(text: str) -> str:
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'


class DeclarationCallShapePass:
    """``@script.indicator(...)`` keeps the call shape of the Pine declaration:
    the arguments the source passed positionally stay positional
    (``indicator('T', 'S', overlay=true)`` -> ``('T', 'S', overlay=True)``), a
    title passed by keyword stays ``title=``, and a Pine v6 title keeps its quotes."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        main = main_function(tree)
        declaration = getattr(context.unit.pine_script, 'script_decl', None)
        if main is None or declaration is None:
            return
        for decorator in main.decorator_list:
            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == 'script'):
                self._reshape(decorator, declaration, context)

    #: how many of a declaration's leading parameters pynecore's decorator takes in Pine's order: its
    #: `script.strategy` has `calc_on_every_history_tick` after `calc_on_every_tick`, so Pine's 10th positional
    #: (`max_bars_back`) and every later one must be passed by keyword
    ALIGNED_POSITIONALS = {'strategy': 9}

    @classmethod
    def _reshape(cls, call: ast.Call, declaration, context: PassContext) -> None:
        positional = getattr(declaration, 'positional_count', 0)
        positional = min(positional, cls.ALIGNED_POSITIONALS.get(call.func.attr, positional))
        order = tuple(getattr(declaration, 'keyword_order', ()))
        title = call.args[0] if call.args else None
        if title is None:  # a title that is an expression: the first keyword
            if positional:
                call.args = [keyword.value for keyword in call.keywords[:positional]]
                call.keywords = call.keywords[positional:]
            return
        if positional == 0:
            call.args = []
            at = order.index('title') if 'title' in order else 0
            call.keywords = call.keywords[:at] + [ast.keyword(arg='title', value=title)] + call.keywords[at:]
        elif positional > 1:
            taken = call.keywords[:positional - 1]
            call.args = [title] + [keyword.value for keyword in taken]
            call.keywords = call.keywords[positional - 1:]
        if (title is not None and getattr(declaration, 'title_double_quoted', False)
                and context.pine_version >= 6 and isinstance(title, ast.Constant)):
            context.spellings[id(title)] = _double_quoted(title.value)


class StrategyDefaultsPass:
    """A Pine v4/v5 ``strategy()`` is compiled with its version's defaults made
    explicit, appended after the source's own arguments in this order: ``margin_long=0, margin_short=0,
    initial_capital=100000`` (v4) or ``1000000`` (v5), ``default_qty_type=strategy.fixed,
    default_qty_value=1, close_by_id=True``. An argument the source gives keeps its value."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        if context.pine_version >= 6:
            return
        main = main_function(tree)
        for decorator in main.decorator_list if main else ():
            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == 'strategy'):
                self._add_defaults(decorator, context.pine_version)

    @staticmethod
    def _add_defaults(call: ast.Call, version: int) -> None:
        given = {keyword.arg for keyword in call.keywords}
        defaults = [
            ('margin_long', ast.Constant(0)),
            ('margin_short', ast.Constant(0)),
            ('initial_capital', ast.Constant(100000 if version <= 4 else 1000000)),
            ('default_qty_type', ast.Attribute(value=ast.Name(id='strategy', ctx=ast.Load()), attr='fixed',
                                               ctx=ast.Load())),
            ('default_qty_value', ast.Constant(1)),
            ('close_by_id', ast.Constant(True)),
        ]
        added = [ast.keyword(arg=name, value=value) for name, value in defaults if name not in given]
        # added before the imports are reconciled, so the `strategy` that `strategy.fixed` reads is imported
        tail = [keyword for keyword in call.keywords if keyword.arg == 'na_bool']
        call.keywords = [keyword for keyword in call.keywords if keyword.arg != 'na_bool'] + added + tail


class NaBoolFlagPass:
    """``na_bool=True`` (Pine's three-state bool) is kept for a Pine v4 script.
    A Pine v5 script keeps it only when it reads a named bool where it may be
    na: ``b[1]`` or ``na(b)``. A Pine v6 bool is never na, so a v6 script never
    has it, not even for ``na()`` of a bool collection or a bool field read
    with history."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        if context.pine_version != 5 or getattr(context.unit.transformed, 'reads_na_of_bool', False):
            return
        main = main_function(tree)
        for decorator in main.decorator_list if main else ():
            if isinstance(decorator, ast.Call):
                decorator.keywords = [keyword for keyword in decorator.keywords if keyword.arg != 'na_bool']


class MathNamespacePass:
    """``abs``, ``max``, ``min``, ``pow`` and ``round`` are always called through
    ``math.``, never as the Python builtins. A
    variable of that name does not change the callee: Pine calls the builtin
    (`max := max(max, src)`), only a function the script declares takes the call."""

    NAMES = frozenset({'abs', 'max', 'min', 'pow', 'round'})

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        bound = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in self.NAMES and node.func.id not in bound):
                node.func = ast.copy_location(ast.Attribute(
                    value=ast.Name(id='math', ctx=ast.Load()), attr=node.func.id, ctx=ast.Load()), node.func)


class LegacyExitStopPass:
    """Pine v4: a ``strategy.exit`` given no price (profit, limit, loss, stop or a trail) is
    passed ``stop=na``, the runtime's default for it (v5 passes nothing)."""

    PRICES = frozenset({'profit', 'limit', 'loss', 'stop', 'trail_price', 'trail_points', 'trail_offset'})
    #: positional slots before Pine v4's `profit`: id, from_entry, qty, qty_percent
    SLOTS_BEFORE_PRICES = 4

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        if context.pine_version > 4:
            return
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and _dotted_name(node.func) == 'strategy.exit'
                    and len(node.args) <= self.SLOTS_BEFORE_PRICES
                    and not any(keyword.arg in self.PRICES for keyword in node.keywords)):
                node.keywords.append(ast.keyword(arg='stop', value=ast.Name(id='na', ctx=ast.Load())))


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return f'{_dotted_name(node.value)}.{node.attr}'
    return node.id if isinstance(node, ast.Name) else ''


class FlatBoolOpPass:
    """``a and (b and c)`` is one ``and`` of three operands: no ``and`` is written
    directly inside an ``and`` (nor ``or`` in ``or``), though Pine sources
    write them."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.BoolOp):
                node.values = self._flat(node)

    def _flat(self, node: ast.BoolOp) -> list:
        values = []
        for value in node.values:
            if isinstance(value, ast.BoolOp) and type(value.op) is type(node.op):
                values.extend(self._flat(value))
            else:
                values.append(value)
        return values


class HexColorPass:
    """A Pine colour literal is ``color.new('#RRGGBB')``, upper case."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        tree.body[:] = [self._visit(node) for node in tree.body]

    def _visit(self, node):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return node
        return _HexColorRewriter().visit(node)


class _HexColorRewriter(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str) and _HEX_COLOR.fullmatch(node.value):
            spelled = node.value.upper()
            if len(spelled) == 9 and spelled.endswith('FF'):
                spelled = spelled[:7]  # an opaque alpha is left out: '#00FF00FF' is '#00FF00'
            call = ast.Call(
                func=ast.Attribute(value=ast.Name(id='color', ctx=ast.Load()), attr='new', ctx=ast.Load()),
                args=[ast.Constant(value=spelled)], keywords=[])
            return ast.copy_location(call, node)
        return node


__all__ = [
    'DeclarationCallShapePass',
    'FlatBoolOpPass',
    'MathNamespacePass',
    'LegacyExitStopPass',
    'NaBoolFlagPass',
    'StrategyDefaultsPass',
    'HexColorPass',
    'statements_in_order',
]


_BLOCK_RESULT_NAME = re.compile(r'__block_result(_\d+)?__')


class BlockKeepPass:
    """An ``if`` used as a value without a final ``else`` (``x = if c ...``,
    ``else if`` arms allowed), whose name is read with history (``x[1]``,
    a ``Series`` annotation), keeps its last value on the bars where no arm
    runs: TradingView computes it so (scripts verified value for value
    against TradingView rely on it, where Pine's manual promises na), and
    the temporary is written as a persistent series::

        __block_keep_1__: PersistentSeries = na
        if c:
            __block_keep_1__ = close
        x: Series = __block_keep_1__

    Such temporaries are numbered on their own, from 1 in the module's order;
    the number the block result would have had is left unused. An ``if`` with
    an ``else``, a ``switch``, a value read without history and a block inside
    a loop stay as they are."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        found: list[tuple[int, list, int]] = []
        self._visit(tree, False, found)
        for number, (_, block, index) in enumerate(sorted(found, key=lambda item: item[0]), start=1):
            self._keep(block, index, f'__block_keep_{number}__')

    def _visit(self, node: ast.AST, in_loop: bool, found: list) -> None:
        for field in ('body', 'orelse', 'finalbody'):
            block = getattr(node, field, None)
            if isinstance(block, list) and not in_loop:
                for index in range(len(block) - 2):
                    slot = self._slot(block, index)
                    if slot is not None:
                        found.append((slot, block, index))
        for child in ast.iter_child_nodes(node):
            self._visit(child, in_loop or isinstance(node, (ast.For, ast.AsyncFor, ast.While)), found)

    @staticmethod
    def _slot(block: list, index: int) -> int | None:
        """The number of the block result at ``block[index]`` when it starts such an ``if``, else None."""
        initial, chain, taken = block[index], block[index + 1], block[index + 2]
        if not (isinstance(initial, ast.Assign) and len(initial.targets) == 1
                and isinstance(initial.targets[0], ast.Name)
                and isinstance(initial.value, ast.Name) and initial.value.id == 'na'
                and isinstance(chain, ast.If)):
            return None
        matched = _BLOCK_RESULT_NAME.fullmatch(initial.targets[0].id)
        if not matched:
            return None
        last = chain
        while len(last.orelse) == 1 and isinstance(last.orelse[0], ast.If):
            last = last.orelse[0]
        if last.orelse:
            return None  # every arm gives a value
        if not (isinstance(taken, ast.AnnAssign) and isinstance(taken.value, ast.Name)
                and taken.value.id == initial.targets[0].id and _is_series_annotation(taken.annotation)):
            return None
        return int(matched.group(1)[1:]) if matched.group(1) else 0

    @staticmethod
    def _keep(block: list, index: int, kept: str) -> None:
        initial, chain, taken = block[index], block[index + 1], block[index + 2]
        temporary = initial.targets[0].id
        block[index] = ast.copy_location(ast.AnnAssign(
            target=ast.Name(id=kept, ctx=ast.Store()), annotation=ast.Name(id='PersistentSeries', ctx=ast.Load()),
            value=ast.Name(id='na', ctx=ast.Load()), simple=1), initial)
        for name in ast.walk(chain):
            if isinstance(name, ast.Name) and name.id == temporary:
                name.id = kept
        taken.value.id = kept


def _is_series_annotation(annotation: ast.expr) -> bool:
    head = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    return isinstance(head, ast.Name) and head.id == 'Series'


#: the helpers ``syminfo.ticker(symbol)`` and ``syminfo.prefix(symbol)`` are written as, and their bodies
_SYMBOL_PART_HELPERS = {
    'ticker': ('syminfo_ticker', 'def syminfo_ticker(symbol):\n    return symbol.partition(":")[2]\n'),
    'prefix': ('syminfo_prefix',
               'def syminfo_prefix(symbol):\n    head, colon, _ = symbol.partition(":")\n'
               '    return head if colon else ""\n'),
}


class SyminfoSymbolPartsPass:
    """``syminfo.ticker(symbol)`` and ``syminfo.prefix(symbol)`` (Pine v5/v6) are functions beside the
    variables of the same name: the symbol's name after the exchange prefix and the prefix itself. PyneCore
    has only the variables (strings of the chart's symbol), so a call fails with ``'str' object is not
    callable``. Each call becomes a call of a helper defined at the start of ``main``, with TradingView's
    results: ``"NASDAQ:AAPL"`` gives ``"AAPL"`` and ``"NASDAQ"``, a symbol without a prefix (``"AAPL"``)
    gives an empty string for both; TradingView does not look up the default exchange."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        bound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        if 'syminfo' in bound:
            return  # a variable of that name: not the namespace
        used: set[str] = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'syminfo'
                    and node.func.attr in _SYMBOL_PART_HELPERS):
                continue
            arguments = list(node.args) + [keyword.value for keyword in node.keywords if keyword.arg == 'symbol']
            if len(arguments) != 1 or len(node.args) + len(node.keywords) != 1:
                continue
            name = _SYMBOL_PART_HELPERS[node.func.attr][0]
            node.func = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node.func)
            node.args, node.keywords = arguments, []
            used.add(node.func.id)
        main = main_function(tree)
        if not used or main is None:
            return
        for attr in ('prefix', 'ticker'):
            name, source = _SYMBOL_PART_HELPERS[attr]
            if name in used:
                main.body.insert(0, ast.copy_location(ast.parse(source).body[0], main.body[0]))
