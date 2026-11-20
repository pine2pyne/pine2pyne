"""Built-in calls spelled the way pine2pyne spells them."""
from __future__ import annotations

import ast

from .base import PassContext

#: ``ta`` members that are series values, not calls (``ta.tr(true)`` is the call form)
TA_SERIES = frozenset({'tr', 'vwap', 'obv', 'accdist', 'iii', 'nvi', 'pvi', 'pvt', 'wad', 'wvad'})
#: Pine v4 hline and plot styles were integers
_HLINE_STYLES = {0: 'style_solid', 1: 'style_dotted', 2: 'style_dashed'}
_PLOT_STYLES = {0: 'style_line', 1: 'style_stepline', 2: 'style_histogram', 3: 'style_cross', 4: 'style_area',
                5: 'style_columns', 6: 'style_circles'}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return f'{_dotted(node.value)}.{node.attr}'
    return node.id if isinstance(node, ast.Name) else ''


def _attribute(base: str, member: str) -> ast.Attribute:
    return ast.Attribute(value=ast.Name(id=base, ctx=ast.Load()), attr=member, ctx=ast.Load())


def _bound_names(tree: ast.Module) -> set[str]:
    bound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    bound |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    bound |= {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    return bound


class _Replacer(ast.NodeTransformer):
    def __init__(self, replace):
        self._replace = replace

    def generic_visit(self, node):
        node = super().generic_visit(node)
        replaced = self._replace(node)
        return node if replaced is None else ast.copy_location(replaced, node)


class TaSeriesPass:
    """``ta.tr``, ``ta.vwap``, ``ta.obv``, ... are values: no call without arguments;
    a Pine v4 bare ``tr``/``vwap``/... is the ``ta.`` value."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        bound = _bound_names(tree)
        legacy = context.pine_version <= 4

        def replace(node):
            if (isinstance(node, ast.Call) and not node.args and not node.keywords
                    and isinstance(node.func, ast.Attribute) and node.func.attr in TA_SERIES
                    and _dotted(node.func.value) == 'ta'):
                return node.func
            if (legacy and isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id in TA_SERIES and node.id not in bound):
                return _attribute('ta', node.id)
            return None
        _Replacer(replace).visit(tree)


class LegacyStrategyDirectionPass:
    """Pine v4 ``strategy.entry(id, true)`` / ``long=true``: the direction is
    ``strategy.long`` / ``strategy.short`` (``direction=`` for the keyword); a bool computed at run time chooses
    between them, ``strategy.long if c else strategy.short``."""

    FUNCTIONS = frozenset({'strategy.entry', 'strategy.order'})

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        if context.pine_version > 4:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted(node.func) in self.FUNCTIONS:
                if len(node.args) >= 2:
                    node.args[1] = self._direction(node.args[1])
                for keyword in node.keywords:
                    if keyword.arg == 'long':
                        keyword.arg, keyword.value = 'direction', self._direction(keyword.value)

    @staticmethod
    def _direction(value: ast.expr) -> ast.expr:
        if isinstance(value, ast.Constant) and value.value in (True, False, 1, 0):
            return _attribute('strategy', 'long' if value.value else 'short')
        if isinstance(value, ast.Attribute) and _dotted(value) in ('strategy.long', 'strategy.short'):
            return value
        # a bool computed at run time: `long=IS_LONG` is `strategy.long if IS_LONG else strategy.short`
        return ast.IfExp(test=value, body=_attribute('strategy', 'long'), orelse=_attribute('strategy', 'short'))


class LegacyLineWidthPass:
    """Pine v4/v5: a ``linewidth`` that is not a constant of at least 1 is ``math.max(1, width)``."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        if context.pine_version >= 6:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == 'linewidth':
                node.value = self._at_least_one(node.value)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'plot' \
                    and len(node.args) > 3:  # plot(series, title, color, linewidth, ...)
                node.args[3] = self._at_least_one(node.args[3])

    @classmethod
    def _at_least_one(cls, width: ast.expr) -> ast.expr:
        if (isinstance(width, ast.Constant) and isinstance(width.value, (int, float)) and width.value >= 1) \
                or cls._is_max(width):
            return width
        return ast.Call(func=_attribute('math', 'max'), args=[ast.Constant(1), width], keywords=[])

    @staticmethod
    def _is_max(value: ast.expr) -> bool:
        return isinstance(value, ast.Call) and _dotted(value.func) == 'math.max'


class LegacyLabelStylePass:
    """Pine v4's ``label.style_labelup`` / ``label.style_labeldown`` are ``label.style_label_up`` /
    ``label.style_label_down``, the only spellings pynecore has."""

    _RENAMES = {'style_labelup': 'style_label_up', 'style_labeldown': 'style_label_down'}

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr in self._RENAMES
                    and isinstance(node.value, ast.Name) and node.value.id == 'label'):
                node.attr = self._RENAMES[node.attr]


class CurrentValuePass:
    """``x[0]`` is ``x``."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        def replace(node):
            if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                    and isinstance(node.slice, ast.Constant) and node.slice.value == 0
                    and isinstance(node.value, (ast.Name, ast.Attribute))
                    and not (isinstance(node.value, ast.Name) and node.value.id.startswith('__'))):
                return node.value
            return None
        _Replacer(replace).visit(tree)


class LegacyHlineStylePass:
    """Pine v4 ``hline(..., linestyle=2)`` / ``plot(..., style=5)``: the style constant,
    ``hline.style_dashed`` / ``plot.style_columns``."""

    _STYLES = {('hline', 'linestyle'): ('hline', _HLINE_STYLES), ('plot', 'style'): ('plot', _PLOT_STYLES)}

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        if context.pine_version > 4:
            return
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                namespace, styles = self._STYLES.get((_dotted(node.func), keyword.arg), (None, {}))
                if (namespace and isinstance(keyword.value, ast.Constant)
                        and type(keyword.value.value) is int and keyword.value.value in styles):
                    keyword.value = _attribute(namespace, styles[keyword.value.value])


class InputOptionsPass:
    """``options=`` is always a tuple: ``options=x`` (one name) is ``options=(x,)``."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted(node.func).startswith('input'):
                for keyword in node.keywords:
                    if keyword.arg == 'options' and not isinstance(keyword.value, (ast.Tuple, ast.List)):
                        keyword.value = ast.Tuple(elts=[keyword.value], ctx=ast.Load())


class ShadowedNamespacePass:
    """A library imported under a built-in namespace's name (``import TradingView/ta/12``)
    reads through ``ta__ren__ = shadowed_namespace(ta__ren____lib, ta)``: the library's
    members first, the built-in's after. Every ``ta.`` of the script reads ``ta__ren__``."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        shadowed = getattr(context.unit.transformed, 'shadowed_namespaces', None) or {}
        if not shadowed:
            return
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in shadowed):
                node.value = ast.Name(id=f'{node.value.id}__ren__', ctx=ast.Load())
        position = max((index + 1 for index, node in enumerate(tree.body)
                        if isinstance(node, (ast.Import, ast.ImportFrom))), default=0)
        tree.body[position:position] = [
            ast.Assign(targets=[ast.Name(id=f'{alias}__ren__', ctx=ast.Store())],
                       value=ast.Call(func=ast.Name(id='shadowed_namespace', ctx=ast.Load()),
                                      args=[ast.Name(id=module, ctx=ast.Load()), ast.Name(id=alias, ctx=ast.Load())],
                                      keywords=[]), lineno=0, col_offset=0)
            for alias, module in shadowed.items()]


class ArrayFromNaPass:
    """A bare ``na`` in ``array.from(...)`` is the na of the array's element type: ``na(T)`` with ``T`` the type of
    its literal arguments (``array.from(na, 1.0)`` is ``array.from_items(na(float), 1.0)``), ``na(int)`` where they
    give none."""

    _TYPES = {float: 'float', int: 'int', str: 'str', bool: 'bool'}

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _dotted(node.func) == 'array.from_items'):
                continue
            if not any(self._is_na(argument) for argument in node.args):
                continue
            kinds = {self._TYPES.get(type(argument.value)) for argument in node.args
                     if isinstance(argument, ast.Constant) and not self._is_na(argument)}
            element = kinds.pop() if len(kinds) == 1 and None not in kinds else 'int'
            node.args = [ast.Call(func=ast.Name(id='na', ctx=ast.Load()), args=[ast.Name(id=element, ctx=ast.Load())],
                                  keywords=[]) if self._is_na(argument) else argument for argument in node.args]

    @staticmethod
    def _is_na(node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id == 'na'


class LegacySecurityBarmergePass:
    """Pine v4 `security(..., lookahead=true)` is `request.security(..., lookahead=barmerge.lookahead_on)`, and so
    for `false` and for `gaps`; Pine v5 keeps the bool."""

    _NAMES = {'lookahead': ('lookahead_on', 'lookahead_off'), 'gaps': ('gaps_on', 'gaps_off')}

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        if context.pine_version >= 5:
            return
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _dotted(node.func) == 'request.security'):
                continue
            for keyword in node.keywords:
                names = self._NAMES.get(keyword.arg)
                if names and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, bool):
                    keyword.value = _attribute('barmerge', names[0] if keyword.value.value else names[1])



class LegacyDuplicateFunctionPass:
    """Pine v4: a function whose parameters and body are those of an earlier function of the same scope is left
    out, and its calls call the earlier one; only the names of the two functions differ, in Pine
    (layout aside: `f(x) => e` and a body of the one line `e` are the same) and in the code written for them
    (`iff(c, a, b)` is not `c ? a : b`). Each call keeps its own series state either way. Pine v5 keeps both.
    Runs before the temporaries of `and`/`or` operands are hoisted, so they are named after the kept function
    (`isValid_1 = isValid()` for a call written `isValidToo()`)."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        if context.pine_version >= 5:
            return
        from ....duplicate_functions import pine_shape
        pine_shapes = {function.line: pine_shape(function) for function in _pine_functions(context)}
        self._pine_shapes = {python_line: pine_shapes[pine_line] for python_line, pine_line in context.origins.items()
                             if pine_line in pine_shapes}
        for scope in ast.walk(tree):
            body = getattr(scope, 'body', None)
            if not isinstance(body, list) or not any(isinstance(node, ast.FunctionDef) for node in body):
                continue
            first: dict[str, str] = {}
            renames: dict[str, str] = {}
            kept = []
            for node in body:
                if isinstance(node, ast.FunctionDef):
                    key = self._shape(node)
                    if key in first:
                        renames[node.name] = first[key]
                        continue
                    first[key] = node.name
                kept.append(node)
            if not renames:
                continue
            scope.body = kept
            for node in ast.walk(scope):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in renames:
                    node.id = renames[node.id]

    def _shape(self, function: ast.FunctionDef):
        pine = self._pine_shapes.get(function.lineno)
        if pine is None:
            return ('unmatched', id(function))  # no Pine function to compare: kept
        return pine, '\n'.join([ast.dump(function.args), ast.dump(function.returns) if function.returns else '',
                                *(ast.dump(decorator) for decorator in function.decorator_list),
                                *(ast.dump(statement) for statement in function.body)])


def _pine_functions(context: PassContext) -> list:
    from ....ast_nodes import FuncDecl
    from ....ast_walk import walk
    script = getattr(context.unit, 'pine_script', None)
    return [node for node in walk(script) if isinstance(node, FuncDecl)] if script is not None else []


class BlockTupleArityPass:
    """A tuple a function gives as the value of its closing `if` (or `switch`, or loop) may be destructured into
    fewer names: `[a, b] = f()` with `f` ending in `if c` / `[x, y, z]` takes the first two, `f()[:2]`.
    A function that returns the tuple directly cannot be destructured so; such a call is left alone."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        arities = {}
        for function in ast.walk(tree):
            if isinstance(function, ast.FunctionDef) and function.body:
                arity = self._block_tuple_arity(function)
                if arity:
                    arities[function.name] = arity
        if not arities:
            return
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)):
                continue
            wanted = len(node.targets[0].elts)
            if arities.get(node.value.func.id, 0) > wanted:
                node.value = ast.Subscript(value=node.value, slice=ast.Slice(upper=ast.Constant(wanted)),
                                           ctx=ast.Load())

    @staticmethod
    def _block_tuple_arity(function: ast.FunctionDef) -> int:
        """The length of the tuple a function returns through a block result, 0 for anything else."""
        last = function.body[-1]
        if not (isinstance(last, ast.Return) and isinstance(last.value, ast.Name)
                and last.value.id.startswith('__block_result')):
            return 0
        for statement in function.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
                    and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == last.value.id:
                return len(statement.value.elts) if isinstance(statement.value, ast.Tuple) else 0
        return 0


class LibraryDrawingTypePass:
    """A type of an imported library named like a built-in drawing type (`Line`, `Label`, `Box`, `Linefill`,
    `Table`, whatever its case but capitalised) is spelled `Line__ren__` in the library's module, so the script reads
    it as `dr.Line__ren__`, in annotations, `na(dr.Line)` and constructors alike."""

    _DRAWING_TYPES = frozenset({'box', 'label', 'line', 'linefill', 'table'})

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        aliases = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.Import)
                   for alias in node.names if alias.name.startswith('lib.')}
        if not aliases:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases \
                    and node.attr[:1].isupper() and node.attr.lower() in self._DRAWING_TYPES:
                node.attr = f'{node.attr}__ren__'


class UdtBoolNaArgumentPass:
    """Pine v5: `na` given to a bool field of a user type's constructor (`T.new(b = na)`, or by position) is
    `NA(bool)` in a script whose bools may be na (`na_bool=True`); the other fields keep
    `na`, and so does Pine v6. Without `na_bool` the runtime's `NA(bool)` is plain False, while Pine v5's field
    stays na (`na(t.b)` holds, `switch t.b` takes its default arm): there the bare `na` is kept, which the
    runtime reads as na whatever the switch."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        if context.pine_version != 5 or not self._bools_may_be_na(tree):
            return
        fields: dict[str, list[tuple[str, bool]]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(_dotted(decorator) == 'udt' for decorator in node.decorator_list):
                fields[node.name] = [(statement.target.id, isinstance(statement.annotation, ast.Name)
                                      and statement.annotation.id == 'bool')
                                     for statement in node.body
                                     if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)]
        if not fields:
            return
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in fields):
                continue
            spec = fields[node.func.id]
            bools = {name for name, is_bool in spec if is_bool}
            node.args = [self._na_bool() if index < len(spec) and spec[index][1] and self._is_na(argument) else argument
                         for index, argument in enumerate(node.args)]
            for keyword in node.keywords:
                if keyword.arg in bools and self._is_na(keyword.value):
                    keyword.value = self._na_bool()

    @staticmethod
    def _bools_may_be_na(tree: ast.Module) -> bool:
        """Whether the script's decorator sets `na_bool=True`."""
        return any(isinstance(node, ast.keyword) and node.arg == 'na_bool' and isinstance(node.value, ast.Constant)
                   and node.value.value is True for node in ast.walk(tree))

    @staticmethod
    def _is_na(node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id == 'na'

    @staticmethod
    def _na_bool() -> ast.expr:
        return ast.Call(func=ast.Name(id='NA', ctx=ast.Load()), args=[ast.Name(id='bool', ctx=ast.Load())], keywords=[])


class RepeatedKeywordPass:
    """A keyword argument naming a parameter that a positional argument already gives is left out
    (`plotcandle(o, h, l, c, 'T', col, wick, wickcolor=w2)`: the positional wins); Python would refuse
    the call ("multiple values for argument")."""

    #: Pine's positional parameter order of the calls this applies to
    _PARAMETERS = {
        'plotcandle': ('open', 'high', 'low', 'close', 'title', 'color', 'wickcolor', 'editable', 'show_last',
                       'bordercolor', 'display', 'format', 'precision', 'force_overlay'),
        'plotbar': ('open', 'high', 'low', 'close', 'title', 'color', 'editable', 'show_last', 'display', 'format',
                    'precision', 'force_overlay'),
        # `label.new(x, y, "T", ..., text = "T")`
        'label.new': ('x', 'y', 'text', 'xloc', 'yloc', 'color', 'style', 'textcolor', 'size', 'textalign',
                      'tooltip', 'text_font_family', 'force_overlay', 'text_formatting'),
        'line.new': ('x1', 'y1', 'x2', 'y2', 'xloc', 'extend', 'color', 'style', 'width', 'force_overlay'),
        'box.new': ('left', 'top', 'right', 'bottom', 'border_color', 'border_width', 'border_style', 'extend', 'xloc',
                    'bgcolor', 'text', 'text_size', 'text_color', 'text_halign', 'text_valign', 'text_wrap',
                    'text_font_family', 'force_overlay', 'text_formatting'),
    }

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted(node.func) in self._PARAMETERS:
                given = set(self._PARAMETERS[_dotted(node.func)][:len(node.args)])
                node.keywords = [keyword for keyword in node.keywords if keyword.arg not in given]
