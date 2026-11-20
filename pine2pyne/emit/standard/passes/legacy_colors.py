"""Pine v4 and v5 transparency, merged into the colour."""
from __future__ import annotations

import ast
import re

from .base import PassContext, function_scopes, own_nodes

_HEX_COLOR = re.compile(r'#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})')
_COLOR_KEYWORDS = {
    'plot': ('color',), 'plotshape': ('color', 'textcolor'), 'plotchar': ('color',),
    'plotarrow': ('colorup', 'colordown'), 'fill': ('color',), 'bgcolor': ('color',),
}
#: v4 positional parameters: (index of color, index of transp)
#: (`plotchar`'s positional colour takes no default transparency: `plotchar(c, 't', '', location.top, red)`)
_POSITIONAL = {'plot': (2, None), 'plotshape': (4, 5), 'plotchar': (None, 5), 'fill': (2, 3), 'bgcolor': (0, 1)}
#: v4's default ``transp`` of the functions that had the argument; a default of 0 leaves the colour as it is
_DEFAULT_TRANSPARENCY = {'plot': 0, 'plotshape': 0, 'plotchar': 0, 'plotarrow': 35, 'fill': 90, 'bgcolor': 90}
#: the helper applying a transparency to a colour that may carry its own
_HELPER = 'transp_color'


#: Pine's named colour constants
COLOUR_CONSTANTS = frozenset({
    'aqua', 'black', 'blue', 'fuchsia', 'gray', 'green', 'lime', 'maroon', 'navy', 'olive', 'orange',
    'purple', 'red', 'silver', 'teal', 'white', 'yellow'})

_OPAQUE, _TRANSPARENT, _UNKNOWN, _NA = 'opaque', 'transparent', 'unknown', 'na'


def _is_constant_colour(node: ast.expr) -> bool:
    """A hex literal or a named constant (``color.red``)."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and bool(_HEX_COLOR.fullmatch(node.value))
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == 'color' and node.attr in COLOUR_CONSTANTS)


def _is_colour_constructor(call: ast.Call) -> bool:
    """`color.new(c, transp=85)`: there transp is the colour's own argument."""
    return (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
            and call.func.value.id == 'color' and call.func.attr in ('new', 'rgb'))


def _literal_number(node: ast.expr):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    return None


def _wrap(colour: ast.expr, transparency: ast.expr) -> ast.Call:
    return ast.Call(func=ast.Attribute(value=ast.Name(id='color', ctx=ast.Load()), attr='new', ctx=ast.Load()),
                    args=[colour, transparency], keywords=[])


def _helper_call(colour: ast.expr, transparency: ast.expr) -> ast.Call:
    return ast.Call(func=ast.Name(id=_HELPER, ctx=ast.Load()), args=[colour, transparency], keywords=[])


def _helper_definition() -> ast.FunctionDef:
    """``def transp_color(c, t): return color.new(c, t) if color.t(c) == 0 else c``"""
    return ast.parse('def transp_color(c, t):\n    return color.new(c, t) if color.t(c) == 0 else c\n').body[0]


class LegacyTransparencyPass:
    """Pine v4 and v5 ``transp``, merged into the colour.

    A transparency given (``transp=t``, positional or named) or a function's non-zero default (90 for
    ``fill`` and ``bgcolor``, 35 for ``plotarrow``; 0, given or default, leaves the colour alone) applies to each colour argument
    by what is known of its own transparency: an opaque colour (a constant, a hex ``#RRGGBB``,
    ``color.rgb(r, g, b)``, ``color.new(c, 0)``, a ternary of such (an ``na`` arm aside), a name bound once to one, ``var`` too) is
    ``color.new(c, t)``; a colour that carries its own (``color.new(c, 30)``, ``color.new(c, transp=30)``, ``#RRGGBBAA``, a name bound once
    to one) keeps it and drops ``t``; any other (a name bound more than once, a call, an input) is
    ``transp_color(c, t)``, which applies ``t`` only where the colour is opaque. ``plot(x, transp=t)`` with no
    colour is ``color.new(color.blue, t)``, the default colour.

    Pine v5 documents no ``transp``, yet published v5 scripts pass it by name and compile: a ``transp=t``
    given by name merges the same way. v5 has no positional ``transp`` (the slot is another parameter) and
    no default transparency. Pine v6 drops it."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        # a user type's constructor or a user function keeps a `transp` of its own (`Br(transp=70)`)
        user_names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef))}

        def builtin(node: ast.Call) -> bool:
            return not (isinstance(node.func, ast.Name) and node.func.id in user_names)

        if context.pine_version >= 6:  # transp is gone: dropped, not merged
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and not _is_colour_constructor(node) and builtin(node):
                    node.keywords = [keyword for keyword in node.keywords if keyword.arg != 'transp']
            return
        helper = False
        for scope in function_scopes(tree):
            bindings = self._bindings(scope)
            for node in own_nodes(scope):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id in _COLOR_KEYWORDS:
                    helper = self._merge(node, bindings, legacy=context.pine_version <= 4) or helper
        for node in ast.walk(tree):  # transp of any other call (barcolor, ...) has no equivalent
            if isinstance(node, ast.Call) and not _is_colour_constructor(node) and builtin(node):
                node.keywords = [keyword for keyword in node.keywords if keyword.arg != 'transp']
        if helper:
            main = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main'), None)
            if main is not None:
                main.body.insert(0, ast.copy_location(_helper_definition(), main.body[0]))

    @staticmethod
    def _bindings(scope: ast.AST) -> dict[str, list[ast.expr]]:
        """Each name's values bound in `scope` (its blocks included)."""
        bindings: dict[str, list[ast.expr]] = {}
        for node in own_nodes(scope):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        bindings.setdefault(target.id, []).append(node.value)
        return bindings

    def _opacity(self, value: ast.expr, bindings: dict, seen: frozenset = frozenset()) -> str:
        if isinstance(value, ast.Name):
            if value.id == 'na':
                return _NA
            found = bindings.get(value.id, [])
            if len(found) != 1 or value.id in seen:
                return _UNKNOWN
            return self._opacity(found[0], bindings, seen | {value.id})
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and _HEX_COLOR.fullmatch(value.value):
            return _OPAQUE if len(value.value) == 7 or value.value[7:].upper() == 'FF' else _TRANSPARENT
        if _is_constant_colour(value):
            return _OPAQUE
        if isinstance(value, ast.Call) and _is_colour_constructor(value):
            own = (value.args[1] if value.func.attr == 'new' and len(value.args) > 1
                   else value.args[3] if value.func.attr == 'rgb' and len(value.args) > 3
                   else next((keyword.value for keyword in value.keywords if keyword.arg == 'transp'), None))
            if own is None:
                return _OPAQUE if value.func.attr == 'rgb' and len(value.args) == 3 else _UNKNOWN
            number = _literal_number(own)
            return _UNKNOWN if number is None else _OPAQUE if number == 0 else _TRANSPARENT
        if isinstance(value, ast.IfExp):
            arms = {self._opacity(arm, bindings, seen) for arm in (value.body, value.orelse)} - {_NA}
            return arms.pop() if len(arms) == 1 else _NA if not arms else _UNKNOWN
        return _UNKNOWN

    def _merge(self, call: ast.Call, bindings: dict, legacy: bool = True) -> bool:
        """Merge the call's transparency into its colours; whether the helper is called. ``legacy`` (Pine v4):
        a positional ``transp`` and the functions' default transparency count too."""
        function = call.func.id
        position = next((index for index, keyword in enumerate(call.keywords) if keyword.arg == 'transp'), None)
        transparency = call.keywords[position].value if position is not None else None
        colour_index, transp_index = _POSITIONAL.get(function, (None, None))
        if legacy and transp_index is not None and len(call.args) > transp_index:
            transparency = call.args.pop(transp_index)
        if legacy and transparency is None and _DEFAULT_TRANSPARENCY[function]:
            transparency = ast.Constant(_DEFAULT_TRANSPARENCY[function])
        if _literal_number(transparency) == 0:
            transparency = None  # `transp=0` leaves the colour as it is
        helper = False

        def merged(value: ast.expr) -> ast.expr:
            nonlocal helper
            kind = self._opacity(value, bindings)
            if kind == _OPAQUE:
                return _wrap(value, transparency)
            if kind == _UNKNOWN:
                helper = True
                return _helper_call(value, transparency)
            return value

        colours = 0
        if transparency is not None:
            if colour_index is not None and len(call.args) > colour_index:
                call.args[colour_index] = merged(call.args[colour_index])
                colours += 1
            for keyword in call.keywords:
                if keyword.arg in _COLOR_KEYWORDS[function]:
                    keyword.value = merged(keyword.value)
                    colours += 1
        if position is not None:
            if function == 'plot' and not colours and transparency is not None:
                # no colour: the transparency applies to plot's default colour
                call.keywords[position] = ast.keyword(arg='color', value=_wrap(
                    ast.Attribute(value=ast.Name(id='color', ctx=ast.Load()), attr='blue', ctx=ast.Load()),
                    transparency))
            else:
                del call.keywords[position]
        return helper
