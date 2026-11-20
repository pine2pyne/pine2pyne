"""The numbers given to the blocks of a script: the suffix of a renamed local.

A local that shadows a name, or is named like a function of the script, is
spelled ``<name>__<n>__``, ``n`` eight hex digits. ``n`` is the id of the
block the local is declared in. The numbering gives some nodes an id, from 1,
in the order a parser finishes them (post-order, in source order):

* a call, except a cast (``int(x)``, ``label(na)``) and Pine v4's ``iff()``,
  which is parsed as a conditional;
* a history read ``x[n]``;
* every block: a function body (a one-line body too), each branch of an
  ``if``/``else if``/``else``, each case of a ``switch``, a loop body;
* a ``type`` or ``enum`` declaration.

A Pine v4 function that repeats an earlier one is left out of the tree, so
neither its block nor its calls take an id.

Pine v4 is converted while it is parsed and the nodes the conversion creates
are numbered too: a constant colour merged with ``transp`` (or with v4's
default transparency) is a ``color.new()`` call. In Pine v4 and v5 a ``linewidth``
that is not a number of at least 1 is ``math.max(1, w)``, a call too. (The session conversion of
``time()``, two more calls, is in the tree already: :mod:`.legacy_sessions`.)

A meter function ``m(x) => m = x`` placed between two constructs shows, by the
suffix of its local, how many ids the construct takes.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional

from .duplicate_functions import repeated_functions
from .legacy_sessions import SESSION_HELPER
from .ast_nodes import (
    ASTNode, Assignment, EnumDecl, ExpressionStatement, ForInLoop, ForLoop, FuncDecl, FunctionCall, Identifier,
    IfExpression, IfStatement, IndexAccess, InputDecl, Literal, MemberAccess, MethodCall, Script, SwitchExpression,
    SwitchStatement, TupleDestructure, TypeDecl, WhileLoop,
)

#: calls that are casts: no node of their own
_CASTS = frozenset({'int', 'float', 'bool', 'string', 'color', 'line', 'label', 'box', 'table', 'linefill',
                    'polyline', 'iff'})
_NAMED_COLOURS = frozenset({
    'aqua', 'black', 'blue', 'fuchsia', 'gray', 'green', 'lime', 'maroon', 'navy', 'olive', 'orange',
    'purple', 'red', 'silver', 'teal', 'white', 'yellow'})
#: Pine v4 calls whose colour takes ``transp``: the colour keywords, (colour, transp) positions
_V4_COLOUR_KEYWORDS = {'plot': ('color',), 'plotshape': ('color', 'textcolor'), 'plotchar': ('color',),
                       'plotarrow': ('colorup', 'colordown'), 'fill': ('color',)}
_V4_COLOUR_POSITIONS = {'plot': (2, None), 'plotshape': (4, 5), 'plotchar': (4, 5), 'fill': (2, 3)}


def hex_suffixed(name: str, block_id: int) -> str:
    """The spelling of a renamed local of block ``block_id``."""
    return f'{name}__{block_id:08x}__'


def _at_least_one(value) -> bool:
    """A number literal of at least 1: a `linewidth` kept as written."""
    return (isinstance(value, Literal) and value.literal_type in ('int', 'float')
            and isinstance(value.value, (int, float)) and value.value >= 1)


class BlockNumbers:
    """The id of every block of a script, by the identity of its statement list
    (or of the function, for a one-line body). Built once per script."""

    def __init__(self, script: Script) -> None:
        version_text = getattr(script.version, 'version', '') if script.version else ''
        self._version = int(version_text) if str(version_text).isdigit() else 6
        self._count = 0
        self._ids: Dict[int, int] = {}
        #: Pine line of a statement -> the id of the innermost block it stands in directly
        self.line_blocks: Dict[int, int] = {}
        items = sorted(list(script.declarations) + list(script.body),
                       key=lambda item: (getattr(item, 'line', 0), getattr(item, 'column', 0)))
        self._constant_colour_names = self._constant_colours(items) if self._version <= 4 else frozenset()
        # the inputs, which the parser keeps apart from the statements: each is numbered where it stands, an input
        # written inside a block among that block's statements
        self._pending_inputs = [item for item in items if isinstance(item, InputDecl) and getattr(item, 'line', 0)]
        # the declaration is numbered where it is written: after the functions a script declares above it
        declaration = script.script_decl
        declaration_line = getattr(declaration, 'line', 0) or 0
        if declaration is not None and not declaration_line:
            self._declaration(declaration)
            declaration = None
        # a Pine v4 function that repeats an earlier one is left out of the tree
        repeated = repeated_functions(items) if self._version <= 4 else set()
        for item in items:
            if declaration is not None and getattr(item, 'line', 0) > declaration_line:
                self._declaration(declaration)
                declaration = None
            if (isinstance(item, InputDecl) and getattr(item, 'line', 0)) or id(item) in repeated:
                continue
            self._inputs_before(getattr(item, 'line', 0))
            self._statement(item)
        if declaration is not None:
            self._declaration(declaration)
        self._inputs_before(None)

    def _declaration(self, declaration) -> None:
        """`indicator(...)`/`strategy(...)`: its arguments, then the call itself."""
        for value in declaration.kwargs.values():
            self._expression(value)
        self._take()

    def _inputs_before(self, line: Optional[int]) -> None:
        """Number the inputs written before `line` (all of them for None)."""
        while self._pending_inputs and (line is None or (line and self._pending_inputs[0].line < line)):
            self._statement(self._pending_inputs.pop(0))

    def of(self, block) -> int:
        """The id of a statement list (or of a FuncDecl's body); 0 when unknown."""
        return self._ids.get(id(block), 0)

    # ------------------------------------------------------------ numbering

    def _take(self) -> int:
        self._count += 1
        return self._count

    def _block(self, statements, owner: Optional[object] = None) -> None:
        if isinstance(statements, list):
            for statement in statements:
                self._inputs_before(getattr(statement, 'line', 0))
                self._statement(statement)
        else:
            self._expression(statements)
        number = self._take()
        self._ids[id(statements)] = number
        if owner is not None:
            self._ids[id(owner)] = number
        for statement in statements if isinstance(statements, list) else []:
            if getattr(statement, 'line', 0):
                self.line_blocks.setdefault(statement.line, number)  # inner blocks are numbered first

    def _statement(self, node) -> None:
        if isinstance(node, FuncDecl):
            if node.name == SESSION_HELPER:
                return  # the session helper takes no id
            for parameter in node.params:
                self._expression(parameter.default)
            self._block(node.body, owner=node)
        elif isinstance(node, IfStatement):
            self._expression(node.condition)
            self._block(node.body)
            for condition, body in node.elseifs:
                self._expression(condition)
                self._block(body)
            if node.else_body is not None:
                self._block(node.else_body)
        elif isinstance(node, ForLoop):
            for value in (node.from_val, node.to_val, node.step):
                self._expression(value)
            self._block(node.body)
        elif isinstance(node, ForInLoop):
            self._expression(node.iterable)
            self._block(node.body)
        elif isinstance(node, WhileLoop):
            self._expression(node.condition)
            self._block(node.body)
        elif isinstance(node, (SwitchStatement, SwitchExpression)):
            self._switch(node)
        elif isinstance(node, InputDecl):
            for value in list(node.args) + list(node.kwargs.values()):
                self._expression(value)
            self._take()
        elif isinstance(node, (TypeDecl, EnumDecl)):
            self._take()
        else:
            self._expression(node)

    def _switch(self, node) -> None:
        self._expression(node.expr)
        for case, body in node.cases:
            self._expression(case)
            self._block(body)
        if node.default is not None:
            self._block(node.default)

    def _expression(self, node) -> None:
        if node is None or not isinstance(node, (ASTNode, list)):
            return
        if isinstance(node, list):
            for item in node:
                self._statement(item)
            return
        if isinstance(node, TupleDestructure):
            return  # the names only: its value is the assignment's
        if isinstance(node, FunctionCall):
            if not isinstance(node.func, str):
                self._expression(node.func)
            for value in node.args:
                self._expression(value)
            for keyword, value in node.kwargs.items():
                self._expression(value)
                if keyword == 'linewidth' and self._version <= 5 and not _at_least_one(value):
                    self._take()  # Pine v4/v5 `linewidth=w` is `math.max(1, w)`, a call made while parsing
            if isinstance(node.func, str) and node.func in _CASTS:
                return
            if self._version <= 4 and isinstance(node.func, str):
                self._v4_conversion(node)
            self._take()
            return
        if isinstance(node, MethodCall):
            self._expression(node.object)
            for value in list(node.args) + list(node.kwargs.values()):
                self._expression(value)
            self._take()
            return
        if isinstance(node, IndexAccess):
            self._expression(node.object)
            self._expression(node.index)
            self._take()
            return
        if isinstance(node, IfExpression):
            self._expression(node.condition)
            self._block(node.true_expr)
            if node.false_expr is not None:
                self._block(node.false_expr)
            return
        if isinstance(node, SwitchExpression):
            self._switch(node)
            return
        if isinstance(node, (Assignment, ExpressionStatement)) or dataclasses.is_dataclass(node):
            for field in dataclasses.fields(node):
                if field.name in ('line', 'column'):
                    continue
                self._child(getattr(node, field.name))

    def _child(self, value) -> None:
        if isinstance(value, ASTNode):
            self._expression(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._child(item)
        elif isinstance(value, dict):
            for item in value.values():
                self._child(item)

    # ------------------------------------------------------------ Pine v4 conversion

    def _v4_conversion(self, call: FunctionCall) -> None:
        """The calls Pine v4's conversion adds around ``call``."""
        if call.func in _V4_COLOUR_KEYWORDS:
            for _ in range(self._merged_colours(call)):
                self._take()  # color.new(c, transp)

    def _merged_colours(self, call: FunctionCall) -> int:
        function = call.func
        transparency = call.kwargs.get('transp')
        colour_index, transp_index = _V4_COLOUR_POSITIONS.get(function, (None, None))
        if transp_index is not None and len(call.args) > transp_index:
            transparency = call.args[transp_index]
        colours = [call.kwargs[key] for key in _V4_COLOUR_KEYWORDS[function] if key in call.kwargs]
        if colour_index is not None and len(call.args) > colour_index:
            colours.append(call.args[colour_index])
        merged = 0
        for colour in colours:
            if self._is_constant_colour(colour):
                merged += 1
            elif transparency is not None and function == 'fill' and not isinstance(colour, Literal):
                merged += 1
        return merged

    def _is_constant_colour(self, value) -> bool:
        if isinstance(value, Literal):
            return value.literal_type == 'color'
        if isinstance(value, Identifier):
            return value.name in self._constant_colour_names or (
                value.name.startswith('color.') and value.name[len('color.'):] in _NAMED_COLOURS)
        return isinstance(value, MemberAccess) and value.object == 'color' and value.member in _NAMED_COLOURS

    def _constant_colours(self, items) -> frozenset:
        """Top-level names bound once, to a constant colour."""
        bindings: Dict[str, list] = {}
        for item in items:
            if isinstance(item, Assignment) and isinstance(item.target, str):
                bindings.setdefault(item.target, []).append(item.value)
        self._constant_colour_names = frozenset()
        return frozenset(name for name, values in bindings.items()
                         if len(values) == 1 and self._is_constant_colour(values[0]))
