"""Every input becomes a main() parameter.

The input calls of a script are numbered in source order. An input that
is the whole value of a top-level assignment (``len = input(14)``) becomes the
parameter of that name; one inside an expression (``w = input.float(80) / 100``)
becomes the parameter ``__input_<n>__`` (n: its 0-based number among all
inputs) and the expression reads that name; the first one of a ``var x = ...``
declaration, and a top-level ``x := input(...)``, is ``x__input``. An input inside a function or a nested block is
``__input_<n>__`` too, whatever it is assigned to.

A Pine v4 input is typed as TradingView's v4-to-v5 conversion types it (see
:func:`typed_input_function`).
"""
from __future__ import annotations

import dataclasses
from typing import List

from .ast_nodes import (
    Assignment, EnumDecl, FunctionCall, Identifier, InputDecl, Literal, MemberAccess, Script, TypeDecl,
    TupleDestructure, UnaryOp,
)
from .ast_walk import is_node, map_children, walk

#: keywords the generic Pine v4 ``input()`` is called with and keeps
_GENERIC_KEYWORDS = frozenset({'title', 'defval', 'type'})
_KIND_FUNCTIONS = {'integer': 'input.int', 'float': 'input.float', 'bool': 'input.bool',
                   'string': 'input.string', 'color': 'input.color', 'source': 'input.source', 'time': 'input.time'}
#: Pine v4 type constants passed positionally as input()'s third argument
_LEGACY_TYPE_CONSTANTS = {'input.integer': 'input.int', 'input.float': 'input.float', 'input.bool': 'input.bool',
                          'input.string': 'input.string', 'input.source': 'input.source', 'input.color': 'input.color',
                          'input.resolution': 'input.timeframe', 'input.symbol': 'input.symbol',
                          'input.session': 'input.session', 'input.time': 'input.time'}
_FUNCTION_KINDS = {function: kind for kind, function in _KIND_FUNCTIONS.items()}
_SOURCES = frozenset({'close', 'open', 'high', 'low', 'volume', 'hl2', 'hlc3', 'ohlc4', 'hlcc4'})


def is_input_call(node) -> bool:
    return isinstance(node, FunctionCall) and isinstance(node.func, str) and (
        node.func == 'input' or node.func.startswith('input.'))


def _default_value(call: FunctionCall):
    return call.args[0] if call.args else call.kwargs.get('defval')


def default_kind(value, constants: dict | None = None) -> str | None:
    """The input kind a default value has on its own: what the generic input() infers.
    ``constants``: top-level names bound to a literal, by the literal's kind."""
    if isinstance(value, UnaryOp) and value.op == '-':
        value = value.operand
    if isinstance(value, Identifier) and constants and value.name in constants:
        return constants[value.name]
    if isinstance(value, FunctionCall) and value.func == 'timestamp':
        return 'time'
    if isinstance(value, Literal):
        return {'int': 'integer', 'float': 'float', 'bool': 'bool', 'string': 'string',
                'color': 'color'}.get(value.literal_type)
    if isinstance(value, Identifier):
        if value.name in _SOURCES:
            return 'source'
        if value.name.startswith('color.'):
            return 'color'
        if value.name.startswith(('size.', 'position.', 'shape.', 'location.', 'xloc.', 'yloc.', 'text.',
                                  'extend.', 'font.', 'line.style_', 'label.style_')):
            return 'string'  # `input(size.normal, options=[...])`: the drawing constants are strings
    if isinstance(value, FunctionCall) and isinstance(value.func, str) and value.func.startswith('color.'):
        return 'color'
    return None


def typed_input_function(call: FunctionCall, version: int = 6, constants: dict | None = None) -> str:
    """The input function written for ``call`` (a Pine v4 rule: TradingView's
    v4-to-v5 conversion). A v4 input stays the generic ``input()`` when it passes
    nothing but title and default and its default already has its kind; any other
    keyword (group, inline, minval, options, ...) makes it ``input.<kind>``.
    ``type=input.time`` stays ``input.time`` even with a ``timestamp(...)`` default
    (``input(timestamp(...))`` stays ``input()``). Pine v5 and later keep the
    function the source calls."""
    if version > 4:
        return call.func
    extra = (set(call.kwargs) - _GENERIC_KEYWORDS) or len(call.args) > 2
    natural = default_kind(_default_value(call), constants)
    if call.func == 'input':
        if not extra:
            return 'input'
        if natural == 'integer' and any(_is_float_literal(call.kwargs.get(key)) for key in ('minval', 'maxval', 'step')):
            return 'input.float'  # a fractional range makes an integer default a float input
        return _KIND_FUNCTIONS.get(natural, 'input')
    if not extra and _FUNCTION_KINDS.get(call.func) == natural and natural != 'time':
        return 'input'  # `type=input.integer` with an integer default adds nothing
    return call.func


def _is_float_literal(value) -> bool:
    """A fractional number as written, its sign included (`maxval=-0.0`)."""
    if isinstance(value, UnaryOp) and value.op in ('-', '+'):
        value = value.operand
    return isinstance(value, Literal) and value.literal_type == 'float'


def _type_constant(value) -> str | None:
    name = value.name if isinstance(value, Identifier) else (
        f'{value.object}.{value.member}' if isinstance(value, MemberAccess) and isinstance(value.object, str) else None)
    return _LEGACY_TYPE_CONSTANTS.get(name) if name else None


def _declaration(name: str, call: FunctionCall, line: int, version: int, constants: dict,
                 type_aliases: dict | None = None) -> InputDecl:
    # a `type=` naming a top-level alias of a type constant (`float_ = input.float`) is that type
    type_value = call.kwargs.get('type')
    if call.func == 'input' and isinstance(type_value, Identifier) and (type_aliases or {}).get(type_value.name):
        call = dataclasses.replace(call, func=type_aliases[type_value.name])
    # a `type=` the parser could not map to an input.* function (a variable) is dropped, and so
    # is `active=`: it only greys the field out (a constant one is dropped too)
    kwargs = {key: value for key, value in call.kwargs.items() if key not in ('type', 'active')}
    args = list(call.args)
    # Pine v4 `input('DJI', title=input.symbol)`: a type constant given as the title is the type
    if call.func == 'input' and 'type' not in call.kwargs and _type_constant(kwargs.get('title')):
        call = dataclasses.replace(call, func=_type_constant(kwargs['title']))
        kwargs = {key: value for key, value in kwargs.items() if key != 'title'}
    # Pine v4 `input(defval, title, input.integer, ...)`: the third positional is the type
    if call.func == 'input' and len(args) >= 3 and _type_constant(args[2]):
        call = dataclasses.replace(call, func=_type_constant(args[2]))
        args = args[:2] + args[3:]
    if args and 'defval' in kwargs:
        # `input.int(100, title=..., defval=100)`: the positional default is kept, the keyword dropped
        kwargs = {key: value for key, value in kwargs.items() if key != 'defval'}
    function = typed_input_function(dataclasses.replace(call, args=args, kwargs=kwargs), version, constants)
    if version <= 4 and 'options' in kwargs:
        # v4 `input(..., minval=2, maxval=600, step=1, options=[...])`: a list of options has no range
        kwargs = {key: value for key, value in kwargs.items() if key not in ('minval', 'maxval', 'step')}
    if function == 'input' and len(args) > 2:
        # the generic input(defval, title, tooltip, inline, group): the rest go by name
        named = dict(zip(('tooltip', 'inline', 'group'), args[2:]))
        args, kwargs = args[:2], {**named, **kwargs}
    return InputDecl(name=name, func=function, args=args,
                     kwargs=kwargs, line=line, column=getattr(call, 'column', 0))


def _literal_constants(items) -> dict:
    """Top-level names bound once to a literal -> the literal's input kind."""
    kinds: dict = {}
    for item in items:
        if isinstance(item, Assignment) and isinstance(item.target, str) and isinstance(item.value, Literal):
            kind = default_kind(item.value)
            if kind:
                kinds[item.target] = kind
    return kinds


def _replace_inputs(node, names: dict):
    """``node`` with each input call in ``names`` (by id) read as its parameter."""
    if id(node) in names:
        return Identifier(name=names[id(node)], line=getattr(node, 'line', 0), column=getattr(node, 'column', 0))
    if not is_node(node):
        return node
    changes = {field.name: map_children(getattr(node, field.name), lambda child: _replace_inputs(child, names))
               for field in dataclasses.fields(node) if field.name not in ('line', 'column')}
    return dataclasses.replace(node, **changes)


def hoist_inputs(script: Script) -> Script:
    """``script`` with every top-level input call made an InputDecl, in source order."""
    items = sorted(list(script.declarations) + list(script.body),
                   key=lambda item: (getattr(item, 'line', 0), getattr(item, 'column', 0) or 0))
    body_ids = {id(item) for item in script.body}
    version_text = getattr(script.version, 'version', '') if script.version else ''
    version = int(version_text) if str(version_text).isdigit() else 6
    constants = _literal_constants(items)
    type_aliases = {item.target: _type_constant(item.value) for item in items
                    if isinstance(item, Assignment) and isinstance(item.target, str) and _type_constant(item.value)}
    # an alias read only as an input's `type=` is the type it names: its assignment goes
    type_reads = {id(value) for node in walk(script) if isinstance(node, FunctionCall) and node.func == 'input'
                  for key, value in node.kwargs.items() if key == 'type'}
    other_reads = {node.name for node in walk(script) if isinstance(node, Identifier) and id(node) not in type_reads}
    unread_aliases = set(type_aliases) - other_reads
    declarations: List = []
    body: List = []
    number = 0
    owner_inputs: set = set()
    for item in items:
        if isinstance(item, Assignment) and isinstance(item.target, str) and item.target in unread_aliases:
            continue
        if isinstance(item, InputDecl):
            call = FunctionCall(func=item.func, args=item.args, kwargs=item.kwargs)
            hoisted = _declaration(item.name, call, item.line, version, constants, type_aliases)
            declarations.append(dataclasses.replace(item, func=hoisted.func, args=hoisted.args, kwargs=hoisted.kwargs))
            number += 1
            continue
        if isinstance(item, (TypeDecl, EnumDecl)):
            (body if id(item) in body_ids else declarations).append(item)
            continue
        if isinstance(item, Assignment) and isinstance(item.target, str) and is_input_call(item.value):
            declarations.append(_declaration(item.target, item.value, item.line, version, constants, type_aliases))
            number += 1
            continue
        top_level = {id(node) for node in walk(_expressions_of(item)) if is_input_call(node)}
        # an input in a function or a nested block is a parameter too, `__input_<n>__` whatever it is assigned to
        # a tuple destructure keeps a copy of its value in its target: its calls are read once, from the value
        destructure = isinstance(item, Assignment) and isinstance(item.target, TupleDestructure)
        calls = sorted((node for node in walk(item.value if destructure else item) if is_input_call(node)),
                       key=lambda node: (getattr(node, 'line', 0), getattr(node, 'column', 0)))
        names = {}
        from .ast_nodes import FuncDecl, Reassignment, VarDecl, VaripDecl
        if isinstance(item, FuncDecl) and calls and isinstance(item.body, list):
            # a function's name bound to a literal that its inputs read (`g = "Group"`, `input(1, group = g)`) is
            # declared where main()'s parameters read it, before main()
            read = {node.name for call in calls for node in walk(call) if isinstance(node, Identifier)}
            lifted = [statement for statement in item.body if isinstance(statement, Assignment)
                      and isinstance(statement.target, str) and statement.target in read
                      and isinstance(statement.value, Literal)]
            if lifted:
                item = dataclasses.replace(item, body=[statement for statement in item.body
                                                       if not any(statement is moved for moved in lifted)])
                # on the line above the function's, ahead of main()'s parameters that read it (they take the
                # function's line)
                body.extend(dataclasses.replace(statement, line=max(item.line - 1, 1), column=0)
                            for statement in lifted)
        for call in calls:
            # `var x = input(...)` (or a method of it), or a top-level `x := input(...)`, reads the parameter
            # `x__input`
            owner = item.name if isinstance(item, (VarDecl, VaripDecl)) else (
                item.target if isinstance(item, Reassignment) and isinstance(item.target, str) else None)
            if owner is not None and id(call) in top_level and call is _chain_head(item.value):
                # a field (`tp1.active := input(...)`) reads its object's name; a name taken again takes a
                # trailing `_` (`tp1__input`, `tp1__input_`, ...)
                name = f"{owner.split('.')[0]}__input"
                while name in owner_inputs:
                    name += '_'
                owner_inputs.add(name)
            else:
                name = f'__input_{number}__'
            names[id(call)] = name
            declarations.append(_declaration(name, call, item.line, version, constants, type_aliases))
            number += 1
        rebuilt = _replace_inputs(item, names) if names else item
        if names and destructure:
            rebuilt = dataclasses.replace(rebuilt, target=dataclasses.replace(rebuilt.target, value=rebuilt.value))
        (body if id(item) in body_ids else declarations).append(rebuilt)
    declarations.sort(key=lambda item: getattr(item, 'line', 0))
    return dataclasses.replace(script, declarations=declarations, body=body)


def _chain_head(value):
    """The value a method chain starts from: `input.enum(...)` of `input.enum(...).to_string()`."""
    from .ast_nodes import MethodCall
    while isinstance(value, MethodCall):
        value = value.object
    return value


def _expressions_of(item):
    """The parts of a top-level statement evaluated at the top level (not a nested block)."""
    from .ast_nodes import ExpressionStatement, IfStatement, Reassignment, VarDecl, VaripDecl
    if isinstance(item, (Assignment, Reassignment, VarDecl, VaripDecl)):
        return item.value
    if isinstance(item, ExpressionStatement):
        return item.expr
    if isinstance(item, IfStatement):
        return item.condition
    return []
