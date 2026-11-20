"""`x[0]` is `x`: a read of the current bar, not of the history.

Pine allows `[0]` on any series; it reads the value of the bar being computed, the same as the bare
name or call (`str.split(s, " ")[0]` is the call's value: Pine has no `[]` on arrays). The index is dropped right after parsing, so no later stage takes `x[0]` for a history read and
types `x` as a series for it (`f(a) => a[0]` is written `def f(a):` with `return a`).
"""
from __future__ import annotations

import dataclasses

from .ast_nodes import FunctionCall, Identifier, IndexAccess, Literal, MemberAccess, Script
from .ast_walk import is_node, map_children


def _is_current_value(node) -> bool:
    return (isinstance(node, IndexAccess) and isinstance(node.index, Literal) and node.index.literal_type == 'int'
            and node.index.value == 0 and (isinstance(node.object, (MemberAccess, FunctionCall)) or (
                isinstance(node.object, Identifier) and not node.object.name.startswith('__'))))


def _without_current_value_indexes(node):
    if _is_current_value(node):
        if isinstance(node.object, Identifier):
            # a name keeps that it was read so: Pine v5 marks a script whose bool is read `b[0]` `na_bool`
            return dataclasses.replace(node.object, read_at_zero=True)
        return _without_current_value_indexes(node.object)
    if not is_node(node):
        return node
    changes = {field.name: map_children(getattr(node, field.name), _without_current_value_indexes)
               for field in dataclasses.fields(node) if field.name not in ('line', 'column')}
    return dataclasses.replace(node, **changes)


def drop_current_value_indexes(script: Script) -> Script:
    """`script` with every `name[0]` read as `name`."""
    return _without_current_value_indexes(script)
