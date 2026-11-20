"""Pine v4 functions that repeat an earlier function.

A Pine v4 function whose parameters and body are those of an earlier function
of the script is left out, and its calls call the earlier one. Only the
names of the two differ; the layout does not count (`f(x) => e` and a body of
the one line `e` are the same). The left-out function is not in the tree
at all: its block and its calls take no node id. Pine v5 and v6 keep both.
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Set


def pine_shape(node):
    """A Pine node without its positions, a function without its name; a body of
    one expression statement is that expression."""
    from .ast_nodes import ExpressionStatement, FuncDecl
    if isinstance(node, FuncDecl):
        body = node.body
        if isinstance(body, list) and len(body) == 1 and isinstance(body[0], ExpressionStatement):
            body = body[0].expr
        node = dataclasses.replace(node, body=body)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return (type(node).__name__,) + tuple(
            (item.name, pine_shape(getattr(node, item.name))) for item in dataclasses.fields(node)
            if item.compare and item.name not in ('line', 'column') and not (isinstance(node, FuncDecl)
                                                                         and item.name == 'name'))
    if isinstance(node, (list, tuple)):
        return tuple(pine_shape(item) for item in node)
    if isinstance(node, dict):
        return tuple(sorted((str(key), repr(pine_shape(value))) for key, value in node.items()))
    if isinstance(node, (set, frozenset)):
        return tuple(sorted(repr(pine_shape(item)) for item in node))
    return node


def repeated_functions(items: Iterable) -> Set[int]:
    """The ids of the functions among `items` (top-level statements in source
    order) that repeat an earlier one."""
    from .ast_nodes import FuncDecl
    seen, repeated = set(), set()
    for item in items:
        if isinstance(item, FuncDecl):
            shape = pine_shape(item)
            if shape in seen:
                repeated.add(id(item))
            seen.add(shape)
    return repeated
