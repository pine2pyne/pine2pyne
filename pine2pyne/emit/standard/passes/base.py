"""Rewrites of the generated Python module towards the standard output layout.

The code generator and the standard layout agree on what a script computes but often spell it
differently (``PersistentSeries`` vs ``Persistent``, ``'#00ced1'`` vs
``color.new('#00CED1')``). Each difference is one :class:`ModulePass`: a
small rewrite of the Python AST, applied in order by the standard emitter.
A new difference is a new pass; existing passes stay untouched.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterator, Protocol

from ....core.contracts import CompilationUnit


@dataclass
class PassContext:
    unit: CompilationUnit
    #: exact text to print for a node (``id(node)``), e.g. a title's source quotes
    spellings: dict[int, str] = field(default_factory=dict)
    #: Python line -> the Pine line of the statement generated there
    origins: dict[int, int] = field(default_factory=dict)

    @property
    def pine_version(self) -> int:
        return self.unit.pine_version or 6


class ModulePass(Protocol):
    def apply(self, tree: ast.Module, context: PassContext) -> None: ...


def main_function(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            return node
    return None


def function_scopes(tree: ast.Module) -> Iterator[ast.AST]:
    """The module and every function in it."""
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Nodes of ``scope`` that are not inside a nested function or class."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def statements_in_order(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements of a block and its nested blocks (not nested functions), in source order."""
    for statement in body:
        yield statement
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for name in ('body', 'orelse', 'finalbody'):
            nested = getattr(statement, name, None)
            if isinstance(nested, list):
                yield from statements_in_order(nested)
