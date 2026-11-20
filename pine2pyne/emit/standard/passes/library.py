"""A library's module and its layout.

The layout::

    <imports>
    from typing import Protocol, Any

    __all__ = [...]                       every exported name, emitted spelling, sorted

    <enums and UDT classes, in source order>

    class _ProtocolInit(Protocol):        one per exported function or method, source order
        def __call__(self, <its parameters>) -> Any: ...

    init: _ProtocolInit = Exported()      one per exported function or method

    <module constants>

    @script.library(...)
    def main(): ...                       @export, @overload, @method in that order

A protocol is named ``_Protocol`` + the function's name with each ``_``-part
capitalised at its first letter (``get_sec_in_day`` -> ``GetSecInDay``, ``renderOnly`` ->
``Renderonly``, ``trailing_3step`` -> ``Trailing3Step``); a
name that already has one gets the protocol's 0-based index across the module
as a suffix (``_ProtocolInit38``). Its parameters are the function's, but a
default of ``na`` (one Python needs after a defaulted parameter) is ``...`` and
a ``Series[T]`` the function's body earned by reading the parameter's history
is ``T`` (a parameter declared ``series T`` keeps ``Series[T]``).
"""
from __future__ import annotations

import ast
import copy
import re

from .base import PassContext, main_function

_DECORATOR_ORDER = {'export': 0, 'overload': 1, 'method': 2}
_EXPORTED_TYPE = re.compile(r'^export\s+(?:type|enum)\s+(\w+)', re.M)
#: `export const int N = 42`: the name of an exported library constant
_EXPORTED_CONST = re.compile(r'^export\s+const\s+[\w.]+(?:<[^>\n]*>)?\s+(\w+)\s*=', re.MULTILINE)


def _source_position(node: ast.ClassDef, context: PassContext) -> int:
    """Where the Pine source declares the `type`/`enum` a class is generated from."""
    match = re.search(rf'^(?:export\s+)?(?:type|enum)\s+{re.escape(_pine_name(node.name))}\b',
                      context.unit.pine_source or '', re.M)
    return match.start() if match else 0


def _capitalised(part: str) -> str:
    """The first letter upper case, the rest lower case: `3step` -> `3Step`, `is2Tap` -> `Is2tap`."""
    match = re.search(r'[A-Za-z]', part)
    if not match:
        return part
    start = match.start()
    return part[:start] + part[start].upper() + part[start + 1:].lower()


def protocol_stem(name: str) -> str:
    """`get_sec_in_day` -> `GetSecInDay`, `trailing_3step` -> `Trailing3Step`; a renamed function keeps its
    Pine name (`fill__ren__` -> `Fill`)."""
    name = re.sub(r'__ren__$|__global__$', '', name)
    return ''.join(_capitalised(part) for part in name.split('_'))


def _pine_name(name: str) -> str:
    """The Pine name of a renamed parameter (`format__ren__`, `adr__00000017__` -> `format`, `adr`)."""
    return re.sub(r'__ren__$|__global__$|__[0-9a-f]{8}__$', '', name)


def _is_exported(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef) and any(
        isinstance(decorator, ast.Name) and decorator.id == 'export' for decorator in node.decorator_list)


def _is_library(tree: ast.Module) -> bool:
    main = main_function(tree)
    return main is not None and any(
        isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == 'library' for decorator in main.decorator_list)


def _exported_call(node: ast.stmt) -> bool:
    value = getattr(node, 'value', None)
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'Exported'


def _assigns_all(node: ast.stmt) -> bool:
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '__all__'
                                                for target in node.targets)


class LibraryPreamblePass:
    """``__all__``, the ``_Protocol*`` classes and the ``Exported()`` declarations of a library."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        main = main_function(tree)
        if main is None:
            return
        for node in ast.walk(main):  # the decorator order, any script
            if isinstance(node, ast.FunctionDef) and len(node.decorator_list) > 1:
                node.decorator_list.sort(key=lambda decorator: _DECORATOR_ORDER.get(
                    getattr(decorator, 'id', ''), -1))
        if not _is_library(tree):
            self._classes_before_constants(tree, context)
            return
        exported = [node for node in main.body if _is_exported(node)]
        declared = self._declared_parameters(context, len(exported))
        protocols, declarations, seen = [], [], set()
        for index, function in enumerate(exported):
            name = f'_Protocol{protocol_stem(function.name)}'
            if name in seen:
                name = f'{name}{index}'
            seen.add(f'_Protocol{protocol_stem(function.name)}')
            protocols.append(self._protocol(name, function, *declared[index]))
            declarations.append(ast.AnnAssign(target=ast.Name(id=function.name, ctx=ast.Store()),
                                              annotation=ast.Name(id=name, ctx=ast.Load()),
                                              value=ast.Call(func=ast.Name(id='Exported', ctx=ast.Load()),
                                                             args=[], keywords=[]), simple=1))
        exported_types = set(_EXPORTED_TYPE.findall(context.unit.pine_source or ''))
        exported_constants = set(_EXPORTED_CONST.findall(context.unit.pine_source or ''))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        public = {function.name for function in exported} | {node.name for node in classes
                                                              if _pine_name(node.name) in exported_types}
        # an exported constant is a module-level assignment (the spelling of its export is not measured)
        public |= {target.id for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                   for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                   if isinstance(target, ast.Name) and _pine_name(target.id) in exported_constants}
        names = ast.Assign(targets=[ast.Name(id='__all__', ctx=ast.Store())],
                           value=ast.List(elts=[ast.Constant(name) for name in sorted(public)], ctx=ast.Load()))

        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        docstring = tree.body[:1] if tree.body and isinstance(tree.body[0], ast.Expr) else []
        rest = [node for node in tree.body if node not in imports and node not in docstring
                and not _exported_call(node) and not _assigns_all(node)]
        # a library alias shadowing a builtin namespace (`ta__ren__ = shadowed_namespace(...)`) comes right after
        # __all__ (TradingView/TechnicalRating)
        shadowing = [node for node in rest if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Name) and node.value.func.id == 'shadowed_namespace']
        constants = [node for node in rest if isinstance(node, (ast.Assign, ast.AnnAssign)) and node not in shadowing]
        tail = [node for node in rest if node not in classes and node not in constants and node not in shadowing]
        classes.sort(key=lambda node: _source_position(node, context))
        if protocols:
            typing = ast.ImportFrom(module='typing', names=[ast.alias(name='Protocol'), ast.alias(name='Any')],
                                    level=0)
            imports.append(typing)
        # a class renamed apart from an exported function of its Pine name is that function's `new`
        # (`necessaryData.new = necessaryData__ren__`), so `T.new(...)` through the library still constructs
        exported_names = {function.name for function in exported}
        constructors = [ast.Assign(targets=[ast.Attribute(value=ast.Name(id=_pine_name(node.name), ctx=ast.Load()),
                                                          attr='new', ctx=ast.Store())],
                                   value=ast.Name(id=node.name, ctx=ast.Load()))
                        for node in sorted(classes, key=lambda node: _source_position(node, context))
                        if node.name != _pine_name(node.name) and _pine_name(node.name) in exported_names]
        declarations += constructors
        for node in [names, *protocols, *declarations, *imports]:
            for child in ast.walk(node):  # new nodes have no source position: line 0 maps to no Pine line
                for attribute in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
                    if not hasattr(child, attribute) and attribute in child._attributes:
                        setattr(child, attribute, 0)
        tree.body[:] = docstring + imports + ([names] if public else []) + shadowing + classes + protocols \
            + declarations + constants + tail

    @staticmethod
    def _classes_before_constants(tree: ast.Module, context: PassContext) -> None:
        """A script's enums and UDTs, in source order, come before its module constants."""
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if not classes:
            return
        main = main_function(tree)
        before_main = tree.body[:tree.body.index(main)] if main in tree.body else list(tree.body)
        movable = [node for node in before_main if isinstance(node, (ast.Assign, ast.AnnAssign))
                   and not (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == 'shadowed_namespace')]
        moved = set(map(id, classes + movable))
        position = min(index for index, node in enumerate(tree.body) if id(node) in moved)
        classes.sort(key=lambda node: _source_position(node, context))
        rest = [node for node in tree.body if id(node) not in moved]
        rest[position:position] = classes + movable
        tree.body[:] = rest

    @staticmethod
    def _declared_parameters(context: PassContext, count: int) -> list[tuple[set[str], set[int]]]:
        """Per exported function, in order: its parameters declared `series T`, and the positions
        of the parameters the source gives no default (Python's `na` there is the protocol's `...`)."""
        empty = [(set(), set()) for _ in range(count)]
        script = context.unit.pine_script
        if script is None:
            return empty
        from ....ast_nodes import FuncDecl
        functions = sorted((node for node in list(script.declarations) + list(script.body)
                            if isinstance(node, FuncDecl) and node.is_export), key=lambda node: node.line)
        found = [({parameter.name for parameter in function.params if parameter.is_series},
                  {position for position, parameter in enumerate(function.params) if parameter.default is None})
                 for function in functions]
        return (found + empty)[:count]

    @staticmethod
    def _protocol(name: str, function: ast.FunctionDef, declared_series: set[str],
                  undefaulted: set[int]) -> ast.ClassDef:
        arguments = copy.deepcopy(function.args)
        arguments.defaults = list(function.args.defaults)  # the same nodes: a default keeps its spelling
        for argument in arguments.args:
            annotation = argument.annotation
            if (isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name)
                    and annotation.value.id == 'Series' and _pine_name(argument.arg) not in declared_series):
                argument.annotation = annotation.slice  # the Series of a history read, not a declared one
        first_default = len(arguments.args) - len(arguments.defaults)
        arguments.defaults = [ast.Constant(...) if first_default + offset in undefaulted
                              and isinstance(default, ast.Name) and default.id == 'na' else default
                              for offset, default in enumerate(arguments.defaults)]
        arguments.args.insert(0, ast.arg(arg='self'))
        call = ast.FunctionDef(name='__call__', args=arguments, body=[ast.Expr(value=ast.Constant(...))],
                               decorator_list=[], returns=ast.Name(id='Any', ctx=ast.Load()), type_params=[])
        call.protocol_call = True  # printed `def __call__(self, <parameters>) -> Any: ...`, `self, ` even alone
        return ast.ClassDef(name=name, bases=[ast.Name(id='Protocol', ctx=ast.Load())], keywords=[],
                            body=[call], decorator_list=[], type_params=[])
