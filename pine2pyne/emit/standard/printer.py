"""Module layout of the generated module, printed from a Python AST.

The printer owns layout only: what a statement *is* comes from the code
generator; how it is spread over lines, indented and separated comes from
here. Expressions go through :class:`~.unparser.ExpressionPrinter`, the
import block through :class:`~.imports.ImportBlockFormatter`, blank lines in
function bodies through a :class:`~.blank_lines.BlankLinePolicy`.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .banner import ModuleBanner
from .blank_lines import AfterDefinitionPolicy, BlankLinePolicy, SourceLayoutPolicy
from .imports import ImportBlockFormatter
from .unparser import ExpressionPrinter, header_strings, source_quoted_strings

INDENT = '    '
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class LayoutSource:
    """What the printer reads: the generated code and where it came from."""

    python_source: str
    pine_source: str = ''
    pine_version: int | None = None
    #: Python line number -> Pine line number of the statement printed there
    origins: Mapping[int, int] = field(default_factory=dict)

    @property
    def keeps_source_layout(self) -> bool:
        """Pine v6 keeps its blank lines and string quotes; earlier versions do not."""
        return (self.pine_version or 6) >= 6


class _ModuleWriter:
    """Prints one module; created per render, so the printer stays stateless."""

    def __init__(self, expression: Callable[[ast.AST], str], blank_lines: BlankLinePolicy,
                 written_as_elif: Callable[[ast.If], bool] = lambda node: True,
                 blank_in_source: Callable[[ast.stmt], bool] = lambda node: False) -> None:
        self._expression = expression
        self._blank_lines = blank_lines
        self._written_as_elif = written_as_elif
        self._blank_in_source = blank_in_source

    # -- functions ---------------------------------------------------------

    def _argument(self, arg: ast.arg, default: ast.expr | None, spaced: bool = False) -> str:
        """`name=default` for main()'s inputs; `name = default` for a user function or an annotation."""
        text = arg.arg
        if arg.annotation is not None:
            text += f': {self._expression(arg.annotation)}'
            if default is not None:
                text += f' = {self._expression(default)}'
        elif default is not None:
            text += f' = {self._expression(default)}' if spaced else f'={self._expression(default)}'
        return text

    def _arguments(self, args: ast.arguments, spaced: bool = False) -> list[str]:
        positional = list(args.posonlyargs) + list(args.args)
        defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        parts = [self._argument(arg, default, spaced) for arg, default in zip(positional, defaults)]
        if args.posonlyargs:
            parts.insert(len(args.posonlyargs), '/')
        if args.vararg is not None:
            parts.append('*' + self._argument(args.vararg, None))
        elif args.kwonlyargs:
            parts.append('*')
        parts.extend(self._argument(arg, default) for arg, default in zip(args.kwonlyargs, args.kw_defaults))
        if args.kwarg is not None:
            parts.append('**' + self._argument(args.kwarg, None))
        return parts

    def _function(self, node: ast.FunctionDef, depth: int, one_parameter_per_line: bool) -> list[str]:
        indent = INDENT * depth
        lines = [f'{indent}@{self._expression(decorator)}' for decorator in node.decorator_list]
        keyword = 'async def' if isinstance(node, ast.AsyncFunctionDef) else 'def'
        returns = f' -> {self._expression(node.returns)}' if node.returns is not None else ''
        parameters = self._arguments(node.args, spaced=not one_parameter_per_line)
        if one_parameter_per_line and parameters:
            lines.append(f'{indent}{keyword} {node.name}(')
            lines.append(',\n'.join(f'{indent}{INDENT}{parameter}' for parameter in parameters))
            lines.append(f'{indent}){returns}:')
        elif getattr(node, 'protocol_call', False) and parameters == ['self']:
            lines.append(f'{indent}{keyword} {node.name}(self, ){returns}:')  # the trailing comma is kept
        else:
            lines.append(f'{indent}{keyword} {node.name}({", ".join(parameters)}){returns}:')
        if _is_stub(node.body):
            lines[-1] += ' ...'
            return lines
        return lines + self.block(node.body, depth + 1)

    def _class(self, node: ast.ClassDef, depth: int) -> list[str]:
        indent = INDENT * depth
        lines = [f'{indent}@{self._expression(decorator)}' for decorator in node.decorator_list]
        bases = [self._expression(base) for base in node.bases]
        bases += [self._expression(keyword) for keyword in node.keywords]
        head = f'{indent}class {node.name}({", ".join(bases)}):' if bases else f'{indent}class {node.name}:'
        body: list[str] = []
        for index, field in enumerate(node.body):
            if index and self._blank_in_source(field):  # a Pine v6 UDT keeps the gaps between its fields
                body.append('')
            body += self.statement(field, depth + 1)
        return lines + [head] + body

    # -- statements --------------------------------------------------------

    def _if(self, node: ast.If, depth: int, keyword: str = 'if') -> list[str]:
        indent = INDENT * depth
        lines = [f'{indent}{keyword} {self._expression(node.test)}:'] + self.block(node.body, depth + 1)
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If) and self._written_as_elif(node.orelse[0]):
                lines += self._if(node.orelse[0], depth, 'elif')
            else:
                lines += [f'{indent}else:'] + self.block(node.orelse, depth + 1)
        return lines

    def _loop(self, node: ast.For | ast.While, depth: int) -> list[str]:
        indent = INDENT * depth
        if isinstance(node, ast.While):
            head = f'{indent}while {self._expression(node.test)}:'
        else:
            target = node.target
            spelled = (', '.join(self._expression(element) for element in target.elts)
                       if isinstance(target, ast.Tuple) and target.elts else self._expression(target))
            head = f'{indent}for {spelled} in {self._expression(node.iter)}:'
        lines = [head] + self.block(node.body, depth + 1)
        if node.orelse:
            lines += [f'{indent}else:'] + self.block(node.orelse, depth + 1)
        return lines

    def statement(self, node: ast.stmt, depth: int, one_parameter_per_line: bool = False) -> list[str]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._function(node, depth, one_parameter_per_line)
        if isinstance(node, ast.ClassDef):
            return self._class(node, depth)
        if isinstance(node, ast.If):
            return self._if(node, depth)
        if isinstance(node, (ast.For, ast.While)):
            return self._loop(node, depth)
        indent = INDENT * depth
        if isinstance(node, (ast.Try, ast.With, ast.Match, ast.AsyncFor, ast.AsyncWith)):
            return [indent + line for line in ast.unparse(node).split('\n')]  # the generator never emits these
        return [indent + self._expression(node)]

    def block(self, body: list[ast.stmt], depth: int) -> list[str]:
        lines: list[str] = []
        previous: ast.stmt | None = None
        for node in body:
            if previous is not None:
                lines += [''] * self._blank_lines.blank_lines_before(previous, node)
            lines += self.statement(node, depth)
            previous = node
        return lines


def _is_stub(body: list[ast.stmt]) -> bool:
    return (len(body) == 1 and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant) and body[0].value.value is ...)


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _is_main_guard(node: ast.stmt) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__')


def _binds_shadowed_namespace(node: ast.stmt) -> bool:
    return (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'shadowed_namespace')


def _declares_an_export(node: ast.stmt | None) -> bool:
    """`name: _ProtocolName = Exported()`."""
    value = getattr(node, 'value', None)
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'Exported'


def _assigned_names(node: ast.stmt) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, 'target', None)]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _moved_lines(source: LayoutSource, tree: ast.Module | None) -> list[int]:
    """The Pine lines of the statements moved out of main(): the module-level assignments and the
    enums and UDTs (found by their declaration in the Pine source)."""
    if tree is None:
        return []
    lines = [source.origins.get(node.lineno, 0) for node in tree.body
             if isinstance(node, (ast.Assign, ast.AnnAssign)) and source.origins.get(node.lineno, 0)]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            pine_name = re.sub(r'__ren__$', '', node.name)
            match = re.search(rf'^(?:export\s+)?(?:type|enum)\s+{re.escape(pine_name)}\b', source.pine_source, re.M)
            if match:
                lines.append(source.pine_source.count('\n', 0, match.start()) + 1)
    return lines


_INPUT_LINE = re.compile(r'(?:var\s+)?(?:\w+\s+)?\w+\s*=\s*input(?:\.\w+)?\s*\(')


def _first_code_line(pine_lines: list[str], after: int, top_level: bool = False) -> int:
    """The first Pine line after line `after` that holds code (not blank, not a comment), starting at the left
    margin if `top_level`; 0 when there is none."""
    for index in range(after, len(pine_lines)):
        text = pine_lines[index]
        if text.strip() and not text.strip().startswith('//') and not (top_level and text[:1].isspace()):
            return index + 1
    return 0


class StandardModulePrinter:
    """Prints a generated module in pine2pyne's standard layout."""

    def __init__(self, banner: ModuleBanner | None = None, imports: ImportBlockFormatter | None = None) -> None:
        self._banner = banner or ModuleBanner()
        self._imports = imports or ImportBlockFormatter()

    def blank_line_policy(self, source: LayoutSource, tree: ast.Module | None = None) -> BlankLinePolicy:
        if source.keeps_source_layout:
            return SourceLayoutPolicy(source.origins, source.pine_source.split('\n'), _moved_lines(source, tree))
        # each `//@version` line marks the first line of code after it: a script pasted below another keeps its
        # own (`//@version=5` again on line 51); an input there, which becomes a parameter, takes the mark along
        pine_lines = source.pine_source.split('\n')
        marked = []
        for match in re.finditer(r'^[ \t]*//\s*@version\s*=', source.pine_source, re.M):
            line = source.pine_source.count('\n', 0, match.start()) + 1
            following = _first_code_line(pine_lines, line)
            if following and re.match(r'(indicator|strategy|study|library)\s*\(', pine_lines[following - 1]):
                # the script's declaration passes the mark on to the first statement of main() after it, past the
                # inputs (which become parameters); without a declaration an input there takes the mark along
                following = _first_code_line(pine_lines, following, top_level=True)
                while following and _INPUT_LINE.match(pine_lines[following - 1]):
                    following = _first_code_line(pine_lines, following, top_level=True)
            if following:
                marked.append((line, following))
        return AfterDefinitionPolicy(source.origins, marked)

    _declares_types = False  # set per render: the module has a UDT or an enum

    def _module_gap(self, previous: ast.stmt | None, current: ast.stmt) -> int:
        """Blank lines between two module-level statements."""
        if _binds_shadowed_namespace(current):
            return 2 if previous is None or not _binds_shadowed_namespace(previous) else 0
        if previous is not None and _binds_shadowed_namespace(previous) and not isinstance(current, _DEFINITIONS):
            return 1
        if previous is None:  # right after the import block
            if isinstance(current, _DEFINITIONS) or '__all__' in _assigned_names(current):
                return 2
            return 1
        if isinstance(previous, _DEFINITIONS) or isinstance(current, _DEFINITIONS):
            return 2
        if '__all__' in _assigned_names(previous) or '__all__' in _assigned_names(current):
            return 2
        if _declares_an_export(previous) != _declares_an_export(current):
            # the Exported() declarations are one group; a constant after them is one line away, two in a
            # library that declares a UDT or an enum (16 of 16 libraries)
            if _declares_an_export(previous):
                return 2 if self._declares_types else 1
            return 2
        return 0

    def render(self, source: LayoutSource, tree: ast.Module | None = None,
               spellings: Mapping[int, str] | None = None) -> str:
        """Print ``tree`` (default: ``source.python_source`` parsed) in the standard layout.

        ``tree`` may be a rewritten parse of ``source.python_source``: nodes that
        kept their positions still find their source spelling and Pine line.
        """
        if tree is None:
            tree = ast.parse(source.python_source)
        source_quoted = source_quoted_strings(tree) if source.keeps_source_layout else frozenset()
        single_quoted = frozenset() if source.keeps_source_layout else header_strings(tree)
        # the strings written as the source has them keep their non-printable characters raw
        raw_strings = source_quoted if source.keeps_source_layout else source_quoted_strings(tree)
        expression = ExpressionPrinter(source.python_source, source_quoted, spellings, single_quoted, raw_strings)
        source_lines = source.python_source.split('\n')

        def written_as_elif(node: ast.If) -> bool:
            """Pine `else if` is `elif`; an `if` nested in an `else` block stays nested."""
            lineno = getattr(node, 'lineno', 0)
            return not lineno or source_lines[lineno - 1].lstrip().startswith('elif ')

        def blank_in_source(node: ast.stmt) -> bool:
            """The generated source has a blank line just above `node` (a gap between two UDT fields)."""
            lineno = getattr(node, 'lineno', 0)
            return lineno > 1 and not source_lines[lineno - 2].strip()

        writer = _ModuleWriter(expression, self.blank_line_policy(source, tree), written_as_elif, blank_in_source)

        self._declares_types = any(
            isinstance(node, ast.ClassDef) and not any(isinstance(base, ast.Name) and base.id == 'Protocol'
                                                       for base in node.bases) for node in tree.body)
        body = [node for node in tree.body if not _is_docstring(node) and not _is_main_guard(node)]
        import_nodes = [node for node in body if isinstance(node, (ast.Import, ast.ImportFrom))]
        others = [node for node in body if not isinstance(node, (ast.Import, ast.ImportFrom))]

        lines = self._banner.header() + self._imports.format(import_nodes)
        previous: ast.stmt | None = None
        for node in others:
            lines += [''] * self._module_gap(previous, node)
            is_main = isinstance(node, ast.FunctionDef) and node.name == 'main'
            lines += writer.statement(node, 0, one_parameter_per_line=is_main)
            previous = node
        lines += ['', ''] + self._banner.footer()
        return '\n'.join(lines) + '\n'
