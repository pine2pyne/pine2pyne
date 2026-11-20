"""Expression printing: :func:`ast.unparse`, with two string rules of its own.

The printer builds a Python AST and prints every expression with the standard
library unparser: minimal parentheses (``a and (not b)`` but ``not a and b``),
``repr`` of numbers (``3.141592653589793``), single-quoted strings. All but
the string literals print exactly as :func:`ast.unparse` prints them. Two
string rules differ:

* printable non-ASCII characters are written as they are (``' '`` is an
  em space in the file, not an escape), only control characters are escaped;
* some string literals of a Pine v6 script keep the quotes the source wrote
  them with (which ones: :func:`source_quoted_strings`).
"""
from __future__ import annotations

import ast
import re
import unicodedata
from typing import Collection, Mapping, Sequence

_ESCAPE = re.compile(r'\\\\|\\x[0-9a-f]{2}|\\u[0-9a-f]{4}|\\U[0-9a-f]{8}')


def string_literal(value: str, single_quoted: bool = True, raw: bool = True) -> str:
    """A single-quoted literal (``'it\\'s'``: the printer never switches to double
    quotes) with ``repr``'s escapes. ``raw``: a string pine2pyne writes as the source
    has it (header, input and whole-assignment strings, :func:`source_quoted_strings`)
    keeps its non-printable non-ASCII characters (``\\u200b``, no-break spaces, the C1 control characters
    U+0080..U+009F a mis-decoded emoji leaves behind) unescaped; any other string keeps ``repr``'s escapes for
    them. The ASCII control characters are always escaped, and so is U+0085, which Python's line splitting
    reads as a line break."""
    def unescape(match: re.Match) -> str:
        text = match.group(0)
        if text == '\\\\':
            return text
        character = chr(int(text[2:], 16))
        if unicodedata.category(character) == 'Cc' and not ('\x80' <= character <= '\x9f' and character != '\x85'):
            return text
        return character
    text = repr(value)
    if single_quoted and text.startswith('"'):  # repr chose double quotes: value has a ' and no "
        text = "'" + text[1:-1].replace("'", "\\'") + "'"
    return _ESCAPE.sub(unescape, text) if raw else text


def header_strings(tree: ast.Module) -> set[int]:
    """``id()`` of the strings a Pine v4/v5 module writes in single quotes whatever they hold
    (``'it\\'s'``): the direct arguments of main()'s decorator and of its parameter defaults (an
    element of a tuple argument too), and a module constant's whole value. A part of a
    concatenation (``'a' + "b's"``) is spelled as ``repr`` spells it."""
    found: set[int] = set()

    def is_string(node) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str)

    def direct(call) -> None:
        if not isinstance(call, ast.Call):
            return
        for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
            if is_string(argument):
                found.add(id(argument))
            elif isinstance(argument, (ast.Tuple, ast.List)):
                found.update(id(element) for element in argument.elts if is_string(element))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            for part in node.decorator_list + node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                direct(part)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and is_string(node.value):
            found.add(id(node.value))
    # and a declaration's whole string value anywhere (`ERR_TXT: str = '... \'{0}\' ...'`, PineCoders/lower_tf)
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and is_string(node.value):
            found.add(id(node.value))
    return found


def source_quoted_strings(tree: ast.Module) -> set[int]:
    """``id()`` of the string literals printed with their source quotes
    in a Pine v6 script: decorator and ``main`` parameter (input) arguments, a
    string that is the whole value of an assignment or an element of an
    assigned tuple, the argument of ``cast_string``. A string elsewhere in the body (a call argument, a
    comparison operand) is re-quoted."""
    keep: set[int] = set()

    def is_string(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str)

    def mark_arguments(call: ast.AST) -> None:
        """The direct arguments of a call, and the elements of a tuple or list passed to it."""
        if not isinstance(call, ast.Call):
            return
        for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
            if is_string(argument):
                keep.add(id(argument))
            elif isinstance(argument, (ast.Tuple, ast.List)):
                keep.update(id(element) for element in argument.elts if is_string(element))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            for decorator in node.decorator_list:
                mark_arguments(decorator)
            for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                mark_arguments(default)
        elif isinstance(node, ast.FunctionDef):
            # a user function's string default keeps its source quotes (and the protocol that shares it)
            keep.update(id(default) for default in node.args.defaults if is_string(default))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'cast_string':
            # `string("lit")` keeps its source quotes: `cast_string("lit")`
            keep.update(id(argument) for argument in node.args if is_string(argument))
        elif (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
              and node.left.id.startswith('__switch')):
            # a switch case keeps its source quotes: `__switch__ == "Top Right"`
            keep.update(id(operand) for operand in node.comparators if is_string(operand))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            if is_string(node.value):
                keep.add(id(node.value))
            elif isinstance(node.value, ast.Tuple):
                keep.update(id(element) for element in node.value.elts if is_string(element))
    return keep


class SourceQuotingUnparser(ast._Unparser):  # the stdlib unparser, extended at one node type
    """Unparses like :func:`ast.unparse`, with pine2pyne's string rules."""

    def __init__(self, source_lines: Sequence[bytes] = (), source_quoted: Collection[int] = frozenset(),
                 spellings: Mapping[int, str] | None = None, single_quoted: Collection[int] = frozenset(),
                 raw_strings: Collection[int] | None = None) -> None:
        super().__init__()
        self._source_lines = source_lines
        self._source_quoted = source_quoted
        self._spellings = spellings or {}
        self._single_quoted = single_quoted
        self._raw_strings = raw_strings

    def _source_text(self, node: ast.AST) -> str | None:
        lineno = getattr(node, 'lineno', None)
        end_lineno = getattr(node, 'end_lineno', None)
        if not lineno or not end_lineno or end_lineno > len(self._source_lines):
            return None
        col, end_col = node.col_offset, node.end_col_offset  # UTF-8 byte offsets
        if lineno == end_lineno:
            raw = self._source_lines[lineno - 1][col:end_col]
        else:
            parts = [self._source_lines[lineno - 1][col:]]
            parts.extend(self._source_lines[lineno:end_lineno - 1])
            parts.append(self._source_lines[end_lineno - 1][:end_col])
            raw = b'\n'.join(parts)
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return None

    def visit_Constant(self, node: ast.Constant) -> None:
        spelled = self._spellings.get(id(node))
        if spelled is not None:
            self.write(spelled)
            return
        if not isinstance(node.value, str) or getattr(self, '_avoid_backslashes', False):
            super().visit_Constant(node)
            return
        if id(node) in self._source_quoted:
            text = self._source_text(node)
            if text and text[0] in '\'"' and '\n' not in text:
                try:
                    same = ast.literal_eval(text) == node.value
                except (ValueError, SyntaxError):
                    same = False
                if same:
                    self.write(text)
                    return
        if node.kind == 'u':
            self.write('u')
        raw = self._raw_strings is None or id(node) in self._raw_strings
        self.write(string_literal(node.value, id(node) in self._single_quoted, raw))


class ExpressionPrinter:
    """Prints one node (an expression or a simple statement) on one line."""

    def __init__(self, python_source: str = '', source_quoted: Collection[int] = frozenset(),
                 spellings: Mapping[int, str] | None = None, single_quoted: Collection[int] = frozenset(),
                 raw_strings: Collection[int] | None = None) -> None:
        self._source_lines = [line.encode('utf-8') for line in python_source.split('\n')]
        self._source_quoted = source_quoted
        self._spellings = spellings or {}
        self._single_quoted = single_quoted
        self._raw_strings = raw_strings

    def __call__(self, node: ast.AST) -> str:
        return SourceQuotingUnparser(self._source_lines, self._source_quoted, self._spellings,
                                     self._single_quoted, self._raw_strings).visit(node)
