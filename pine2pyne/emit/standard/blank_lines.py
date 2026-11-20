"""Where pine2pyne leaves blank lines inside a function body.

For a Pine v4/v5 script every blank line in ``main`` follows a nested
``def``/``class`` or stands where the ``//@version`` line separates two
statements (a script written below others); a Pine v6 script keeps the blank
lines of its source.

A Pine v6 statement *marks* a gap when the Pine line just above it is blank or
a comment. The statements pine2pyne moves out of ``main`` to the module (the
constants the inputs read) keep their marks: between two statements of
``main``, ``n`` marks (the moved statements' and the second statement's own)
print ``(2n + 1) // 3`` blank lines. Inputs, which become ``main``'s
parameters, leave no mark. The formula holds for ``n`` = 1..6.
"""
from __future__ import annotations

import ast
from typing import Mapping, Protocol, Sequence

_BLOCK_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_loop_variable(statement: ast.stmt) -> bool:
    """`i = __loop_0__.value`, the first line of a loop body read through `pine_loop`."""
    return (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Attribute)
            and statement.value.attr == 'value' and isinstance(statement.value.value, ast.Name)
            and statement.value.value.id.startswith('__loop_'))


class BlankLinePolicy(Protocol):
    def blank_lines_before(self, previous: ast.stmt, current: ast.stmt) -> int: ...


class AfterDefinitionPolicy:
    """Pine v4/v5: one blank line after a nested function or class, and one where the ``//@version`` line
    stands between two statements (statements written above it), none elsewhere.

    ``origins`` maps a Python line number to the Pine line the statement printed there came from;
    ``version_line`` holds the Pine lines that follow a ``//@version`` line: the first line of code after each
    (a script pasted below another keeps its own); a statement starting there takes the gap."""

    def __init__(self, origins: Mapping[int, int] | None = None,
                 version_line: int | Sequence[tuple[int, int]] = 0) -> None:
        self._origins = origins or {}
        # (the `//@version` line, the line its mark falls on); a bare line marks the next statement after it
        self._version_lines = ([(version_line, 0)] if version_line else []) if isinstance(version_line, int) \
            else list(version_line)

    def blank_lines_before(self, previous: ast.stmt, current: ast.stmt) -> int:
        if isinstance(previous, _BLOCK_DEFINITIONS):
            return 1
        if self._version_lines:
            previous_line = self._origins.get(previous.lineno, 0)
            current_line = self._origins.get(current.lineno, 0)
            if previous_line > 0 and any(previous_line < version < current_line and marked in (0, current_line)
                                         for version, marked in self._version_lines):
                return 1  # a statement written above the `//@version` line, the marked one below it
        return 0


class SourceLayoutPolicy:
    """Pine v6: a blank line where the Pine line just above the statement is
    blank or a comment (a dropped comment leaves a blank line behind).

    ``origins`` maps a Python line number to the Pine line the statement
    printed there came from.
    """

    def __init__(self, origins: Mapping[int, int], pine_lines: Sequence[str],
                 moved_lines: Sequence[int] = ()) -> None:
        self._origins = origins
        self._pine_lines = pine_lines
        #: the Pine lines of the statements moved out of main() to the module
        self._moved_lines = sorted(set(moved_lines))

    def _pine_line(self, node: ast.stmt) -> int:
        decorators = getattr(node, 'decorator_list', None)
        first = min(decorator.lineno for decorator in decorators) if decorators else node.lineno
        return self._origins.get(first, 0)

    def _marks(self, line: int) -> bool:
        above = self._pine_lines[line - 2].strip() if 0 <= line - 2 < len(self._pine_lines) else ''
        return (not above or above.startswith('//')) and not self._continues_a_comma_list(line)

    def _continues_a_comma_list(self, line: int) -> bool:
        """The statement is an item of a comma-joined list written over several lines (`a := 1,`, a blank line,
        `b := 2`): one Pine statement, with no gap of its own."""
        for index in range(line - 2, -1, -1):
            code = self._pine_lines[index].split('//')[0].rstrip()
            if code:
                return code.endswith(',')
        return False

    def blank_lines_before(self, previous: ast.stmt, current: ast.stmt) -> int:
        if _is_loop_variable(previous):
            return 0  # `i = __loop_0__.value` opens a loop body: its first statement takes no gap
        line = self._pine_line(current)
        previous_line = self._pine_line(previous)
        after_definition = isinstance(previous, _BLOCK_DEFINITIONS)
        # `a = 1, b = 2` on one Pine line; a value's last assignment after its lowered block (`x = switch`)
        if not after_definition and (line <= 1 or line <= previous_line):
            return 0
        # a nested def or class marks the gap after it once, whatever its source has (467 of 467 gaps)
        marks = (1 if after_definition else 0) + (1 if line > 1 and self._marks(line) else 0)
        if 0 < previous_line < line:
            marks += sum(1 for moved in self._moved_lines if previous_line < moved < line and self._marks(moved))
        return max((2 * marks + 1) // 3, 1 if after_definition else 0)
