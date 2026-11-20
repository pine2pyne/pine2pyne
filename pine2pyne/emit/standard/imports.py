"""The import block and its layout.

The rules:

* ``from`` imports come first, one per module, modules in alphabetical order;
  plain ``import`` statements (library imports) follow, sorted;
* names of a ``pynecore`` module are sorted (code-point order), ``typing``
  keeps the order it was given;
* a ``from`` line up to 100 characters stays on one line; a longer one is
  parenthesised and its names filled greedily: the first inner line takes up to
  102 characters less the ``from <module> import `` prefix (77 for
  ``pynecore.lib``, 75 for ``pynecore.types``), the following ones 99.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ImportBlockFormatter:
    single_line_limit: int = 100
    #: the first inner line's width is this less the `from <module> import ` prefix
    first_line_budget: int = 102
    next_line_width: int = 99
    indent: str = '    '
    unsorted_modules: frozenset = frozenset({'typing'})

    def format(self, statements: list[ast.stmt]) -> list[str]:
        from_imports: dict[str, list[str]] = {}
        plain_imports: list[str] = []
        for statement in statements:
            if isinstance(statement, ast.ImportFrom):
                module = '.' * statement.level + (statement.module or '')
                names = from_imports.setdefault(module, [])
                for alias in statement.names:
                    spelled = alias.name if alias.asname is None else f'{alias.name} as {alias.asname}'
                    if spelled not in names:
                        names.append(spelled)
            elif isinstance(statement, ast.Import):
                line = ast.unparse(statement)
                if line not in plain_imports:
                    plain_imports.append(line)
        lines: list[str] = []
        for module in sorted(from_imports):
            names = from_imports[module]
            if module not in self.unsorted_modules:
                names = sorted(names)
            lines.extend(self._from_import(module, names))
        return lines + sorted(plain_imports)  # library imports sorted by path

    def _from_import(self, module: str, names: list[str]) -> list[str]:
        single = f'from {module} import {", ".join(names)}'
        if len(single) <= self.single_line_limit:
            return [single]
        rows: list[list[str]] = [[]]
        for name in names:
            row = rows[-1]
            width = (self.first_line_budget - len(f'from {module} import ') if len(rows) == 1
                     else self.next_line_width)
            candidate = self.indent + ', '.join(row + [name]) + ','
            if row and len(candidate) > width:
                rows.append([name])
            else:
                row.append(name)
        body = [self.indent + ', '.join(row) + ',' for row in rows[:-1]]
        body.append(self.indent + ', '.join(rows[-1]))
        return [f'from {module} import ('] + body + [')']
