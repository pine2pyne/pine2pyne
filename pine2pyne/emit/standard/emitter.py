"""The standard output profile."""
from __future__ import annotations

import ast
from typing import Sequence

from ...codegen import CodeGenerator
from ...core.contracts import CompilationUnit, OutputProfile
from ...core.recursion import recursion_headroom
from ...errors import CodeGenError
from .passes import ModulePass, PassContext
from .printer import LayoutSource, StandardModulePrinter


class StandardEmitter:
    """Generates code with :class:`~pine2pyne.codegen.CodeGenerator`, then
    re-prints it with :class:`StandardModulePrinter`.

    Three separate concerns: the generator decides *what* the program says,
    the passes give every construct its standard spelling, the printer lays it out.
    """

    profile = OutputProfile.STANDARD

    def __init__(self, printer: StandardModulePrinter, passes: Sequence[ModulePass] = ()) -> None:
        self._printer = printer
        self._passes = tuple(passes)

    def emit(self, unit: CompilationUnit) -> str:
        # the AST passes walk an expression recursively: a long concatenation or `switch` is deep
        with recursion_headroom():
            return self._emit(unit)

    def _emit(self, unit: CompilationUnit) -> str:
        # Pine v4/v5 strict and/or is StrictBooleanHoistingPass's job, not all((...))
        unit.transformed.strict_boolean_operators = False
        generator = CodeGenerator(keep_keyword_string_quotes=True)
        python_source = generator.generate(unit.transformed)
        # an output element may span several lines (the import block does)
        first_line_of: list[int] = []
        line_number = 1
        for element in generator.output:
            first_line_of.append(line_number)
            line_number += element.count('\n') + 1
        origins = {first_line_of[index]: pine_line for index, pine_line in generator.line_origins.items()
                   if index < len(first_line_of)}
        source = LayoutSource(
            python_source=python_source,
            pine_source=unit.pine_source,
            pine_version=unit.pine_version,
            origins=origins,
        )
        try:
            tree = ast.parse(python_source)
        except SyntaxError as error:  # a construct the generator spelled as no valid Python
            raise CodeGenError(f'the generated code is not valid Python: {error.msg}',
                               line=origins.get(error.lineno or 0, 0)) from error
        context = PassContext(unit=unit, origins=origins)
        for module_pass in self._passes:
            module_pass.apply(tree, context)
        return self._printer.render(source, tree, context.spellings)
