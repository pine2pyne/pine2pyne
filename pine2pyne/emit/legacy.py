"""pine2pyne's own output layout (the historical default)."""
from __future__ import annotations

from ..codegen import CodeGenerator
from ..core.contracts import CompilationUnit, OutputProfile


class LegacyEmitter:
    """Prints the transformed tree with :class:`~pine2pyne.codegen.CodeGenerator` as is."""

    profile = OutputProfile.LEGACY

    def emit(self, unit: CompilationUnit) -> str:
        return CodeGenerator().generate(unit.transformed)
