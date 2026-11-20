"""The transpile use case, composed from injected stages."""
from __future__ import annotations

from typing import Sequence

from ..core.contracts import (
    AstNormalizer,
    CodeEmitter,
    CompilationUnit,
    ScriptTransformer,
    SemanticGuard,
    SourceFilter,
    SourceParser,
)
from ..core.recursion import recursion_headroom


class TranspileService:
    """Pine source -> Python source.

    The service only orders the stages; it knows nothing about how any of
    them works (S and D in SOLID). Swap a stage by registering another
    implementation of its contract in the container.
    """

    def __init__(
        self,
        source_filter: SourceFilter,
        parser: SourceParser,
        normalizers: Sequence[AstNormalizer],
        guard: SemanticGuard,
        transformer: ScriptTransformer,
        emitter: CodeEmitter,
    ) -> None:
        self._source_filter = source_filter
        self._parser = parser
        self._normalizers = tuple(normalizers)
        self._guard = guard
        self._transformer = transformer
        self._emitter = emitter

    @property
    def emitter(self) -> CodeEmitter:
        return self._emitter

    def _front_end(self, source: str):
        script = self._parser.parse(source)
        for normalizer in self._normalizers:
            script = normalizer.normalize(script)
        self._guard.check(script)
        return script

    def transpile(self, source: str) -> str:
        """Python source for ``source``; an empty string for a comment-only source.

        Raises:
            Pine2PyneError: if any stage refuses the script.
        """
        if self._source_filter.is_blank(source):
            return ''
        with recursion_headroom():
            script = self._front_end(source)
            transformed = self._transformer.transform(script)
            return self._emitter.emit(CompilationUnit(pine_source=source, pine_script=script, transformed=transformed))

    def validate(self, source: str) -> None:
        """Run the whole pipeline and discard the text (``--validate``): a script is valid exactly when
        :meth:`transpile` accepts it, so a refusal of a later stage is reported here too.

        Raises:
            Pine2PyneError: if any stage refuses the script.
        """
        self.transpile(source)
