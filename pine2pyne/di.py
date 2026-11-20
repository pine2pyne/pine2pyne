"""Composition root: the dishka container that wires the pipeline.

Every concrete class is named here and nowhere else in the application
layer. A provider groups one responsibility; an output profile is one
provider, so a new output layout is a new provider (open/closed)::

    from pine2pyne.di import make_transpiler_container
    from pine2pyne.core import OutputProfile, Transpiler

    container = make_transpiler_container(OutputProfile.STANDARD)
    print(container.get(Transpiler).transpile(pine_source))
    container.close()

All stages are stateless (the single-use Transformer and CodeGenerator are
created per call inside their stages), so everything lives in ``Scope.APP``
and one container serves any number of scripts.
"""
from __future__ import annotations

from typing import Sequence

from dishka import Container, Provider, Scope, from_context, make_container, provide

from .core.contracts import (
    AstNormalizer,
    CodeEmitter,
    OutputProfile,
    ScriptTransformer,
    SemanticGuard,
    SourceFilter,
    SourceParser,
    Transpiler,
    TranspilerSettings,
)
from .emit.legacy import LegacyEmitter
from .emit.standard import ImportBlockFormatter, ModuleBanner, StandardEmitter, StandardModulePrinter
from .emit.standard.passes import ModulePass, default_passes
from .pipeline.service import TranspileService
from .pipeline.stages import (
    CommentOnlySourceFilter,
    CurrentValueNormalizer,
    FreshTransformer,
    InputHoistingNormalizer,
    LegacyFunctionNameNormalizer,
    LegacySessionNormalizer,
    PineSemanticGuard,
    RepairingSourceParser,
)


class SettingsProvider(Provider):
    """Configuration handed in by the caller as container context."""

    settings = from_context(provides=TranspilerSettings, scope=Scope.APP)


class FrontEndProvider(Provider):
    """Pine text -> checked Pine AST."""

    scope = Scope.APP

    source_filter = provide(CommentOnlySourceFilter, provides=SourceFilter)
    parser = provide(RepairingSourceParser, provides=SourceParser)
    guard = provide(PineSemanticGuard, provides=SemanticGuard)

    @provide
    def normalizers(self) -> Sequence[AstNormalizer]:
        return (CurrentValueNormalizer(), LegacyFunctionNameNormalizer(), LegacySessionNormalizer(),
                InputHoistingNormalizer())


class MiddleEndProvider(Provider):
    """Pine AST -> PyneCore-shaped tree."""

    scope = Scope.APP

    transformer = provide(FreshTransformer, provides=ScriptTransformer)


class LegacyOutputProvider(Provider):
    """pine2pyne's own layout."""

    scope = Scope.APP

    emitter = provide(LegacyEmitter, provides=CodeEmitter)


class StandardOutputProvider(Provider):
    """The standard layout."""

    scope = Scope.APP

    @provide
    def banner(self) -> ModuleBanner:
        return ModuleBanner()

    @provide
    def imports(self) -> ImportBlockFormatter:
        return ImportBlockFormatter()

    @provide
    def printer(self, banner: ModuleBanner, imports: ImportBlockFormatter) -> StandardModulePrinter:
        return StandardModulePrinter(banner=banner, imports=imports)

    @provide
    def passes(self) -> Sequence[ModulePass]:
        return default_passes()

    @provide
    def emitter(self, printer: StandardModulePrinter, passes: Sequence[ModulePass]) -> CodeEmitter:
        return StandardEmitter(printer, passes)


class ApplicationProvider(Provider):
    """The use case, assembled from whatever the other providers bound."""

    scope = Scope.APP

    @provide
    def transpiler(
        self,
        source_filter: SourceFilter,
        parser: SourceParser,
        normalizers: Sequence[AstNormalizer],
        guard: SemanticGuard,
        transformer: ScriptTransformer,
        emitter: CodeEmitter,
    ) -> Transpiler:
        return TranspileService(source_filter, parser, normalizers, guard, transformer, emitter)


#: The output provider of each profile. Register a new profile here.
OUTPUT_PROVIDERS: dict[OutputProfile, type[Provider]] = {
    OutputProfile.LEGACY: LegacyOutputProvider,
    OutputProfile.STANDARD: StandardOutputProvider,
}


def build_providers(profile: OutputProfile) -> list[Provider]:
    return [
        SettingsProvider(),
        FrontEndProvider(),
        MiddleEndProvider(),
        OUTPUT_PROVIDERS[profile](),
        ApplicationProvider(),
    ]


def make_transpiler_container(
    profile: OutputProfile | str | None = None,
    *extra_providers: Provider,
) -> Container:
    """A container for one output profile. ``extra_providers`` come last, so
    they may override any binding (tests swap a stage this way)."""
    resolved = OutputProfile.parse(profile)
    return make_container(
        *build_providers(resolved), *extra_providers,
        context={TranspilerSettings: TranspilerSettings(profile=resolved)},
    )


_containers: dict[OutputProfile, Container] = {}


def default_container(profile: OutputProfile | str | None = None) -> Container:
    """The process-wide container of a profile, built on first use."""
    resolved = OutputProfile.parse(profile)
    container = _containers.get(resolved)
    if container is None:
        container = _containers.setdefault(resolved, make_transpiler_container(resolved))
    return container


def get_transpiler(profile: OutputProfile | str | None = None) -> Transpiler:
    return default_container(profile).get(Transpiler)
