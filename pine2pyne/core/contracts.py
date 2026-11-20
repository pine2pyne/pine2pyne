"""Abstractions of the transpilation pipeline (the D in SOLID).

Every stage of pine2pyne is described here as a small Protocol. The
application service (:class:`pine2pyne.pipeline.service.TranspileService`)
depends only on these abstractions; concrete stages live in
:mod:`pine2pyne.pipeline.stages` and :mod:`pine2pyne.emit`, and are wired
together by the dishka container in :mod:`pine2pyne.di`.

Stages are deliberately narrow (the I in SOLID): a parser only parses, a
guard only refuses, an emitter only prints. A new output target is a new
:class:`CodeEmitter` implementation plus a provider, with no edit of the
service (the O in SOLID).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:  # the engine modules are heavy; contracts only name them
    from ..ast_nodes import Script
    from ..transformer import PyneTransformedScript


class OutputProfile(str, Enum):
    """The layout the emitter writes."""

    #: the standard layout (the default)
    STANDARD = 'standard'
    #: pine2pyne's earlier layout
    LEGACY = 'legacy'

    @classmethod
    def parse(cls, value: 'OutputProfile | str | None') -> 'OutputProfile':
        if value is None:
            return cls.STANDARD
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            names = ', '.join(profile.value for profile in cls)
            raise ValueError(f'unknown output profile {value!r} (expected one of: {names})') from None


@dataclass(frozen=True)
class TranspilerSettings:
    """Application-wide configuration, handed to the container as context."""

    profile: OutputProfile = OutputProfile.STANDARD


@dataclass
class CompilationUnit:
    """Everything an emitter may need about one script.

    ``pine_source`` and ``pine_script`` let an emitter recover source layout
    (blank lines, quote style, the Pine version); ``transformed`` is the
    PyneCore-shaped tree the code generator prints.
    """

    pine_source: str
    pine_script: 'Script'
    transformed: 'PyneTransformedScript'
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def pine_version(self) -> int | None:
        version = getattr(self.pine_script, 'version', None)
        text = getattr(version, 'version', None)
        return int(text) if text and str(text).isdigit() else None


@runtime_checkable
class SourceFilter(Protocol):
    """Decides whether a source has anything to transpile at all."""

    def is_blank(self, source: str) -> bool: ...


@runtime_checkable
class SourceParser(Protocol):
    """Pine source text -> Pine AST (lexing, parsing and known repairs)."""

    def parse(self, source: str) -> 'Script': ...


@runtime_checkable
class AstNormalizer(Protocol):
    """Pine AST -> Pine AST rewrite that keeps the program's meaning
    (e.g. spelling Pine v1-v4 call names the way v5 spells them)."""

    def normalize(self, script: 'Script') -> 'Script': ...


@runtime_checkable
class SemanticGuard(Protocol):
    """Refuses what TradingView's compiler refuses; returns nothing."""

    def check(self, script: 'Script') -> None: ...


@runtime_checkable
class ScriptTransformer(Protocol):
    """Pine AST -> PyneCore-shaped tree."""

    def transform(self, script: 'Script') -> 'PyneTransformedScript': ...


@runtime_checkable
class CodeEmitter(Protocol):
    """PyneCore-shaped tree -> Python source text, in one output profile."""

    profile: OutputProfile

    def emit(self, unit: CompilationUnit) -> str: ...


@runtime_checkable
class Transpiler(Protocol):
    """The use case: Pine source text -> Python source text."""

    def transpile(self, source: str) -> str: ...

    def validate(self, source: str) -> None: ...
