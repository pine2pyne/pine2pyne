"""Concrete pipeline stages: thin adapters over the transpiler engine.

Each class implements one contract from :mod:`pine2pyne.core.contracts` and
delegates to the engine module that already does the work, so the engine
keeps its own tests while the application layer sees only abstractions.
"""
from __future__ import annotations

from ..ast_nodes import Script
from ..current_value import drop_current_value_indexes
from ..input_hoisting import hoist_inputs
from ..legacy_function_names import qualify_legacy_function_names
from ..legacy_sessions import qualify_legacy_sessions
from ..parse_repair import parse_source
from ..pine_semantics import enforce
from ..transformer import PyneTransformedScript, Transformer


class CommentOnlySourceFilter:
    """A source made of comments and whitespace has nothing to transpile."""

    def is_blank(self, source: str) -> bool:
        if not source:
            return True
        for line in source.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith('//'):
                return False
        return True


class RepairingSourceParser:
    """Lex and parse; a rejected known spelling of valid Pine is repaired."""

    def parse(self, source: str) -> Script:
        script, _repairs = parse_source(source)
        return script


class CurrentValueNormalizer:
    """`x[0]` is read as `x` (see :mod:`pine2pyne.current_value`)."""

    def normalize(self, script: Script) -> Script:
        return drop_current_value_indexes(script)


class LegacyFunctionNameNormalizer:
    """Pine v1-v4 call names are spelled the way Pine v5 spells them."""

    def normalize(self, script: Script) -> Script:
        return qualify_legacy_function_names(script)


class LegacySessionNormalizer:
    """Pine v4 sessions of time()/time_close() get v4's default days (see :mod:`pine2pyne.legacy_sessions`)."""

    def normalize(self, script: Script) -> Script:
        return qualify_legacy_sessions(script)


class InputHoistingNormalizer:
    """Every top-level input call becomes a main() parameter (see :mod:`pine2pyne.input_hoisting`)."""

    def normalize(self, script: Script) -> Script:
        return hoist_inputs(script)


class PineSemanticGuard:
    """Refuse what TradingView's compiler refuses, at the Pine line."""

    def check(self, script: Script) -> None:
        enforce(script)


class FreshTransformer:
    """:class:`~pine2pyne.transformer.Transformer` is single-use:
    this stage builds a fresh one for every script, so the stage itself is
    stateless and safe to share."""

    def transform(self, script: Script) -> PyneTransformedScript:
        return Transformer().transform(script)
