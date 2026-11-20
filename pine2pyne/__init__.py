"""
pine2pyne: Pine Script v4, v5 and v6 to PyneCore Python transpiler.

A tool to convert TradingView Pine Script indicators and strategies
to PyneCore Python code for local backtesting and analysis.
"""

__version__ = '1.0.0'
__author__ = 'Pine2Pyne Transpiler'

from .lexer import Lexer, tokenize
from .parser import Parser, parse
from .parse_repair import parse_source
from .legacy_function_names import qualify_legacy_function_names
from .pine_semantics import enforce as enforce_pine_semantics
from .transformer import Transformer, PyneTransformedScript
from .codegen import CodeGenerator, generate_code
from .errors import (
    Pine2PyneError,
    LexerError,
    ParserError,
    TransformerError,
    CodeGenError,
    SemanticError,
    SemanticWarning,
    UnsupportedFeatureError,
)


from .core.contracts import OutputProfile, Transpiler
from .pipeline.stages import CommentOnlySourceFilter


def _is_comment_only(source: str) -> bool:
    """Check if source contains only comments and whitespace."""
    return CommentOnlySourceFilter().is_blank(source)


def transpile(source: str, profile: 'OutputProfile | str | None' = None) -> str:
    """
    Transpile Pine Script source code to PyneCore Python.

    The pipeline (parse -> normalize -> semantic guard -> transform -> emit) is
    assembled by the dishka container in :mod:`pine2pyne.di`; ``profile`` picks
    the output layout:

    * ``'standard'`` (default) - the standard layout;
    * ``'legacy'`` - pine2pyne's earlier layout.

    Args:
        source: Pine Script v4, v5 or v6 source code
        profile: output profile (an :class:`OutputProfile` or its name)

    Returns:
        PyneCore-compatible Python source code (empty string if source is comment-only)

    Raises:
        Pine2PyneError: If any stage of transpilation fails
    """
    from .di import get_transpiler
    # A file saved as "UTF-8 with BOM" (PowerShell 5.1 `Set-Content -Encoding utf8`, Notepad) starts with
    # U+FEFF: an encoding mark, not Pine source
    return get_transpiler(profile).transpile(source.removeprefix('\ufeff'))


__all__ = [
    'transpile',
    'OutputProfile',
    'Transpiler',
    'Lexer',
    'Parser',
    'Transformer',
    'CodeGenerator',
    'tokenize',
    'parse',
    'generate_code',
    'Pine2PyneError',
    'LexerError',
    'ParserError',
    'TransformerError',
    'CodeGenError',
    'SemanticError',
    'SemanticWarning',
    'UnsupportedFeatureError',
    'PyneTransformedScript',
]
