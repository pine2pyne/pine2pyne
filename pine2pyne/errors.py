"""
Custom error types for the Pine Script transpiler with source location tracking.
"""


class Pine2PyneError(Exception):
    """Base exception for all transpiler errors."""
    def __init__(self, message: str, line: int = 0, column: int = 0, source_file: str = None):
        self.message = message
        self.line = line
        self.column = column
        self.source_file = source_file
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format error message with location information."""
        location = f"{self.line}:{self.column}"
        if self.source_file:
            location = f"{self.source_file}:{location}"
        return f"Error at {location}: {self.message}"


class LexerError(Pine2PyneError):
    """Error during lexical analysis (tokenization)."""
    pass


class ParserError(Pine2PyneError):
    """Error during parsing (syntax error)."""
    pass


class TransformerError(Pine2PyneError):
    """Error during AST transformation."""
    pass


class CodeGenError(Pine2PyneError):
    """Error during code generation."""
    pass


class SemanticError(Pine2PyneError):
    """A program TradingView's compiler refuses, refused at its Pine line and
    column. `code` names the rule, e.g. 'history-target'; `more`
    counts the further refusals in the same script."""
    def __init__(self, code: str, message: str, line: int = 0, column: int = 0, more: int = 0):
        suffix = f' (and {more} more refusal{"s" if more != 1 else ""})' if more else ''
        super().__init__(f'Pine refuses: {message} [{code}]{suffix}', line, column)
        self.code = code


class SemanticWarning(UserWarning):
    """A program TradingView refuses that is still transpiled, output unchanged:
    a same-scope redeclaration, a read
    before the declaration; and, though TradingView compiles it, a `ta.*` call
    that runs on only some bars. Carries the rule's `code` and the Pine position;
    under `-W error` it propagates as an exception like a refusal."""
    def __init__(self, code: str, message: str, line: int = 0, column: int = 0):
        super().__init__(f'{line}:{column}: {message} [{code}]')
        self.code, self.text, self.line, self.column = code, message, line, column


class UnsupportedFeatureError(Pine2PyneError):
    """Feature not yet supported by the transpiler."""
    def __init__(self, feature: str, line: int = 0, column: int = 0):
        message = f"Unsupported feature: {feature}"
        super().__init__(message, line, column)
        self.feature = feature
