"""
Recursive descent parser for Pine Script v4, v5 and v6.

Converts token stream from lexer into an Abstract Syntax Tree (AST).
"""
import dataclasses
import warnings
from typing import List, Optional, Union
from .tokens import Token, TokenType
from .ast_nodes import *
from .errors import ParserError, SemanticError, UnsupportedFeatureError
from .once_lowering import lower_once_structures
from .strategy_when_lowering import lower_strategy_when_arguments
from .scope_renamer import fresh_name


def _pine_version(version: Optional[VersionAnnotation]) -> int:
    """The //@version number, or 0 when absent or not a number."""
    text = version.version if version else ''
    return int(text) if str(text).isdigit() else 0


# `x += 5` is read as `x := x + 5`: the operator each compound assignment token stands for
COMPOUND_ASSIGNMENT_OPERATORS = {
    TokenType.PLUS_ASSIGN: '+',
    TokenType.MINUS_ASSIGN: '-',
    TokenType.MULT_ASSIGN: '*',
    TokenType.DIV_ASSIGN: '/',
    TokenType.MOD_ASSIGN: '%',
}
# The variable that holds the receiver of `arr.get(0).v += 1`
RECEIVER_NAME_STEM = 'assigned_field_receiver'
#: the form keywords a parameter can be declared with; two overloads of one name are distinct when these differ
PARAMETER_QUALIFIER_WORDS = frozenset({'simple', 'series', 'const', 'input'})


def _as_statement_list(statement) -> list:
    """What parse_statement returned as a list: one statement, several (a source statement
    lowered to more than one), or none."""
    if isinstance(statement, list):
        return statement
    return [statement] if statement else []


class Parser:
    """Recursive descent parser for Pine Script v4, v5 and v6."""

    def __init__(self, tokens: List[Token]):
        # COMPILER_ANNOTATION tokens (//@variable, //@param, //@function, ...) are
        # documentation only and carry no syntax. They can appear anywhere a
        # comment can — including inside blocks and expressions — so handling them
        # at each parse site would mean a case in every rule. Drop them up front
        # instead; the meaningful VERSION_ANNOTATION is preserved.
        self.tokens = [t for t in tokens if t.type is not TokenType.COMPILER_ANNOTATION]
        self._late_version = None  # a `//@version` met inside a block (functions declared above it)
        self.pos = 0
        self._token_names: Optional[set] = None  # every name the tokens hold, read on first use

    def current_token(self) -> Token:
        """Get current token without advancing."""
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[self.pos]

    def peek_token(self, offset: int = 1) -> Token:
        """Look ahead at token at pos + offset."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[pos]

    def advance(self) -> Token:
        """Consume and return current token."""
        token = self.current_token()
        if token.type != TokenType.EOF:
            self.pos += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        """Consume token of expected type or raise error."""
        token = self.current_token()
        if token.type != token_type:
            error = ParserError(
                f"Expected {token_type.name}, got {token.type.name}",
                token.line,
                token.column
            )
            error.token_index = self._current_index()  # read by parse_repair
            raise error
        return self.advance()

    def match(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        return self.current_token().type in token_types

    def skip_newlines(self) -> None:
        """Skip any NEWLINE tokens."""
        while self.match(TokenType.NEWLINE):
            self.advance()

    def _looks_like_generic_call(self) -> bool:
        """
        Check if the current position looks like a generic function call.
        Pattern: <type1, type2, ...>(args)
        This prevents `close < ema` from being parsed as generics.
        """
        saved_pos = self.pos
        try:
            # We're currently at <, skip it
            self.pos += 1

            # Skip through the generic parameters
            depth = 1
            while depth > 0 and self.pos < len(self.tokens):
                token = self.tokens[self.pos]
                if token.type == TokenType.LT:
                    depth += 1
                elif token.type == TokenType.GT:
                    depth -= 1
                    if depth == 0:
                        # Found closing >, check next token
                        self.pos += 1
                        if self.pos < len(self.tokens):
                            next_token = self.tokens[self.pos]
                            # Generic calls MUST be followed by ( for function call
                            self.pos = saved_pos
                            return next_token.type == TokenType.LPAREN
                        self.pos = saved_pos
                        return False
                elif token.type not in (TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER,
                                        TokenType.COMMA, TokenType.LT, TokenType.GT):
                    # Only type names and commas are valid inside <...> generics.
                    # Any operator (and, or, /, +, etc.) means this is a comparison, not generics.
                    self.pos = saved_pos
                    return False
                self.pos += 1

            self.pos = saved_pos
            return False
        except Exception:
            self.pos = saved_pos
            return False

    def error(self, message: str) -> ParserError:
        """Create parser error at current position."""
        token = self.current_token()
        error = ParserError(message, token.line, token.column)
        error.token_index = self._current_index()  # read by parse_repair
        return error

    def _current_index(self) -> int:
        """Index in self.tokens of current_token()."""
        return min(self.pos, len(self.tokens) - 1)

    # ========================================================================
    # Top-level parsing
    # ========================================================================

    def parse(self) -> Script:
        """Parse entire Pine Script source."""
        script = Script()

        self.skip_newlines()

        # Parse version annotation
        if self.match(TokenType.VERSION_ANNOTATION):
            version_token = self.advance()
            script.version = VersionAnnotation(version=version_token.value)
            self.skip_newlines()

        # Parse script declaration (indicator/strategy/library)
        if self._at_script_declaration(script.version):
            script.script_decl = self.parse_script_declaration(script.version)
            self.skip_newlines()

        # Parse imports
        while self.match(TokenType.IMPORT):
            script.imports.append(self.parse_import())
            self.skip_newlines()

        # Parse global declarations and body
        # Track actual source line numbers for ordering and blank line detection
        while not self.match(TokenType.EOF):
            self.skip_newlines()

            if self.match(TokenType.EOF):
                break

            # Handle VERSION_ANNOTATION tokens that appear in the body
            # (e.g., when pre-version declarations like DEFAULT_PYRAMIDING = 6
            # appear before //@version=6 in the source)
            if self.match(TokenType.VERSION_ANNOTATION):
                if not script.version:
                    version_token = self.advance()
                    script.version = VersionAnnotation(version=version_token.value)
                else:
                    self.advance()  # Skip duplicate version annotations
                continue
            if not script.version and self._late_version is not None:
                script.version = VersionAnnotation(version=self._late_version)

            # Handle COMPILER_ANNOTATION tokens in the body
            if self.match(TokenType.COMPILER_ANNOTATION):
                self.advance()  # Skip compiler annotations
                continue

            # Handle strategy/indicator/library declarations that appear after
            # pre-declaration code (when the initial top-level check missed them)
            if not script.script_decl and self._at_script_declaration(script.version):
                script.script_decl = self.parse_script_declaration(script.version)
                self.skip_newlines()
                continue

            # Record the starting position of each declaration/statement
            start_token = self.current_token()
            self._statement_start = self.pos

            # `export type T` / `export enum E` / `export const T N = v` (a library's API): parsed as the
            # declaration itself
            exported_declaration = False
            if self.match(TokenType.EXPORT) and (self.peek_token().type in (TokenType.TYPE, TokenType.ENUM)
                                                 or self.peek_token().value == 'const'):
                exported_declaration = True
                self.advance()

            # Check for function declarations
            if self.is_function_declaration():
                node = self.parse_function_declaration()
                self._stamp_top_level(node, start_token)
                script.declarations.append(node)
            # Check for var/varip declarations
            elif self.match(TokenType.VAR, TokenType.VARIP):
                node = self.parse_var_declaration()
                self._stamp_top_level(node, start_token)
                script.declarations.append(node)
            # Check for type declarations
            elif self.match(TokenType.TYPE):
                node = self.parse_type_declaration()
                node.is_export = exported_declaration
                self._stamp_top_level(node, start_token)
                script.declarations.append(node)
            # Check for enum declarations
            elif self.match(TokenType.ENUM):
                node = self.parse_enum_declaration()
                node.is_export = exported_declaration
                self._stamp_top_level(node, start_token)
                script.declarations.append(node)
            # Handle import statements that appear in the body (not just at top)
            elif self.match(TokenType.IMPORT):
                import_node = self.parse_import()
                script.imports.append(import_node)
            # Check for input declarations
            elif self.is_input_declaration():
                saved_pos = self.pos
                try:
                    node = self.parse_input_declaration()
                    self._stamp_top_level(node, start_token)
                    script.declarations.append(node)
                except Exception:
                    # Not a simple input declaration (e.g., input.string(...) == "On")
                    # Fall back to parsing as a regular statement
                    self.pos = saved_pos
                    for stmt in _as_statement_list(self.parse_statement()):
                        self._stamp_top_level(stmt, start_token)
                        script.body.append(stmt)
            # Otherwise, parse as statement
            else:
                for stmt in _as_statement_list(self.parse_statement()):
                    self._stamp_top_level(stmt, start_token)
                    script.body.append(stmt)

            self.skip_newlines()

        script = lower_once_structures(script)
        return lower_strategy_when_arguments(script, _pine_version(script.version))

    def _at_script_declaration(self, version: Optional[VersionAnnotation]) -> bool:
        """At an indicator/strategy/library call; before v5 a study() call is the indicator."""
        if not self.match(TokenType.IDENTIFIER):
            return False
        name = self.current_token().value
        if name == 'study':
            return _pine_version(version) < 5 and self.peek_token().type == TokenType.LPAREN
        return name in ('indicator', 'strategy', 'library')

    def parse_script_declaration(self, version: Optional[VersionAnnotation] = None
                                 ) -> Union[IndicatorDecl, StrategyDecl, LibraryDecl]:
        """Parse indicator/strategy/library declaration, binding positionals to Pine's names.

        `title` holds a string-literal title; any other title expression stays in kwargs
        under 'title' and `title` is None.
        """
        name_token = self.advance()  # indicator, strategy, library, or v4 study
        kind = 'indicator' if name_token.value == 'study' else name_token.value
        self.expect(TokenType.LPAREN)
        positionals, keywords = self._parse_declaration_arguments(name_token.value)
        self.expect(TokenType.RPAREN)
        kwargs = self._bind_declaration_positionals(kind, name_token, positionals, keywords, version)

        title_value = kwargs.pop('title', None)
        title_double_quoted = isinstance(title_value, Literal) and title_value.is_double_quoted
        if title_value is None:
            title = f"Untitled {name_token.value}"
        elif isinstance(title_value, Literal) and title_value.literal_type == 'string':
            title = title_value.value
        else:
            title = None
            kwargs = {'title': title_value, **kwargs}

        declaration_class = {'indicator': IndicatorDecl, 'strategy': StrategyDecl}.get(kind, LibraryDecl)
        # a codegen refusal reports the declaration's line
        return declaration_class(title=title, kwargs=kwargs, line=name_token.line,
                                 positional_count=len(positionals), title_double_quoted=title_double_quoted,
                                 keyword_order=tuple(keywords))

    def _parse_declaration_arguments(self, function_name: str) -> tuple:
        """Return (positionals, keywords) up to the closing paren.

        A positional after a keyword is refused, as Pine refuses it in every call.
        """
        positionals, keywords = [], {}
        while not self.match(TokenType.RPAREN):
            self.skip_newlines()  # Handle newlines before parameters
            if self.match(TokenType.RPAREN):  # Check again after skipping newlines
                break
            if self.match(TokenType.COMMA):
                self.advance()
                self.skip_newlines()  # Handle newlines after comma
                continue
            if (self.match(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER) and
                    self.peek_token().type == TokenType.ASSIGN):
                key_token = self.advance()
                if key_token.value in keywords:  # published scripts do this; the last value wins, as before
                    warnings.warn(f"{key_token.line}:{key_token.column}: {function_name}() argument "
                                  f"'{key_token.value}' is given twice; the last value is kept")
                self.expect(TokenType.ASSIGN)
                keywords[key_token.value] = self.parse_expression()
            elif keywords:
                raise self.positional_after_keyword(list(keywords)[-1])
            else:
                positionals.append(self.parse_expression())
        return positionals, keywords

    def _bind_declaration_positionals(self, kind: str, name_token: Token, positionals: list,
                                      keywords: dict, version: Optional[VersionAnnotation]) -> dict:
        """Name each positional by Pine's parameter order; return them, then the keywords."""
        from .pine_builtins import SCRIPT_DECLARATION_POSITIONAL_PARAMS, VERSION_INVARIANT_POSITIONAL_COUNT
        names = SCRIPT_DECLARATION_POSITIONAL_PARAMS[kind]
        invariant_count = VERSION_INVARIANT_POSITIONAL_COUNT[kind]
        function_name = name_token.value
        if len(positionals) > len(names):
            raise self._declaration_error(
                f"{function_name}() takes at most {len(names)} positional arguments, got {len(positionals)}",
                name_token)
        if len(positionals) > invariant_count and _pine_version(version) < 5:
            raise self._declaration_error(
                f"{function_name}() positional argument {invariant_count + 1} ('{names[invariant_count]}') "
                f"depends on the Pine version; pass it by keyword or declare //@version=5 or later",
                name_token)
        if function_name == 'study':
            keywords = self._study_keywords_as_indicator(keywords, name_token)
        bound = dict(zip(names, positionals))
        for name in bound:
            if name in keywords:
                # `study("A", title="B")` runs on TradingView; the positional value is kept
                warnings.warn(f"{name_token.line}:{name_token.column}: {function_name}() argument '{name}' is "
                              f"given by position and by keyword; the positional value is kept")
        return {**bound, **{key: value for key, value in keywords.items() if key not in bound}}

    def _study_keywords_as_indicator(self, keywords: dict, name_token: Token) -> dict:
        """v4 study() keywords under indicator()'s names; one indicator() lacks is refused."""
        from .pine_builtins import SCRIPT_DECLARATION_POSITIONAL_PARAMS, STUDY_KEYWORD_RENAMES
        renamed = {STUDY_KEYWORD_RENAMES.get(key, key): value for key, value in keywords.items()}
        unknown = [key for key in renamed if key not in SCRIPT_DECLARATION_POSITIONAL_PARAMS['indicator']]
        if unknown:
            raise self._declaration_error(f"study() argument '{unknown[0]}' has no indicator() equivalent",
                                          name_token)
        return renamed

    def _declaration_error(self, message: str, token: Token) -> ParserError:
        """A declaration binding error at `token`."""
        error = ParserError(message, token.line, token.column)
        error.token_index = next(index for index, candidate in enumerate(self.tokens) if candidate is token)
        return error

    def parse_import(self) -> ImportDecl:
        """Parse import statement: import user/library/version as alias"""
        import_token = self.expect(TokenType.IMPORT)

        # Parse user/library/version as separate tokens joined by DIV (/)
        # e.g., doqkhanh/tafirstlib/2 → IDENTIFIER DIV IDENTIFIER DIV INT_LITERAL
        user = self._hyphenated_name()
        library = ''
        version = None

        if self.match(TokenType.DIV):
            self.advance()  # consume /
            library = self._hyphenated_name()

        if self.match(TokenType.DIV):
            self.advance()  # consume /
            # Version can be an integer or identifier
            if self.match(TokenType.INT_LITERAL):
                version = self.advance().value
            elif self.match(TokenType.IDENTIFIER):
                version = self.advance().value

        # Parse optional 'as alias'
        alias = None
        if self.match(TokenType.IDENTIFIER) and self.current_token().value == 'as':
            self.advance()
            alias = self.expect(TokenType.IDENTIFIER).value

        return ImportDecl(user=user, library=library, version=version, alias=alias, line=import_token.line)

    def _hyphenated_name(self) -> str:
        """A publisher or library name of an import, hyphens included (`Trading-IQ`)."""
        name = self.expect(TokenType.IDENTIFIER).value
        while (self.match(TokenType.MINUS) and self.peek_token().type == TokenType.IDENTIFIER
               and self.peek_token().line == self.current_token().line):
            self.advance()
            name += '-' + self.advance().value
        return name

    def is_function_declaration(self) -> bool:
        """Check if current position is a function declaration."""
        # Look for: [export] [method] name(params) =>
        saved_pos = self.pos

        # Skip export/method keywords
        if self.match(TokenType.EXPORT):
            self.pos += 1
        # a function or method may be named like a builtin type (`box(x) =>`, `method box(...)`)
        names = (TokenType.IDENTIFIER, TokenType.FUNCTION, TokenType.TYPE_IDENTIFIER)
        if self.match(TokenType.METHOD):
            self.pos += 1

        # Check for identifier followed by ( and eventually =>; `function` is a valid name (`export function (...)`)
        if not (self.match(*names) and self.peek_token().type == TokenType.LPAREN):
            self.pos = saved_pos
            return False

        # Scan ahead to find => arrow (function declarations must have this)
        # Skip past the parameter list
        paren_depth = 0
        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if token.type == TokenType.LPAREN:
                paren_depth += 1
            elif token.type == TokenType.RPAREN:
                paren_depth -= 1
                if paren_depth == 0:
                    # Found closing paren, check next token for =>
                    self.pos += 1
                    if self.pos < len(self.tokens) and self.tokens[self.pos].type == TokenType.ARROW:
                        self.pos = saved_pos
                        return True
                    else:
                        self.pos = saved_pos
                        return False
            self.pos += 1

        self.pos = saved_pos
        return False

    def parse_function_declaration(self) -> FuncDecl:
        """Parse function declaration."""
        is_export = False
        is_method = False

        if self.match(TokenType.EXPORT):
            is_export = True
            self.advance()

        if self.match(TokenType.METHOD):
            is_method = True
            self.advance()

        name = (self.advance().value if self.match(TokenType.FUNCTION, TokenType.TYPE_IDENTIFIER)
                else self.expect(TokenType.IDENTIFIER).value)

        # Parse parameters
        self.expect(TokenType.LPAREN)
        self.skip_newlines()  # Skip newlines after opening paren
        params = []

        while not self.match(TokenType.RPAREN):
            self.skip_newlines()  # Skip newlines before each parameter

            if self.match(TokenType.RPAREN):
                break

            if self.match(TokenType.COMMA):
                self.advance()
                continue

            # Parse parameter: [type] name [= default]
            # Type can be: simple (float), generic (map<K,V>), or array (int[]), or custom type (TradeInfo)
            is_series = False
            type_hint = None
            qualifier = None
            if self.match(TokenType.TYPE_IDENTIFIER):
                # Look ahead to disambiguate type hint vs param name:
                # TYPE_IDENTIFIER + IDENTIFIER → type hint (e.g. "float x")
                # TYPE_IDENTIFIER + LT → generic type hint (e.g. "map<string, int> this")
                # TYPE_IDENTIFIER + RPAREN/COMMA/ASSIGN → param name (e.g. "array" as name)
                next_tok = self.peek_token()
                if next_tok.type in (TokenType.IDENTIFIER, TokenType.LT, TokenType.TYPE_IDENTIFIER) \
                        or self._array_type_ahead():
                    # a parameter may be declared with a form keyword; two overloads of one name differ when
                    # their keywords differ, so the word is kept beside the type
                    if self.current_token().value in PARAMETER_QUALIFIER_WORDS and (
                            next_tok.type == TokenType.TYPE_IDENTIFIER
                            or (next_tok.type == TokenType.IDENTIFIER
                                and self.peek_token(2).type in (TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER))):
                        qualifier = self.current_token().value
                    is_series = self.current_token().value == 'series' and (
                        next_tok.type == TokenType.TYPE_IDENTIFIER
                        or (next_tok.type == TokenType.IDENTIFIER
                            and self.peek_token(2).type in (TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER)))
                    type_hint = self.parse_generic_type()
                # else: fall through, treat TYPE_IDENTIFIER as param name
            elif self.match(TokenType.IDENTIFIER) and self._array_type_ahead():
                # `FVG[] arr`: an array of a user type
                element = self.advance().value
                self.advance()  # [
                self.advance()  # ]
                type_hint = f'array<{element}>'
            elif self.match(TokenType.IDENTIFIER) and self.peek_token().type == TokenType.IDENTIFIER:
                # For method parameters like "TradeInfo this" or "int val"
                # First IDENTIFIER is the type, second is the parameter name
                type_hint = self.advance().value

            # Param name can be IDENTIFIER or TYPE_IDENTIFIER used as a name
            if self.match(TokenType.IDENTIFIER):
                param_name = self.advance().value
            elif self.match(TokenType.TYPE_IDENTIFIER):
                param_name = self.advance().value
            else:
                param_name = self.expect(TokenType.IDENTIFIER).value

            default = None
            if self.match(TokenType.ASSIGN):
                self.advance()
                default = self.parse_expression()

            params.append(Parameter(name=param_name, type_hint=type_hint, default=default,
                                    is_series=is_series, qualifier=qualifier))
            self.skip_newlines()  # Skip newlines after parameter

        self.expect(TokenType.RPAREN)

        # Parse => and body
        self.expect(TokenType.ARROW)
        self.skip_newlines()

        # Check if single-line or multi-line body
        if self.match(TokenType.INDENT):
            # Multi-line body
            self.advance()
            body = self.parse_block()
            self.expect(TokenType.DEDENT)
        else:
            # Single-line body — may have comma-separated statements
            # e.g., fn(x) => a = expr1, b = expr2, return_expr
            body = self._parse_single_line_body()

        return FuncDecl(name=name, params=params, body=body,
                        is_method=is_method, is_export=is_export)

    def _parse_single_line_body(self):
        """Parse single-line function body with optional comma-separated statements.
        Pine Script allows: fn(x) => stmt1, stmt2, return_expr
        Each comma-separated item can be an assignment or expression.
        Returns a single expression (no commas) or a list of statements (commas present).
        """
        # Parse first item — could be var declaration, assignment, or expression
        first = self._parse_positioned_single_line_item()

        # Check for comma (multiple statements)
        if not self.match(TokenType.COMMA):
            return first  # Single expression, return as-is

        # Multiple comma-separated statements
        items = [first]
        while self.match(TokenType.COMMA):
            self.advance()  # consume comma
            self.skip_newlines()
            items.append(self._parse_positioned_single_line_item())

        # `f(id) => line.set_y1(id, 1), line.set_y2(id, 2)`: an expression before the last one is a statement
        # of the body; the last one stays the value the function returns
        return [item if isinstance(item, Statement) or index == len(items) - 1 else
                ExpressionStatement(expr=item, line=getattr(item, 'line', 0), column=getattr(item, 'column', 0))
                for index, item in enumerate(items)]

    def _parse_single_line_item(self):
        """Parse a single item in a comma-separated single-line function body.
        Can be: var type name = expr, name = expr, or just expr.
        """
        # Handle 'var' declarations: var type name = expr
        if self.match(TokenType.VAR):
            return self.parse_statement()

        # Try assignment: name = expr (look ahead for IDENTIFIER followed by ASSIGN)
        saved_pos = self.pos
        if self.match(TokenType.TYPE_IDENTIFIER):
            # Could be typed assignment: float x = expr
            type_hint = self.parse_generic_type()
            if self.match(TokenType.IDENTIFIER):
                var_name = self.advance().value
                if self.match(TokenType.ASSIGN):
                    self.advance()
                    value = self.parse_expression()
                    return Assignment(target=var_name, value=value, type_hint=type_hint)
            self.pos = saved_pos

        if self.match(TokenType.IDENTIFIER):
            maybe_name = self.current_token().value
            next_pos = self.pos + 1
            if next_pos < len(self.tokens) and self.tokens[next_pos].type == TokenType.ASSIGN:
                var_name = self.advance().value
                self.advance()  # consume =
                value = self.parse_expression()
                return Assignment(target=var_name, value=value)
            # Not an assignment — fall through to expression parse
            self.pos = saved_pos

        return self.parse_expression()

    def _array_type_ahead(self) -> bool:
        """At `T[] name`: an array type in the bracket spelling, followed by a name."""
        return (self.peek_token().type == TokenType.LBRACKET and self.peek_token(2).type == TokenType.RBRACKET
                and self.peek_token(3).type in (TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER))

    def parse_generic_type(self) -> str:
        """
        Parse type syntax:
        - Modifier + type: simple int, series float, const string, input float
        - Generic: type<T> or map<K, V>
        - Array bracket: type[] (e.g., string[], int[])
        """
        # Pine Script type modifiers (no Python equivalent, consume and drop)
        TYPE_MODIFIERS = {'simple', 'series', 'const', 'input'}

        base_type = self.advance().value  # Get base type (map, array, string, int, etc.)

        # Handle type modifiers: simple int, series float, etc.
        # The modifier is consumed; `const` is remembered for the declaration being parsed
        self.last_type_modifier = None
        if base_type in TYPE_MODIFIERS and (
                self.match(TokenType.TYPE_IDENTIFIER)
                # `simple Timezone tz`, `series collectedData value`: a modifier before a user type
                or (self.match(TokenType.IDENTIFIER) and self.peek_token().type in (
                    TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER, TokenType.LBRACKET))):
            self.last_type_modifier = base_type
            base_type = self.advance().value

        # Check for array bracket syntax: type[]
        if self.match(TokenType.LBRACKET):
            self.advance()  # consume [
            self.expect(TokenType.RBRACKET)  # consume ]
            # Convert to generic array syntax: string[] -> array<string>
            return f"array<{base_type}>"

        # Check for generic parameters: type<T>
        if self.match(TokenType.LT):
            self.advance()  # consume <

            generic_params = []
            while not self.match(TokenType.GT):
                if self.match(TokenType.COMMA):
                    self.advance()
                    continue

                # Parse type parameter (can be TYPE_IDENTIFIER or IDENTIFIER)
                if self.match(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER):
                    generic_params.append(self.advance().value)
                else:
                    raise self.error(f"Expected type parameter, got {self.current_token().type.name}")

            self.expect(TokenType.GT)  # consume >

            # Format as Python generic: map<string, float> -> dict[str, float]
            return f"{base_type}<{', '.join(generic_params)}>"

        return base_type

    def parse_var_declaration(self) -> Union[VarDecl, VaripDecl]:
        """Parse var/varip declaration."""
        is_var = self.match(TokenType.VAR)
        self.advance()  # var or varip

        # Parse optional type hint (with generic support or UDT name)
        type_hint = None
        # A type keyword directly followed by '=' is being used as the variable
        # NAME, not a type hint — Pine lets a variable shadow a type namespace
        # (e.g. `var matrix = matrix.new<float>(...)`, `var array = ...`). A real
        # type hint is instead followed by a name or a generic '<'. Skip the
        # type-hint branch so the keyword is consumed as the name below.
        if self.match(TokenType.TYPE_IDENTIFIER) and self.peek_token().type != TokenType.ASSIGN:
            type_hint = self.parse_generic_type()
        elif self.match(TokenType.IDENTIFIER):
            # Could be a UDT type name like Statistics, TradeInfo
            next_type = self.peek_token().type
            if next_type == TokenType.IDENTIFIER:
                # Simple UDT: TradeInfo myVar = ...
                type_hint = self.advance().value
            elif next_type in (TokenType.LBRACKET, TokenType.LT):
                # UDT array/generic: TestResult[] results = ..., MyType<T> x = ...
                type_hint = self.advance().value  # UDT name
                if self.match(TokenType.LBRACKET):
                    self.advance()  # consume [
                    self.expect(TokenType.RBRACKET)  # consume ]
                    type_hint = f"array<{type_hint}>"
                elif self.match(TokenType.LT):
                    # Parse generic params
                    self.advance()  # consume <
                    generic_params = []
                    while not self.match(TokenType.GT):
                        if self.match(TokenType.COMMA):
                            self.advance()
                            continue
                        if self.match(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER):
                            generic_params.append(self.advance().value)
                        else:
                            raise self.error(f"Expected type parameter, got {self.current_token().type.name}")
                    self.expect(TokenType.GT)
                    type_hint = f"{type_hint}<{', '.join(generic_params)}>"

        # Parse variable name. Accept a TYPE_IDENTIFIER here too: when no type
        # hint was taken, a type keyword in this position is the variable name
        # (`var matrix = ...`), which Pine permits.
        # After a type, the name may be a type keyword too (`var color color = na`)
        if self.match(TokenType.TYPE_IDENTIFIER) and self.peek_token().type == TokenType.ASSIGN:
            name = self.advance().value
        else:
            name = self.expect(TokenType.IDENTIFIER).value

        # Parse = value
        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()

        if is_var:
            return VarDecl(name=name, value=value, type_hint=type_hint)
        else:
            return VaripDecl(name=name, value=value, type_hint=type_hint)

    def parse_type_declaration(self) -> TypeDecl:
        """Parse type (UDT) declaration."""
        self.expect(TokenType.TYPE)
        name = self.expect(TokenType.IDENTIFIER).value

        self.skip_newlines()
        self.expect(TokenType.INDENT)

        fields = []
        field_lines = []
        while not self.match(TokenType.DEDENT):
            self.skip_newlines()
            if self.match(TokenType.DEDENT):
                break
            field_lines.append(self.current_token().line)

            # Parse field: [var|varip] type name [= default_value]
            # Type can be a built-in type (TYPE_IDENTIFIER), generic (array<T>), or UDT (IDENTIFIER)
            # varip behaves like var since we don't use realtime candles
            field_is_var = False
            if self.match(TokenType.VAR, TokenType.VARIP):
                self.advance()  # consume var/varip
                field_is_var = True
            if self.match(TokenType.TYPE_IDENTIFIER):
                type_hint = self.parse_generic_type()
            elif self.match(TokenType.IDENTIFIER):
                type_hint = self.advance().value
                if self.match(TokenType.LBRACKET) and self.peek_token().type == TokenType.RBRACKET:
                    # `piv [] swingH`: an array of a user type, `[]` written apart from its name
                    self.advance()
                    self.advance()
                    type_hint += '[]'
            else:
                raise self.error(f"Expected type, got {self.current_token().type.name}")
            if type_hint.endswith('[]'):
                type_hint = f'array<{type_hint[:-2]}>'  # `piv[] swings` is `array<piv> swings`
            # a field may be named like a type: `box box`
            field_token = (self.advance() if self.match(TokenType.TYPE_IDENTIFIER)
                           else self.expect(TokenType.IDENTIFIER))
            field_name = field_token.value

            # Handle optional default value; it carries its field's line, so a
            # refused default names the Pine line
            default_value = None
            if self.match(TokenType.ASSIGN):
                self.advance()  # consume =
                default_value = self.parse_expression()
                if isinstance(default_value, ASTNode) and not default_value.line:
                    default_value = dataclasses.replace(default_value, line=field_token.line)

            fields.append((field_name, type_hint, default_value))
            self.skip_newlines()

        self.expect(TokenType.DEDENT)

        return TypeDecl(name=name, fields=fields, field_lines=field_lines)

    def parse_enum_declaration(self) -> EnumDecl:
        """Parse enum declaration."""
        self.expect(TokenType.ENUM)
        # Enum name can be IDENTIFIER or TYPE_IDENTIFIER (e.g., "enum polyline", "enum ta")
        if self.match(TokenType.IDENTIFIER):
            name = self.advance().value
        elif self.match(TokenType.TYPE_IDENTIFIER):
            name = self.advance().value
        else:
            name = self.expect(TokenType.IDENTIFIER).value

        self.skip_newlines()
        self.expect(TokenType.INDENT)

        members = []
        while not self.match(TokenType.DEDENT):
            self.skip_newlines()
            if self.match(TokenType.DEDENT):
                break

            # Parse enum member: name [= value]; can also be TYPE_IDENTIFIER
            if self.match(TokenType.IDENTIFIER):
                member = self.advance().value
            elif self.match(TokenType.TYPE_IDENTIFIER):
                member = self.advance().value
            else:
                member = self.expect(TokenType.IDENTIFIER).value

            # Check for optional value assignment
            if self.match(TokenType.ASSIGN):
                self.advance()  # consume =
                # Parse the value (can be string, int, etc.)
                value = self.parse_expression()
                # Store as tuple (name, value) or just name
                members.append((member, value))
            else:
                members.append(member)

            self.skip_newlines()

        self.expect(TokenType.DEDENT)

        return EnumDecl(name=name, members=members)

    def is_input_declaration(self) -> bool:
        """Check if current position is an input declaration.
        Matches: name = input.*() or type name = input.*()
        """
        saved_pos = self.pos

        # Skip optional type hint (e.g., float, int, string)
        if self.match(TokenType.TYPE_IDENTIFIER):
            self.pos += 1

        # Must have an identifier (the variable name)
        if not self.match(TokenType.IDENTIFIER):
            self.pos = saved_pos
            return False

        self.pos += 1  # Skip identifier

        # Check for = input.
        result = (self.match(TokenType.ASSIGN) and
                  ((self.peek_token().type == TokenType.IDENTIFIER and
                    self.peek_token().value.startswith('input.')) or
                   self._legacy_typed_input_call_ahead()))

        self.pos = saved_pos
        return result

    def _legacy_typed_input_call_ahead(self) -> bool:
        """At `= input(` ... `type=<kind>` ... `)` with a kind parse_function_call rewrites to an input.* call:
        such a call is an input declaration, as its v5 spelling is."""
        from .pine_builtins import LEGACY_INPUT_TYPE_ARGUMENT_FUNCTIONS
        if self.peek_token().value != 'input' or self.peek_token(2).type != TokenType.LPAREN:
            return False
        depth, offset = 0, 2
        while True:
            token = self.peek_token(offset)
            if token.type == TokenType.EOF:
                return False
            if token.type in (TokenType.LPAREN, TokenType.LBRACKET):
                depth += 1
            elif token.type in (TokenType.RPAREN, TokenType.RBRACKET):
                depth -= 1
                if depth == 0:
                    return False
            elif (depth == 1 and token.type == TokenType.IDENTIFIER and token.value == 'type'
                  and self.peek_token(offset + 1).type == TokenType.ASSIGN):
                kind = self.peek_token(offset + 2)
                # a bare `float`/`bool`/`string`/`color` lexes as TYPE_IDENTIFIER; parse_function_call rewrites it too
                return (kind.type in (TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER)
                        and kind.value.removeprefix('input.') in LEGACY_INPUT_TYPE_ARGUMENT_FUNCTIONS)
            offset += 1

    def parse_input_declaration(self) -> InputDecl:
        """Parse input declaration. Handles: name = input.*() and type name = input.*()"""
        # Skip optional type hint
        if self.match(TokenType.TYPE_IDENTIFIER):
            self.advance()

        name = self.advance().value
        self.expect(TokenType.ASSIGN)

        # Parse input.*() call
        func_call = self.parse_expression()

        if not isinstance(func_call, FunctionCall):
            raise self.error("Expected input function call")

        # Extract function name and arguments
        if isinstance(func_call.func, str):
            func = func_call.func
        elif isinstance(func_call.func, MemberAccess):
            func = f"{func_call.func.object}.{func_call.func.member}"
        else:
            func = "input"

        return InputDecl(name=name, func=func, args=func_call.args, kwargs=func_call.kwargs)

    # ========================================================================
    # Statement parsing
    # ========================================================================

    def parse_statement(self) -> Union[Statement, List[Statement], None]:
        """Parse one source statement: a statement, a list of statements when the source statement is
        lowered to several (an assignment to a field of a call result), or None."""
        self.skip_newlines()

        # If statement
        if self.match(TokenType.IF):
            return self.parse_if_statement()

        # For loop
        if self.match(TokenType.FOR):
            return self.parse_for_loop()

        # While loop
        if self.match(TokenType.WHILE):
            return self.parse_while_loop()

        # Switch statement
        if self.match(TokenType.SWITCH):
            return self.parse_switch_statement()

        # Break
        if self.match(TokenType.BREAK):
            self.advance()
            return BreakStatement()

        # Continue
        if self.match(TokenType.CONTINUE):
            self.advance()
            return ContinueStatement()

        # once structure (Pine v6): `once [condition]` + an indented block
        if self._at_once_structure():
            return self.parse_once_statement()

        # Var/varip declarations (can appear inside blocks in Pine Script v6)
        if self.match(TokenType.VAR, TokenType.VARIP):
            return self.parse_var_declaration()

        # Local variable declaration with explicit type: type name = value
        # Examples: float x = 10.0, string[] ids = array.new_string()
        # Also handles user-defined types: Signal signalState = ...
        # A leading type keyword is only a declaration when followed by the var
        # NAME (IDENTIFIER), a generic '<', or an array '[]'. Anything else means
        # the keyword is being used as a VALUE — a type-conversion call
        # `bool(na)` (LPAREN), a bare variable reference `matrix` (NEWLINE, when a
        # variable shadows the type namespace), member access `matrix.set(...)`
        # (DOT), or history access `matrix[0]` — all of which are expressions.
        # A following TYPE_IDENTIFIER also means a decl — a qualifier/modifier
        # chain such as `const matrix<T> m`, `series int x`, `simple string s`.
        _pk = self.peek_token().type
        _is_typed_decl = _pk in (TokenType.LT, TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER) or (
            _pk == TokenType.LBRACKET
            and self.pos + 2 < len(self.tokens)
            and self.tokens[self.pos + 2].type == TokenType.RBRACKET
        )
        if self.match(TokenType.TYPE_IDENTIFIER) and _is_typed_decl:
            type_hint = self.parse_generic_type()
            qualifier = getattr(self, 'last_type_modifier', None)
            is_const = qualifier == 'const'
            var_name = (self.advance().value if self.match(TokenType.TYPE_IDENTIFIER)
                        and self.peek_token().type == TokenType.ASSIGN
                        else self.expect(TokenType.IDENTIFIER).value)
            self.expect(TokenType.ASSIGN)
            value = self.parse_expression()
            # Return as a local variable assignment with type hint
            return Assignment(target=var_name, value=value, type_hint=type_hint, is_const=is_const,
                              qualifier=qualifier)

        # Check for user-defined type declaration: UDT varName = value
        # Heuristic: IDENTIFIER IDENTIFIER ASSIGN (e.g., Signal signalState =)
        if (self.match(TokenType.IDENTIFIER) and
            self.pos + 1 < len(self.tokens) and
            self.tokens[self.pos + 1].type == TokenType.IDENTIFIER and
            self.pos + 2 < len(self.tokens) and
            self.tokens[self.pos + 2].type == TokenType.ASSIGN):
            type_name = self.advance().value
            var_name = self.advance().value
            self.expect(TokenType.ASSIGN)
            value = self.parse_expression()
            return Assignment(target=var_name, value=value, type_hint=type_name)

        # Check for UDT array declaration: UDT[] varName = value
        # Heuristic: IDENTIFIER LBRACKET RBRACKET IDENTIFIER ASSIGN
        if (self.match(TokenType.IDENTIFIER) and
            self.pos + 1 < len(self.tokens) and
            self.tokens[self.pos + 1].type == TokenType.LBRACKET and
            self.pos + 2 < len(self.tokens) and
            self.tokens[self.pos + 2].type == TokenType.RBRACKET and
            self.pos + 3 < len(self.tokens) and
            self.tokens[self.pos + 3].type == TokenType.IDENTIFIER and
            self.pos + 4 < len(self.tokens) and
            self.tokens[self.pos + 4].type == TokenType.ASSIGN):
            udt_name = self.advance().value
            self.advance()  # consume [
            self.advance()  # consume ]
            var_name = self.advance().value
            self.expect(TokenType.ASSIGN)
            value = self.parse_expression()
            return Assignment(target=var_name, value=value, type_hint=f"array<{udt_name}>")

        # Check for nested function declarations: name(params) =>
        if self.is_function_declaration():
            return self.parse_function_declaration()

        # Assignment or expression statement
        start_token = self.current_token()
        expr = self.parse_expression()

        # Check for assignment/reassignment
        if self.match(TokenType.ASSIGN):
            operator_token = self.advance()
            self._refuse_history_target(expr, operator_token)
            call_result_field = self._is_call_result_field(expr)
            if not call_result_field:
                self._refuse_unassignable_target(expr, operator_token, tuple_allowed=True)
            value = self.parse_expression()

            if call_result_field:
                return self._lower_call_result_field_assignment(expr, operator_token, value, start_token)
            if isinstance(expr, Identifier):
                return Assignment(target=expr.name, value=value)
            elif isinstance(expr, ArrayLiteral):
                # Tuple destructuring
                names = [e.name if isinstance(e, Identifier) else str(_without_positions(e)) for e in expr.elements]
                return Assignment(target=TupleDestructure(names=names, value=value), value=value)

        elif self.match(TokenType.REASSIGN):
            operator_token = self.advance()
            self._refuse_history_target(expr, operator_token)
            call_result_field = self._is_call_result_field(expr)
            if not call_result_field:
                self._refuse_unassignable_target(expr, operator_token)
            value = self.parse_expression()

            if call_result_field:
                return self._lower_call_result_field_assignment(expr, operator_token, value, start_token)
            if isinstance(expr, Identifier):
                return Reassignment(target=expr.name, value=value)

        # Compound assignment operators: +=, -=, *=, /=, %=
        elif self.match(TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN, TokenType.MULT_ASSIGN,
                        TokenType.DIV_ASSIGN, TokenType.MOD_ASSIGN):
            op_token = self.advance()
            self._refuse_history_target(expr, op_token)
            call_result_field = self._is_call_result_field(expr)
            if not call_result_field:
                self._refuse_unassignable_target(expr, op_token)
            right_value = self.parse_expression()

            if call_result_field:
                return self._lower_call_result_field_assignment(expr, op_token, right_value, start_token)
            if isinstance(expr, Identifier):
                # Convert x += 5 to x := x + 5
                op_symbol = COMPOUND_ASSIGNMENT_OPERATORS[op_token.type]

                # Create binary expression: x + 5
                from .ast_nodes import BinaryOp
                new_value = BinaryOp(left=expr, op=op_symbol, right=right_value, compound=True)

                # Return reassignment: x := (x + 5)
                return Reassignment(target=expr.name, value=new_value)

        # Expression statement
        return ExpressionStatement(expr=expr)

    @staticmethod
    def _refuse_history_target(target: Expression, operator_token: Token) -> None:
        """`x[1] := v`, `x[1] += v`, `x[1] = v`: TradingView refuses an
        assignment to a history reference. The tree has no node that could keep
        it (the statement used to be dropped), so it is refused here, at the
        operator."""
        if isinstance(target, IndexAccess):
            raise SemanticError('history-target',
                                f'cannot assign with {operator_token.value!r} to a history reference (x[n])',
                                operator_token.line, operator_token.column)

    @staticmethod
    def _is_call_result_field(target: Expression) -> bool:
        """`arr.get(0).v`: a field of the value an expression returns."""
        return isinstance(target, MemberAccess) and not isinstance(target.object, str)

    def _lower_call_result_field_assignment(self, target: MemberAccess, operator_token: Token, value: Expression,
                                            start_token: Token) -> List[Statement]:
        """`arr.get(0).v += 1` -> `assigned_field_receiver1 = arr.get(0)` then
        `assigned_field_receiver1.v := assigned_field_receiver1.v + 1`.

        The receiver is the object a call returns, so the field changes in that object; binding it
        to a variable first evaluates the call once, as Pine does, and uses the dotted-name
        assignment every later stage already handles. The variable is fresh against every
        identifier of the script."""
        receiver = self._fresh_receiver_name()
        field = f'{receiver}.{target.member}'
        position = self._position_of(start_token)
        if operator_token.type is TokenType.ASSIGN:
            assigned = Assignment(target=field, value=value)
        elif operator_token.type is TokenType.REASSIGN:
            assigned = Reassignment(target=field, value=value)
        else:
            current = Identifier(name=field, **position)
            assigned = Reassignment(target=field, value=BinaryOp(
                left=current, op=COMPOUND_ASSIGNMENT_OPERATORS[operator_token.type], right=value, compound=True,
                **position))
        statements = [Assignment(target=receiver, value=target.object), assigned]
        for statement in statements:
            statement.line, statement.column = position.get('line', 0), position.get('column', 0)
        return statements

    def _fresh_receiver_name(self) -> str:
        """`assigned_field_receiver<N>`, the lowest N no identifier of the script holds."""
        if self._token_names is None:
            self._token_names = {part for token in self.tokens if isinstance(token.value, str)
                                 for part in token.value.split('.')}
        return fresh_name(RECEIVER_NAME_STEM, self._token_names)

    @staticmethod
    def _refuse_unassignable_target(target: Expression, operator_token: Token, tuple_allowed: bool = False) -> None:
        """`arr.get(0).v += 1` is valid Pine, but the tree keeps an assignment only to a name (a
        dotted field path included) or, for `=`, a tuple. Any other target used to be dropped
        without a word, leaving a bare expression statement, so it is refused at the operator.
        A tuple that is not allowed (`[a, b] := f()`, `[a, b] += ...`) has no receiver
        and no field, so it gets its own text."""
        if isinstance(target, Identifier) or (tuple_allowed and isinstance(target, ArrayLiteral)):
            return
        if isinstance(target, ArrayLiteral):
            # Pine's tuple declaration is `<tuple_of_identifiers> = <call or structure>`, with one new name per value:
            # the names it declares are reassigned one by one
            raise SemanticError('tuple-reassignment',
                                f"a tuple is declared with '=' only, not assigned with {operator_token.value!r}; "
                                "assign each name of the tuple with its own ':=' statement",
                                operator_token.line, operator_token.column)
        advice = 'to the result of an expression; assign the receiver to a variable first and assign its field'
        raise UnsupportedFeatureError(f'assignment with {operator_token.value!r} {advice}',
                                      operator_token.line, operator_token.column)

    def parse_block(self) -> List[Statement]:
        """Parse a block of statements (indented)."""
        statements = []

        while not self.match(TokenType.DEDENT, TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.DEDENT, TokenType.EOF):
                break
            if self.match(TokenType.VERSION_ANNOTATION, TokenType.COMPILER_ANNOTATION):
                # a `//@version` written after functions declared above it (a comment line closes no block, so its
                # token stands before the block's end): the script's version, read by parse()
                token = self.advance()
                if token.type == TokenType.VERSION_ANNOTATION and self._late_version is None:
                    self._late_version = token.value
                continue

            statements.extend(_as_statement_list(self._parse_positioned_statement()))

            # Handle comma-separated statements on the same line
            # e.g., colNum = 8, rowNum = 8
            while self.match(TokenType.COMMA):
                self.advance()  # consume comma
                if self._comma_ends_block():
                    break  # `f(a), g(b),` closing its block: the trailing comma joins nothing
                statements.extend(_as_statement_list(self._parse_positioned_statement()))

        return statements

    def _comma_ends_block(self) -> bool:
        """After a trailing comma: the block ends before another statement (a dedent or the end)."""
        position = self.pos
        while position < len(self.tokens) and self.tokens[position].type == TokenType.NEWLINE:
            position += 1
        return position > self.pos and (position >= len(self.tokens)
                                        or self.tokens[position].type in (TokenType.DEDENT, TokenType.EOF))

    # ------------------------------------------------------------------ positions

    @staticmethod
    def _position_of(token: Token) -> dict:
        """The Pine line and 1-based column of `token`, as node fields: a nested
        statement, an identifier, a call and a `[]` carry the position a
        refusal points at. The one source of the positions refusals report."""
        return {'line': token.line, 'column': token.column}

    def _stamp_top_level(self, node, start_token: Token) -> None:
        """A top-level item's line, as always, and its column. Only the column
        goes through _position_of, so switching positions off keeps the line."""
        node.line = start_token.line
        node.column = self._position_of(start_token).get('column', 0)
        self._stamp_end_line(node)

    def _stamp_nested(self, node, start_token: Token):
        """A freshly parsed nested statement's position, unless it has one."""
        if node is not None and not node.line:
            position = self._position_of(start_token)
            node.line, node.column = position.get('line', 0), position.get('column', 0)
        self._stamp_end_line(node)
        return node

    _LAYOUT_TOKENS = frozenset({TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT, TokenType.EOF})

    def _stamp_end_line(self, node) -> None:
        """A declaration's last line (the line of the last token it consumed) and the line its value
        starts on (the first token after its `=`)."""
        if not isinstance(node, (Assignment, Reassignment, VarDecl, VaripDecl)) or node.end_line:
            return
        index = self.pos - 1
        while index > 0 and self.tokens[index].type in self._LAYOUT_TOKENS:
            index -= 1
        if index >= 0:
            node.end_line = self.tokens[index].line
        start = getattr(self, '_statement_start', None)
        if start is None:
            return
        for position in range(start, self.pos):
            if self.tokens[position].type in (TokenType.ASSIGN, TokenType.REASSIGN):
                after = position + 1
                while after < self.pos and self.tokens[after].type in self._LAYOUT_TOKENS:
                    after += 1
                if after < self.pos:
                    node.value_line = self.tokens[after].line
                return

    def _parse_positioned_statement(self):
        start_token = self.current_token()
        start_index = self._statement_start = self.pos
        statement = self.parse_statement()
        self._statement_start = start_index
        if isinstance(statement, list):  # one source statement lowered to several
            return [self._stamp_nested(item, start_token) for item in statement]
        return self._stamp_nested(statement, start_token)

    def _parse_positioned_single_line_item(self):
        start_token = self.current_token()
        start_index = self._statement_start = self.pos
        item = self._parse_single_line_item()
        self._statement_start = start_index
        if isinstance(item, (Assignment, VarDecl, VaripDecl)):
            return self._stamp_nested(item, start_token)
        return item

    def parse_if_statement(self) -> IfStatement:
        """Parse if statement."""
        self.expect(TokenType.IF)
        condition = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)
        body = self.parse_block()
        self.expect(TokenType.DEDENT)

        # Parse else if clauses
        elseifs = []
        while self.match(TokenType.ELSE):
            self.advance()
            if self.match(TokenType.IF):
                self.advance()
                elif_condition = self.parse_expression()
                self.skip_newlines()
                self.expect(TokenType.INDENT)
                elif_body = self.parse_block()
                self.expect(TokenType.DEDENT)
                elseifs.append((elif_condition, elif_body))
            else:
                # Else clause
                self.skip_newlines()
                self.expect(TokenType.INDENT)
                else_body = self.parse_block()
                self.expect(TokenType.DEDENT)
                return IfStatement(condition=condition, body=body, elseifs=elseifs, else_body=else_body)

        return IfStatement(condition=condition, body=body, elseifs=elseifs)

    _ASSIGNMENT_OPERATORS = frozenset({
        TokenType.ASSIGN, TokenType.REASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
        TokenType.MULT_ASSIGN, TokenType.DIV_ASSIGN, TokenType.MOD_ASSIGN,
    })
    # Tokens after `once` that make it a name: `once = 1`, `once := x`, `once.f()`, `once[1]`
    _ONCE_AS_NAME_FOLLOWERS = _ASSIGNMENT_OPERATORS | {
        TokenType.DOT, TokenType.LBRACKET, TokenType.COMMA, TokenType.ARROW, TokenType.EOF,
    }

    def _at_once_structure(self) -> bool:
        """At a Pine v6 `once` structure: the identifier `once` whose logical
        line is followed by an indented block. `once` is not a keyword here, so
        a variable or function named `once` (Pine v5) stays a name."""
        token = self.current_token()
        if token.type != TokenType.IDENTIFIER or token.value != 'once':
            return False
        if self.peek_token().type in self._ONCE_AS_NAME_FOLLOWERS or self.is_function_declaration():
            return False
        index, depth = self.pos + 1, 0
        while index < len(self.tokens):
            kind = self.tokens[index].type
            if kind in (TokenType.LPAREN, TokenType.LBRACKET):
                depth += 1
            elif kind in (TokenType.RPAREN, TokenType.RBRACKET):
                depth -= 1
            elif kind == TokenType.EOF or (kind == TokenType.NEWLINE and depth <= 0):
                break
            index += 1
        while index < len(self.tokens) and self.tokens[index].type == TokenType.NEWLINE:
            index += 1
        return index < len(self.tokens) and self.tokens[index].type == TokenType.INDENT

    def parse_once_statement(self) -> OnceStatement:
        """`once [condition]` + an indented block. Pine gives it no `else`."""
        self.advance()  # once
        condition = None if self.match(TokenType.NEWLINE) else self.parse_expression()
        self.skip_newlines()
        self.expect(TokenType.INDENT)
        body = self.parse_block()
        self.expect(TokenType.DEDENT)
        if self.match(TokenType.ELSE):
            raise self.error("a 'once' structure has no 'else'")
        return OnceStatement(condition=condition, body=body)

    def parse_for_loop(self) -> Union[ForLoop, ForInLoop]:
        """Parse for loop (range or for-in style)."""
        self.expect(TokenType.FOR)

        # Check for tuple destructuring [a, b] in ...
        if self.match(TokenType.LBRACKET):
            return self.parse_for_in_loop()

        # Skip optional type hint: for int i = ... or for float x in ...; a type name followed by `in` or `=` is
        # the variable itself (`for box in boxes`)
        if self.match(TokenType.TYPE_IDENTIFIER) and self.peek_token().type in (TokenType.IN, TokenType.ASSIGN):
            var_name = self.current_token().value
            self.advance()
        else:
            if self.match(TokenType.TYPE_IDENTIFIER):
                self.advance()  # consume type hint (int, float, etc.)
            # Check for regular 'for var in iterable'
            var_name = self.expect(TokenType.IDENTIFIER).value

        if self.match(TokenType.IN):
            # for-in loop
            self.advance()
            iterable = self.parse_expression()

            self.skip_newlines()
            self.expect(TokenType.INDENT)
            body = self.parse_block()
            self.expect(TokenType.DEDENT)

            return ForInLoop(vars=[var_name], iterable=iterable, body=body)

        # for var = from to to_val [by step]
        self.expect(TokenType.ASSIGN)
        from_val = self.parse_expression()

        self.expect(TokenType.TO)
        to_val = self.parse_expression()

        step = None
        if self.match(TokenType.BY):
            self.advance()
            step = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)
        body = self.parse_block()
        self.expect(TokenType.DEDENT)

        return ForLoop(var=var_name, from_val=from_val, to_val=to_val, step=step, body=body)

    def parse_for_in_loop(self) -> ForInLoop:
        """Parse for-in loop with tuple destructuring."""
        self.expect(TokenType.LBRACKET)

        vars = []
        while not self.match(TokenType.RBRACKET):
            if self.match(TokenType.COMMA):
                self.advance()
                continue

            vars.append(self.expect(TokenType.IDENTIFIER).value)

        self.expect(TokenType.RBRACKET)
        self.expect(TokenType.IN)

        iterable = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)
        body = self.parse_block()
        self.expect(TokenType.DEDENT)

        return ForInLoop(vars=vars, iterable=iterable, body=body)

    def parse_while_loop(self) -> WhileLoop:
        """Parse while loop."""
        self.expect(TokenType.WHILE)
        condition = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)
        body = self.parse_block()
        self.expect(TokenType.DEDENT)

        return WhileLoop(condition=condition, body=body)

    def _inline_statements(self) -> list:
        """The statements of a one-line block, separated by commas (`a := 1, b := 2`)."""
        statements = _as_statement_list(self._parse_positioned_statement())
        while self.match(TokenType.COMMA):
            self.advance()
            statements.extend(_as_statement_list(self._parse_positioned_statement()))
        return statements

    def parse_switch_statement(self) -> SwitchStatement:
        """Parse switch statement."""
        self.expect(TokenType.SWITCH)

        # Optional expression
        expr = None
        if not self.match(TokenType.NEWLINE):
            expr = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)

        cases = []
        default = None

        while not self.match(TokenType.DEDENT):
            self.skip_newlines()
            if self.match(TokenType.DEDENT):
                break

            # Check for default case (starts with =>)
            if self.match(TokenType.ARROW):
                self.advance()  # consume =>
                self.skip_newlines()
                if self.match(TokenType.INDENT):
                    self.advance()
                    default = self.parse_block()
                    self.expect(TokenType.DEDENT)
                else:
                    default = self._inline_statements()
                self.skip_newlines()
                continue

            # Parse case expression
            case_expr = self.parse_expression()

            self.expect(TokenType.ARROW)
            self.skip_newlines()

            # Parse case body
            if self.match(TokenType.INDENT):
                self.advance()
                case_body = self.parse_block()
                self.expect(TokenType.DEDENT)
            else:
                # Single line, possibly several statements: `'FX' => a := 1, b := 2`
                case_body = self._inline_statements()

            cases.append((case_expr, case_body))

        self.expect(TokenType.DEDENT)

        return SwitchStatement(expr=expr, cases=cases, default=default)

    # ========================================================================
    # Expression parsing
    # ========================================================================

    def parse_expression(self) -> Expression:
        """Parse expression (entry point for expression parsing)."""
        return self.parse_ternary()

    def parse_ternary(self) -> Expression:
        """Parse ternary operator: cond ? true_expr : false_expr"""
        expr = self.parse_or()

        if self.match(TokenType.TERNARY):
            question_line = self.current_token().line
            self.advance()
            true_expr = self.parse_ternary()
            colon_line = self.current_token().line
            self.expect(TokenType.COLON)
            false_expr = self.parse_ternary()
            return TernaryOp(condition=expr, true_expr=true_expr, false_expr=false_expr, question_line=question_line,
                             colon_line=colon_line)

        return expr

    def parse_or(self) -> Expression:
        """Parse logical OR."""
        left = self.parse_and()

        while self.match(TokenType.OR):
            op = self.advance().value
            right = self.parse_and()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_and(self) -> Expression:
        """Parse logical AND."""
        left = self.parse_equality()

        while self.match(TokenType.AND):
            op = self.advance().value
            right = self.parse_equality()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_equality(self) -> Expression:
        """Parse == and != operators."""
        left = self.parse_comparison()

        while self.match(TokenType.EQ, TokenType.NEQ):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_comparison(self) -> Expression:
        """Parse <, >, <=, >= operators."""
        left = self.parse_addition()

        while self.match(TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            op = self.advance().value
            right = self.parse_addition()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_addition(self) -> Expression:
        """Parse + and - operators."""
        left = self.parse_multiplication()

        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplication()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_multiplication(self) -> Expression:
        """Parse *, /, % operators."""
        left = self.parse_unary()

        while self.match(TokenType.MULT, TokenType.DIV, TokenType.MOD):
            op = self.advance().value
            right = self.parse_unary()
            left = BinaryOp(left=left, op=op, right=right)

        return left

    def parse_unary(self) -> Expression:
        """Parse unary operators (-, not)."""
        if self.match(TokenType.MINUS, TokenType.NOT):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op=op, operand=operand)

        return self.parse_postfix()

    def parse_postfix(self) -> Expression:
        """Parse postfix operations (function calls, member access, indexing)."""
        expr = self.parse_primary()

        # Block-level expressions (switch/if) already consumed their DEDENT;
        # don't greedily attach postfix [ or . from the NEXT statement
        if isinstance(expr, (SwitchExpression, IfExpression)):
            return expr

        while True:
            # Function call (may have generic parameters)
            # Parse as generics only for function/method names, not after index access
            # This prevents `arr[0] < 5` and `close < ema` from being parsed as generics
            # Heuristic: generics MUST be followed by ( for function call
            if (self.match(TokenType.LT) and
                self.peek_token().type in (TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER) and
                isinstance(expr, (Identifier, MemberAccess)) and
                self._looks_like_generic_call()):  # Additional check
                # Generic function call: func<T>() or obj.method<T>()
                generics = self.parse_generic_params()
                # Add generics to the expression (store as string annotation)
                if isinstance(expr, Identifier):
                    expr.name = f"{expr.name}<{generics}>"
                elif isinstance(expr, MemberAccess):
                    expr.member = f"{expr.member}<{generics}>"
                # Continue to parse the actual function call
                if self.match(TokenType.LPAREN):
                    expr = self.parse_function_call(expr)
            elif self.match(TokenType.LPAREN):
                expr = self.parse_function_call(expr)
            # Member access
            elif self.match(TokenType.DOT):
                self.advance()
                # a method named like a builtin type (`'none'.box(b)`) is a member too
                member = (self.advance().value if self.match(TokenType.TYPE_IDENTIFIER)
                          else self.expect(TokenType.IDENTIFIER).value)

                # Check for generic parameters after member (support both built-in and user-defined types)
                if (self.match(TokenType.LT) and
                    self.peek_token().type in (TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER) and
                    self._looks_like_generic_call()):
                    generics = self.parse_generic_params()
                    member = f"{member}<{generics}>"

                # Check if this is a method call
                if self.match(TokenType.LPAREN):
                    expr = self.parse_method_call(expr, member)
                else:
                    expr = MemberAccess(object=expr, member=member)
            # Index access
            elif self.match(TokenType.LBRACKET):
                bracket = self.advance()
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                expr = IndexAccess(object=expr, index=index, **self._position_of(bracket))
            else:
                break

        return expr

    def parse_generic_params(self) -> str:
        """Parse generic parameters in angle brackets: <T> or <K, V>."""
        self.expect(TokenType.LT)  # consume <

        params = []
        while not self.match(TokenType.GT):
            if self.match(TokenType.COMMA):
                self.advance()
                continue

            # Parse type parameter
            if self.match(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER):
                params.append(self.advance().value)
            else:
                raise self.error(f"Expected type parameter, got {self.current_token().type.name}")

        self.expect(TokenType.GT)  # consume >

        return ', '.join(params)

    def positional_after_keyword(self, keyword: str) -> ParserError:
        """Pine v4, v5 and v6 refuse an argument without its parameter name after one with it, in any call."""
        return self.error(f'Syntax error after the argument for "{keyword}". Arguments without their parameter '
                          f'name cannot be used after arguments with parameter names.')

    def parse_function_call(self, func: Expression) -> FunctionCall:
        """Parse function call arguments."""
        self.expect(TokenType.LPAREN)

        args = []
        kwargs = {}
        repeated: list[str] = []  # keywords given twice (refused from Pine v6)
        last_keyword = None

        while not self.match(TokenType.RPAREN):
            if self.match(TokenType.COMMA):
                self.advance()
                continue

            self.skip_newlines()
            if self.match(TokenType.RPAREN):
                break

            # Check for keyword argument (can be IDENTIFIER or TYPE_IDENTIFIER like 'color')
            if self.match(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER) and self.peek_token().type == TokenType.ASSIGN:
                key = last_keyword = self.advance().value
                self.advance()  # =
                value = self.parse_expression()
                # a keyword given twice keeps its FIRST value: MEASURED on TradingView 2026-11-13 (Pine v4,
                # `label.new(..., text = "FIRST", ..., text = "SECOND")` reads back "FIRST")
                if key in kwargs:
                    repeated.append(key)
                kwargs.setdefault(key, value)
            elif last_keyword is not None:
                raise self.positional_after_keyword(last_keyword)
            else:
                # Positional argument
                args.append(self.parse_expression())

        self.expect(TokenType.RPAREN)

        # Convert func expression to string if it's an identifier or member access
        if isinstance(func, Identifier):
            func_name = func.name
        elif isinstance(func, MemberAccess):
            func_name = func
        else:
            func_name = func

        # Pine v2-v4 `input(defval, title, type=input.float)` is the v5 `input.float(defval, title)`
        if func_name == 'input' and isinstance(kwargs.get('type'), Identifier):
            from .pine_builtins import LEGACY_INPUT_TYPE_ARGUMENT_FUNCTIONS
            typed_function = LEGACY_INPUT_TYPE_ARGUMENT_FUNCTIONS.get(kwargs['type'].name.removeprefix('input.'))
            if typed_function is not None:
                func_name = typed_function
                kwargs = {key: value for key, value in kwargs.items() if key != 'type'}

        # a call is placed at its callee's name, where a refusal points
        return FunctionCall(func=func_name, args=args, kwargs=kwargs, repeated_kwargs=repeated,
                            line=getattr(func, 'line', 0), column=getattr(func, 'column', 0))

    def parse_method_call(self, obj: Expression, method: str) -> MethodCall:
        """Parse method call arguments."""
        self.expect(TokenType.LPAREN)

        args = []
        kwargs = {}
        last_keyword = None

        while not self.match(TokenType.RPAREN):
            if self.match(TokenType.COMMA):
                self.advance()
                continue

            # Check for keyword argument (can be IDENTIFIER or TYPE_IDENTIFIER like 'color')
            if self.match(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER) and self.peek_token().type == TokenType.ASSIGN:
                key = last_keyword = self.advance().value
                self.advance()  # =
                value = self.parse_expression()
                # a keyword given twice keeps its FIRST value: MEASURED on TradingView 2026-11-13 (Pine v4,
                # `label.new(..., text = "FIRST", ..., text = "SECOND")` reads back "FIRST")
                kwargs.setdefault(key, value)
            elif last_keyword is not None:
                raise self.positional_after_keyword(last_keyword)
            else:
                # Positional argument
                args.append(self.parse_expression())

        self.expect(TokenType.RPAREN)

        return MethodCall(object=obj, method=method, args=args, kwargs=kwargs)

    def parse_primary(self) -> Expression:
        """Parse primary expressions (literals, identifiers, parenthesized expressions)."""
        if self.pos > 0 and self.tokens[self.pos - 1].type in self._ASSIGNMENT_OPERATORS and self._at_once_structure():
            raise self.error("a 'once' structure returns no value: it cannot be assigned")
        # Literals
        if self.match(TokenType.INT_LITERAL):
            token = self.advance()
            return Literal(value=token.value, literal_type='int')

        if self.match(TokenType.FLOAT_LITERAL):
            token = self.advance()
            return Literal(value=token.value, literal_type='float')

        if self.match(TokenType.STRING_LITERAL):
            token = self.advance()
            return Literal(value=token.value, literal_type='string', is_double_quoted=token.is_double_quoted)

        if self.match(TokenType.TRUE):
            self.advance()
            return Literal(value=True, literal_type='bool')

        if self.match(TokenType.FALSE):
            self.advance()
            return Literal(value=False, literal_type='bool')

        if self.match(TokenType.COLOR_LITERAL):
            token = self.advance()
            return Literal(value=token.value, literal_type='color', is_double_quoted=token.is_double_quoted)

        if self.match(TokenType.NA_LITERAL):
            self.advance()
            return NaLiteral()

        # Array literal
        if self.match(TokenType.LBRACKET):
            self.advance()
            elements = []

            while not self.match(TokenType.RBRACKET):
                if self.match(TokenType.COMMA):
                    self.advance()
                    continue

                elements.append(self.parse_expression())

            self.expect(TokenType.RBRACKET)
            return ArrayLiteral(elements=elements)

        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            if isinstance(expr, (BinaryOp, TernaryOp, Identifier, FunctionCall, IndexAccess)):
                expr = dataclasses.replace(expr, parenthesized=True)
            return expr

        # Identifier (`function` too, when a user function is named so: `function(x)`)
        if self.match(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER) or (
                self.match(TokenType.FUNCTION) and self.peek_token().type == TokenType.LPAREN):
            token = self.advance()
            return Identifier(name=token.value, **self._position_of(token))

        # If expression
        if self.match(TokenType.IF):
            return self.parse_if_expression()

        # Switch expression
        if self.match(TokenType.SWITCH):
            return self.parse_switch_expression()

        # For/for-in as value expression: x = for i in arr ...
        if self.match(TokenType.FOR):
            return self.parse_for_loop()

        # While as a value expression: x = while cond ...
        if self.match(TokenType.WHILE):
            return self.parse_while_loop()

        raise self.error(f"Unexpected token in expression: {self.current_token().type.name}")

    def parse_if_expression(self) -> IfExpression:
        """Parse if expression (returns a value)."""
        self.expect(TokenType.IF)
        condition = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)

        # Parse true branch (can be expression or statements)
        true_expr = self.parse_expression_or_block()

        self.skip_newlines()
        self.expect(TokenType.DEDENT)

        # Parse else/else-if branch
        false_expr = None
        if self.match(TokenType.ELSE):
            self.advance()
            self.skip_newlines()
            if self.match(TokenType.IF):
                # else if → recurse into another if-expression
                false_expr = dataclasses.replace(self.parse_if_expression(), chained=True)
            else:
                self.expect(TokenType.INDENT)
                false_expr = self.parse_expression_or_block()
                self.skip_newlines()
                self.expect(TokenType.DEDENT)

        return IfExpression(condition=condition, true_expr=true_expr, false_expr=false_expr)

    def parse_switch_expression(self) -> SwitchExpression:
        """Parse switch expression (returns a value)."""
        self.expect(TokenType.SWITCH)

        # Optional expression to switch on
        expr = None
        if not self.match(TokenType.NEWLINE):
            expr = self.parse_expression()

        self.skip_newlines()
        self.expect(TokenType.INDENT)

        cases = []
        default = None

        while not self.match(TokenType.DEDENT):
            self.skip_newlines()
            if self.match(TokenType.DEDENT):
                break

            # Check for default case (starts with =>)
            if self.match(TokenType.ARROW):
                self.advance()  # consume =>
                self.skip_newlines()
                # Parse default value — may be a multi-statement block
                if self.match(TokenType.INDENT):
                    self.advance()  # consume INDENT
                    default = self.parse_expression_or_block()
                    self.skip_newlines()
                    self.expect(TokenType.DEDENT)
                else:
                    default = self._comma_separated_arm(self.parse_expression())
                self.skip_newlines()
                continue

            # Parse case condition
            case_condition = self.parse_expression()

            self.expect(TokenType.ARROW)
            self.skip_newlines()

            # Parse case value — may be a multi-statement block
            if self.match(TokenType.INDENT):
                self.advance()  # consume INDENT
                case_value = self.parse_expression_or_block()
                self.skip_newlines()
                self.expect(TokenType.DEDENT)
            else:
                if self.match(TokenType.SWITCH, TokenType.IF):
                    # Pine reads the arm's value up to the end of the `=>` line: an if/switch value, whose
                    # arms follow on lines of their own, must start on its own line under the case
                    raise self.error('Mismatched input "|PE|" expecting set "end of line without line '
                                     'continuation" (an if/switch case value starts on its own line)')
                case_value = self._comma_separated_arm(self.parse_expression())

            cases.append((case_condition, case_value))
            self.skip_newlines()

        self.expect(TokenType.DEDENT)

        return SwitchExpression(expr=expr, cases=cases, default=default)

    def _comma_separated_arm(self, first):
        """`=> runtime.error("..."), na`: an arm of several expressions is a block whose last one is its
        value (the call, then the block result)."""
        if self.match(TokenType.COMMA) and self.peek_token().type in (TokenType.NEWLINE, TokenType.DEDENT,
                                                                      TokenType.EOF):
            self.advance()  # `'Dashed' => line.style_dashed ,`: a comma ending the arm joins nothing
        if not self.match(TokenType.COMMA):
            return first
        statements = [ExpressionStatement(expr=first, line=getattr(first, 'line', 0))]
        while self.match(TokenType.COMMA):
            self.advance()
            if self.match(TokenType.NEWLINE, TokenType.DEDENT, TokenType.EOF):
                break
            expression = self.parse_expression()
            statements.append(ExpressionStatement(expr=expression, line=getattr(expression, 'line', 0)))
        return statements

    def parse_expression_or_block(self) -> Union[Expression, List[Statement]]:
        """Parse either a single expression or a block of statements.
        For multi-statement blocks, returns a list of statements where the
        last expression is the return value.
        """
        # Try to parse as expression first
        start_pos = self.pos
        try:
            expr = self.parse_expression()
            # A loop parsed as an expression is a statement: a branch is either
            # a plain expression or a list of statements, never a bare loop
            if isinstance(expr, (ForLoop, ForInLoop, WhileLoop)):
                expr = [expr]
            # If next is DEDENT, it's a single expression (end of block)
            if self.match(TokenType.DEDENT):
                return expr
            # If next is NEWLINE, check if more statements follow
            if self.match(TokenType.NEWLINE):
                self.skip_newlines()
                # If DEDENT follows the newlines, it's a single expression
                if self.match(TokenType.DEDENT):
                    return expr
                # Otherwise there are more statements — fall through to block parse
        except:
            pass

        # Otherwise, parse as block
        self.pos = start_pos
        return self.parse_block()


def _without_positions(value):
    """`value` with every line and column 0, as the parser built nodes before
    they carried positions: a non-name element of a destructuring target is emitted
    as its repr, which must not change."""
    if isinstance(value, list):
        return [_without_positions(item) for item in value]
    if isinstance(value, dict):
        return {key: _without_positions(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, line=0, column=0, **{
            field.name: _without_positions(getattr(value, field.name))
            for field in dataclasses.fields(value) if field.name not in ('line', 'column')})
    return value


def parse(tokens: List[Token]) -> Script:
    """Convenience function to parse Pine Script tokens into AST.

    A rejected known spelling of valid Pine is repaired, as in transpile().
    """
    from .parse_repair import parse_tokens
    return parse_tokens(tokens)
