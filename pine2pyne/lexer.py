"""
Lexer (tokenizer) for Pine Script v4, v5 and v6.

Handles indentation-significant syntax, multi-line continuations,
and all Pine Script token types.
"""
import re
import dataclasses
from typing import List, Optional
from .errors import LexerError
from .tokens import Token, TokenType, KEYWORDS, TYPE_KEYWORDS


class Lexer:
    """Tokenizes Pine Script v4, v5 and v6 source code."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        self.indent_stack: List[int] = [0]  # Track indentation levels
        self.paren_depth = 0  # Track depth inside (), [], {} for indentation-free zones
        self.pine_version = 6  # from //@version

    def current_char(self) -> Optional[str]:
        """Get current character without advancing."""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]

    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Look ahead at character at pos + offset."""
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]

    def advance(self) -> Optional[str]:
        """Consume and return current character, updating position."""
        if self.pos >= len(self.source):
            return None
        char = self.source[self.pos]
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char

    def skip_whitespace(self, skip_newlines: bool = False) -> None:
        """Skip spaces and tabs (and optionally newlines)."""
        while self.current_char() in (' ', '\t') or (skip_newlines and self.current_char() == '\n'):
            self.advance()

    def read_line_comment(self) -> Token:
        """Read // style comment until end of line."""
        start_line = self.line
        start_col = self.column

        # Check for compiler annotations. pos is on the first '/' of '//', so the
        # '@' of '//@version' is at offset 2 — peek_char()'s default offset of 1
        # only ever sees the second '/'.
        if self.peek_char(2) == '@':
            return self.read_annotation()
        # `// @version=5`: the version still counts with spaces after `//`
        offset = 2
        while self.peek_char(offset) in (' ', '\t'):
            offset += 1
        if offset > 2 and self.source[self.pos + offset:self.pos + offset + 8] == '@version':
            while offset > 2:
                self.advance()
                offset -= 1
            return self.read_annotation()

        # Regular comment
        comment = ''
        while self.current_char() and self.current_char() != '\n':
            comment += self.advance()

        return Token(TokenType.COMMENT, comment.strip(), start_line, start_col, comment)

    def read_annotation(self) -> Token:
        """Read //@annotation style compiler directive."""
        start_line = self.line
        start_col = self.column

        # Skip //
        self.advance()
        self.advance()
        # Skip @
        self.advance()

        # Read annotation name
        annotation = ''
        while self.current_char() and self.current_char() not in (' ', '\t', '\n', '='):
            annotation += self.advance()

        # Check for version annotation
        if annotation == 'version':
            self.skip_whitespace()
            if self.current_char() == '=':
                self.advance()
                self.skip_whitespace()
                version = ''
                while self.current_char() and self.current_char() != '\n':
                    version += self.advance()
                if version.strip().isdigit():
                    self.pine_version = int(version.strip())
                return Token(TokenType.VERSION_ANNOTATION, version.strip(), start_line, start_col)

        # Read rest of annotation
        value = ''
        while self.current_char() and self.current_char() != '\n':
            value += self.advance()

        return Token(TokenType.COMPILER_ANNOTATION, f"{annotation}{value}".strip(), start_line, start_col)

    def _read_triple_quoted_string(self, quote: str, start_line: int, start_col: int) -> Token:
        """`'''...'''`: a string over several lines, its quotes and line breaks kept as written."""
        for _ in range(3):
            self.advance()
        string = ''
        while self.current_char() is not None and self.source[self.pos:self.pos + 3] != quote * 3:
            char = self.advance()
            if char == '\\' and self.current_char() in ('n', 't', '"', "'", '\\'):
                escaped = self.advance()
                string += {'n': '\n', 't': '\t'}.get(escaped, escaped)
            else:
                string += char
        if self.current_char() is None:
            raise LexerError('Unterminated triple-quoted string', start_line, start_col)
        for _ in range(3):
            self.advance()
        return Token(TokenType.STRING_LITERAL, string, start_line, start_col, string)

    def read_string(self, quote: str) -> Token:
        """Read string literal (single or double quoted)."""
        start_line = self.line
        start_col = self.column
        if self.source[self.pos:self.pos + 3] == quote * 3:
            return self._read_triple_quoted_string(quote, start_line, start_col)

        # Skip opening quote
        self.advance()

        string = ''
        while self.current_char() and self.current_char() != quote:
            char = self.current_char()
            if char == '\\':
                # Handle escape sequences — unescape to actual characters
                self.advance()
                next_char = self.current_char()
                if next_char == 'n':
                    string += '\n'
                    self.advance()
                elif next_char == 't':
                    string += '\t'
                    self.advance()
                elif next_char in ('"', "'", '\\'):
                    string += next_char
                    self.advance()
                elif next_char is not None and next_char != '\n':
                    # any other escaped character is itself, `\r` too: `\S` is `S` (Pine; only `\n` and `\t` are
                    # escape sequences of their own)
                    string += self.advance()
                else:
                    string += char
            elif char == '\n':
                # a string may go on on the next line; the line break is not part of it, and outside
                # parentheses it is not counted: the lines after the string are numbered one
                # less (PineCoders/Time v5; a tooltip inside a call keeps the count)
                self.advance()
                if self.paren_depth == 0:
                    self.line -= 1
                # the line break and the next line's indentation are one space, in Pine v5 and v6 alike, outside
                # parentheses and inside a call (measured on TradingView); spaces before the break stay
                while self.current_char() in (' ', '\t'):
                    self.advance()
                string += ' '
            else:
                string += self.advance()

        if not self.current_char():
            raise LexerError(f"Unterminated string literal", start_line, start_col)

        # Skip closing quote
        self.advance()

        # Mark if double quoted (for preserving quote style in output)
        is_double_quoted = (quote == '"')
        return Token(TokenType.STRING_LITERAL, string, start_line, start_col,
                    f"{quote}{string}{quote}", is_double_quoted=is_double_quoted)

    def read_number(self) -> Token:
        """Read integer or float literal."""
        start_line = self.line
        start_col = self.column

        number = ''
        has_dot = False
        has_e = False

        while self.current_char():
            char = self.current_char()

            if char.isdigit():
                number += self.advance()
            elif char == '.' and not has_dot and not has_e:
                next_ch = self.peek_char()
                if next_ch and next_ch.isdigit():
                    # 100.5 — normal decimal
                    has_dot = True
                    number += self.advance()
                elif next_ch is None or not (next_ch.isalpha() or next_ch == '_' or next_ch == '.'):
                    # 100. — trailing dot float (not member access like 100.toString)
                    has_dot = True
                    number += self.advance()
                else:
                    break
            elif char in ('e', 'E') and not has_e:
                has_e = True
                number += self.advance()
                # Handle optional +/- after e
                if self.current_char() in ('+', '-'):
                    number += self.advance()
                # An exponent marker with no digits after it (1e, 1e+) would reach
                # float() below and leak a raw ValueError past the lexer's error
                # boundary. Reject it here with position info instead.
                if not (self.current_char() or '').isdigit():
                    raise LexerError(
                        f"Malformed exponent in numeric literal: {number!r}",
                        start_line, start_col,
                    )
            else:
                break

        # a digit run followed by a letter or an underscore (`3barsDown`) is neither a number nor an
        # identifier: TradingView stops at the tail (identifiers start with a letter or an underscore)
        following = self.current_char()
        if following and (following.isalpha() or following == '_'):
            tail = ''
            while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
                tail += self.advance()
            raise LexerError(f'Syntax error at input "{tail}": an identifier cannot start with a digit',
                             start_line, start_col)

        if has_dot or has_e:
            return Token(TokenType.FLOAT_LITERAL, float(number), start_line, start_col, number)
        else:
            return Token(TokenType.INT_LITERAL, int(number), start_line, start_col, number)

    def read_color_literal(self) -> Token:
        """Read #RRGGBB or #RRGGBBAA color literal."""
        start_line = self.line
        start_col = self.column

        color = self.advance()  # #

        # Read hex digits
        while self.current_char() and self.current_char() in '0123456789abcdefABCDEF':
            color += self.advance()

        # Validate length (6 or 8 hex digits after #)
        if len(color) not in (7, 9):
            raise LexerError(f"Invalid color literal: {color}", start_line, start_col)

        return Token(TokenType.COLOR_LITERAL, color, start_line, start_col, color)

    def read_identifier(self) -> Token:
        """Read identifier or keyword."""
        start_line = self.line
        start_col = self.column

        identifier = ''
        while self.current_char() and (self.current_char().isalnum() or self.current_char() in ('_', '.')):
            identifier += self.advance()
            if identifier.endswith('.') and self.current_char() in (' ', '\t'):
                # `get. box.get_left`: blanks after a member dot belong to no token
                offset = 0
                while self.peek_char(offset) in (' ', '\t'):
                    offset += 1
                following = self.peek_char(offset)
                if following and (following.isalpha() or following == '_'):
                    for _ in range(offset):
                        self.advance()

        # `type` is a keyword only at the head of a user-defined-type declaration
        if identifier == 'type' and not self._heads_type_declaration():
            return Token(TokenType.IDENTIFIER, identifier, start_line, start_col, identifier)

        # `method` is a keyword only where it heads a method declaration (`method name(`);
        # elsewhere it is a name (`f(string method) =>`, `method == "SMA"`)
        if identifier == 'method' and not self._heads_method_declaration():
            return Token(TokenType.IDENTIFIER, identifier, start_line, start_col, identifier)

        # `in` is a keyword from Pine v5 on (`for x in a`); a v4 script may name a parameter `in`
        if identifier == 'in' and self.pine_version < 5:
            return Token(TokenType.IDENTIFIER, identifier, start_line, start_col, identifier)

        # Check if it's a keyword
        if identifier in KEYWORDS:
            token_type = KEYWORDS[identifier]
            return Token(token_type, identifier, start_line, start_col, identifier)

        # Check if it's a type keyword
        if identifier in TYPE_KEYWORDS:
            return Token(TokenType.TYPE_IDENTIFIER, identifier, start_line, start_col, identifier)

        return Token(TokenType.IDENTIFIER, identifier, start_line, start_col, identifier)

    def _heads_method_declaration(self) -> bool:
        """After the word `method`: a name and then `(` on the same line."""
        offset = 0
        while self.peek_char(offset) in (' ', '\t'):
            offset += 1
        start = offset
        while (self.peek_char(offset) or ' ').isalnum() or self.peek_char(offset) == '_':
            offset += 1
        if offset == start:
            return False
        while self.peek_char(offset) in (' ', '\t'):
            offset += 1
        return self.peek_char(offset) == '('

    def _heads_type_declaration(self) -> bool:
        """True for `type Name`: first on its statement (or after `export`), a name following on the same line.

        Anywhere else `type` is an ordinary name (the Pine v2-v4 `type=` argument of input(), a variable,
        a parameter, a field). The parser accepted a TYPE token only at the head of a declaration, so a
        source that parsed before is not changed by this.
        """
        previous = self.tokens[-1].type if self.tokens else None
        if previous not in (None, TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT, TokenType.EXPORT):
            return False
        offset = 0
        while self.peek_char(offset) in (' ', '\t'):
            offset += 1
        following = self.peek_char(offset)
        if following is None or not (following.isalpha() or following == '_'):
            return False
        name_end = offset
        while (self.peek_char(name_end) or ' ').isalnum() or self.peek_char(name_end) == '_':
            name_end += 1
        following_word = self.source[self.pos + offset:self.pos + name_end]
        return following_word not in ('and', 'or', 'in', 'to', 'by')

    def _line_has_arrow(self) -> bool:
        """Whether the tokens of the line just ended hold a `=>`."""
        for token in reversed(self.tokens):
            if token.type in (TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT):
                return False
            if token.type == TokenType.ARROW:
                return True
        return False

    def _is_continuation_line(self) -> bool:
        """Check if this line is a continuation of the previous expression.

        Pine Script treats indented lines as continuations when:
        1. The line starts with a binary operator keyword (or/and)
        2. The previous line ended with an operator that needs a right operand (?, :, and, or, +, -, etc.)
        """
        # Check if line starts with a continuation keyword
        if self.current_char() and self.current_char().isalpha():
            word = ''
            temp_pos = self.pos
            # the whole identifier: `or_sesh := ...` starts a statement, not an `or` operand
            while temp_pos < len(self.source) and (self.source[temp_pos].isalnum() or self.source[temp_pos] == '_'):
                word += self.source[temp_pos]
                temp_pos += 1
            if word in ('or', 'and'):
                return True
        # A line that starts with the `?` or the `:` of a ternary, or with a comma, continues the one above
        if self.current_char() in ('?', ':', ',') and self.peek_char() != '=':
            return True
        # so does one that starts with a comparison or a multiplication (`== 'Wicks'`), which no
        # statement can start with
        if self.source[self.pos:self.pos + 2] in ('==', '!=', '>=', '<=') or (
                self.current_char() in ('*', '/') and self.peek_char() not in ('/', '=')):
            return True

        # Check if previous line ended with an operator that needs a right operand
        # These operators always indicate the expression continues on the next line
        CONTINUATION_END_TOKENS = {
            TokenType.TERNARY,   # ?
            TokenType.COLON,     # : (ternary false branch)
            TokenType.AND,       # and
            TokenType.OR,        # or
            TokenType.PLUS,      # +
            TokenType.MINUS,     # -
            TokenType.MULT,      # *
            TokenType.DIV,       # /
            TokenType.MOD,       # %
            TokenType.COMMA,     # ,
            # Assignment/comparison operators can never end a complete expression,
            # so a line ending with one always continues onto the next indented
            # line, e.g. `x =` \n `  "a" + "b"` (line-wrapped RHS after `=`).
            TokenType.ASSIGN,       # =
            TokenType.REASSIGN,     # :=
            TokenType.PLUS_ASSIGN,  # +=
            TokenType.MINUS_ASSIGN, # -=
            TokenType.MULT_ASSIGN,  # *=
            TokenType.DIV_ASSIGN,   # /=
            TokenType.MOD_ASSIGN,   # %=
            TokenType.GT,           # >
            TokenType.LT,           # <
            TokenType.GTE,          # >=
            TokenType.LTE,          # <=
            TokenType.EQ,           # ==
            TokenType.NEQ,          # !=
            TokenType.IF,           # `if` with its condition on the next line
            TokenType.WHILE,        # `while` likewise
        }
        if self.tokens:
            last_token = self.tokens[-1]
            if last_token.type in CONTINUATION_END_TOKENS:
                return True

        return False

    def handle_indentation(self, indent_level: int) -> List[Token]:
        """
        Generate INDENT/DEDENT tokens based on indentation change.

        Uses lenient matching: if exact level not in stack, finds closest level.
        This handles real-world Pine Script files with varied indentation styles.
        """
        tokens = []
        current_level = self.indent_stack[-1]

        if indent_level > current_level:
            # Increase indentation
            self.indent_stack.append(indent_level)
            tokens.append(Token(TokenType.INDENT, None, self.line, self.column))
        elif indent_level < current_level:
            # Decrease indentation (may generate multiple DEDENT tokens)
            while self.indent_stack and self.indent_stack[-1] > indent_level:
                self.indent_stack.pop()
                tokens.append(Token(TokenType.DEDENT, None, self.line, self.column))

            # Lenient matching: if exact level not found, find closest match
            if not self.indent_stack:
                # Should never happen, but reset to base
                self.indent_stack.append(0)
            elif self.indent_stack[-1] != indent_level:
                # Find closest level at or below target
                # This handles cases where continuation lines use different indentation
                closest_level = max(level for level in self.indent_stack if level <= indent_level)

                # If we found a closer match, pop until we reach it
                while self.indent_stack and self.indent_stack[-1] > closest_level:
                    self.indent_stack.pop()
                    tokens.append(Token(TokenType.DEDENT, None, self.line, self.column))

                # If we need to add the new level
                if closest_level < indent_level:
                    self.indent_stack.append(indent_level)
                    tokens.append(Token(TokenType.INDENT, None, self.line, self.column))

        return tokens

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source code."""
        self.tokens = []
        at_line_start = True

        while self.pos < len(self.source):
            char = self.current_char()

            # Handle newlines and indentation
            if char == '\n':
                # Skip empty lines and lines with only comments
                saved_pos = self.pos
                saved_line = self.line
                saved_col = self.column

                self.advance()  # Skip \n

                # Count indentation on next line
                indent_level = 0
                while self.current_char() in (' ', '\t'):
                    if self.current_char() == ' ':
                        indent_level += 1
                    else:  # tab
                        indent_level += 4
                    self.advance()

                # Check if line is empty or comment-only
                if self.current_char() in ('\n', None) or (self.current_char() == '/' and self.peek_char() == '/'):
                    # Empty line or comment line - skip indentation tracking
                    if self.current_char() is None:
                        break
                    continue

                # Check for line continuation: increased indent + starts with binary operator
                # In Pine Script, indented lines starting with or/and continue the previous expression
                # Pine also wraps at the block's own indentation, or below it, when that is not a multiple
                # of 4 (`f(x) =>` / `      a ? b :` / `      c`; a 4-space body wrapped at 3 spaces)
                wraps_in_place = (indent_level <= self.indent_stack[-1] and indent_level % 4 != 0
                                  and self.tokens and self.tokens[-1].type in (
                                      TokenType.TERNARY, TokenType.COLON, TokenType.AND, TokenType.OR,
                                      TokenType.PLUS, TokenType.MINUS, TokenType.MULT, TokenType.DIV))
                # a statement cannot end with a comma: `var a = 1,` / `var b = 2` is one line; a switch arm
                # ending in a comma (`'Dashed' => line.style_dashed ,`) ends there
                wraps_in_place = wraps_in_place or (indent_level == self.indent_stack[-1]
                                                    and self.tokens and self.tokens[-1].type == TokenType.COMMA
                                                    and not self._line_has_arrow())
                # no statement starts with a comma: one at an indent that is no multiple of 4, below the block's
                # too, continues the line above (`x := true` / `   , y += 1`)
                wraps_in_place = wraps_in_place or (indent_level % 4 != 0 and self.current_char() == ','
                                                    and self.tokens and self.tokens[-1].type != TokenType.NEWLINE)
                # the argument list of a call wrapped below its name (`tb.merge_cells` / `       (0, 0, 5, 0)`)
                calls_above = (indent_level > self.indent_stack[-1] and indent_level % 4 != 0
                               and self.current_char() == '(' and self.tokens
                               and self.tokens[-1].type == TokenType.IDENTIFIER)
                if (self.paren_depth == 0
                     and (indent_level > self.indent_stack[-1] or wraps_in_place)
                     and (calls_above or self._is_continuation_line())):
                    # Suppress NEWLINE and INDENT — treat as seamless continuation
                    at_line_start = False
                    continue

                # Inside parentheses/brackets/braces: suppress NEWLINE and indentation
                # Multi-line expressions inside delimiters are seamless
                if self.paren_depth > 0:
                    at_line_start = False
                    continue

                # Emit NEWLINE token for previous line
                self.tokens.append(Token(TokenType.NEWLINE, '\n', saved_line, saved_col))

                # Handle indentation changes
                if self.paren_depth == 0:
                    indent_tokens = self.handle_indentation(indent_level)
                    self.tokens.extend(indent_tokens)

                at_line_start = False
                continue

            # Skip inline whitespace
            if char in (' ', '\t'):
                self.skip_whitespace()
                continue

            # Comments and annotations
            if char == '/' and self.peek_char() == '/':
                token = self.read_line_comment()
                # Don't add comments to token stream (or add them with a flag)
                # For now, skip comments except annotations
                # a second `//@version` (a script pasted below another) is a comment
                if token.type == TokenType.COMPILER_ANNOTATION or (
                        token.type == TokenType.VERSION_ANNOTATION
                        and not any(seen.type == TokenType.VERSION_ANNOTATION for seen in self.tokens)):
                    self.tokens.append(token)
                continue

            # String literals
            if char in ('"', "'"):
                self.tokens.append(self.read_string(char))
                continue

            # Color literals
            if char == '#':
                self.tokens.append(self.read_color_literal())
                continue

            # Numbers
            if char.isdigit():
                self.tokens.append(self.read_number())
                continue

            # Identifiers and keywords
            if char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
                continue

            # Operators and delimiters
            start_line = self.line
            start_col = self.column

            # Two-character operators
            if char == ':' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.REASSIGN, ':=', start_line, start_col, ':='))
                continue

            # Compound assignment operators
            if char == '+' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.PLUS_ASSIGN, '+=', start_line, start_col, '+='))
                continue

            if char == '-' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.MINUS_ASSIGN, '-=', start_line, start_col, '-='))
                continue

            if char == '*' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.MULT_ASSIGN, '*=', start_line, start_col, '*='))
                continue

            if char == '/' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.DIV_ASSIGN, '/=', start_line, start_col, '/='))
                continue

            if char == '%' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.MOD_ASSIGN, '%=', start_line, start_col, '%='))
                continue

            if char == '=' and self.peek_char() == '>':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ARROW, '=>', start_line, start_col, '=>'))
                continue

            if char == '=' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.EQ, '==', start_line, start_col, '=='))
                continue

            if char == '!' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.NEQ, '!=', start_line, start_col, '!='))
                continue

            if char == '>' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.GTE, '>=', start_line, start_col, '>='))
                continue

            if char == '<' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.LTE, '<=', start_line, start_col, '<='))
                continue

            # Single-character operators and delimiters
            single_char_tokens = {
                '=': TokenType.ASSIGN,
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULT,
                '/': TokenType.DIV,
                '%': TokenType.MOD,
                '>': TokenType.GT,
                '<': TokenType.LT,
                '?': TokenType.TERNARY,
                ':': TokenType.COLON,
                '.': TokenType.DOT,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                ',': TokenType.COMMA,
            }

            if char in single_char_tokens:
                token_type = single_char_tokens[char]

                # Track delimiter depth for indentation-free zones
                if char in ('(', '[', '{'):
                    self.paren_depth += 1
                elif char in (')', ']', '}'):
                    self.paren_depth = max(0, self.paren_depth - 1)

                self.advance()
                self.tokens.append(Token(token_type, char, start_line, start_col, char))
                continue

            # Unknown character
            raise LexerError(f"Unexpected character: {char!r}", self.line, self.column)

        # Emit final DEDENT tokens to return to base indentation
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, None, self.line, self.column))

        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))

        return _joined_dotted_names(self.tokens)


def _joined_dotted_names(tokens: List[Token]) -> List[Token]:
    """`fib .new(...)`, `map  .new<K, V>()`: a name (or a type namespace), a dot and a name on one line are one
    dotted name, as when written without spaces."""
    joined: List[Token] = []
    for token in tokens:
        if (token.type == TokenType.IDENTIFIER and len(joined) >= 2 and joined[-1].type == TokenType.DOT
                and joined[-2].type in (TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER)
                and joined[-2].line == joined[-1].line == token.line):
            # a type namespace too (`map  .new<K, V>()`)
            joined.pop()
            head = joined.pop()
            token = dataclasses.replace(head, type=TokenType.IDENTIFIER, value=f'{head.value}.{token.value}')
        joined.append(token)
    return joined


def tokenize(source: str) -> List[Token]:
    """Convenience function to tokenize Pine Script source."""
    lexer = Lexer(source)
    return lexer.tokenize()
