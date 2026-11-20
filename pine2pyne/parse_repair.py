"""Parse-failure recovery.

The lexer and the grammar are unchanged. Only when the lexer or the parser
REJECTS a source does this module look at what was rejected. If it is one of
the spellings of valid Pine below, it is rewritten into the spelling the grammar
already accepts, and the source is parsed again. A source that parses today
never reaches a repair, so its output cannot change.

  row 1  `\\xa0` used as whitespace         -> a space (outside strings and comments)
  CRLF   `\\r\\n` / `\\r` line ends in a str -> `\\n`, as reading the file in text mode does
  row 2  unary `+x`                        -> `x`
  row 3  `.99`                             -> `0.99`
  row 4  `a = 1, b = 2` at the top level   -> two statements, only when a statement
                                              (assignment, declaration or call) follows the comma
  row 5  a wrapped line led by an operator, on an indent no block claims -> joined
  row 8  `0 => k := k + 1` in an expression switch -> the multi-line arm (any assignment
         right after `=>` the parser rejects, e.g. a one-line function body, gets the
         same multi-line form)

Rows 2, 3 and 4 are repaired at every site of the kind in one pass (G6): no site
they match can parse as written, so rewriting one the parser has not reached yet
changes nothing it would have accepted. Rows 5 and 8 depend on which INDENT a
block claims, which only the parser knows, so they are repaired one site per parse.
Each parse costs one pass of the file's budget; when a chain dead-ends or the
budget runs out, the FIRST error is raised, with the message and line:col the
unrepaired source gave, and the error the chain ended on as its __cause__.

UNPROVEN on TradingView, pinned by the parse-repair tests: `\\xa0` as
whitespace; comma statements; `var p = 0, q = 0` makes only `p` a var here; a
statement led by `+` on the indent grid stays refused; unary `+` on a non-number
(`+"a"`) is accepted; an operator-led wrap indented by a multiple of 4 is joined, as
the lexer already joins a trailing-operator wrap on any indent.
"""
from typing import List, Optional, Set, Tuple

from .ast_nodes import Script
from .errors import LexerError, ParserError
from .lexer import Lexer
from .parser import Parser
from .tokens import Token, TokenType

MINIMUM_REPAIR_PASSES = 200
NO_BREAK_SPACE = '\xa0'
CARRIAGE_RETURN = '\r'

OPERAND_END_TYPES = {
    TokenType.INT_LITERAL, TokenType.FLOAT_LITERAL, TokenType.STRING_LITERAL,
    TokenType.BOOL_LITERAL, TokenType.COLOR_LITERAL, TokenType.NA_LITERAL,
    TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER, TokenType.TRUE, TokenType.FALSE,
    TokenType.RPAREN, TokenType.RBRACKET,
}
LINE_START_TYPES = {TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT}
LINE_END_TYPES = LINE_START_TYPES | {TokenType.EOF}
OPENING_TYPES = {TokenType.LPAREN, TokenType.LBRACKET}
CLOSING_TYPES = {TokenType.RPAREN, TokenType.RBRACKET}
CONTINUATION_LEAD_TYPES = {
    TokenType.PLUS, TokenType.MINUS, TokenType.MULT, TokenType.DIV, TokenType.MOD,
    TokenType.TERNARY, TokenType.COLON, TokenType.AND, TokenType.OR,
    TokenType.GT, TokenType.LT, TokenType.GTE, TokenType.LTE, TokenType.EQ, TokenType.NEQ,
    TokenType.ASSIGN, TokenType.REASSIGN,
}
ASSIGNMENT_TYPES = {
    TokenType.ASSIGN, TokenType.REASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
    TokenType.MULT_ASSIGN, TokenType.DIV_ASSIGN, TokenType.MOD_ASSIGN,
}


def parse_source(source: str) -> Tuple[Script, List[str]]:
    """Lex and parse `source`; return (Script, repairs), one 'kind@line:column' per repair pass."""
    repairs: List[str] = []
    tokens, first_lexer_error = _tokenize(source, repairs)
    try:
        return parse_tokens(tokens, repairs), repairs
    except ParserError as error:
        if first_lexer_error is None:
            raise
        raise first_lexer_error from error          # the unrepaired source's first error


def parse_tokens(tokens: List[Token], repairs: Optional[List[str]] = None) -> Script:
    """Parse `tokens`, repairing each rejected known spelling; append to `repairs` per pass."""
    repairs = [] if repairs is None else repairs
    first_error, budget, passes = None, None, 0
    while True:
        parser = Parser(tokens)
        try:
            return parser.parse()
        except ParserError as error:
            if first_error is None:                  # sized only once a source is refused
                first_error, budget = error, repair_pass_budget(parser.tokens)
            repaired = _repair(parser.tokens, error)
            if repaired is None:
                raise first_error from (None if error is first_error else error)
            if passes == budget:
                raise _budget_refusal(first_error, budget) from error
            tokens, kind = repaired
            repairs.append(f'{kind}@{error.line}:{error.column}')
            passes += 1


def repair_pass_budget(tokens: List[Token]) -> int:
    """Parses a source may spend on repairs, sized to its candidate sites (G6).

    Each row 5 or row 8 repair fixes one candidate per parse, and each may expose new
    sites for every batched kind, so a source with n candidates needs at most
    n * (1 + len(BATCHED_REPAIRS)) passes beyond the batched kinds' own.
    """
    context = _LineContext(tokens)
    wraps = sum(1 for index in range(1, len(tokens) - 1)
                if tokens[index].type == TokenType.INDENT and tokens[index - 1].type == TokenType.NEWLINE
                and tokens[index + 1].type in CONTINUATION_LEAD_TYPES)
    arm_assignments = sum(1 for index, token in enumerate(tokens)
                          if token.type in ASSIGNMENT_TYPES and context.has_arrow(index))
    return MINIMUM_REPAIR_PASSES + (wraps + arm_assignments) * (1 + len(BATCHED_REPAIRS))


def _budget_refusal(first_error: ParserError, budget: int) -> ParserError:
    """The first error, naming the repair budget the source ran out of."""
    refusal = ParserError(f'{first_error.message} (still refused after {budget} repair passes, '
                          f'the budget for this file)', first_error.line, first_error.column)
    refusal.token_index = getattr(first_error, 'token_index', None)
    return refusal


def _tokenize(source: str, repairs: List[str]) -> Tuple[List[Token], Optional[LexerError]]:
    """Lex `source`; on a no-break space or a carriage return, rewrite it once and lex again.

    Returns the tokens and the first lexer error a rewrite got past (None if none did).
    """
    rewrites = {NO_BREAK_SPACE: ('no_break_space', _no_break_space_outside_strings),
                CARRIAGE_RETURN: ('carriage_return', _universal_newlines)}
    first_error = None
    while True:
        try:
            return Lexer(source).tokenize(), first_error
        except LexerError as error:
            first_error = first_error or error
            character = _rejected_character(error)
            if character not in rewrites:
                raise first_error from (None if error is first_error else error)
            kind, rewrite = rewrites.pop(character)
            repairs.append(f'{kind}@{error.line}:{error.column}')
            source = rewrite(source)


def _rejected_character(error: LexerError) -> Optional[str]:
    """The character of an `Unexpected character: <repr>` lexer error, else None."""
    for character in (NO_BREAK_SPACE, CARRIAGE_RETURN):
        if error.message == f'Unexpected character: {character!r}':
            return character
    return None


def _universal_newlines(source: str) -> str:
    """`\\r\\n` and a lone `\\r` become `\\n`, as `Path.read_text` does for the CLI."""
    return source.replace('\r\n', '\n').replace('\r', '\n')


def _no_break_space_outside_strings(source: str) -> str:
    """Every `\\xa0` outside a string literal or a `//` comment becomes a space."""
    characters, quote, index = list(source), None, 0
    while index < len(characters):
        character = characters[index]
        if quote:
            if character == '\\':
                index += 1
            elif character == quote:  # the lexer lets a string span lines
                quote = None
        elif character in ('"', "'"):
            quote = character
        elif character == '/' and source[index:index + 2] == '//':
            index = source.find('\n', index)
            index = len(source) if index < 0 else index
            continue
        elif character == NO_BREAK_SPACE:
            characters[index] = ' '
        index += 1
    return ''.join(characters)


def _repair(tokens: List[Token], error: ParserError) -> Optional[Tuple[List[Token], str]]:
    """(rewritten tokens, kind) for the known spelling at the error's token, else None."""
    index = getattr(error, 'token_index', None)
    if index is None or not 0 <= index < len(tokens):
        return None
    context = _LineContext(tokens)
    for kind, is_site, rewrite_site in BATCHED_REPAIRS:
        if is_site(tokens, index, context):
            sites = {position for position in range(len(tokens)) if is_site(tokens, position, context)}
            return _rewrite_sites(tokens, sites, rewrite_site), kind
    for repair in (_operator_led_wrap, _switch_arm_statement):
        result = repair(tokens, index)
        if result is not None:
            return result, repair.__name__.lstrip('_')
    return None


def _opens_type_arguments(token: Token) -> bool:
    """A name whose `<` opens type arguments: `map<`, `array<`, `matrix<`, `map.new<`, `T.new<`."""
    name = token.value if isinstance(token.value, str) else ''
    return token.type in (TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER) and (
        name in ('map', 'array', 'matrix') or name.endswith('.new'))


class _LineContext:
    """Per token: the bracket depth within its line, the block nesting, and whether its line has `=>`."""

    def __init__(self, tokens: List[Token]):
        self.depth: List[int] = []
        self.nesting: List[int] = []
        self.line: List[int] = []
        self.lines_with_arrow: Set[int] = set()
        #: per token, how deep it stands in a type's angle brackets (`map.new<string, float>`)
        self.generic: List[int] = []
        depth = nesting = line = generic = 0
        for position, token in enumerate(tokens):
            if token.type in LINE_START_TYPES:
                depth, line, generic = 0, line + 1, 0
                nesting += (token.type == TokenType.INDENT) - (token.type == TokenType.DEDENT)
            if token.type == TokenType.LT and position and _opens_type_arguments(tokens[position - 1]):
                generic += 1
            self.depth.append(depth)
            self.nesting.append(nesting)
            self.line.append(line)
            self.generic.append(generic)
            if token.type == TokenType.GT and generic:
                generic -= 1
            depth += (token.type in OPENING_TYPES) - (token.type in CLOSING_TYPES)
            if token.type == TokenType.ARROW:
                self.lines_with_arrow.add(line)

    def has_arrow(self, index: int) -> bool:
        return self.line[index] in self.lines_with_arrow


def _rewrite_sites(tokens: List[Token], sites: Set[int], rewrite_site) -> List[Token]:
    """A new token list with `rewrite_site(tokens, index)` in place of each site's token."""
    output: List[Token] = []
    index = 0
    while index < len(tokens):
        if index in sites:
            replacement, consumed = rewrite_site(tokens, index)
            output.extend(replacement)
            index += consumed
        else:
            output.append(tokens[index])
            index += 1
    return output


def _is_unary_plus(tokens: List[Token], index: int, context: _LineContext) -> bool:
    """Row 2: a `+` with no left operand. At a line start only on a `=>` line (`+1 => a`) or inside a block,
    where a line `+ 1` on the block's grid is a statement of its own; at the script's top level Pine refuses it."""
    if tokens[index].type != TokenType.PLUS or index == 0:
        return False
    previous = tokens[index - 1].type
    if previous in OPERAND_END_TYPES:
        return False                                 # a binary `+`, e.g. `(a) + +b`
    return previous not in LINE_START_TYPES or context.has_arrow(index) or tokens[index].column > 1


def _drop_token(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
    return [], 1


def _is_leading_dot_number(tokens: List[Token], index: int, context: _LineContext) -> bool:
    """Row 3: `.` written right against the digits of a number (`. 5` stays refused)."""
    dot = tokens[index]
    number = tokens[index + 1] if index + 1 < len(tokens) else None
    return (dot.type == TokenType.DOT and number is not None
            and number.type in (TokenType.INT_LITERAL, TokenType.FLOAT_LITERAL)
            and (number.line, number.column) == (dot.line, dot.column + 1)
            and (number.lexeme or '')[:1].isdigit() and '.' not in number.lexeme)


def _merge_leading_dot_number(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
    dot, number = tokens[index], tokens[index + 1]
    lexeme = '0.' + number.lexeme
    return [Token(TokenType.FLOAT_LITERAL, float(lexeme), dot.line, dot.column, lexeme)], 2


#: keywords that start a block statement
BLOCK_KEYWORD_TYPES = frozenset({TokenType.IF, TokenType.FOR, TokenType.WHILE, TokenType.SWITCH})


def _is_comma_statement(tokens: List[Token], index: int, context: _LineContext) -> bool:
    """Row 4: a comma between two top-level statements, outside brackets and `=>` lines (G3).

    Both sides must be a statement, so the comma of `map<string, float> m = ...` (bracket
    depth 0, but inside `<...>`) is never a site: `map<string` is not a statement; nor is
    a comma at a line start or after another comma, whose item before it is empty. A
    comma ending its line joins the statement that starts the next line at the same indent,
    a block statement (`if`, `for`, ...) too.
    """
    if (tokens[index].type != TokenType.COMMA or context.depth[index] != 0
            or context.nesting[index] != 0 or context.has_arrow(index)):
        return False
    after = _item_after_comma(tokens, index)
    if not _is_statement_item(tokens, _item_start(tokens, index, context), index):
        return False
    if after < len(tokens) and tokens[after].type in BLOCK_KEYWORD_TYPES and tokens[after].line > tokens[index].line:
        return True  # `a = 1,` ending its line before an `if` / `for` / `while` / `switch`: the comma ends `a = 1`
    return _is_statement_item(tokens, after, _item_end(tokens, after))


def _item_after_comma(tokens: List[Token], comma: int) -> int:
    """First token of the item after `comma`; for `a = 1,` ending its line, the next line's first."""
    position = comma + 1
    while position < len(tokens) and tokens[position].type == TokenType.NEWLINE:
        position += 1
    return position


def _item_start(tokens: List[Token], comma: int, context: _LineContext) -> int:
    """First token of the item before `comma`: after the line start or the previous top-level comma."""
    position = comma - 1
    while position >= 0 and tokens[position].type not in LINE_START_TYPES and not (
            tokens[position].type == TokenType.COMMA and context.depth[position] == 0
            and context.generic[position] == 0):
        position -= 1
    return position + 1


def _item_end(tokens: List[Token], start: int) -> int:
    """Index after the item from `start`: the next top-level comma or the line end."""
    depth, position = 0, start
    while position < len(tokens):
        kind = tokens[position].type
        if kind in LINE_END_TYPES or (kind == TokenType.COMMA and depth == 0):
            break
        depth += (kind in OPENING_TYPES) - (kind in CLOSING_TYPES)
        position += 1
    return position


def _is_statement_item(tokens: List[Token], start: int, end: int) -> bool:
    """tokens[start:end] is an assignment or declaration (`=` at its top level), or a call."""
    depth = 0
    for position in range(start, end):
        kind = tokens[position].type
        if depth == 0 and kind in ASSIGNMENT_TYPES:
            return True                              # `b = 2`, `var float b = 2`, `[p, q] = f()`
        depth += (kind in OPENING_TYPES) - (kind in CLOSING_TYPES)
    return _is_call_item(tokens, start, end)


def _is_call_item(tokens: List[Token], start: int, end: int) -> bool:
    """tokens[start:end] is exactly `name(...)`, `name.method(...)` or `name .method(...)`."""
    if start >= end or tokens[start].type != TokenType.IDENTIFIER:
        return False
    paren = start + 1
    while paren + 1 < end and tokens[paren].type == TokenType.DOT and tokens[paren + 1].type == TokenType.IDENTIFIER:
        paren += 2                                   # `pivots .append(...)`: a spaced member call
    if paren >= end or tokens[paren].type != TokenType.LPAREN:
        return False
    depth = 0
    for position in range(paren, end):
        depth += (tokens[position].type in OPENING_TYPES) - (tokens[position].type in CLOSING_TYPES)
        if depth == 0:
            return position == end - 1
    return False


def _comma_to_newline(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
    comma = tokens[index]
    return [Token(TokenType.NEWLINE, '\n', comma.line, comma.column)], 1


BATCHED_REPAIRS = (
    ('unary_plus', _is_unary_plus, _drop_token),
    ('leading_dot_number', _is_leading_dot_number, _merge_leading_dot_number),
    ('comma_statement', _is_comma_statement, _comma_to_newline),
)


def _operator_led_wrap(tokens: List[Token], index: int) -> Optional[List[Token]]:
    """Row 5: an INDENT no block claimed, before an operator: join the wrapped lines."""
    if tokens[index].type == TokenType.NEWLINE and index + 1 < len(tokens):
        index += 1                                   # `Expected COLON, got NEWLINE` names the NEWLINE
    if not (tokens[index].type == TokenType.INDENT and index >= 1
            and tokens[index - 1].type == TokenType.NEWLINE and index + 1 < len(tokens)
            and tokens[index + 1].type in CONTINUATION_LEAD_TYPES):
        return None
    output = tokens[:index - 1]
    position = index + 1
    while position < len(tokens):
        token = tokens[position]
        if token.type == TokenType.INDENT:
            return None                              # a nested indent under a wrap: not handled
        if token.type == TokenType.DEDENT:
            return output + tokens[position + 1:]    # the wrap closes at EOF: drop its DEDENT
        if token.type == TokenType.NEWLINE:
            following = tokens[position + 1] if position + 1 < len(tokens) else None
            if following is not None and following.type in CONTINUATION_LEAD_TYPES:
                position += 1                        # another wrapped line at the same indent
                continue
            if following is not None and following.type == TokenType.DEDENT:
                return output + [token] + tokens[position + 2:]   # drop the matching DEDENT
            return None                              # a non-operator line at the wrap indent
        output.append(token)
        position += 1
    return None


def _switch_arm_statement(tokens: List[Token], index: int) -> Optional[List[Token]]:
    """Row 8: an assignment on a `=>` arm line becomes the arm's indented body."""
    if tokens[index].type not in ASSIGNMENT_TYPES:
        return None
    arrow = index - 1
    while arrow >= 0 and tokens[arrow].type not in LINE_START_TYPES | {TokenType.ARROW}:
        arrow -= 1
    if arrow < 0 or tokens[arrow].type != TokenType.ARROW:
        return None
    end = index
    while end < len(tokens) and tokens[end].type not in (TokenType.NEWLINE, TokenType.EOF):
        end += 1
    near = tokens[arrow]
    newline, indent, dedent = (Token(kind, '\n' if kind == TokenType.NEWLINE else None, near.line, near.column)
                               for kind in (TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT))
    line_end = tokens[end:end + 1] if tokens[end].type == TokenType.NEWLINE else [newline]
    rest = tokens[end + 1:] if tokens[end].type == TokenType.NEWLINE else tokens[end:]
    return tokens[:arrow + 1] + [newline, indent] + tokens[arrow + 1:end] + line_end + [dedent] + rest
