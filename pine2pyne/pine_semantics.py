"""
Pine semantics: the checks TradingView's compiler makes that the grammar does not.

One read-only pass over the parsed Script, run by transpile() and by
`--validate` between the parser and the transformer. It keeps a Pine scope
model - the global scope, one scope per function (its parameters and body) and
one per if/for/while/switch body - and collects diagnostics at a Pine line and
1-based column: an identifier or call where one is at fault, else the
statement.

- A refusal is a program TradingView refuses whose transpile would give a
  silent wrong value, drop a statement or fail at a Python line; enforce()
  raises SemanticError for the first one and counts the rest.
- A warning is a program TradingView refuses that working files depend on:
  a same-scope redeclaration, a read before the
  declaration (from a function or block too) or in its own declaration.
  enforce() reports each once per (line, column, name, code) as a
  SemanticWarning, at its statement, when the script has no refusal; the
  output is unchanged. Also a warning, though TradingView compiles it: a
  `ta.*` call that runs on only some bars, whose history differs from the
  every-bar call (conditional_history_calls.py). Never hoisted.

The pass never rewrites the tree, so a script it accepts transpiles exactly as
before. An assignment to a history reference (`x[1] := v`) is refused by the
parser instead: the tree keeps no trace of it.

This scope model is a second one beside scope_renamer's. scope_decisions()
exposes how it resolves each read, and a test compares that with the
renamer's resolution, so a drift between the two fails.
"""
import dataclasses
import re
import itertools
import typing
import warnings
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from .ast_nodes import BreakStatement, ContinueStatement, LibraryDecl
from .ast_nodes import (LibraryDecl, StrategyDecl, IndicatorDecl, 
    ArrayLiteral, Assignment, BinaryOp, EnumDecl, Expression, ExpressionStatement, ForInLoop, ForLoop, FuncDecl,
    FunctionCall, Identifier, IfExpression, IfStatement, IndexAccess, InputDecl, Literal, MemberAccess,
    MethodCall, NaLiteral, Reassignment, ReturnStatement, Script, Statement, SwitchExpression,
    SwitchStatement, TernaryOp, TupleDestructure, TypeDecl, UnaryOp, VarDecl, VaripDecl, WhileLoop,
)
from .ast_walk import walk
from .block_value_lowering import value_tails
from .bool_builtins import BOOL_BUILTINS
from .conditional_history_calls import conditional_history_calls
from .errors import SemanticError, SemanticWarning
from .library_collision import (
    FieldReadRule, HeadBinding, declared_type, global_binding_types, input_type, script_type_evidence,
)
from .namespace_members import NAMESPACE_MEMBERS
from .pine_builtins import FUNCTION_RENAMES, MODULE_RENAMES
from .pine_builtins import PYNECORE_LIB_MODULES, SERIES_FUNCTIONS, SERIES_VARIABLES, input_positional_param_names
from .reference_signatures import PARAMETER_QUALIFIERS, UNIQUE_CONSTANTS
from .symbol_table import SymbolTable
from .udt_field_defaults import constant_heads, default_message, invalid_defaults, positioned_part

REFUSE, WARN = 'refuse', 'warn'
# Before Pine v3 a declaration may read its own name (v2 `x = nz(x[1]) + 1`)
LAST_SELF_REFERENCE_VERSION = 2
# Pine v5 reserved these words (migration guide to v5): a variable, function or parameter cannot bear one.
# MEASURED on TradingView 2026-11-14: v5 `text = "a"`, `class = 5.0` and `do = request.security(...)` fail
# (`'text' cannot be used as a variable or function name.`), v6 `text = "a"`, `class = 5.0` and `range = high - low`
# fail with CE10150; v4 `text = "a"` and `range = high - low` compile. An element of a tuple declaration may bear
# one: v6 `[a, range] = f(close)` and `[a, text] = f(close)` compile
RESERVED_NAMES_FROM_VERSION = 5
RESERVED_NAMES = frozenset('catch class do ellipse in is polygon range return struct text throw try'.split())
# Pine v5 refuses a strategy.exit() that sets no exit level (migration guide to v5): MEASURED on TradingView
# 2026-11-14, v5 `strategy.exit("x", "L")` fails with the message below, v4 compiles it (and it closes nothing)
STRATEGY_EXIT_EFFECT_FROM_VERSION = 5
STRATEGY_EXIT_PARAMETERS = ('id', 'from_entry', 'qty', 'qty_percent', 'profit', 'limit', 'loss', 'stop', 'trail_price',
                            'trail_points', 'trail_offset')
STRATEGY_EXIT_EFFECTS = frozenset({'profit', 'limit', 'loss', 'stop', 'trail_price', 'trail_points', 'trail_offset'})
STRATEGY_EXIT_MESSAGE = ("strategy.exit must have at least one of the following parameters: 'profit', 'limit', 'loss', "
                         "'stop' or one of the following pairs: 'trail_offset' and 'trail_price' / 'trail_points'. "
                         "To close the position at market price, use 'strategy.close'")
DISCARD = '_'
# `_` discards a value: MEASURED on TradingView 2026-11-14, reading it fails in Pine v4 too (`_ = close * 2`,
# `plot(_)`: `Undeclared identifier '<_1>'`), as in v5 and v6
DISCARD_FROM_VERSION = 4
# Pine v6 removed the history of a field of a user-defined type, `t.x[1]` (migration guide to v6, "History of
# UDT fields"); v5 accepts it
FIELD_HISTORY_REFUSED_FROM_VERSION = 6
# TradingView refuses a history read of a loop variable: MEASURED 2026-10-06 for a Pine v6 `for i = a to b`
# counter (VN301! 5m): `Error at 5:31: Variable "i" doesn't have history values (CE10078)`, 5:31 being the counter
# in `nz(i[1], -1)`; MEASURED 2026-11-13 (BINANCE:BTCUSDT 1D) for Pine v4 (`Variable 'i' doesn't have history
# values.`) and v5, and for both variables of a v5 `for [i, v] in arr` and the variable of `for v in arr`. Pine
# v1-v3 are not measured
LOOP_COUNTER_HISTORY_REFUSED_FROM_VERSION = 4
# MEASURED on TradingView 2026-11-13 (Pine v5): a tuple declaration whose right side has another number of values
# is a syntax error, a function's tuple or a `request.security` tuple alike: `[a, b] = f()` with `f()` =>
# `[close, open, high]` gives `Syntax error: The quantities of tuple elements on each side of the assignment
# operator do not match. The right side has 3 but the left side has 2.`, and `[a, b, c] = f()` with a pair gives
# `... The right side has 2 but the left side has 3.`
# a `switch` is a value only as the whole right side of `=`, `:=` or a compound assignment (`s += switch ...`), never as
# an operand of an operator (`1 + switch k`, `c and switch k`): TradingView refuses those with CE10150
SWITCH_OPERAND_MESSAGE = '"switch" cannot be used as a variable or function name. (CE10150)'
# Pine v4-v6 refuse an untyped declaration whose value is the na literal, `x = na`
# and `var x = na` alike: the variable has no type to take from it
UNTYPED_NA_MESSAGE = 'Value with NA type cannot be assigned to a variable that was defined without type keyword'
# Pine v5 and v6 refuse a field or a method read straight off a history index,
# `t[1].v` or `a[1].size()`; `(t[1]).v` compiles. Pine v4 has neither (not measured)
MEMBER_OF_HISTORY_MESSAGE = "User variable identifiers should not contain '.'"
MEMBER_OF_HISTORY_REFUSED_FROM_VERSION = 5
NA_BLOCK_OF_A_BOOL_MESSAGE = ('Return type of one of the "if" or "switch" blocks is not compatible with return '
                              'type of other block(s) (series bool; simple na)')
UNTYPED_NA_REFUSED_FROM_VERSION = 4
#: the version whose `array.every`/`array.some` take an array of bool only (measured on v6; earlier
#: versions read a number as a truth value, so the rule does not reach back)
BOOL_ARRAY_PREDICATES_FROM_VERSION = 6
TUPLE_WIDTH_MESSAGE = ('Syntax error: The quantities of tuple elements on each side of the assignment operator do '
                       'not match. The right side has {right} but the left side has {left}.')
SECURITY_FUNCTIONS = frozenset({'request.security', 'security'})
# MEASURED on TradingView 2026-11-13 (Pine v5): an int counter cannot take a fractional step, `for i = 0 to 3 by
# 0.5` gives `The increment/decrement value type ('literal float') of the 'for' loop is not compatible with counter
# 'i' type ('series int')` (a `const float` variable as the step is refused too, its message not read)
# MEASURED on TradingView 2026-11-13 (Pine v5): `b = input.source(a, "B")` with `a = input.source(close, "A")` does
# not compile; the default of input.source is one of the chart's own sources. MEASURED on TradingView 2026-11-14
# (Pine v4): `input(a, "B", type=input.source)` with another input, a user variable, `sma(close, 3)` or `vwap` does
# not compile either ("Arguments of input function must be of constant type, or 'source' builtin variables");
# `ohlc4` does
INPUT_SOURCE_CHECKED_FROM_VERSION = 4
INPUT_SOURCE_V4_MESSAGE = "Syntax error: Arguments of input function must be of constant type, or 'source' builtin variables."
INPUT_SOURCE_VALUES = ('open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4', 'hlcc4', 'volume')
INPUT_SOURCE_MESSAGE = ("Invalid value for the 'defval' parameter of the 'input.source' function. Possible values: "
                        "[" + ', '.join(INPUT_SOURCE_VALUES) + "].")
#: named-constant parameters whose value TradingView checks against its own namespace: a constant of another
#: namespace is refused at compile time and the message lists the ones the parameter takes. MEASURED on
#: TradingView 2026-11-16 (BINANCE:BTCUSDT 1D, Pine v6): `plot(close, style = scale.left)` and
#: `hline(0.0, linestyle = plot.style_area)` are both CE10068 with the lists written out here. Only a parameter
#: whose list was measured is checked, and only a dotted constant of a built-in namespace is refused -- a
#: variable or a field could hold a legal value, and guessing there would refuse a script TradingView compiles.
ENUM_ARGUMENTS = {
    ('plot', 'style'): ('plot.style_line', 'plot.style_linebr', 'plot.style_histogram', 'plot.style_cross',
                        'plot.style_area', 'plot.style_areabr', 'plot.style_columns', 'plot.style_circles',
                        'plot.style_stepline', 'plot.style_stepline_diamond', 'plot.style_steplinebr'),
    ('hline', 'linestyle'): ('hline.style_solid', 'hline.style_dotted', 'hline.style_dashed'),
    # MEASURED 2026-11-17 (BINANCE:BTCUSDT 1D, Pine v6), each from the CE10068 its own snippet drew
    ('plot', 'display'): ('display.pane', 'display.data_window', 'display.status_line', 'display.price_scale',
                          'display.pine_screener', 'display.none', 'display.all'),
    ('indicator', 'scale'): ('scale.right', 'scale.left', 'scale.none'),
    # MEASURED 2026-11-17: strategy() answers with the same three values. The neighbouring `format`
    # parameter is not of this kind at all: format.price and friends are plain const strings whose value
    # is the word itself, so a string literal is legal there.
    ('strategy', 'scale'): ('scale.right', 'scale.left', 'scale.none'),
    ('request.security', 'gaps'): ('barmerge.gaps_off', 'barmerge.gaps_on'),
    ('request.security', 'lookahead'): ('barmerge.lookahead_off', 'barmerge.lookahead_on'),
    ('array.sort', 'order'): ('order.ascending', 'order.descending'),
    # MEASURED 2026-11-17, one measurement per parameter
    ('array.sort_indices', 'order'): ('order.ascending', 'order.descending'),
    ('strategy.entry', 'direction'): ('strategy.long', 'strategy.short'),
    # MEASURED 2026-11-17. The list of possible values belongs to the parameter, not to the name of its type:
    # `display` is three different lists, each cut down to what the surface can show. A fill or a bar colour is
    # only on or off; an input never appears on the pane or on the price scale; a plot can appear anywhere.
    ('fill', 'display'): ('display.none', 'display.all'),
    ('bgcolor', 'display'): ('display.none', 'display.all'),
    ('label.new', 'text_formatting'): ('text.format_none', 'text.format_bold', 'text.format_italic'),
    ('plotshape', 'display'): ('display.pane', 'display.data_window', 'display.status_line',
                               'display.price_scale', 'display.pine_screener', 'display.none', 'display.all'),
    # MEASURED 2026-11-17: the whole plot* family answers with this one list — plotchar, plotarrow,
    # plotcandle and plotbar were each measured, not inferred from plot and plotshape.
    ('plotchar', 'display'): ('display.pane', 'display.data_window', 'display.status_line',
                               'display.price_scale', 'display.pine_screener', 'display.none', 'display.all'),
    ('plotarrow', 'display'): ('display.pane', 'display.data_window', 'display.status_line',
                               'display.price_scale', 'display.pine_screener', 'display.none', 'display.all'),
    ('plotcandle', 'display'): ('display.pane', 'display.data_window', 'display.status_line',
                               'display.price_scale', 'display.pine_screener', 'display.none', 'display.all'),
    ('plotbar', 'display'): ('display.pane', 'display.data_window', 'display.status_line',
                               'display.price_scale', 'display.pine_screener', 'display.none', 'display.all'),
    # MEASURED 2026-11-17. `text_formatting` is one list wherever it appears, unlike `display`: the three
    # surfaces that take formatted text offer the same three values.
    ('box.new', 'text_formatting'): ('text.format_none', 'text.format_bold', 'text.format_italic'),
    ('table.cell', 'text_formatting'): ('text.format_none', 'text.format_bold', 'text.format_italic'),
    # MEASURED 2026-11-17. A bar colour and a level are only on or off, like a fill or a background.
    ('barcolor', 'display'): ('display.none', 'display.all'),
    ('hline', 'display'): ('display.none', 'display.all'),
}

#: the values `display` of an input*() call takes. An input is written in the settings, the data window and the
#: status line, never on the pane or on the price scale, and its list is shorter than a plot's by those two.
#: MEASURED 2026-11-17 on input.int and input.bool; the Pine Script Reference Manual gives `display` the same
#: type for every input*() form, so the whole family is held to the same list.
INPUT_DISPLAY_VALUES = ('display.data_window', 'display.status_line', 'display.pine_screener',
                        'display.none', 'display.all')

#: types whose values TradingView refuses to compare with == or !=, and how it names the type in the refusal.
#: MEASURED 2026-11-17 (BINANCE:BTCUSDT 1D, Pine v6), one measurement per type: a box, a table, a polyline and a
#: linefill are refused with CE10123, while a line and a label compare and the comparison is identity. The
#: split does not follow "a drawing": the two types Pine has carried since v4 compare, the later ones do not.
UNCOMPARABLE_TYPES = {'box': 'series box', 'table': 'series table', 'polyline': 'series polyline',
                      'linefill': 'series linefill'}
#: a collection is refused the same way, but the compiler writes the type as the script declares it, with its
#: element types and no qualifier: `array<float>`, `matrix<float>`, `map<string, float>` (MEASURED 2026-11-17,
#: one measurement each). An object of a user type is refused with the type's own name, also without a qualifier.
COLLECTION_TYPE_PREFIXES = ('array<', 'matrix<', 'map<')
#: a collection is never an element of another collection. MEASURED 2026-11-18 (BINANCE:BTCUSDT 1D,
#: Pine v6): the type template that would hold one is CE10025, "Cannot use a collection in a type
#: template of another collection. Create a user-defined type with that collection as a field and use
#: it instead", and array.from() given an array is CE10122. The compiler writes the element type it
#: knows (`array<int>` in the measurement) and the type it expected from the other arguments
#: (`series int` there); this names the type the script declares, and keeps the measured wording.
COLLECTION_TYPE_WORDS = frozenset({'array', 'matrix', 'map'})
#: MEASURED on TradingView 2026-11-18 (BINANCE:BTCUSDT 1D, Pine v6) and 2026-11-19 (the same chart, Pine v5):
#: a collection in an `array.from` argument is refused in BOTH versions, and the v5 compiler writes the same
#: sentence with apostrophes where v6 uses double quotes. The quote is the only difference, so one template
#: carries both and `COLLECTION_AS_ELEMENT_FROM_VERSION` is where the rule starts. Pine v4 is not measured.
COLLECTION_AS_ELEMENT_MESSAGE = ('Cannot call {q}array.from{q} with argument {q}arg_{position}{q}={q}{name}{q}. '
                                 'An argument of {q}{actual}{q} type was used but one from {q}series int{q} '
                                 'is expected')
COLLECTION_AS_ELEMENT_FROM_VERSION = 5
#: the quote the compiler of a version puts around a name in a message
MESSAGE_QUOTE = {4: "'", 5: "'", 6: '"'}
COMPARISON_TYPE_MESSAGE = ('Cannot call "operator {op}" with argument "expr0"="{written}". An argument of '
                           '"{actual}" type was used but a "{expected}" is expected.')
#: what an operator names as the type it expected, where it reads a value of a type it has no overload for.
#: MEASURED 2026-11-17 (BINANCE:BTCUSDT 1D, Pine v6), one measurement per operator, all eleven of them: each names
#: the first overload of its own family, and the three families answer with three different types.
OPERATOR_EXPECTED_TYPES = {'==': 'simple string', '!=': 'simple string',
                           '<': 'simple float', '>': 'simple float',
                           '<=': 'simple float', '>=': 'simple float',
                           '+': 'const int', '-': 'const int', '*': 'const int',
                           '/': 'const int', '%': 'const int',
                           # MEASURED 2026-11-17: the bool operators name `const bool`, and the compiler calls
                           # the ternary `operator ?:` and holds its condition to the same type
                           'and': 'const bool', 'or': 'const bool'}
#: the name the compiler gives the ternary, and the type it expects of its condition (MEASURED 2026-11-17)
TERNARY_OPERATOR, TERNARY_CONDITION_TYPE = '?:', 'const bool'
#: the same for the unary operators, which have overload tables of their own: MEASURED 2026-11-17, the unary
#: minus names `simple int` where the binary `-` names `const int` — one symbol, two tables — and `not` names
#: `simple bool`.
UNARY_OPERATOR_EXPECTED_TYPES = {'-': 'simple int', 'not': 'simple bool'}
#: the two drawing types that compare for equality but not for order: `line == line` and `label == label`
#: compile, `line < line` and `label < label` do not (MEASURED 2026-11-17, a measurement per type per operator).
ORDER_ONLY_UNCOMPARABLE = {'line': 'series line', 'label': 'series label'}
#: the built-in constants whose type is none of string, int, float, bool or color, with the type TradingView
#: names when it refuses one. Taken from the v6 reference (the `variables_v6.json` of the docs scraper,
#: 2026-03-12) and written out here so the package needs no asset at run time: forty-four constants in
#: thirteen families, a set that only grows with a new Pine version.
NON_STRING_CONSTANTS = {
    'backadjustment.inherit': 'const backadjustment',
    'backadjustment.off': 'const backadjustment',
    'backadjustment.on': 'const backadjustment',
    'barmerge.gaps_off': 'const barmerge_gaps',
    'barmerge.gaps_on': 'const barmerge_gaps',
    'barmerge.lookahead_off': 'const barmerge_lookahead',
    'barmerge.lookahead_on': 'const barmerge_lookahead',
    'display.all': 'const plot_simple_display',
    'display.data_window': 'const plot_display',
    'display.none': 'const plot_simple_display',
    'display.pane': 'const plot_display',
    'display.pine_screener': 'const plot_display',
    'display.price_scale': 'const plot_display',
    'display.status_line': 'const plot_display',
    'hline.style_dashed': 'const hline_style',
    'hline.style_dotted': 'const hline_style',
    'hline.style_solid': 'const hline_style',
    'order.ascending': 'const sort_order',
    'order.descending': 'const sort_order',
    'plot.linestyle_dashed': 'const plot_line_style',
    'plot.linestyle_dotted': 'const plot_line_style',
    'plot.linestyle_solid': 'const plot_line_style',
    'plot.style_area': 'const plot_style',
    'plot.style_areabr': 'const plot_style',
    'plot.style_circles': 'const plot_style',
    'plot.style_columns': 'const plot_style',
    'plot.style_cross': 'const plot_style',
    'plot.style_histogram': 'const plot_style',
    'plot.style_line': 'const plot_style',
    'plot.style_linebr': 'const plot_style',
    'plot.style_stepline': 'const plot_style',
    'plot.style_stepline_diamond': 'const plot_style',
    'plot.style_steplinebr': 'const plot_style',
    'scale.left': 'const scale_type',
    'scale.none': 'const scale_type',
    'scale.right': 'const scale_type',
    'settlement_as_close.inherit': 'const settlement',
    'settlement_as_close.off': 'const settlement',
    'settlement_as_close.on': 'const settlement',
    'strategy.long': 'const strategy_direction',
    'strategy.short': 'const strategy_direction',
    'text.format_bold': 'const text_format',
    'text.format_italic': 'const text_format',
    'text.format_none': 'const text_format',
}

#: parameters that take a string, and the very type TradingView says it expects there. MEASURED on TradingView
#: 2026-11-16 (BINANCE:BTCUSDT 1D, Pine v6): each of these given `scale.left` is CE10123 worded as below. Only
#: a constant from NON_STRING_CONSTANTS is refused -- a variable, a field or a plain string could be legal.
STRING_ARGUMENTS = {
    ('label.new', 'style'): 'series string',
    ('line.new', 'style'): 'series string',
    ('box.new', 'border_style'): 'series string',
    ('table.new', 'position'): 'series string',
    ('plotshape', 'style'): 'input string',
    ('plotshape', 'location'): 'input string',
    # MEASURED 2026-11-17, the same wording with the qualifier each parameter names
    ('label.new', 'xloc'): 'series string',
    ('label.new', 'textalign'): 'series string',
    ('line.new', 'extend'): 'series string',
    ('box.new', 'extend'): 'series string',
    ('strategy', 'default_qty_type'): 'const string',
    ('strategy', 'commission_type'): 'const string',
    # MEASURED 2026-11-17, one measurement per parameter, each run as its own snippet
    ('label.new', 'yloc'): 'series string',
    ('line.new', 'xloc'): 'series string',
    ('box.new', 'text_halign'): 'series string',
    ('table.cell', 'text_halign'): 'series string',
    ('plotchar', 'location'): 'input string',
    ('label.new', 'size'): 'series string',
    ('box.new', 'text_valign'): 'series string',
    ('table.cell', 'text_valign'): 'series string',
}
ARGUMENT_TYPE_MESSAGE = ('Cannot call "{name}" with argument "{parameter}"="{written}". An argument of '
                         '"{actual}" type was used but a "{expected}" is expected.')
ENUM_ARGUMENT_MESSAGE = 'Invalid argument "{parameter}" in "{name}" call. Possible values: [{values}]'
#: `fill()` has a plot pair and an hline pair and takes no mixed one: given a plot where the other argument
#: is an hline, Pine resolves the hline overload and names the position of the plot argument.
#: MEASURED 2026-11-17 (BINANCE:BTCUSDT 1D, Pine v6) in both orders: the plot first names "hline1",
#: the plot second names "hline2", and the wording is the same CE10123 otherwise.
FILL_PAIR_MESSAGE = ('Cannot call "fill" with argument "hline{position}"="{name}". An argument of "plot" '
                     'type was used but a "hline" is expected.')
#: the two calls whose result `fill()` pairs; a name bound to one of them carries that kind
DRAWING_PAIR_CALLS = ('plot', 'hline')
FOR_STEP_TYPE_MESSAGE = ("The increment/decrement value type ('literal float') of the 'for' loop is not "
                         "compatible with counter '{}' type ('series int')")
# `for ... in` walks a collection; TradingView refuses a scalar there (CE10123, measured on a series float)
FOR_IN_COLLECTION_MESSAGE = ("'for ... in' walks a collection: '{shown}' is of {type} type, but an array, "
                             "a matrix or a map is expected")
#: the built-in types a value of which is never a collection, so `for ... in` cannot walk it
SCALAR_TYPE_WORDS = frozenset({'int', 'float', 'bool', 'string', 'color'})
# Pine v6 has no na bool (Pine v6 migration guide; PyneCore documentation): na() of a bool, nz() or fixnan() of
# one, `bool b = na` and `b := na` are refused. Only a value known to be a bool is refused: a literal, a comparison,
# `not`, `and`/`or`, a bool builtin or input, a typed bool parameter, a name declared from one of these (or typed
# bool), its history, a ternary of such arms, a user function whose value is one; a field of a user type is not
# (`na(t.flag)` is accepted)
BOOL_NEVER_NA_FROM_VERSION = 6
BOOL_OPERATORS = frozenset({'==', '!=', '<', '>', '<=', '>=', 'and', 'or'})
BOOL_NEVER_NA_MESSAGE = 'a bool is never na in Pine v6'
# Pine v6's namespaces: the heads of the dotted names of the TradingView v6
# reference (pinescript-docs-scraper pine_reference_types.json, 2026-03-12)
PINE_NAMESPACES = frozenset({
    'adjustment', 'aggressor', 'alert', 'array', 'backadjustment', 'barmerge', 'barstate', 'box', 'chart', 'color',
    'currency', 'dayofweek', 'display', 'dividends', 'earnings', 'extend', 'font', 'footprint', 'format',
    'hline', 'input', 'label', 'line', 'linefill', 'location', 'log', 'map', 'math', 'matrix', 'order',
    'plot', 'polyline', 'position', 'request', 'runtime', 'scale', 'session', 'settlement_as_close',
    'shape', 'size', 'splits', 'str', 'strategy', 'syminfo', 'ta', 'table', 'text', 'ticker',
    'timeframe', 'volume_row', 'xloc', 'yloc',
})
# Pine v6's bare built-in variables (the same reference)
PINE_BUILTIN_VARIABLES = frozenset({
    'aggressor_side', 'ask', 'bar_index', 'bid', 'close', 'dayofmonth', 'dayofweek', 'false', 'high', 'hl2', 'hlc3',
    'hlcc4', 'hour', 'last_bar_index', 'last_bar_time', 'low', 'minute', 'month', 'na', 'ohlc4', 'open',
    'second', 'time', 'time_close', 'time_tradingday', 'timenow', 'true', 'volume', 'weekofyear', 'year',
})
# The bare built-in variables of Pine v4 and before that v5 moved into namespaces
LEGACY_BUILTIN_VARIABLES = frozenset({
    'accdist', 'iii', 'interval', 'isdaily', 'isdwm', 'isintraday', 'ismonthly', 'isweekly', 'n', 'nii',
    'nvi', 'obv', 'period', 'pvi', 'pvt', 'ticker', 'tickerid', 'tr', 'vwap', 'wad', 'wvad',
})
LAST_LEGACY_VERSION = 4
# Pine v3 and before: colours, styles, locations, shapes, sizes and days as bare constants
PRE_V4_CONSTANTS = frozenset({
    'aqua', 'black', 'blue', 'fuchsia', 'gray', 'green', 'lime', 'maroon', 'navy', 'olive', 'orange',
    'purple', 'red', 'silver', 'teal', 'white', 'yellow', 'line', 'stepline', 'histogram', 'cross', 'area',
    'columns', 'circles', 'solid', 'dotted', 'dashed', 'abovebar', 'belowbar', 'top', 'bottom', 'absolute',
    'xcross', 'circle', 'triangleup', 'triangledown', 'flag', 'arrowup', 'arrowdown', 'square', 'diamond',
    'labelup', 'labeldown', 'auto', 'tiny', 'small', 'normal', 'large', 'huge', 'monday', 'tuesday',
    'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
})
LAST_PRE_V4_VERSION = 3
# An enum cannot take a namespace's name (`str` and `input` included, docs 7144);
# a UDT can (docs 6876: "UDT names shadow the language's namespaces")
RESERVED_ENUM_NAMES = PINE_NAMESPACES
# Names a bare read may mean without a declaration in the script: never out of scope
BUILTIN_NAMES = (PINE_BUILTIN_VARIABLES | PINE_NAMESPACES | frozenset(SymbolTable().builtin_symbols)
                 | SERIES_VARIABLES | PYNECORE_LIB_MODULES | frozenset(NAMESPACE_MEMBERS))
PARAMETER, LOOP_VARIABLE, INPUT, VARIABLE = 'parameter', 'loop variable', 'input', 'variable'
LOOP_COUNTER = 'loop counter'  # a variable of `for i = a to b` or `for ... in`: a loop variable that has no history
STATEMENT_TYPES = typing.get_args(Statement)
SCOPE_KIND_TEXT = {'if': 'an if block', 'for': 'a for loop', 'while': 'a while loop',
                   'switch': 'a switch', 'function': 'a function'}
UNDECLARED_MESSAGE = ("'{}' is declared only inside a block or function that has ended here; "
                      "TradingView reports it as undeclared")


class Diagnostic(NamedTuple):
    severity: str
    code: str
    line: int
    column: int
    name: str
    message: str


class ReadDecision(NamedTuple):
    """How the pass resolved one read: `binding` is (scope number, name), or
    None when no declaration in the script is visible; `shadows` says whether
    that binding hides one of the same name in an enclosing scope (a global
    counts wherever it is declared, as scope_renamer binds every global first);
    `later_global` marks an unresolved read of a global declared further down."""
    binding: Optional[Tuple[int, str]]
    shadows: bool
    later_global: bool


@dataclasses.dataclass
class _Scope:
    number: int
    kind: str
    names: Dict[str, str] = dataclasses.field(default_factory=dict)  # name -> how it was declared
    types: Dict[str, Optional[str]] = dataclasses.field(default_factory=dict)  # name -> its declared Pine type
    shadowing: set = dataclasses.field(default_factory=set)  # names declared here that hide an outer one
    bools: set = dataclasses.field(default_factory=set)  # names declared here that are known to hold a bool
    series: set = dataclasses.field(default_factory=set)  # names declared here whose value is known to be series
    consts: Dict[str, float] = dataclasses.field(default_factory=dict)  # name -> the number its value folds to
    explicit_types: Dict[str, str] = dataclasses.field(default_factory=dict)  # name -> the type keyword it was declared with
    declared_ahead: frozenset = frozenset()  # names the scope's own statements declare, walked or not
    ended: set = dataclasses.field(default_factory=set)  # names bound in blocks nested here that have ended
    #: the block runs a series number of times: a loop body, or a block a series condition governs
    governed_series: bool = False


def check_script(script: Script) -> List[Diagnostic]:
    """Every diagnostic for `script`, in source order."""
    return _SemanticCheck(script).run()


def enforce(script: Script) -> None:
    """Raise SemanticError for the first refusal, naming how many more there
    are; a script with no refusal gets one SemanticWarning per (line, column,
    name, code). A refused script is not warned about: the refusal is its
    verdict, also under `-W error`."""
    diagnostics = check_script(script)
    refusals = [diagnostic for diagnostic in diagnostics if diagnostic.severity == REFUSE]
    if refusals:
        first = refusals[0]
        raise SemanticError(first.code, first.message, first.line, first.column, len(refusals) - 1)
    warned = {(diagnostic.line, diagnostic.column, diagnostic.name, diagnostic.code): diagnostic
              for diagnostic in diagnostics if diagnostic.severity == WARN}
    for diagnostic in warned.values():
        warnings.warn(SemanticWarning(diagnostic.code, diagnostic.message, diagnostic.line, diagnostic.column),
                      stacklevel=3)


def scope_decisions(script: Script) -> Dict[Tuple[int, int, str], ReadDecision]:
    """(line, column, name) of every positioned bare-name read -> how the pass resolved it."""
    check = _SemanticCheck(script)
    check.run()
    return check.decisions


def _declared_names(node) -> List[str]:
    if isinstance(node, (VarDecl, VaripDecl, InputDecl)):
        return [node.name]
    if isinstance(node, Assignment):
        return list(node.target.names) if isinstance(node.target, TupleDestructure) else [node.target]
    return []


def _literal_of(node, literal_type: str) -> bool:
    return isinstance(node, Literal) and node.literal_type == literal_type


def _constant_name(node) -> Optional[str]:
    """The written name of a built-in namespaced constant, `plot.style_area`, or None for anything else."""
    if isinstance(node, MemberAccess) and isinstance(node.object, str) and node.object in PINE_NAMESPACES:
        return f'{node.object}.{node.member}'
    if isinstance(node, Identifier) and '.' in node.name and node.name.split('.', 1)[0] in PINE_NAMESPACES:
        return node.name
    return None


def _literal_number(node) -> Optional[float]:
    """The value of an int or float literal, with a leading minus, or None for anything else."""
    if isinstance(node, UnaryOp) and node.op == '-':
        inner = _literal_number(node.operand)
        return None if inner is None else -inner
    return node.value if isinstance(node, Literal) and node.literal_type in ('int', 'float') else None


def _shown_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


#: the calls that fold to a number when their arguments do. MEASURED on TradingView 2026-11-18
#: (BINANCE:BTCUSDT 1D, Pine v6): `int(math.max(101, 101))` reaches the range check as the
#: constant 101, both as a direct argument and through a variable bound to it. The set holds only calls
#: whose value is exact and needs no bar: a cast and the numeric picks/rounds of `math.*`.
FOLDABLE_CALLS: Dict[str, int] = {
    'int': 1, 'float': 1,
    'math.abs': 1, 'math.ceil': 1, 'math.floor': 1, 'math.round': 1, 'math.sign': 1,
    'math.max': -1, 'math.min': -1,  # -1: any number of arguments
}
#: `/` is deliberately absent: whether Pine folds `300 / 3` as an int or a float depends on the operands'
#: own types (see pine2pyne/const_int_division.py), and a wrong answer here would refuse a script
#: TradingView accepts. `%` is absent for the same reason.
FOLDABLE_OPERATORS = frozenset({'+', '-', '*'})

#: Reading an element out of a collection gives a per-bar value. MEASURED on TradingView 2026-11-19
#: (BINANCE:BTCUSDT 1D, Pine v6):
#: `alertcondition(title = array.get(names, 0))` is CE10123, `argument "title"="call "array.get"
#: (series string)"` -- a collection's CONTENTS change from bar to bar, so what comes out of one is series
#: even where the collection itself is not.
#:
#: Only the element getters are here, and the boundary rests on evidence, not caution: the published
#: library jason5480/string_utils/6 declares `const map<string, string> LEXICON =
#: map.new<string, string>()` and compiles on TradingView, so a CONSTRUCTOR is const-able and the whole
#: namespace cannot be series. What `array.size()` and the other inspectors are qualified as is not
#: measured, so they are left alone.
COLLECTION_READS = frozenset({
    'array.get', 'array.first', 'array.last', 'array.pop', 'array.shift', 'array.remove',
    'matrix.get', 'map.get',
})


_PRICE_BUILTINS = frozenset({'open', 'high', 'low', 'close', 'volume', 'hl2', 'hlc3', 'ohlc4', 'hlcc4'})
_FAMILY = {'int': 'number', 'float': 'number', 'string': 'string', 'bool': 'bool'}
ARITHMETIC_OPERATORS = frozenset({'+', '-', '*', '/', '%'})
COMPARISON_OPERATORS = frozenset({'==', '!=', '<', '>', '<=', '>='})
ORDERING_OPERATORS = frozenset({'<', '>', '<=', '>='})
OPERATOR_ARGUMENT_MESSAGE = ('Cannot call "operator {op}" with argument "{slot}"="{shown}". An argument of "{used}" '
                             'type was used but a "{expected}" is expected.')
NA_COMPARISON_MESSAGE = 'Cannot compare a value to "na" directly. Use the "na()" function instead.'
IF_CONDITION_MESSAGE = 'The condition of the "if" statement must evaluate to a "bool" value.'
INPUT_MINVAL_MESSAGE = 'Input\'s "defval" value must be between "minval" and "maxval"'
INPUT_OPTIONS_MESSAGE = "input's defval should be in options, but {shown} is not in [{options}]"
TRANSPARENCY_MESSAGE = 'color.new: transp argument value should be between 0 and 100'
PLOT_SERIES_MESSAGE = ('Cannot call "plot" with argument "series"="{shown}". An argument of "series bool" type was used but a "series float" is expected.')
# A date string timestamp(dateString) does not read: a trailing Z as the zone letter. The spellings it reads are
# many (an ISO date with an optional time and offset, "01 Jan 2020 00:00 +0000", "01-01-2018", "01 1 2000 00:00 +0000"),
# so only the measured unreadable shape is refused.
TIMESTAMP_UNREADABLE = re.compile(r'\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*Z$')
TIMESTAMP_FORMAT_MESSAGE = 'timestamp(s): unrecognized datetime format'
# the int-typed coordinates of the drawing constructors (a bar index or a UNIX time), by parameter and position
INT_COORDINATE_PARAMETERS = {'label.new': {'x': 0}, 'line.new': {'x1': 0, 'x2': 2}, 'box.new': {'left': 0, 'right': 2}}
# the element written into an array, by parameter and position
STRING_TEXT_PARAMETERS = {
    'label.new': {'text': 3, 'tooltip': None},
    'label.set_text': {'text': 1},
    'label.set_tooltip': {'tooltip': 1},
    'box.set_text': {'text': 1},
    'table.cell': {'text': 3},
}
ARRAY_VALUE_PARAMETERS = {'array.push': 1, 'array.unshift': 1, 'array.set': 2, 'array.insert': 2, 'array.fill': 1}
# the predicates over a whole array take an array of bool and nothing else: an array<float> is refused, although
# the manual of earlier versions read every element as a truth value (measured 2026-11-16: CE10123)
#: the built-ins that return nothing (`→ void` in the Pine Script Reference Manual v6, snapshot 2026-11-14, plus
#: `array.sort` and `matrix.sort`, whose manual pages give no syntax line and which were measured on TradingView
#: 2026-11-16): naming the result of such a call is CE10098 there, so it is refused here too
VOID_FUNCTIONS: frozenset[str] = frozenset({
    'alert', 'alertcondition', 'array.clear', 'array.fill', 'array.insert', 'array.push', 'array.reverse',
    'array.set', 'array.sort', 'array.unshift', 'barcolor', 'bgcolor', 'box.delete', 'box.set_bgcolor',
    'box.set_border_color', 'box.set_border_style', 'box.set_border_width', 'box.set_bottom',
    'box.set_bottom_right_point', 'box.set_extend', 'box.set_left', 'box.set_lefttop', 'box.set_right',
    'box.set_rightbottom', 'box.set_text', 'box.set_text_color', 'box.set_text_font_family',
    'box.set_text_formatting', 'box.set_text_halign', 'box.set_text_size', 'box.set_text_valign',
    'box.set_text_wrap', 'box.set_top', 'box.set_top_left_point', 'box.set_xloc', 'indicator', 'label.delete',
    'label.set_color', 'label.set_point', 'label.set_size', 'label.set_style', 'label.set_text',
    'label.set_text_font_family', 'label.set_text_formatting', 'label.set_textalign', 'label.set_textcolor',
    'label.set_tooltip', 'label.set_x', 'label.set_xloc', 'label.set_xy', 'label.set_y', 'label.set_yloc',
    'library', 'line.delete', 'line.set_color', 'line.set_extend', 'line.set_first_point', 'line.set_second_point',
    'line.set_style', 'line.set_width', 'line.set_x1', 'line.set_x2', 'line.set_xloc', 'line.set_xy1',
    'line.set_xy2', 'line.set_y1', 'line.set_y2', 'linefill.delete', 'linefill.set_color', 'map.clear',
    'map.put_all', 'matrix.add_col', 'matrix.add_row', 'matrix.fill', 'matrix.reshape', 'matrix.reverse',
    'matrix.set', 'matrix.sort', 'matrix.swap_columns', 'matrix.swap_rows', 'max_bars_back', 'plotarrow', 'plotbar',
    'plotcandle', 'plotchar', 'plotshape', 'polyline.delete', 'runtime.error', 'strategy', 'strategy.cancel',
    'strategy.cancel_all', 'strategy.close', 'strategy.close_all', 'strategy.entry', 'strategy.exit',
    'strategy.order', 'strategy.risk.allow_entry_in', 'strategy.risk.max_cons_loss_days',
    'strategy.risk.max_drawdown', 'strategy.risk.max_intraday_filled_orders', 'strategy.risk.max_intraday_loss',
    'strategy.risk.max_position_size', 'table.cell', 'table.cell_set_bgcolor', 'table.cell_set_height',
    'table.cell_set_text', 'table.cell_set_text_color', 'table.cell_set_text_font_family',
    'table.cell_set_text_formatting', 'table.cell_set_text_halign', 'table.cell_set_text_size',
    'table.cell_set_text_valign', 'table.cell_set_tooltip', 'table.cell_set_width', 'table.clear', 'table.delete',
    'table.merge_cells', 'table.set_bgcolor', 'table.set_border_color', 'table.set_border_width',
    'table.set_frame_color', 'table.set_frame_width', 'table.set_position',
})
VOID_ASSIGNMENT_MESSAGE = 'Void expression cannot be assigned to a variable'
BOOL_ARRAY_MEMBERS = ('every', 'some')
#: the `strategy.*` constants that name a direction, an order size or an OCA group rather than the account:
#: only the account side is refused in an indicator, so a constant alone never trips the rule
STRATEGY_CONSTANT_PREFIXES = ('strategy.long', 'strategy.short', 'strategy.cash', 'strategy.fixed',
                              'strategy.percent_of_equity', 'strategy.direction.', 'strategy.oca.',
                              'strategy.commission.')
STRATEGY_IN_INDICATOR_MESSAGE = ('You cannot use strategy functions ({name}) in indicator script. Please replace '
                                 '"indicator" with "strategy".')
# the index of a collection element: TradingView expects a `series int` and refuses a string or a bool there
# (measured 2026-11-15: array.get with a literal string index is CE10123)
COLLECTION_INDEX_PARAMETERS = {'array.get': {'index': 1}, 'array.set': {'index': 1}, 'array.remove': {'index': 1},
                               'array.insert': {'index': 1}}
# the key and the value of a map call, by parameter and position (measured 2026-11-15: a number key of a
# string-keyed map is CE10123)
MAP_KEY_VALUE_PARAMETERS = {'map.put': {'key': 1, 'value': 2}, 'map.get': {'key': 1}, 'map.contains': {'key': 1},
                            'map.remove': {'key': 1}}
# a parameter that names a timeframe: a number there is refused (measured 2026-11-15: request.security with
# timeframe = 5 is CE10123, the manual's qualifier is `simple string`)
TIMEFRAME_PARAMETERS = {'request.security': {'timeframe': 1}, 'request.security_lower_tf': {'timeframe': 1}}
# a parameter that takes a colour: a string or a number there is refused (measured 2026-11-15 for color.new)
COLOR_PARAMETERS = {'color.new': {'color': 0}, 'color.t': {'color': 0}, 'color.r': {'color': 0},
                    'color.g': {'color': 0}, 'color.b': {'color': 0}}
# the ta.* functions whose first parameter is the series `source` (ta.tr takes a bool, ta.valuewhen a condition,
# ta.pivothigh a count: those are left alone)
TA_SOURCE_FIRST = frozenset({
    'ta.sma', 'ta.ema', 'ta.wma', 'ta.rma', 'ta.hma', 'ta.vwma', 'ta.swma', 'ta.alma', 'ta.linreg', 'ta.rsi', 'ta.cci',
    'ta.cmo', 'ta.cog', 'ta.change', 'ta.mom', 'ta.roc', 'ta.stdev', 'ta.variance', 'ta.dev', 'ta.highest', 'ta.lowest',
    'ta.highestbars', 'ta.lowestbars', 'ta.percentrank', 'ta.percentile_linear_interpolation',
    'ta.percentile_nearest_rank', 'ta.median', 'ta.mode', 'ta.range', 'ta.cum', 'ta.falling', 'ta.rising', 'ta.macd',
    'ta.bb', 'ta.bbw', 'ta.kc', 'ta.kcw', 'ta.mfi', 'ta.vwap', 'ta.max', 'ta.min'})
# Pine v6 drops the implicit conversions between numbers and bools and the comparison with the na literal
NA_COMPARISON_REFUSED_FROM_VERSION = 6
BOOL_NUMBER_MISMATCH_FROM_VERSION = 6
INPUT_VALUE_TYPES = {'input': 'int', 'input.int': 'int', 'input.float': 'float', 'input.bool': 'bool',
                     'input.string': 'string', 'input.color': 'color', 'input.timeframe': 'string',
                     'input.session': 'string', 'input.symbol': 'string', 'input.price': 'float', 'input.time': 'int',
                     'input.text_area': 'string'}


def _shown_bool(node) -> str:
    """How the compiler spells a bool argument it refuses: a name as written, an operator as a call of it."""
    if isinstance(node, Identifier):
        return node.name
    if isinstance(node, BinaryOp):
        return f'call "operator {node.op}" (series bool)'
    if isinstance(node, UnaryOp):
        return f'call "operator {node.op}" (series bool)'
    return 'expr'


def _value_kind(node) -> Optional[tuple]:
    """(qualifier, type, shown) of a value whose type is certain: a literal (`('literal', 'string', 'a')`), a price
    built-in (`('series', 'float', 'close')`); None for anything else."""
    if isinstance(node, Literal) and node.literal_type in _FAMILY:
        shown = str(node.value).lower() if node.literal_type == 'bool' else str(node.value)
        return 'literal', node.literal_type, shown
    if isinstance(node, UnaryOp) and node.op == '-':
        inner = _value_kind(node.operand)
        if inner is not None and _FAMILY[inner[1]] == 'number':
            return 'literal', inner[1], '-' + inner[2]
    if isinstance(node, Identifier):
        if node.name in ('true', 'false'):
            return 'literal', 'bool', node.name
        if node.name in _PRICE_BUILTINS:
            return 'series', 'float', node.name
    return None


def _bool_number(a: str, b: str) -> bool:
    return {_FAMILY[a], _FAMILY[b]} == {'bool', 'number'}


def _expected_of(kind: tuple) -> str:
    return f"{'const' if kind[0] == 'literal' else kind[0]} {kind[1]}"


def _array_element_type(value) -> Optional[str]:
    """The element type an array constructor names: `array.new<float>()`, `array.new_float()`, `array.from(1, 2)`."""
    if not isinstance(value, FunctionCall) or not isinstance(value.func, str):
        return None
    name = value.func
    if name.startswith('array.new<') and name.endswith('>'):
        return name[len('array.new<'):-1]
    if name.startswith('array.new_'):
        return name[len('array.new_'):]
    if name == 'array.from' and value.args:
        kind = _value_kind(value.args[0])
        return kind[1] if kind is not None else None
    return None


def _map_types(value) -> Optional[tuple]:
    """The key and value types a map constructor names: `map.new<string, float>()`."""
    if not isinstance(value, FunctionCall) or not isinstance(value.func, str):
        return None
    name = value.func
    if not (name.startswith('map.new<') and name.endswith('>')):
        return None
    key, _, element = name[len('map.new<'):-1].partition(',')
    return (key.strip(), element.strip()) if element else None


def _literal_string(node) -> Optional[str]:
    """The text of a string literal, or None for anything else (a name, a call, nothing)."""
    return node.value if isinstance(node, Literal) and node.literal_type == 'string' else None


def _not_of_enumeration(node, allowed: Sequence[str]) -> bool:
    """True for an argument a parameter typed as an enumeration cannot take: a string, or any constant of a
    namespace that is not one of the possible values. A string is refused even though the values are written as
    words: the parameter is not of string type, and the compiler answers with the same list of possible values
    as for a constant of the wrong family. Every constant inside the argument is held to the list, not only one
    written directly: the display constants add and subtract, and the type of the whole expression is the wide
    one, so a parameter that takes only two of them refuses `display.all - display.status_line` as well
    (MEASURED 2026-11-17). A name, a call or anything else whose value is not certain here is left alone."""
    if _literal_string(node) is not None:
        return True
    return any(written is not None and written not in allowed
               for written in (_constant_name(inner) for inner in walk(node)))


def _pine_version(script: Script) -> int:
    version = script.version.version if script.version else ''
    return int(version) if version.isdigit() else 1  # no annotation: Pine v1


SCRIPT_DECLARATION_CALLS = frozenset({'indicator', 'strategy', 'library', 'study'})
LIBRARY_TITLE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')
LIBRARY_TITLE_MESSAGE = ('Invalid argument "title" in "library" call. It cannot contain spaces, special characters or '
                         'begin with a digit.')
LIBRARY_EXPORT_MESSAGE = 'A library must contain at least one exported function, method, or type.'
#: a declaration that names another timeframe cannot be combined with code that draws: the drawing would have
#: to be made in a context that is not the chart's. MEASURED on TradingView 2026-11-18 (BINANCE:BTCUSDT 1D) on
#: `table.new` and on `label.new`; both give the same CE10080, and so does `timeframe = ""`, the chart's own
#: timeframe -- the check looks at whether the argument is there, not at what it says. MEASURED again
#: 2026-11-18: `alert()` is refused the same way, while `alertcondition()` is NOT -- it compiles and the
#: script runs in the weekly context -- so the message's word «alerts» covers only the call that fires one.
TIMEFRAME_DRAWING_CALLS = frozenset({'line.new', 'line.copy', 'label.new', 'label.copy', 'box.new', 'box.copy',
                                     'table.new', 'polyline.new', 'alert'})
TIMEFRAME_SIDE_EFFECT_MESSAGE = ('Cannot include a "timeframe" argument in the "indicator()" function when the '
                                 'code produces side effects (e.g., the outputs of functions that create '
                                 'drawings or alerts). Scripts cannot evaluate side effects in other contexts.')
EXPORT_OUTSIDE_LIBRARY_MESSAGE = 'Only libraries can contain exported functions.'
#: MEASURED 2026-11-18 (BINANCE:BTCUSDT 1D, Pine v6): an exported function that reads a variable an input gave is
#: refused with this sentence, which names both the function and the variable. A plain global is allowed -- the
#: same library compiles when the variable holds a literal -- so only the bindings of `input.*` count here.
EXPORT_INPUT_MESSAGE = ('The exported function "{func}" depends on the "{name}" input variable, which is not '
                        'allowed.')
DECLARATION_COUNT_MESSAGE = ('Scripts must contain one declaration statement: "indicator()", "strategy()" or "library()". '
                             'Your script has {}.')
FUNCTION_REDEFINED_MESSAGE = ('Function "{name}" already defined. The "{name}" function has overloads using the same number '
                              'of required parameters without them having distinct types. Function overloads with the same '
                              'number of required parameters must have explicit parameter types that are unique among '
                              'overloads.')
SAME_PARAMETERS_MESSAGE = ('The "{name}" function has overloads with the same parameters. The type of parameters must be '
                           'different in overloaded versions of functions.')
METHOD_RECEIVER_TYPE_MESSAGE = 'The first method argument must be explicitly typed'
LOOP_VARIABLE_MUTABLE_MESSAGE = 'Variable "{}" cannot be mutable'
TUPLE_TO_SCALAR_MESSAGE = 'Invalid assignment. Cannot assign a tuple to a variable "{}".'
LOOP_SCOPE_KINDS = frozenset({'for', 'while'})

TIMEFRAME_GAPS_ALONE_MESSAGE = ('"timeframe_gaps" has no effect because the "indicator()" call has no '
                                '"timeframe" argument')


def _type_word(type_hint: Optional[str]) -> Optional[str]:
    """A type hint without its form keyword (`series float` -> `float`)."""
    return type_hint.split()[-1] if type_hint else None


class _SemanticCheck:
    def __init__(self, script: Script):
        self.script = script
        self.version = _pine_version(script)
        self.constant_heads = constant_heads(script)
        self.builtin_names = BUILTIN_NAMES | (LEGACY_BUILTIN_VARIABLES if self.version <= LAST_LEGACY_VERSION
                                              else frozenset()) \
            | (PRE_V4_CONSTANTS if self.version <= LAST_PRE_V4_VERSION else frozenset())
        self.top_items = sorted(list(script.declarations) + list(script.body), key=lambda item: item.line)
        self.diagnostics: List[Diagnostic] = []
        self.decisions: Dict[Tuple[int, int, str], ReadDecision] = {}
        self.line, self.column = 0, 0
        self.declaring: frozenset = frozenset()  # the names the statement being walked declares
        self.scope_numbers = itertools.count()
        self.global_names = {name for item in self.top_items for name in _declared_names(item)}
        self.field_read_rule = FieldReadRule(script)
        self.global_types = global_binding_types(script)
        self.type_evidence = script_type_evidence(script)
        every_node = list(walk(self.top_items))
        self.local_only = _names_declared_in(every_node) - self.global_names
        self.function_names = {node.name for node in every_node if isinstance(node, FuncDecl)}
        self.declared_functions: set = set()  # the user-defined functions walked so far
        self.array_elements: Dict[str, str] = {}  # array variable -> the element type its constructor names
        self.map_types: Dict[str, tuple] = {}  # map variable -> the (key, value) types its constructor names
        self.drawing_kind: Dict[str, str] = {}  # global name -> 'plot' or 'hline', the id kind it holds
        self.type_names = {node.name for node in every_node if isinstance(node, (TypeDecl, EnumDecl))}
        self.udt_names = {node.name for node in every_node if isinstance(node, TypeDecl)}
        self.udt_fields = {node.name: {field_item[0]: field_item[1] for field_item in node.fields}
                           for node in every_node if isinstance(node, TypeDecl)}
        self.method_function_names = {node.name for node in every_node if isinstance(node, FuncDecl) and node.is_method}
        declared_count: Dict[str, int] = {}
        for item in self.top_items:
            if isinstance(item, FuncDecl) and not item.is_method:
                declared_count[item.name] = declared_count.get(item.name, 0) + 1
        self.overloaded = {name for name, count in declared_count.items() if count > 1}
        self.single_functions = {item.name: item for item in self.top_items if isinstance(item, FuncDecl)
                                 and not item.is_method and item.name not in self.overloaded}
        self.bool_functions = self.functions_of_bool_value()
        self.static_requests = _declares_static_requests(script.script_decl)

    # ------------------------------------------------------------ reporting

    def report(self, severity: str, code: str, name: str, message: str, node=None) -> None:
        """A diagnostic at `node`'s own position when it has one, else the statement's."""
        line, column = (node.line, node.column) if getattr(node, 'line', 0) else (self.line, self.column)
        self.diagnostics.append(Diagnostic(severity, code, line, column, name, message))

    def at(self, node) -> None:
        if getattr(node, 'line', 0):
            self.line, self.column = node.line, node.column

    def at_expression(self, node) -> None:
        """An else-if condition or a switch case comes after a block that moved the
        position: take its own, from its first positioned part."""
        self.at(next((part for part in walk(node) if getattr(part, 'line', 0)), None))

    # ------------------------------------------------------------ scopes

    def new_scope(self, kind: str, statements=()) -> _Scope:
        declared = [name for statement in statements for name in _declared_names(statement)] \
            if isinstance(statements, list) else []
        return _Scope(next(self.scope_numbers), kind, declared_ahead=frozenset(declared))

    @staticmethod
    def declared_later(stack: List[_Scope], name: str) -> bool:
        """`name` is declared further down a scope that is still open (it is not bound yet)."""
        return any(name in scope.declared_ahead for scope in stack)

    def out_of_scope(self, stack: List[_Scope], name: str) -> bool:
        """An unbound `name` that only a scope which has ended declared so far: bound in
        an ended if/for/while/switch block of the reader's own scopes (the emitted
        Python would read that block's stale value; a name the statement itself
        declares is a self-reference instead), or local to other scopes and not
        declared later in an open one."""
        return ((name not in self.declaring and any(name in scope.ended for scope in self.own_scopes(stack)))
                or (name in self.local_only and not self.declared_later(stack, name)))

    @staticmethod
    def own_scopes(stack: List[_Scope]) -> List[_Scope]:
        """The open scopes a read sees as its own: from the innermost function
        scope in (a function reading a global is its own case), else all."""
        functions = [index for index, scope in enumerate(stack) if scope.kind == 'function']
        return stack[functions[-1]:] if functions else stack

    def reserved_name(self, name: str) -> None:
        """From Pine v5 a reserved word cannot name a variable, a function or a parameter."""
        if self.version >= RESERVED_NAMES_FROM_VERSION and name in RESERVED_NAMES:
            self.report(REFUSE, 'reserved-name', name, f"'{name}' cannot be used as a variable or function name.")

    def declare(self, stack: List[_Scope], name: str, how: str, type_name: Optional[str] = None) -> None:
        if name == DISCARD:
            return
        scope = stack[-1]
        earlier = scope.names.get(name)
        if earlier is None:
            outer = any(name in enclosing.names for enclosing in stack[:-1])
            if outer or (len(stack) > 1 and name in self.global_names):
                scope.shadowing.add(name)
        elif INPUT in (earlier, how):  # an input is declared only at global scope (parser.py top-level loop)
            self.report(REFUSE, 'duplicate-input', name, f"'{name}' is already declared in this scope "
                        f'(an input and another declaration cannot share a name)')
        elif earlier not in (LOOP_VARIABLE, LOOP_COUNTER):
            # MEASURED on TradingView 2026-11-14 (v6): `x = 1.0` then `x = 2.0`, and a parameter declared again in
            # the function's body, fail to compile: `"x" is already defined`
            self.report(REFUSE, 'redeclared', name, f'"{name}" is already defined')
        scope.names.setdefault(name, how)
        scope.types.setdefault(name, type_name)  # the first declaration's, as scope_renamer binds it

    @staticmethod
    def depth_of(stack: List[_Scope], name: str) -> Optional[int]:
        for depth in range(len(stack) - 1, -1, -1):
            if name in stack[depth].names:
                return depth
        return None

    def read(self, stack: List[_Scope], identifier: Identifier) -> None:
        name = identifier.name
        depth = self.depth_of(stack, name)
        if identifier.line:
            binding = (stack[depth].number, name) if depth is not None else None
            shadows = depth is not None and name in stack[depth].shadowing
            later_global = depth is None and name in self.global_names
            self.decisions[identifier.line, identifier.column, name] = ReadDecision(binding, shadows, later_global)
        if name == DISCARD:
            if self.version >= DISCARD_FROM_VERSION:
                self.report(REFUSE, 'discard-read', name, "'_' discards a value and cannot be read", identifier)
            return
        if depth is not None or name in self.builtin_names or name in self.function_names or name in self.type_names:
            return
        if self.out_of_scope(stack, name):
            self.report(REFUSE, 'out-of-scope', name, UNDECLARED_MESSAGE.format(name), identifier)
        elif name in self.global_names or self.declared_later(stack, name):
            self.read_before_declaration(name)

    def read_before_declaration(self, name: str) -> None:
        """A read above the name's declaration in a scope that is still open, from
        any inner scope, or in its own declaration. MEASURED on TradingView
        2026-11-14 (v6): a function reading a global declared below it, and a
        global read above its declaration, fail to compile: `Undeclared
        identifier "g"`."""
        if name not in self.declaring:
            self.report(REFUSE, 'used-before-declaration', name, f'Undeclared identifier "{name}"')
        elif self.version > LAST_SELF_REFERENCE_VERSION:
            # MEASURED on TradingView 2026-11-14 (v4, v6): `x = nz(x[1]) + 1` gives `Undeclared identifier "x"`
            self.report(REFUSE, 'self-reference', name, f'Undeclared identifier "{name}"')

    # ------------------------------------------------------------ statements

    def run(self) -> List[Diagnostic]:
        self.recursion()
        self.script_declaration()
        self.imported_once()
        self.function_overloads()
        stack = [self.new_scope('global')]
        for item in self.top_items:
            self.statement(item, stack)
        self.conditional_history_calls()
        self.inline_input_titles()
        return sorted(self.diagnostics, key=lambda diagnostic: (diagnostic.line, diagnostic.column))

    def imported_once(self) -> None:
        """A library may be imported once: TradingView refuses a second import of the same user/library/version,
        whatever the alias (CE10233)."""
        seen = set()
        for item in self.script.imports:
            key = (item.user, item.library, item.version)
            if key in seen:
                self.report(REFUSE, 'import-twice', item.library, IMPORT_TWICE_MESSAGE, item)
                return
            seen.add(key)

    def script_declaration(self) -> None:
        """A script carries one declaration statement; `export` belongs to a library, whose title is spelled like an
        identifier and which exports at least one function, method or type."""
        declaration = self.script.script_decl
        if declaration is None:
            return
        # the declaration carries named-constant parameters too, and it is not a FunctionCall node, so the
        # check that call_arguments makes has to be repeated here
        kind = 'strategy' if isinstance(declaration, StrategyDecl) else (
            'library' if isinstance(declaration, LibraryDecl) else 'indicator')
        for parameter, expected in ((p, e) for (f, p), e in STRING_ARGUMENTS.items() if f == kind):
            written = _constant_name(declaration.kwargs.get(parameter))
            actual = NON_STRING_CONSTANTS.get(written or '')
            if actual is not None:
                self.report(REFUSE, 'argument-type', kind,
                            ARGUMENT_TYPE_MESSAGE.format(name=kind, parameter=parameter, written=written,
                                                         actual=actual, expected=expected), declaration)
        for parameter, allowed in ((p, v) for (f, p), v in ENUM_ARGUMENTS.items() if f == kind):
            if _not_of_enumeration(declaration.kwargs.get(parameter), allowed):
                self.report(REFUSE, 'enum-argument', kind,
                            ENUM_ARGUMENT_MESSAGE.format(parameter=parameter, name=kind,
                                                         values=', '.join(allowed)), declaration)
        self.declaration_limits(declaration)
        self.timeframe_with_a_drawing(declaration, kind)
        self.gaps_without_a_timeframe(declaration, kind)
        self.declaration_argument_names(declaration, kind)
        extra = [item.expr for item in self.top_items if isinstance(item, ExpressionStatement)
                 and isinstance(item.expr, FunctionCall) and item.expr.func in SCRIPT_DECLARATION_CALLS]
        for call in extra:
            self.report(REFUSE, 'declaration-count', call.func, DECLARATION_COUNT_MESSAGE.format(1 + len(extra)), call)
        exported = [item for item in self.top_items
                    if isinstance(item, (FuncDecl, TypeDecl, EnumDecl)) and item.is_export]
        if isinstance(declaration, LibraryDecl):
            if isinstance(declaration.title, str) and not LIBRARY_TITLE.fullmatch(declaration.title):
                self.report(REFUSE, 'library-title', declaration.title, LIBRARY_TITLE_MESSAGE, declaration)
            if not exported:
                self.report(REFUSE, 'library-export', declaration.title or 'library', LIBRARY_EXPORT_MESSAGE,
                            declaration)
            self.exported_input_reads(exported)
        else:
            for item in exported:
                self.report(REFUSE, 'export-outside-library', item.name, EXPORT_OUTSIDE_LIBRARY_MESSAGE, item)

    def timeframe_with_a_drawing(self, declaration, kind: str) -> None:
        """An indicator that names another timeframe calculates in that context, where a drawing cannot be made."""
        if kind != 'indicator' or 'timeframe' not in declaration.kwargs:
            return
        for node in walk(self.top_items):
            # a call's head is not always a name: a dynamic call carries a node there, and some of those
            # nodes are unhashable, so the membership test only runs on a spelled-out name
            if isinstance(node, FunctionCall) and isinstance(node.func, str) \
                    and node.func in TIMEFRAME_DRAWING_CALLS:
                self.report(REFUSE, 'timeframe-side-effect', node.func, TIMEFRAME_SIDE_EFFECT_MESSAGE, declaration)
                return

    def declaration_argument_names(self, declaration, kind: str) -> None:
        """A script declaration names only the arguments it has. The manual's parameter list is the whole of it,
        so a name outside the list -- a near miss, or one that belongs to another declaration -- is refused with
        the same words the compiler uses. Older versions spelled some arguments differently and are left alone."""
        if self.version < REMOVED_PARAMETERS_FROM_VERSION:
            return
        signature = PARAMETER_QUALIFIERS.get(kind)
        if signature is None:
            return
        for key in declaration.kwargs:
            if key not in signature[0]:
                self.report(REFUSE, 'declaration-argument-name', key,
                            f'The "{kind}" function does not have an argument with the name "{key}"', declaration)

    def gaps_without_a_timeframe(self, declaration, kind: str) -> None:
        """The gaps flag means something only beside a timeframe of the script's own: on its own it is refused,
        not quietly ignored."""
        if kind != 'indicator' or 'timeframe' in declaration.kwargs:
            return
        if self.version < REMOVED_PARAMETERS_FROM_VERSION:
            # older versions spelled the flag differently and are not measured: leave them alone
            return
        if 'timeframe_gaps' in declaration.kwargs:
            self.report(REFUSE, 'timeframe-gaps-alone', 'timeframe_gaps', TIMEFRAME_GAPS_ALONE_MESSAGE, declaration)

    def exported_input_reads(self, exported: list) -> None:
        """An exported function is self-contained in one respect: it cannot read a variable an input gave, nor
        one computed from such a variable -- the refusal follows the chain and names the input at its head. A
        variable that holds anything else may be read from the enclosing scope, and a name the function itself
        binds is its own."""
        inputs = {item.name: item.name for item in self.top_items if isinstance(item, InputDecl) and item.name}
        if not inputs:
            return
        # the refusal follows the chain, and names the input at its head: a global computed from an input
        # carries the same ban as the input itself
        changed = True
        while changed:
            changed = False
            for item in self.top_items:
                name = item.target if isinstance(item, Assignment) and isinstance(item.target, str) else (
                    item.name if isinstance(item, (VarDecl, VaripDecl)) else None)
                if not name or name in inputs or getattr(item, 'value', None) is None:
                    continue
                source = next((inputs[node.name] for node in walk([item.value])
                               if isinstance(node, Identifier) and node.name in inputs), None)
                if source is not None:
                    inputs[name] = source
                    changed = True
        for item in exported:
            if not isinstance(item, FuncDecl):
                continue
            body = item.body if isinstance(item.body, list) else [item.body]
            own = {parameter.name for parameter in item.params} | _names_declared_in(list(walk(body)))
            for node in walk(body):
                if isinstance(node, Identifier) and node.name in inputs and node.name not in own:
                    self.report(REFUSE, 'export-input', item.name,
                                EXPORT_INPUT_MESSAGE.format(func=item.name, name=inputs[node.name]), node)
                    break

    def declaration_limits(self, declaration) -> None:
        """A declaration cannot raise a drawing count above the maximum TradingView keeps, nor the history buffer
        above 5000 bars."""
        kwargs = getattr(declaration, 'kwargs', None) or {}
        call = 'library' if isinstance(declaration, LibraryDecl) else (
            'strategy' if isinstance(declaration, StrategyDecl) else 'indicator')
        for parameter, limit in DRAWING_COUNT_LIMITS.items():
            value = _literal_number(kwargs.get(parameter))
            if value is not None and value > limit:
                self.report(REFUSE, 'drawing-count-range', parameter,
                            DRAWING_COUNT_MESSAGE.format(parameter=parameter, value=_shown_number(value),
                                                         limit=limit), declaration)
        precision = _literal_number(kwargs.get('precision'))
        if precision is not None and precision < 0:
            self.report(REFUSE, 'precision-negative', 'precision', NEGATIVE_PRECISION_MESSAGE.format(call=call),
                        declaration)
        for parameter, (low, high) in DECLARATION_RANGES.items():
            value = _literal_number(kwargs.get(parameter))
            if value is None or low <= value <= high:
                continue
            code = 'max-bars-back-range' if parameter == 'max_bars_back' else 'drawing-count-range'
            self.report(REFUSE, code, parameter,
                        DECLARATION_RANGE_MESSAGE.format(value=_shown_number(value), parameter=parameter,
                                                         call=call, low=low, high=high), declaration)

    def function_overloads(self) -> None:
        """Two declarations of one function name are overloads only when their required parameters differ in
        number, or every parameter is explicitly typed and the type lists differ; otherwise the name is redefined.

        A parameter's form keyword is part of its type here: `simple string this` and `string this` are two
        overloads, and libraries do declare both. A method is held to the same rule, with its receiver counted as
        the first parameter: MEASURED 2026-11-17 (BINANCE:BTCUSDT 1D, Pine v6), a pair of methods on one
        user-defined type answers with the same message as a pair of plain functions, calling the method a
        function. Methods and functions of one name are kept apart, which is not measured and only narrows this.
        """
        by_name: Dict[tuple, List[FuncDecl]] = {}
        for item in self.top_items:
            if isinstance(item, FuncDecl):
                by_name.setdefault((item.name, item.is_method), []).append(item)
        for (name, _), declarations in by_name.items():
            seen: List[tuple] = []
            for declaration in declarations:
                required = sum(1 for parameter in declaration.params if parameter.default is None)
                types = tuple((_type_word(parameter.type_hint), parameter.qualifier)
                              for parameter in declaration.params)
                untyped = any(word is None for word, _ in types)
                same = [earlier_types for earlier_required, earlier_types in seen if required == earlier_required
                        and (untyped or any(word is None for word, _ in earlier_types) or types == earlier_types)]
                if same:
                    typed = not untyped and all(all(word is not None for word, _ in earlier) for earlier in same)
                    message = (SAME_PARAMETERS_MESSAGE if typed else FUNCTION_REDEFINED_MESSAGE).format(name=name)
                    self.report(REFUSE, 'function-redefined', name, message, declaration)
                seen.append((required, types))

    def user_call_arguments(self, node: FunctionCall) -> None:
        """A call of a user-defined function names only its parameters, gives every required parameter a value and
        passes no more positional arguments than the function declares."""
        name = node.func
        if name in self.method_function_names or name not in self.function_names:
            return
        declarations = [item for item in self.top_items
                        if isinstance(item, FuncDecl) and item.name == name and not item.is_method]
        if not declarations:
            return
        parameter_names = {parameter.name for declaration in declarations for parameter in declaration.params}
        for key in list(node.kwargs) + list(node.repeated_kwargs):
            if key not in parameter_names:
                self.report(REFUSE, 'argument-name', key,
                            f'The "{name}" function does not have an argument with the name "{key}"', node)
        most = max(len(declaration.params) for declaration in declarations)
        if len(node.args) > most:
            self.report(REFUSE, 'argument-count', name,
                        f'Too many arguments passed into the "{name}()" function call. Passed {len(node.args)} '
                        f'arguments but expected {most}.', node)
        if len(declarations) == 1:
            for index, parameter in enumerate(declarations[0].params):
                if index >= len(node.args) and parameter.name not in node.kwargs and parameter.default is None:
                    self.report(REFUSE, 'argument-missing', parameter.name,
                                f'No value assigned to the "{parameter.name}" parameter in {name}()', node)

    def unknown_field(self, dotted: str, stack: List[_Scope], node) -> None:
        """`p.z` where `p` is a variable of a user-defined type that declares no field `z`; a longer path
        (`a.b.c`) is followed while the declared field types name user-defined types."""
        head, *path = dotted.split('.')
        if not path or not self.field_read_rule.is_field_read(
                Identifier(name=dotted), lambda name: self.binding_in_scope(name, stack)):
            return  # `position.top_right` beside a variable named `position` still reads the namespace
        type_name = self.variable_type(head, stack)
        for member in path:
            fields = self.udt_fields.get(_type_word(type_name) or '')
            if fields is None:
                return
            if member not in fields:
                self.report(REFUSE, 'unknown-field', member, f'Object has no field {member}', node)
                return
            type_name = fields[member]

    def inline_input_titles(self) -> None:
        """Inputs that share `inline` and `group` and each carry a title may not align in the Settings/Inputs tab:
        TradingView's compiler warns (CW10026) on every titled input of such a line after its first one. A warning
        only: the inputs are emitted as written."""
        titled: Dict[tuple, int] = {}
        for item in self.top_items:
            if not isinstance(item, InputDecl):
                continue
            inline = _literal_string(item.kwargs.get('inline'))
            if inline is None:
                continue
            title = _literal_string(item.kwargs['title'] if 'title' in item.kwargs
                                    else item.args[1] if len(item.args) > 1 else None)
            if not title:
                continue
            key = (inline, _literal_string(item.kwargs.get('group')) or '')
            titled[key] = titled.get(key, 0) + 1
            if titled[key] > 1:
                self.report(WARN, 'inline-input-titles', item.name,
                            f'The inputs with the `inline` argument "{inline}" might not align correctly in the '
                            f'"Settings/Inputs" tab. To ensure that inline inputs align consistently, specify a title '
                            f'for only the first input on each line, and use an empty title for all others.', item)

    def conditional_history_calls(self) -> None:
        """A `ta.*` call that runs on only some bars keeps a history that differs from the
        every-bar call. A warning only: the call stays where the script has it."""
        for call in conditional_history_calls(self.script):
            self.report(WARN, 'conditional-history-call', call.func,
                        f"'{call.func}' is called only when its condition holds: its series history "
                        f'differs from an unconditional call (the output keeps the call where it is, as TradingView '
                        f'does; TradingView advises calling it on every bar, in the global scope)', call)

    def block(self, statements, stack: List[_Scope], kind: str, names=(), how=LOOP_VARIABLE,
              types: Optional[Dict[str, Optional[str]]] = None, governed: bool = False) -> None:
        inner = stack + [self.new_scope(kind, statements if statements is not None else ())]
        inner[-1].governed_series = governed
        for name in names:
            if how != LOOP_COUNTER or len(names) == 1:  # the variables of a `for [a, b] in` tuple are left alone
                self.reserved_name(name)
            self.declare(inner, name, how, (types or {}).get(name))
            if (types or {}).get(name) == 'bool':
                inner[-1].bools.add(name)
        if isinstance(statements, list):
            for statement in statements:
                self.statement(statement, inner)
        elif statements is not None:
            self.expression(statements, inner)
        if kind != 'function':  # a Python def keeps its locals; an if/for block leaks them
            stack[-1].ended.update(inner[-1].names, inner[-1].ended)

    def member_of_history(self, node, member: str) -> None:
        """Refuse `x[n].member` (a field or a method) written without parentheses."""
        if (self.version >= MEMBER_OF_HISTORY_REFUSED_FROM_VERSION and isinstance(node.object, IndexAccess)
                and not node.object.parenthesized):
            self.report(REFUSE, 'member-of-history', member, MEMBER_OF_HISTORY_MESSAGE, node.object)

    def statement(self, node, stack: List[_Scope]) -> None:
        self.at(node)
        if isinstance(node, FuncDecl):
            self.function(node, stack)
        elif isinstance(node, (VarDecl, VaripDecl, Assignment)):
            enclosing = self.declaring
            self.declaring = enclosing | frozenset(_declared_names(node))
            self.expression(node.value, stack)
            self.declaring = enclosing
            self.at(node)  # a block in the value moved the position
            if isinstance(getattr(node, 'target', None), TupleDestructure):
                right = self.tuple_width(node.value)
                if right is not None and right != len(node.target.names):
                    self.report(REFUSE, 'tuple-width', ', '.join(node.target.names),
                                TUPLE_WIDTH_MESSAGE.format(right=right, left=len(node.target.names)))
            elif isinstance(node.value, FunctionCall) and (self.tuple_width(node.value) or 0) > 1:
                self.report(REFUSE, 'tuple-to-scalar', _declared_names(node)[0],
                            TUPLE_TO_SCALAR_MESSAGE.format(_declared_names(node)[0]))
            self.void_value(node)
            for name in _declared_names(node):
                if not isinstance(getattr(node, 'target', None), TupleDestructure):
                    self.reserved_name(name)
                first = name not in stack[-1].names
                type_name = declared_type(node, self.type_evidence._replace(
                    variable_type=lambda variable: self.variable_type(variable, stack)))
                self.declare(stack, name, VARIABLE, type_name)
                if first and not isinstance(getattr(node, 'target', None), TupleDestructure) and (
                        type_name == 'bool' or (type_name is None and self.is_bool(node.value, stack))):
                    stack[-1].bools.add(name)
            self.bool_declared_na(node)
            self.block_value_types(node.value)
            if not isinstance(getattr(node, 'target', None), TupleDestructure):
                self.declared_value_type(node)
                self.drawing_binding(node, stack)
                element = _array_element_type(node.value)
                if element is not None:
                    self.array_elements[_declared_names(node)[0]] = element
                pair = _map_types(node.value)
                if pair is not None:
                    self.map_types[_declared_names(node)[0]] = pair
                self.declared_form(node, stack)
                self.remember_const(node, stack)
                if node.type_hint in ('int', 'float', 'string'):
                    stack[-1].explicit_types[_declared_names(node)[0]] = node.type_hint
                self.block_value_default(node.value, stack)
            if (self.version >= UNTYPED_NA_REFUSED_FROM_VERSION and node.type_hint is None
                    and _is_na(node.value)
                    and not isinstance(getattr(node, 'target', None), TupleDestructure)):
                self.report(REFUSE, 'untyped-na', _declared_names(node)[0], UNTYPED_NA_MESSAGE)
        elif isinstance(node, InputDecl):
            self.expression([node.args, list(node.kwargs.values())], stack)
            self.input_arguments(node, stack)
            self.reserved_name(node.name)
            self.declare(stack, node.name, INPUT, declared_type(node))
            if _is_a_bool_input(node):
                stack[-1].bools.add(node.name)
        elif isinstance(node, EnumDecl):
            if node.name in RESERVED_ENUM_NAMES:
                self.report(REFUSE, 'named-like-namespace', node.name,
                            f"an enum cannot be named '{node.name}', the name of a built-in namespace")
            seen_members: set = set()
            for member in node.members:
                member_name = member[0] if isinstance(member, tuple) else member
                if member_name in seen_members:
                    self.report(REFUSE, 'duplicate-field', member_name, f'Duplicated field: "{member_name}".', node)
                seen_members.add(member_name)
        elif isinstance(node, TypeDecl):
            seen_fields: set = set()
            for field_name, *_ in node.fields:
                if field_name in seen_fields:
                    self.report(REFUSE, 'duplicate-field', field_name, f"Duplicated field: '{field_name}'.", node)
                seen_fields.add(field_name)
            for field_name, default in invalid_defaults(node, self.constant_heads):
                self.report(REFUSE, 'udt-field-default', field_name, default_message(node, field_name),
                            positioned_part(default))
        elif isinstance(node, Reassignment):
            self.expression(node.value, stack)
            self.at(node)  # a block in the value moved the position
            self.reassignment(node.target, stack)
            self.reassigned_value_type(node, stack)
            self.reassigned_form(node, stack)
            if isinstance(node.target, str):
                self.forget_const(node.target, stack)
            self.bool_reassigned_na(node, stack)
            self.block_value_types(node.value)
        elif isinstance(node, (IfStatement, ForLoop, ForInLoop, WhileLoop)):
            self.control_flow(node, stack)
        elif isinstance(node, SwitchStatement):
            # a switch answers CE10205 under the `if` keyword: TradingView compiles it into if-chains
            self.statement_size(node, [body for _, body in node.cases] + [node.default], 'switch', stack)
            self.switch(node, stack)
        elif isinstance(node, (BreakStatement, ContinueStatement)):
            if not any(scope.kind in LOOP_SCOPE_KINDS for scope in self.own_scopes(stack)):
                word = 'break' if isinstance(node, BreakStatement) else 'continue'
                self.report(REFUSE, 'break-outside-loop', word, f'"{word}" is only allowed inside loops.', node)
        elif isinstance(node, (ExpressionStatement, ReturnStatement)):
            self.expression(node.expr, stack)
        elif isinstance(node, Expression):
            self.expression(node, stack)

    def input_arguments(self, node: InputDecl, stack: List[_Scope]) -> None:
        """An input's arguments must be const, so they cannot read another input (manual,
        Inputs: "results from other input*() calls"); `active` takes an input bool, and
        input.source's `defval` a series float."""
        parameters = ['defval', 'title'] + input_positional_param_names(node.func, node.args)
        # a row that does not name `active` ends at `display`: a positional argument after it is `active`
        positional = (parameters[:parameters.index('display') + 1]
                      if 'display' in parameters and 'active' not in parameters else parameters)
        arguments = list(zip(positional, node.args)) + list(node.kwargs.items())
        if node.func == 'input.source' and self.version >= INPUT_SOURCE_CHECKED_FROM_VERSION:
            defval = dict(arguments).get('defval')
            if defval is not None and not (isinstance(defval, Identifier) and defval.name in INPUT_SOURCE_VALUES):
                self.report(REFUSE, 'input-source-defval', node.name,
                            INPUT_SOURCE_V4_MESSAGE if self.version == 4 else INPUT_SOURCE_MESSAGE, _positioned(defval))
        defval = dict(arguments).get('defval')
        self.input_bounds(node, dict(arguments))
        if node.func == 'input.int' and self.version >= REMOVED_PARAMETERS_FROM_VERSION:
            # MEASURED on TradingView 2026-11-15: input.int(5, "t", step = 0) is CE10040; a float input steps by
            # a fraction, so only the int form is checked
            step = _literal_number(dict(arguments).get('step'))
            if step is not None and step < 1:
                self.report(REFUSE, 'input-step', node.name,
                            'The "input.int()" function cannot have a "step" value less than 1. Use a value of 1 or '
                            'greater in the function call.', node)
        if _not_of_enumeration(dict(arguments).get('display'), INPUT_DISPLAY_VALUES):
            self.report(REFUSE, 'enum-argument', node.func,
                        ENUM_ARGUMENT_MESSAGE.format(parameter='display', name=node.func,
                                                     values=', '.join(INPUT_DISPLAY_VALUES)), node)
        declared = INPUT_VALUE_TYPES.get(node.func)
        # the v4 `input()` takes its type from the default, so only the named forms are checked
        if declared in _FAMILY and node.func not in ('input', 'input.source'):
            # measured 2026-11-15: input.int("a") is CE10123
            kind = _value_kind(defval)
            if kind is not None and _bool_number(kind[1], declared) \
                    and self.version < REMOVED_PARAMETERS_FROM_VERSION:
                kind = None  # before v6 a number stands in for a bool (`input.bool(0)` of a v5 script)
            if kind is not None and _FAMILY[kind[1]] != _FAMILY[declared]:
                self.report(REFUSE, 'argument-type', 'defval',
                            f'Cannot call "{node.func}" with argument "defval"="{kind[2]}". An argument of '
                            f'"{kind[0]} {kind[1]}" type was used but a "const {declared}" is expected.',
                            _positioned(defval))
        if isinstance(defval, FunctionCall) and isinstance(defval.func, str) and defval.func in INPUT_VALUE_TYPES:
            # an input's default is a constant: the value of another input is not
            inner, outer = INPUT_VALUE_TYPES[defval.func], INPUT_VALUE_TYPES.get(node.func, 'int')
            self.report(REFUSE, 'input-defval', node.name,
                        f'Cannot call "{node.func}" with argument "defval"="call "{defval.func}" (input {inner})". '
                        f'An argument of "input {inner}" type was used but a "const {outer}" is expected.',
                        _positioned(defval))
        for parameter, argument in arguments:
            if parameter == 'active' or (parameter == 'defval' and node.func == 'input.source'):
                continue
            for read in walk(argument):
                if isinstance(read, Identifier) and self.depth_of(stack, read.name) == 0 \
                        and stack[0].names[read.name] == INPUT:
                    self.report(REFUSE, 'input-argument-reads-input', read.name,
                                f"the '{parameter}' argument of input '{node.name}' reads the input "
                                f"'{read.name}'; TradingView requires a const argument there", read)

    def input_bounds(self, node: InputDecl, arguments: Dict[str, object]) -> None:
        """An input's default must lie between its `minval` and `maxval` and be one of its `options`, as the
        compiler checks when all three are literals."""
        if self.version < REMOVED_PARAMETERS_FROM_VERSION:
            return
        defval = _literal_number(arguments.get('defval'))
        if defval is not None:
            low, high = _literal_number(arguments.get('minval')), _literal_number(arguments.get('maxval'))
            if (low is not None and defval < low) or (high is not None and defval > high):
                self.report(REFUSE, 'input-bounds', node.name, INPUT_MINVAL_MESSAGE, node)
                return
        options = arguments.get('options')
        if not isinstance(options, ArrayLiteral) or not options.elements:
            return
        values = [_literal_number(element) if _literal_number(element) is not None else _literal_string(element)
                  for element in options.elements]
        if any(value is None for value in values):
            return
        given = defval if defval is not None else _literal_string(arguments.get('defval'))
        if given is None or given in values:
            return
        shown = _shown_number(given) if not isinstance(given, str) else f'"{given}"'
        listed = ', '.join(_shown_number(value) if not isinstance(value, str) else f'"{value}"' for value in values)
        self.report(REFUSE, 'input-options', node.name,
                    INPUT_OPTIONS_MESSAGE.format(shown=shown, options=listed), node)

    def control_flow(self, node, stack: List[_Scope]) -> None:
        if isinstance(node, IfStatement):
            self.expression(node.condition, stack)
            self.if_condition(node.condition, stack)
            self.statement_size(node, [node.body, node.else_body] + [body for _, body in node.elseifs],
                                'if', stack)
            governed = self.is_series(node.condition, stack)
            self.block(node.body, stack, 'if', governed=governed)
            for condition, body in node.elseifs:
                self.at_expression(condition)
                self.expression(condition, stack)
                self.if_condition(condition, stack)
                governed = governed or self.is_series(condition, stack)  # an arm runs when the ones above did not
                self.block(body, stack, 'if', governed=governed)
            self.block(node.else_body, stack, 'if', governed=governed)
        elif isinstance(node, ForLoop):
            self.expression([node.from_val, node.to_val, node.step], stack)
            self.statement_size(node, [node.body], 'for', stack)
            self.at(node)
            if _literal_of(node.from_val, 'int') and _literal_of(node.step, 'float'):
                self.report(REFUSE, 'for-step-type', node.var, FOR_STEP_TYPE_MESSAGE.format(node.var))
            self.block(node.body, stack, 'for', [node.var], LOOP_COUNTER, governed=True)
        elif isinstance(node, ForInLoop):
            self.expression(node.iterable, stack)
            self.statement_size(node, [node.body], 'for', stack)
            self.for_in_iterable(node, stack)
            self.block(node.body, stack, 'for', node.vars, LOOP_COUNTER, governed=True)
        else:
            self.expression(node.condition, stack)
            self.statement_size(node, [node.body], 'while', stack)
            self.block(node.body, stack, 'while', governed=True)

    def for_in_iterable(self, node: ForInLoop, stack: List[_Scope]) -> None:
        """`for ... in` walks an array, a matrix or a map. A value that is certainly a scalar - a literal, a price
        built-in, a variable declared of a primitive type - is refused, as TradingView refuses one."""
        kind = _value_kind(node.iterable)
        if kind is not None:
            shown, type_text = kind[2], _expected_of(kind)
        elif isinstance(node.iterable, Identifier):
            word = _type_word(self.variable_type(node.iterable.name, stack))
            if word not in SCALAR_TYPE_WORDS:
                return
            shown, type_text = node.iterable.name, word
        else:
            return
        self.at(node)
        self.report(REFUSE, 'for-in-collection', shown,
                    FOR_IN_COLLECTION_MESSAGE.format(shown=shown, type=type_text), node)

    def function(self, node: FuncDecl, stack: List[_Scope]) -> None:
        self.reserved_name(node.name)
        self.declared_functions.add(node.name)
        if node.is_method and node.params and node.params[0].type_hint is None:
            self.report(REFUSE, 'method-receiver-type', node.params[0].name, METHOD_RECEIVER_TYPE_MESSAGE, node)
        if len(stack) > 1:
            self.report(REFUSE, 'nested-function', node.name, f"function '{node.name}' is declared inside "
                        f'{SCOPE_KIND_TEXT[stack[-1].kind]}; TradingView allows a function declaration only '
                        f'at global scope')
        self.expression([parameter.default for parameter in node.params], stack)
        self.block(node.body, stack, 'function', [parameter.name for parameter in node.params], PARAMETER,
                   {parameter.name: declared_type(parameter) for parameter in node.params})
        self.method_receiver_read(node)

    def method_receiver_read(self, node: FuncDecl) -> None:
        """A method whose receiver (its first parameter) is of a reference type - a collection, a user-defined
        type, a drawing - must read it; a primitive receiver (int, float, string, ...) may go unused."""
        if not node.is_method or not node.params:
            return
        receiver = node.params[0]
        if self.is_reference_type(receiver.type_hint) and receiver.name not in _names_read(node.body):
            self.report(REFUSE, 'method-receiver-unused', receiver.name,
                        f'The "{receiver.name}" argument is not used. The first method argument should be used.')

    def is_reference_type(self, type_hint: Optional[str]) -> bool:
        if not type_hint:
            return False
        name = type_hint.split()[-1]  # without a form keyword
        return (name.split('<', 1)[0] in {'array', 'matrix', 'map'} or name in REFERENCE_BUILTIN_TYPES
                or name in self.udt_names)

    def reassignment(self, target: str, stack: List[_Scope]) -> None:
        if '.' in target:  # a field: `p.v := x` is valid anywhere
            self.unknown_field(target, stack, None)
            return
        depth = self.depth_of(stack, target)
        functions = [index for index, scope in enumerate(stack) if scope.kind == 'function']
        if depth is None:
            if self.out_of_scope(stack, target):
                self.report(REFUSE, 'out-of-scope', target, UNDECLARED_MESSAGE.format(target))
            elif functions and target in self.global_names:
                self.report(REFUSE, 'global-modified-in-function', target,
                            f"a function cannot modify the global variable '{target}'")
            else:
                self.report(REFUSE, 'reassigned-undeclared', target, f"'{target}' is not declared before "
                            f"this ':='; TradingView refuses a reassignment of an undeclared variable")
        elif functions and depth < functions[-1]:
            self.report(REFUSE, 'global-modified-in-function', target,
                        f"a function cannot modify the {'global' if depth == 0 else 'enclosing'} variable '{target}'")
        elif stack[depth].names.get(target) == LOOP_COUNTER:
            self.report(REFUSE, 'loop-variable-mutable', target, LOOP_VARIABLE_MUTABLE_MESSAGE.format(target))
        elif stack[depth].names.get(target) == PARAMETER:
            self.report(REFUSE, 'parameter-modified', target, f"a function cannot modify its parameter '{target}'")

    def switch(self, node, stack: List[_Scope]) -> None:
        self.expression(node.expr, stack)
        governed = self.is_series(node.expr, stack) or any(self.is_series(case, stack) for case, _ in node.cases)
        for case, body in node.cases:
            self.at_expression(case)
            self.expression(case, stack)
            self.block(body, stack, 'switch', governed=governed)
        self.block(node.default, stack, 'switch', governed=governed)

    # ------------------------------------------------------------ expressions

    def expression(self, node, stack: List[_Scope]) -> None:
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                self.expression(item, stack)
        elif isinstance(node, dict):
            self.expression(list(node.values()), stack)
        elif isinstance(node, Identifier):
            if '.' not in node.name:  # `N.m` names a namespace, a type or a library unless N is in scope
                self.read(stack, node)
            else:
                self.unknown_field(node.name, stack, node)
                self.strategy_in_indicator(node.name, node)
        elif isinstance(node, IndexAccess):
            if isinstance(node.object, IndexAccess):
                self.report(REFUSE, 'double-history', '', 'the [] history operator can be used only once on '
                            'the same value (close[1][2])', node)
            elif self.version >= REMOVED_PARAMETERS_FROM_VERSION and isinstance(node.object, Literal):
                self.report(REFUSE, 'history-on-literal', '',
                            'The history-referencing operator can not be used on literals or built-in constants.', node)
            elif self.version >= FIELD_HISTORY_REFUSED_FROM_VERSION and self.field_read_rule.is_field_read(
                    node.object, lambda name: self.binding_in_scope(name, stack)):
                self.report(REFUSE, 'history-on-udt-field', '', self.field_history_message(node), node)
            elif self.version >= LOOP_COUNTER_HISTORY_REFUSED_FROM_VERSION and \
                    (counter := self.loop_counter_in(node.object, stack)) is not None:
                self.report(REFUSE, 'loop-counter-history', counter.name,
                            f'Variable "{counter.name}" doesn\'t have history values', counter)
            self.expression([node.object, node.index], stack)
        elif isinstance(node, FunctionCall):
            if not isinstance(node.func, str):
                self.expression(node.func, stack)
            elif node.func in self.function_names and node.func not in self.declared_functions:
                # a user-defined function is known from its declaration on: a call above it does not compile
                self.report(REFUSE, 'function-before-declaration', node.func,
                            f"Could not find function or function reference '{node.func}'", node)
            self.expression([node.args, list(node.kwargs.values())], stack)
            self.na_of_a_bool(node, stack)
            self.call_arguments(node, stack)
            if isinstance(node.func, str):
                self.user_call_arguments(node)
        elif isinstance(node, MethodCall):
            self.member_of_history(node, node.method)
            self.expression([node.object, node.args, list(node.kwargs.values())], stack)
        elif isinstance(node, MemberAccess):
            self.member_of_history(node, node.member)
            if not isinstance(node.object, str):
                self.expression(node.object, stack)
        elif isinstance(node, IfExpression):
            self.expression(node.condition, stack)
            self.if_condition(node.condition, stack)
            governed = self.is_series(node.condition, stack)
            self.block(node.true_expr, stack, 'if', governed=governed)
            self.block(node.false_expr, stack, 'if', governed=governed)
            if not node.chained:
                self.na_block_of_a_bool(node, stack)
        elif isinstance(node, TernaryOp):
            self.expression([node.condition, node.true_expr, node.false_expr], stack)
            self.na_arm_of_a_bool(node, stack)
            self.ternary_arm_types(node)
            self.comparison_types(node, stack)
        elif isinstance(node, SwitchExpression):
            self.switch(node, stack)
            self.na_block_of_a_bool(node, stack)
        elif isinstance(node, (BinaryOp, UnaryOp)):
            operands = [node.operand] if isinstance(node, UnaryOp) else [node.left, node.right]
            for operand in operands:
                if isinstance(operand, SwitchExpression) and not getattr(node, 'compound', False):
                    self.report(REFUSE, 'switch-operand', 'switch', SWITCH_OPERAND_MESSAGE, operand)
            self.expression(operands, stack)
            self.operand_types(node)
            self.comparison_types(node, stack)
        elif isinstance(node, STATEMENT_TYPES):
            self.statement(node, stack)
        elif dataclasses.is_dataclass(node):
            self.expression([getattr(node, field.name) for field in dataclasses.fields(node)
                             if field.name not in ('line', 'column')], stack)

    # ------------------------------------------------------------ the arms of a block value agree in type

    def call_arguments(self, node: FunctionCall, stack: List[_Scope]) -> None:
        """The argument rules of Pine v6 the compiler enforces: a keyword given twice, a parameter v6 removed
        (`transp`, `when`), a series `offset` and a `linewidth` below 1 of a plot-family call; and, with
        `dynamic_requests = false`, a request.*() call inside a loop or conditional block."""
        if not isinstance(node.func, str):
            return
        name = node.func
        for (func, parameter, position), (low, high) in CALL_ARGUMENT_RANGES.items():
            if name != func:
                continue
            given = node.kwargs.get(parameter)
            if given is None and len(node.args) > position:
                given = node.args[position]
            # the range is checked after constant folding, as TradingView checks it (`folded_number`)
            value = self.folded_number(given, stack)
            if value is not None and not low <= value <= high:
                self.report(REFUSE, 'argument-range', parameter,
                            DECLARATION_RANGE_MESSAGE.format(value=_shown_number(value), parameter=parameter,
                                                             call=name, low=low, high=high), node)
        if name == 'strategy.exit' and self.version >= STRATEGY_EXIT_EFFECT_FROM_VERSION:
            given = set(node.kwargs) | set(STRATEGY_EXIT_PARAMETERS[:len(node.args)])
            if not given & STRATEGY_EXIT_EFFECTS:
                self.report(REFUSE, 'exit-without-effect', name, STRATEGY_EXIT_MESSAGE, node)
        if self.version >= REMOVED_PARAMETERS_FROM_VERSION:
            for key in node.repeated_kwargs:
                self.report(REFUSE, 'repeated-parameter', key,
                            f'Two or more arguments are passed to the "{key}" parameter. You can pass only one argument.',
                            node)
            for key, functions in REMOVED_PARAMETERS.items():
                if key in node.kwargs and name in functions:
                    self.report(REFUSE, 'removed-parameter', key,
                                f'The "{name}" function does not have an argument with the name "{key}"', node)
            if name in PLOT_FAMILY:
                offset = node.kwargs.get('offset')
                if offset is not None and self.is_series(offset, stack):
                    self.report(REFUSE, 'offset-form', name,
                                f'Cannot call "{name}" with argument "offset". An argument of "series int" type was '
                                f'used but a "simple int" is expected.', node)
                linewidth = node.kwargs.get('linewidth')
                if isinstance(linewidth, Literal) and linewidth.literal_type in ('int', 'float') and linewidth.value < 1:
                    self.report(REFUSE, 'linewidth-range', name,
                                f'The "{name}()" function cannot have a "linewidth" value less than 1. Use a value of 1 '
                                f'or greater in the function call.', node)
            self.strict_parameters(node, stack)
        if name == 'max_bars_back':
            num = node.kwargs.get('num', node.args[1] if len(node.args) > 1 else None)
            if isinstance(num, Literal) and num.literal_type in ('int', 'float') and not 0 <= num.value <= MAX_BARS_BACK_LIMIT:
                self.report(REFUSE, 'max-bars-back-range', name,
                            f'Invalid value "{num.value}" for "num" parameter of the "max_bars_back()" function. It must '
                            f'be between 0 and {MAX_BARS_BACK_LIMIT}', node)
        if name == 'str.substring':
            self.substring_positions(node)
        if name == 'str.repeat' and self.version >= REMOVED_PARAMETERS_FROM_VERSION:
            # MEASURED on TradingView 2026-11-15: str.repeat("a", -1) is CE10039
            count = _literal_number(node.kwargs.get('repeat', node.args[1] if len(node.args) > 1 else None))
            if count is not None and count < 0:
                self.report(REFUSE, 'repeat-negative', name,
                            'The "str.repeat()" function cannot have a negative "repeat" value. Use a positive '
                            'value in the function call.', node)
            # MEASURED on TradingView 2026-11-19: a repeat that would build a string longer than 40960
            # characters is CE10146, and the message names the length it would have had. The check is on the
            # produced constant, so it only fires when both the text and the count are literals; a separator
            # would make the string longer still and is not counted here
            source = node.kwargs.get('source', node.args[0] if node.args else None)
            if (count is not None and count > 0 and isinstance(source, Literal)
                    and getattr(source, 'literal_type', None) == 'string'):
                produced = len(str(source.value)) * int(count)
                if produced > STRING_LENGTH_LIMIT:
                    self.report(REFUSE, 'string-too-long', name,
                                f'String is too long ({produced}), max allowed length is '
                                f'{STRING_LENGTH_LIMIT}.', node)
        if name == 'strategy.exit' and self.version >= REMOVED_PARAMETERS_FROM_VERSION:
            # MEASURED on TradingView 2026-11-18: strategy.exit(..., qty_percent = 150) is CE10041, "Invalid
            # value "150" for "qty_percent" parameter of the "strategy.exit()" function. It must be between 0
            # and 100", and a percentage of -50 is the same error with its own number, so both sides are
            # measured. Zero stays allowed here on purpose: it compiles and stops the script at run time
            # instead (RE10025), the way a zero position-size cap does
            percent = _literal_number(node.kwargs.get('qty_percent'))
            if percent is not None and not 0 <= percent <= 100:
                self.report(REFUSE, 'qty-percent-range', name,
                            f'Invalid value "{_shown_number(percent)}" for "qty_percent" parameter of the '
                            f'"strategy.exit()" function. It must be between 0 and 100', node)
        if name in ('strategy.entry', 'strategy.close', 'strategy.order') and self.version >= REMOVED_PARAMETERS_FROM_VERSION:
            # MEASURED on TradingView 2026-11-18: a negative "qty" is CE10039 for all three of these, each message
            # naming its own function; a qty of 0, and a qty below the symbol's minimum order size, compile and
            # simply place no order. strategy.exit is the exception: a negative quantity compiles there and the
            # exit closes its absolute value, so that call is left alone
            quantity = _literal_number(node.kwargs.get('qty', node.args[2] if len(node.args) > 2 else None))
            if quantity is not None and quantity < 0:
                self.report(REFUSE, 'order-qty-negative', name,
                            f'The "{name}()" function cannot have a negative "qty" value. Use a positive value '
                            f'in the function call.', node)
        if name == 'strategy.risk.max_position_size' and self.version >= REMOVED_PARAMETERS_FROM_VERSION:
            # MEASURED on TradingView 2026-11-18: strategy.risk.max_position_size(-1) is CE10039, the same wording
            # the str.repeat rule above carries; a cap of 0 compiles and stops the script at run time instead
            contracts = _literal_number(node.kwargs.get('contracts', node.args[0] if node.args else None))
            if contracts is not None and contracts < 0:
                self.report(REFUSE, 'position-size-negative', name,
                            'The "strategy.risk.max_position_size()" function cannot have a negative "contracts" '
                            'value. Use a positive value in the function call.', node)
        if name == 'color.new' and self.version >= REMOVED_PARAMETERS_FROM_VERSION:
            # MEASURED on TradingView 2026-11-15: color.new(color.red, 150) and a transparency of -5 are CE10274
            transp = node.kwargs.get('transp', node.args[1] if len(node.args) > 1 else None)
            # MEASURED 2026-11-18: a transparency the compiler can fold is checked too (`folded_number`)
            value = self.folded_number(transp, stack)
            if value is not None and not 0 <= value <= 100:
                self.report(REFUSE, 'transparency-range', name, TRANSPARENCY_MESSAGE, node)
        if name == 'array.from' and self.version >= COLLECTION_AS_ELEMENT_FROM_VERSION:
            for position, argument in enumerate(node.args):
                if not isinstance(argument, Identifier):
                    continue
                written = self.variable_type(argument.name, stack)
                if written and written.split('<')[0] in COLLECTION_TYPE_WORDS:
                    self.report(REFUSE, 'collection-as-element', argument.name,
                                COLLECTION_AS_ELEMENT_MESSAGE.format(
                                    q=MESSAGE_QUOTE.get(self.version, '"'), position=position,
                                    name=argument.name, actual=written), node)
                    break
        if name == 'plot' and self.version >= BOOL_NUMBER_MISMATCH_FROM_VERSION:
            # MEASURED on TradingView 2026-11-15: plot() of a bool is CE10123; plotshape() and plotchar() take one
            series = node.kwargs.get('series', node.args[0] if node.args else None)
            if series is not None and self.is_bool(series, stack):
                self.report(REFUSE, 'plot-series-type', name,
                            PLOT_SERIES_MESSAGE.format(shown=_shown_bool(series)), node)
        for parameter, expected in ((p, e) for (f, p), e in STRING_ARGUMENTS.items() if f == name):
            written = _constant_name(node.kwargs.get(parameter))
            actual = NON_STRING_CONSTANTS.get(written or '')
            if actual is not None:
                self.report(REFUSE, 'argument-type', name,
                            ARGUMENT_TYPE_MESSAGE.format(name=name, parameter=parameter, written=written,
                                                         actual=actual, expected=expected), node)
        for parameter, allowed in ((p, v) for (f, p), v in ENUM_ARGUMENTS.items() if f == name):
            if _not_of_enumeration(node.kwargs.get(parameter), allowed):
                self.report(REFUSE, 'enum-argument', name,
                            ENUM_ARGUMENT_MESSAGE.format(parameter=parameter, name=name,
                                                         values=', '.join(allowed)), node)
        if name == 'fill':
            self.fill_pair(node, stack)
        if name == 'timestamp' and len(node.args) == 1 and not node.kwargs:
            text = _literal_string(node.args[0])
            if text is not None and TIMESTAMP_UNREADABLE.search(text.strip()):
                self.report(REFUSE, 'timestamp-format', name, TIMESTAMP_FORMAT_MESSAGE, node)
        self.argument_base_types(node)
        self.namespace_member(node, stack)
        self.strategy_in_indicator(name, node)
        if name in GLOBAL_ONLY_CALLS and len(stack) > 1:
            self.report(REFUSE, 'local-scope', name, f'Cannot use "{name}" in local scope', node)
        if self.static_requests and name.startswith('request.') and any(
                scope.kind in CONDITIONAL_SCOPE_KINDS for scope in stack[1:]):
            self.report(REFUSE, 'static-request', name,
                        'Cannot use "request.*()" call within loops or conditional structures, when "dynamic_requests" '
                        'is "false" in the "indicator()" function.', node)

    def comparison_types(self, node, stack: List[_Scope]) -> None:
        """A value of a type an operator has no overload for is refused where that operator reads it. Only a
        name declared with such a type keyword is refused: anything else could hold a value the operator takes.
        Equality and order part ways here: a line and a label compare for equality but not for order."""
        if isinstance(node, TernaryOp):
            # only the condition: the arms carry the value and may legitimately be of any type
            return self.operand_of_operator(node.condition, TERNARY_OPERATOR, TERNARY_CONDITION_TYPE, node, stack)
        if isinstance(node, UnaryOp):
            expected = UNARY_OPERATOR_EXPECTED_TYPES.get(node.op)
            operands = (node.operand,)
        elif isinstance(node, BinaryOp):
            expected = OPERATOR_EXPECTED_TYPES.get(node.op)
            operands = (node.left, node.right)
        else:
            return
        if expected is None:
            return
        for operand in operands:
            if self.operand_of_operator(operand, node.op, expected, node, stack):
                return

    def operand_of_operator(self, operand, op: str, expected: str, node, stack: List[_Scope]) -> bool:
        """Refuse one operand an operator has no overload for, and say whether it was refused. Only the
        binding the name actually reads here is looked at: an inner declaration of the same name hides an
        outer one of another type, and a published library does exactly that (a string local over an array
        parameter), so reading any enclosing scope would refuse a script TradingView compiles."""
        if not isinstance(operand, Identifier) or '.' in operand.name:
            return False
        depth = self.depth_of(stack, operand.name)
        if depth is None:
            return False
        actual = self.uncomparable_type(stack[depth].types.get(operand.name) or '', op)
        if actual is None:
            return False
        self.report(REFUSE, 'comparison-type', f'operator {op}',
                    COMPARISON_TYPE_MESSAGE.format(op=op, written=operand.name, actual=actual,
                                                   expected=expected), node)
        return True

    def uncomparable_type(self, declared: str, op: str) -> Optional[str]:
        """How the compiler names a type this operator has no overload for, or None for a type it takes. A
        collection and an object of a user type are written as the script declares them; the drawing types
        carry the `series` qualifier. Only `==` takes a line or a label: for order and for arithmetic no type
        here is a number, so they are refused as well."""
        if declared in UNCOMPARABLE_TYPES:
            return UNCOMPARABLE_TYPES[declared]
        if op not in ('==', '!=') and declared in ORDER_ONLY_UNCOMPARABLE:
            return ORDER_ONLY_UNCOMPARABLE[declared]
        if declared.startswith(COLLECTION_TYPE_PREFIXES) or declared in self.udt_names:
            return declared
        return None

    def operand_types(self, node) -> None:
        """An operand whose type is certain (a literal, a price built-in) where an operator needs another kind is
        refused as the compiler refuses it: a string or bool in arithmetic, a comparison of a string with a number,
        `not` on a number; and a comparison with the na literal, which only na() can make."""
        if isinstance(node, UnaryOp):
            if node.op == 'not' and self.version >= BOOL_NUMBER_MISMATCH_FROM_VERSION:
                kind = _value_kind(node.operand)
                if kind is not None and kind[1] != 'bool':
                    self.report(REFUSE, 'operand-types', node.op, OPERATOR_ARGUMENT_MESSAGE.format(
                        op='not', slot='expr0', shown=kind[2], used=f'{kind[0]} {kind[1]}', expected='simple bool'), node)
            return
        if node.op in COMPARISON_OPERATORS and (_is_na(node.left) or _is_na(node.right)):
            if self.version >= NA_COMPARISON_REFUSED_FROM_VERSION:
                self.report(REFUSE, 'na-comparison', node.op, NA_COMPARISON_MESSAGE, node)
            return
        left, right = _value_kind(node.left), _value_kind(node.right)
        if node.op in ORDERING_OPERATORS and self.version >= BOOL_NUMBER_MISMATCH_FROM_VERSION:
            # MEASURED on TradingView 2026-11-15: `"a" < "b"` is CE10123, the operator takes numbers only
            for slot, kind in (('expr0', left), ('expr1', right)):
                if kind is not None and _FAMILY[kind[1]] != 'number':
                    self.report(REFUSE, 'operand-types', node.op, OPERATOR_ARGUMENT_MESSAGE.format(
                        op=node.op, slot=slot, shown=kind[2], used=f'{kind[0]} {kind[1]}',
                        expected='simple float'), node)
                    return
        if left is None or right is None or _FAMILY[left[1]] == _FAMILY[right[1]]:
            return
        if _bool_number(left[1], right[1]) and self.version < BOOL_NUMBER_MISMATCH_FROM_VERSION:
            return
        if node.op in ARITHMETIC_OPERATORS:
            # the number is the operator's own kind: the other operand is the one reported
            if _FAMILY[left[1]] == 'number':
                slot, wrong, reference = 'expr1', right, left
            elif _FAMILY[right[1]] == 'number':
                slot, wrong, reference = 'expr0', left, right
            else:
                return
        elif node.op in COMPARISON_OPERATORS:
            slot, wrong, reference = 'expr1', right, left  # the first operand sets the kind
        else:
            return
        self.report(REFUSE, 'operand-types', node.op, OPERATOR_ARGUMENT_MESSAGE.format(
            op=node.op, slot=slot, shown=wrong[2], used=f'{wrong[0]} {wrong[1]}', expected=_expected_of(reference)), node)

    def ternary_arm_types(self, node: TernaryOp) -> None:
        """Ternary arms of certain and different kinds (a number against a string or a bool) are refused: the
        first arm sets the kind the second must have."""
        first, second = _value_kind(node.true_expr), _value_kind(node.false_expr)
        if first is None or second is None or _FAMILY[first[1]] == _FAMILY[second[1]]:
            return
        if _bool_number(first[1], second[1]) and self.version < BOOL_NUMBER_MISMATCH_FROM_VERSION:
            return
        qualifier = 'series' if first[0] == 'series' else 'simple'
        self.report(REFUSE, 'ternary-arm-types', '?:', OPERATOR_ARGUMENT_MESSAGE.format(
            op='?:', slot='expr2', shown=second[2], used=f'{second[0]} {second[1]}', expected=f'{qualifier} {first[1]}'),
            node)

    def drawing_binding(self, node, stack: List[_Scope]) -> None:
        """Remember that a global name holds the id of a plot or of an hline: the two are not interchangeable.

        Only a global binding is remembered, and only the first one, because a local name of the same spelling
        belongs to another variable and must not be read as this kind.
        """
        value = getattr(node, 'value', None)
        if len(stack) != 1 or not isinstance(value, FunctionCall) or value.func not in DRAWING_PAIR_CALLS:
            return
        names = _declared_names(node)
        if len(names) == 1:
            self.drawing_kind.setdefault(names[0], value.func)

    def fill_pair(self, node: FunctionCall, stack: List[_Scope]) -> None:
        """`fill()` fills between two plots or between two hlines, never between one of each."""
        pair = (node.kwargs.get('plot1', node.kwargs.get('hline1')),
                node.kwargs.get('plot2', node.kwargs.get('hline2')))
        positional = [argument for argument in node.args[:2]]
        given = [positional[i] if i < len(positional) else pair[i] for i in (0, 1)]
        kinds = [self.drawing_kind_of(argument, stack) for argument in given]
        if None in kinds or kinds[0] == kinds[1]:
            return
        position = 1 if kinds[0] == 'plot' else 2
        plot_argument = given[position - 1]
        self.report(REFUSE, 'fill-pair', 'fill',
                    FILL_PAIR_MESSAGE.format(position=position, name=plot_argument.name), node)

    def drawing_kind_of(self, argument, stack: List[_Scope]) -> Optional[str]:
        """The kind a name holds, read only where the name still binds the global declaration."""
        if not isinstance(argument, Identifier) or '.' in argument.name:
            return None
        if self.depth_of(stack, argument.name) not in (0, None):
            return None
        return self.drawing_kind.get(argument.name)

    def declared_value_type(self, node) -> None:
        """A typed declaration whose value is certainly of another kind, or a float where an int is declared, is
        refused: Pine converts an int to a float, nothing else without a cast."""
        hint = getattr(node, 'type_hint', None)
        kind = _value_kind(getattr(node, 'value', None))
        if hint not in _FAMILY or kind is None:
            return
        if _FAMILY[hint] == _FAMILY[kind[1]] and not (hint == 'int' and kind[1] == 'float'):
            return
        if _bool_number(hint, kind[1]) and self.version < BOOL_NUMBER_MISMATCH_FROM_VERSION:
            return
        name = _declared_names(node)[0]
        self.report(REFUSE, 'declared-type', name,
                    f'Cannot assign a value of the "{_expected_of(kind)}" type to the "{name}" variable. The variable '
                    f'is declared with the "const {hint}" type.', node)

    def reassigned_value_type(self, node: Reassignment, stack: List[_Scope]) -> None:
        """A `:=` of a value certainly of another kind than the variable's declared or first-assigned type."""
        kind = _value_kind(node.value)
        declared = self.variable_type(node.target, stack) if '.' not in node.target else None
        if kind is None or declared not in _FAMILY or _FAMILY[declared] == _FAMILY[kind[1]]:
            return
        if _bool_number(declared, kind[1]) and self.version < BOOL_NUMBER_MISMATCH_FROM_VERSION:
            return
        self.report(REFUSE, 'declared-type', node.target,
                    f'Cannot assign a value of the "{kind[1]}" type to the "{node.target}" variable. The variable is '
                    f'declared with the "{declared}" type.', node)

    def namespace_member(self, node: FunctionCall, stack: List[_Scope]) -> None:
        """A call into a built-in namespace that has no such member: Pine v6 has no `ta.sum()` (the rolling sum
        is `math.sum()`), and TradingView refuses the script instead of running it."""
        # the runtime spells a few Pine names differently (`array.from` is `from_items`, `str` is `string`)
        name = FUNCTION_RENAMES.get(node.func, node.func)
        namespace, _, member = name.partition('.')
        member = member.partition('<')[0]  # a generic call names its types: `array.new<float>`
        if not member or '.' in member:
            return
        members = NAMESPACE_MEMBERS.get(MODULE_RENAMES.get(namespace, namespace))
        if members is None or member in members:
            return
        namespace = node.func.partition('.')[0]
        # a variable, a parameter or a library alias of that name is not the namespace
        aliases = {item.alias or item.library for item in self.script.imports}
        if self.depth_of(stack, namespace) is not None or namespace in aliases:
            return
        if namespace in self.function_names or namespace in self.type_names:
            return
        self.report(REFUSE, 'namespace-member', name, NAMESPACE_MEMBER_MESSAGE.format(name=name), node)

    def void_value(self, node) -> None:
        """Refuse naming the result of a call that returns nothing: a setter, a drawing deletion, an in-place
        sort and the script declarations all return void, and TradingView refuses `x = f()` for them."""
        value = node.value
        if not isinstance(value, FunctionCall) or not isinstance(value.func, str):
            return
        if value.func not in VOID_FUNCTIONS or value.func in self.function_names:
            return
        names = _declared_names(node)
        self.report(REFUSE, 'void-assignment', names[0] if names else value.func, VOID_ASSIGNMENT_MESSAGE, value)

    def strategy_in_indicator(self, name: str, node) -> None:
        """An indicator has no position and places no orders: a `strategy.*` call or account reading in one is
        refused, as TradingView refuses it (the direction and order-size constants are left alone)."""
        if not name.startswith('strategy.') or not isinstance(self.script.script_decl, IndicatorDecl):
            return
        if name.startswith(STRATEGY_CONSTANT_PREFIXES):
            return
        self.report(REFUSE, 'strategy-in-indicator', name, STRATEGY_IN_INDICATOR_MESSAGE.format(name=name), node)

    def argument_base_types(self, node: FunctionCall) -> None:
        """An argument whose type is certain and wrong for the parameter: a float for a drawing's bar-index
        coordinate, a string or bool as a ta.* source, an element of another kind written into an array."""
        name = node.func

        def argument(parameter: str, position: int):
            return node.kwargs.get(parameter, node.args[position] if len(node.args) > position else None)

        def refuse(parameter: str, kind: tuple, expected: str) -> None:
            self.report(REFUSE, 'argument-type', parameter,
                        f'Cannot call "{name}" with argument "{parameter}"="{kind[2]}". An argument of '
                        f'"{kind[0]} {kind[1]}" type was used but a "{expected}" is expected.', node)

        for parameter, position in INT_COORDINATE_PARAMETERS.get(name, {}).items():
            kind = _value_kind(argument(parameter, position))
            if kind is not None and kind[1] == 'float':
                refuse(parameter, kind, 'series int')
        if name in TA_SOURCE_FIRST:
            kind = _value_kind(argument('source', 0))
            if kind is not None and _FAMILY[kind[1]] != 'number':
                refuse('source', kind, 'series float')
            # a window counts bars: measured 2026-11-15, ta.sma(close, 3.5) is CE10123. Only a literal is
            # refused; a series float length is left to the qualifier rules
            length = _value_kind(argument('length', 1))
            if length is not None and length[0] == 'literal' and length[1] != 'int':
                refuse('length', length, 'series int')
        for parameter, position in STRING_TEXT_PARAMETERS.get(name, {}).items():
            kind = _value_kind(node.kwargs.get(parameter) if position is None else argument(parameter, position))
            if kind is not None and kind[1] != 'string':
                refuse(parameter, kind, 'series string')
        if name == 'plot':
            series = _value_kind(argument('series', 0))
            if series is not None and _FAMILY[series[1]] != 'number':
                refuse('series', series, 'series float')
            width = _value_kind(node.kwargs.get('linewidth'))
            if width is not None and width[1] == 'float':
                refuse('linewidth', width, 'input int')
        for parameter, position in COLLECTION_INDEX_PARAMETERS.get(name, {}).items():
            kind = _value_kind(argument(parameter, position))
            if kind is not None and _FAMILY.get(kind[1]) != 'number':
                refuse(parameter, kind, 'series int')
        for parameter, position in TIMEFRAME_PARAMETERS.get(name, {}).items():
            kind = _value_kind(argument(parameter, position))
            if kind is not None and kind[1] != 'string':
                refuse(parameter, kind, 'simple string')
        for parameter, position in COLOR_PARAMETERS.get(name, {}).items():
            kind = _value_kind(argument(parameter, position))
            if kind is not None:  # a colour is never a literal or a price built-in, so any certain kind is wrong
                refuse(parameter, kind, 'const color')
        element = _array_element_type(node) if name.startswith('array.new') else None
        if element in _FAMILY:
            # measured 2026-11-15: array.new<float>(2, "x") is CE10123
            kind = _value_kind(argument('initial_value', 1))
            if kind is not None and _FAMILY[kind[1]] != _FAMILY[element]:
                # the compiler names the function without its type argument
                self.report(REFUSE, 'argument-type', 'initial_value',
                            f'Cannot call "array.new" with argument "initial_value"="{kind[2]}". An argument of '
                            f'"{kind[0]} {kind[1]}" type was used but a "series {element}" is expected.', node)
        if name in MAP_KEY_VALUE_PARAMETERS:
            target = argument('id', 0)
            types = self.map_types.get(target.name) if isinstance(target, Identifier) else None
            for parameter, position in MAP_KEY_VALUE_PARAMETERS[name].items():
                declared = (types or (None, None))[0 if parameter == 'key' else 1]
                kind = _value_kind(argument(parameter, position))
                if declared in _FAMILY and kind is not None and _FAMILY[kind[1]] != _FAMILY[declared]:
                    refuse(parameter, kind, f'series {declared}')
        if name in ARRAY_VALUE_PARAMETERS:
            target = argument('id', 0)
            element = self.array_elements.get(target.name) if isinstance(target, Identifier) else None
            kind = _value_kind(argument('value', ARRAY_VALUE_PARAMETERS[name]))
            if element in _FAMILY and kind is not None and _FAMILY[element] != _FAMILY[kind[1]]:
                refuse('value', kind, f'series {element}')
        if name == 'str.tostring':
            # the chart formatter prints a collection of numbers, bools or strings; an array of a declared
            # type or enum has no spelling, and the compiler names its first overload (measured 2026-11-16)
            target = argument('value', 0)
            element = (_array_element_type(target) if isinstance(target, FunctionCall)
                       else self.array_elements.get(target.name) if isinstance(target, Identifier) else None)
            if element is not None and element in self.type_names:
                shown = target.name if isinstance(target, Identifier) else f'call "{target.func}"'
                self.report(REFUSE, 'argument-type', 'value',
                            f'Cannot call "str.tostring" with argument "value"="{shown}". An argument of '
                            f'"array<{element}>" type was used but a "series float" is expected.', node)
        namespace, _, member = name.rpartition('.')
        if (member in BOOL_ARRAY_MEMBERS and member not in self.method_function_names
                and self.version >= BOOL_ARRAY_PREDICATES_FROM_VERSION):
            if namespace == 'array':
                target = argument('id', 0)
                element = (_array_element_type(target) if isinstance(target, FunctionCall)
                           else self.array_elements.get(target.name) if isinstance(target, Identifier) else None)
                shown = f'call "{target.func}"' if isinstance(target, FunctionCall) else getattr(target, 'name', '')
            else:  # the receiver form, `values.every()`
                element, shown = self.array_elements.get(namespace), namespace
            if element is not None and element != 'bool':
                self.report(REFUSE, 'argument-type', 'id',
                            f'Cannot call "array.{member}" with argument "id"="{shown} (array<{element}>)". '
                            f'An argument of "array<{element}>" type was used but a "array<bool>" is expected.', node)

    def substring_positions(self, node: FunctionCall) -> None:
        """A literal begin_pos of str.substring() below 0, or above a literal end_pos, is refused as the compiler
        refuses it; a position past the source is checked at run time only."""
        begin = node.kwargs.get('begin_pos', node.args[1] if len(node.args) > 1 else None)
        end = node.kwargs.get('end_pos', node.args[2] if len(node.args) > 2 else None)
        begin_value, end_value = _literal_number(begin), _literal_number(end)
        if begin_value is None:
            return
        if begin_value < 0 or (end_value is not None and begin_value > end_value):
            bound = f' or lower than the "end_pos": ({_shown_number(end_value)})' if end_value is not None else ''
            self.report(REFUSE, 'substring-range', node.func,
                        f'Invalid value "{_shown_number(begin_value)}" for "begin_pos" parameter of the '
                        f'"str.substring" function. It must be >= 0{bound}', node)

    def strict_parameters(self, node: FunctionCall, stack: List[_Scope]) -> None:
        """A certainly-series argument (a price or bar built-in, a history read, a ta.*/request.* value, a name
        declared from one of those, a variable the script reassigns) for a parameter the Pine Script Reference
        Manual qualifies as const, input or simple is refused, as TradingView refuses it."""
        signature = PARAMETER_QUALIFIERS.get(node.func)
        if signature is None:
            return
        order, strict = signature
        bound = list(zip(order, node.args)) + list(node.kwargs.items())
        for param, argument in bound:
            kind = strict.get(param)
            if kind is None or not self.argument_is_series(argument, stack):
                continue
            qualifier, _, base = kind.partition(' ')
            base = 'float' if base == 'int/float' else base  # the manual's spelling, the compiler's message
            shown = _shown_argument(argument, f'series {base}') or param
            self.report(REFUSE, 'simple-parameter', param,
                        f'Cannot call "{node.func}" with argument "{param}"="{shown}". An argument of "series {base}" '
                        f'type was used but a "{qualifier} {base}" is expected.', node)
            return
        self.parameter_families(node, bound, strict)

    def parameter_families(self, node: FunctionCall, bound: list, strict: Dict[str, str]) -> None:
        """A literal of another family than the parameter takes: a number where the manual says string, a string
        where it says a number (measured 2026-11-15: a numeric timeframe of request.security is CE10123)."""
        for param, argument in bound:
            kind_text = strict.get(param)
            if kind_text is None:
                continue
            qualifier, _, base = kind_text.partition(' ')
            family = _FAMILY.get('float' if base == 'int/float' else base)
            kind = _value_kind(argument)
            if family is None or kind is None:
                continue
            # measured 2026-11-15: plot(close, offset = 1.5) is CE10123 - a whole number is expected there
            fractional = base == 'int' and kind[1] == 'float'
            if _FAMILY[kind[1]] == family and not fractional:
                continue
            if _bool_number(kind[1], base) and self.version < REMOVED_PARAMETERS_FROM_VERSION:
                continue  # before v6 a number stands in for a bool
            base = 'float' if base == 'int/float' else base
            self.report(REFUSE, 'argument-type', param,
                        f'Cannot call "{node.func}" with argument "{param}"="{kind[2]}". An argument of '
                        f'"{kind[0]} {kind[1]}" type was used but a "{qualifier} {base}" is expected.', node)
            return

    def argument_is_series(self, argument, stack: List[_Scope]) -> bool:
        return self.is_series(argument, stack)

    def if_condition(self, condition, stack: List[_Scope]) -> None:
        """From Pine v6 an if condition must be a bool: a number or string literal, a price or bar built-in, or a
        name declared with an int, float or string type keyword is refused."""
        if self.version < REMOVED_PARAMETERS_FROM_VERSION:
            return
        if _is_na(condition):
            # MEASURED on TradingView 2026-11-15: `if na` is CE10101, a bare na is of no type here
            self.report(REFUSE, 'if-condition-type', '', IF_CONDITION_MESSAGE, condition)
            return
        if isinstance(condition, Literal):
            typed = condition.literal_type in ('int', 'float', 'string')
        elif isinstance(condition, Identifier) and '.' not in condition.name:
            name = condition.name
            depth = self.depth_of(stack, name)
            if depth is None:
                typed = name in SERIES_VARIABLES and name not in self.local_only
            else:
                typed = any(name in scope.explicit_types for scope in stack[:depth + 1])
        else:
            return
        if typed:
            self.report(REFUSE, 'if-condition-type', '',
                        IF_CONDITION_MESSAGE, condition)

    def block_value_default(self, value, stack: List[_Scope]) -> None:
        """From Pine v6 an if/switch value whose arms are all built-in constants of a unique type (a plot style, an
        hline style, a display mode, a text format) needs an else/default block: such a type is never na."""
        if self.version < REMOVED_PARAMETERS_FROM_VERSION or not isinstance(value, (IfExpression, SwitchExpression)):
            return
        arms = _value_arms(value)
        if not arms or not all(_unique_constant(arm) for arm in arms):
            return
        if not _has_default(value):
            self.report(REFUSE, 'block-value-default', '',
                        'Return type of the "if" or "switch" blocks of constant type should have "else" or "default" '
                        'block.', value)
        elif any(self.is_series(condition, stack) for condition in _block_conditions(value)):
            # a bar-dependent choice among built-in constants is refused; an input-dependent one is accepted
            self.report(REFUSE, 'block-value-condition', '',
                        'const built-in variable can not be used as return value in if statement in case its '
                        'condition is not constant.', value)

    def declared_form(self, node, stack: List[_Scope]) -> None:
        """`simple float x = <series value>` (also `const`, `input`) is refused: the form keyword promises a value
        known before the first bar, and a value read from the bars is series. A declaration whose value is
        certainly series is remembered, so a later `simple float y = x` is refused too. Only a value that is
        certainly series counts: a price or bar built-in, a history read, a call of ta.*, request.* or
        math.random(), a name declared from one of those; anything else is left alone."""
        name = _declared_names(node)[0]
        series = self.is_series(node.value, stack)
        if series:
            stack[-1].series.add(name)
        qualifier = getattr(node, 'qualifier', None)
        if series and qualifier in ('simple', 'const', 'input'):
            self.report(REFUSE, 'qualifier-form', name,
                        f'A variable declared with the "{qualifier}" keyword cannot accept values of the '
                        f'"series {node.type_hint}" form. Assign a "{qualifier}" value to this variable or remove '
                        f'the "{qualifier}" keyword from its declaration.')

    def reassigned_form(self, node: Reassignment, stack: List[_Scope]) -> None:
        """A variable is series from a reassignment on: `x := <series value>`, or a reassignment inside a block
        that runs a series number of times - a loop body, or a block whose condition (or one of the conditions
        above its arm) is series. A reassignment under an input or simple condition, or unconditional, keeps the
        variable's form, and so does one written below the read: the form a read sees is the one the statements
        above it gave the variable."""
        if not isinstance(node.target, str) or '.' in node.target:
            return
        depth = self.depth_of(stack, node.target)
        if depth is None:
            return
        if self.is_series(node.value, stack) or any(scope.governed_series for scope in stack[depth + 1:]):
            stack[depth].series.add(node.target)

    def folded_number(self, node, stack: List[_Scope]) -> Optional[float]:
        """The number `node` folds to before the first bar, or None when it cannot be told.

        MEASURED on TradingView 2026-11-18 (BINANCE:BTCUSDT 1D, Pine v6): a range is checked AFTER constant
        folding, not only on a literal -- `math.round(third, int(math.max(101, 101)))` earns the same CE10041
        as a bare 101, and so does the same value reached through a variable (`q = int(math.max(101, 101))`);
        `color.new(color.red, int(math.max(150, 150)))` is CE10274 the same way. A value computed from the
        bars still passes, so nothing series folds here: a name is read only out of `scope.consts`, which
        holds a binding whose value folded and which no reassignment has touched.

        :param node: The expression to fold
        :param stack: The scopes in force, innermost last
        :return: The constant value, or None
        """
        if node is None:
            return None
        if isinstance(node, Literal):
            return node.value if node.literal_type in ('int', 'float') else None
        if isinstance(node, UnaryOp) and node.op in ('-', '+'):
            inner = self.folded_number(node.operand, stack)
            return None if inner is None else (-inner if node.op == '-' else inner)
        if isinstance(node, Identifier):
            depth = self.depth_of(stack, node.name)
            if depth is None or any(node.name in scope.series for scope in stack):
                return None
            for scope in reversed(stack[:depth + 1]):
                if node.name in scope.consts:
                    return scope.consts[node.name]
            return None
        if isinstance(node, BinaryOp) and node.op in FOLDABLE_OPERATORS:
            left = self.folded_number(node.left, stack)
            right = self.folded_number(node.right, stack)
            if left is None or right is None:
                return None
            return left + right if node.op == '+' else left - right if node.op == '-' else left * right
        if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in FOLDABLE_CALLS:
            arity = FOLDABLE_CALLS[node.func]
            if node.kwargs or not node.args or (arity > 0 and len(node.args) != arity):
                return None
            values = [self.folded_number(argument, stack) for argument in node.args]
            if any(value is None for value in values):
                return None
            if node.func == 'int':
                return float(int(values[0]))  # Pine's int() truncates toward zero, as Python's does
            if node.func in ('float', 'math.abs', 'math.ceil', 'math.floor', 'math.round', 'math.sign'):
                import math as _math
                return {'float': lambda v: v, 'math.abs': abs,
                        'math.ceil': lambda v: float(_math.ceil(v)),
                        'math.floor': lambda v: float(_math.floor(v)),
                        'math.round': lambda v: float(round(v)),
                        'math.sign': lambda v: float((v > 0) - (v < 0))}[node.func](values[0])
            return max(values) if node.func == 'math.max' else min(values)
        return None

    def folded_string_length(self, node, stack: List[_Scope]) -> Optional[int]:
        """The length of the constant string `node` folds to, or None when it is not a constant string.

        The compiler folds `str.repeat` of constants -- CE10146 names the length of a string that is too
        long, which it could not do otherwise -- so a repeat of a literal counts at its produced length.
        A concatenation of constants counts as the sum.

        :param node: The expression to measure
        :param stack: The scopes in force, innermost last
        :return: The produced length, or None
        """
        text = _literal_string(node)
        if text is not None:
            return len(text)
        if isinstance(node, BinaryOp) and node.op == '+':
            left = self.folded_string_length(node.left, stack)
            right = self.folded_string_length(node.right, stack)
            return None if left is None or right is None else left + right
        if isinstance(node, FunctionCall) and node.func == 'str.repeat':
            source = node.kwargs.get('source', node.args[0] if node.args else None)
            count = self.folded_number(node.kwargs.get('count', node.args[1] if len(node.args) > 1 else None),
                                       stack)
            length = self.folded_string_length(source, stack) if source is not None else None
            if length is None or count is None or count < 0:
                return None
            # a separator would make it longer still; it is not counted, which keeps this a lower bound
            return length * int(count)
        return None

    def statement_size(self, node, bodies: list, kind: str, stack: List[_Scope]) -> None:
        """Refuse a block statement whose body folds a constant string past the statement's budget.

        BISECTED 2026-11-19 (BINANCE:BTCUSDT 1D, Pine v6): 36969 characters of folded constant inside an
        `if` body compile, 36970 do not, and every block kind is refused at 36970 while a function body is
        refused at no length at all. `BLOCK_STATEMENT_CONSTANT_LIMIT` carries the threshold and what else
        the session measured about the budget's shape.

        :param node: The block statement, for the position of last resort
        :param bodies: Its bodies (an `if` has its arms, a `switch` its cases and default)
        :param kind: Its keyword, as `BLOCK_STATEMENT_KEYWORDS` spells it
        :param stack: The scopes in force, innermost last
        """
        if self.version < REMOVED_PARAMETERS_FROM_VERSION:
            return
        keyword = BLOCK_STATEMENT_KEYWORDS[kind]
        for body in bodies:
            for inner in walk(body):
                length = self.folded_string_length(inner, stack)
                # above the allowed string length the `string-too-long` rule owns the case: which of the
                # two refusals TradingView gives inside a block is not measured
                if length is not None and BLOCK_STATEMENT_CONSTANT_LIMIT <= length <= STRING_LENGTH_LIMIT:
                    self.report(REFUSE, 'statement-too-long', keyword,
                                BLOCK_STATEMENT_TOO_LONG_MESSAGE.format(keyword=keyword),
                                _positioned(inner) or node)
                    return

    def remember_const(self, node, stack: List[_Scope]) -> None:
        """Record a declaration whose value folds to a number, so a later read of the name folds too."""
        names = _declared_names(node)
        if len(names) != 1 or isinstance(getattr(node, 'target', None), TupleDestructure):
            return
        value = self.folded_number(node.value, stack)
        if value is not None:
            stack[-1].consts[names[0]] = value

    def forget_const(self, target: str, stack: List[_Scope]) -> None:
        """A reassigned name is no longer a compile-time constant, whatever the new value folds to: the
        value a read sees depends on the bar the reassignment ran on."""
        for scope in stack:
            scope.consts.pop(target, None)

    def is_series(self, expression, stack: List[_Scope]) -> bool:
        """Whether `expression` certainly carries a series value (see `declared_form`)."""
        for node in walk(expression):
            if isinstance(node, IndexAccess):
                return True
            if isinstance(node, Identifier):
                # a dotted spelling is never a binding of its own: it is a field read, a built-in
                # namespace or an enum member, which `is_udt_field_read` tells apart
                if self.is_udt_field_read(node.name, stack):
                    return True
                if self.depth_of(stack, node.name) is None:
                    if node.name in SERIES_VARIABLES and node.name not in self.local_only:
                        return True
                elif any(node.name in scope.series for scope in stack):
                    return True
            elif isinstance(node, FunctionCall) and isinstance(node.func, str) and (
                    node.func in SERIES_FUNCTIONS or node.func == 'math.random'
                    or node.func.startswith(('ta.', 'request.'))
                    or node.func in COLLECTION_READS):
                return True
        return False

    def is_udt_field_read(self, spelling: str, stack: List[_Scope]) -> bool:
        """Whether `spelling` is a dotted read of a field of a user-defined type's instance.

        MEASURED 2026-11-19 (BINANCE:BTCUSDT 1D, Pine v6):
        `alertcondition(title = h.name)` with `h` a UDT instance is CE10123, `argument "title"="h.name"` of
        `series string`. An object in Pine is series-qualified, so reading a field of one gives series.

        The test is on the PREFIX being a variable of a declared type, which is what keeps the two other
        dotted spellings out: a built-in namespace (`syminfo.tickerid`, a simple value) has no binding, and
        an enum member (`Side.up`, a const) names a type rather than a variable.

        :param spelling: The identifier as written, e.g. ``h.name``
        :param stack: The scopes in force, innermost last
        :return: True for a field read of a user-defined type
        """
        head, _, path = spelling.partition('.')
        if not path:
            return False
        declared = self.variable_type(head, stack)
        fields = self.udt_fields.get(_type_word(declared) or '')
        return bool(fields) and path.partition('.')[0] in fields

    def block_value_types(self, value) -> None:
        """`x = if c ... else ...` or `x = switch ...` whose arms end in values of different types is refused:
        a variable has one type. Only arms whose type is certain are compared (a literal, a price or bar
        built-in); an arm ending in a call, a name or `na` is left alone."""
        arms = [(family, arm) for arm in _value_arms(value) if (family := _value_family(arm)) is not None]
        if len({family for family, _ in arms}) > 1:
            first_family, first_arm = arms[0]
            other = next(item for item in arms if item[0] != first_family)
            self.report(REFUSE, 'block-value-types', '',
                        'Return type of one of the "if" or "switch" blocks is not compatible with return type of '
                        f'other block(s) ({_value_shown(first_arm)}; {_value_shown(other[1])})', other[1])

    # ------------------------------------------------------------ Pine v6: a bool is never na

    def na_of_a_bool(self, call: FunctionCall, stack: List[_Scope]) -> None:
        """`na(b)`, `nz(b)`, `fixnan(b)` of a value known to be a bool, refused from Pine v6."""
        if self.version < BOOL_NEVER_NA_FROM_VERSION:
            return
        name = 'na' if isinstance(call.func, NaLiteral) else call.func
        if name not in ('na', 'nz', 'fixnan'):
            return
        argument = call.args[0] if call.args else call.kwargs.get('source', call.kwargs.get('x'))
        if argument is None or not self.is_bool(argument, stack):
            return
        message = (f'na() of a bool: {BOOL_NEVER_NA_MESSAGE}' if name == 'na' else
                   f'{name}() takes a number, not a bool: {BOOL_NEVER_NA_MESSAGE}')
        self.report(REFUSE, 'bool-never-na', name, message, _positioned(argument))

    def na_arm_of_a_bool(self, node: TernaryOp, stack: List[_Scope]) -> None:
        """`c ? b : na` or `c ? na : b` with `b` known to be a bool, refused from Pine v6: MEASURED on TradingView
        2026-11-13, `x = c ? close > open : na` gives `Cannot call "operator ?:" with argument "expr2"="na". An
        argument of "simple na" type was used but a "series bool" is expected. (CE10123)`; `c ? true : na` the same
        with "simple bool", `c ? na : close > open` with "expr1"."""
        if self.version < BOOL_NEVER_NA_FROM_VERSION:
            return
        for slot, arm, other in (('expr1', node.true_expr, node.false_expr), ('expr2', node.false_expr, node.true_expr)):
            if _is_na(arm) and not _is_na(other) and self.is_bool(other, stack):
                self.report(REFUSE, 'bool-never-na', slot,
                            f'Cannot call "operator ?:" with argument "{slot}"="na". An argument of "simple na" type '
                            f'was used but a bool is expected (CE10123): {BOOL_NEVER_NA_MESSAGE}', _positioned(arm))

    def na_block_of_a_bool(self, node, stack: List[_Scope]) -> None:
        """An if/switch value with a bool arm and an `na` arm, refused from Pine v6: MEASURED on TradingView
        2026-11-14, `b = if c1` + `flag` + `else if c2` + `na` gives `Return type of one of the "if" or "switch"
        blocks is not compatible with return type of other block(s) (series bool; simple na)`."""
        if self.version < BOOL_NEVER_NA_FROM_VERSION:
            return
        tails = list(value_tails(node))
        na_tail = next((tail for tail in tails if _is_na(tail)), None)
        if na_tail is not None and any(not _is_na(tail) and self.is_bool(tail, stack) for tail in tails):
            self.report(REFUSE, 'bool-never-na', 'if', NA_BLOCK_OF_A_BOOL_MESSAGE, _positioned(na_tail))

    def bool_declared_na(self, node) -> None:
        """`bool b = na` (`var` and `varip` too), refused from Pine v6."""
        hint = getattr(node, 'type_hint', None)
        if self.version >= BOOL_NEVER_NA_FROM_VERSION and hint and hint.split()[-1] == 'bool' \
                and _is_na(node.value):
            name = node.name if isinstance(node, (VarDecl, VaripDecl)) else node.target
            self.report(REFUSE, 'bool-never-na', name, f"'{name}' is declared a bool and na: {BOOL_NEVER_NA_MESSAGE}")

    def bool_reassigned_na(self, node: Reassignment, stack: List[_Scope]) -> None:
        """`b := na` of a name known to hold a bool, refused from Pine v6."""
        if self.version >= BOOL_NEVER_NA_FROM_VERSION and '.' not in node.target and _is_na(node.value) \
                and self.is_bool(Identifier(name=node.target), stack):
            self.report(REFUSE, 'bool-never-na', node.target,
                        f"'{node.target}' holds a bool and cannot be assigned na: {BOOL_NEVER_NA_MESSAGE}")

    def is_bool(self, expression, stack: List[_Scope]) -> bool:
        """`expression` is known to be a bool here (BOOL_NEVER_NA_FROM_VERSION); False when not known."""
        def named(name: str) -> bool:
            depth = self.depth_of(stack, name)
            return depth is not None and name in stack[depth].bools

        def bound(name: str) -> bool:
            return self.depth_of(stack, name) is not None or name in self.global_names \
                or name in self.local_only or name in self.function_names or name in self.type_names
        return _is_bool(expression, named, self.bool_functions, bound)

    def functions_of_bool_value(self) -> set:
        """The functions declared once whose value is known to be a bool: their last expression is one, read
        with their typed bool parameters and the names their body declares from one, in declaration order."""
        found: set = set()
        declared = self.global_names | self.local_only | self.function_names | self.type_names

        def bound(name: str) -> bool:
            return name in declared
        for item in self.top_items:
            if not isinstance(item, FuncDecl) or item.is_method or item.name in self.overloaded:
                continue
            names = {parameter.name for parameter in item.params if declared_type(parameter) == 'bool'}
            body = item.body if isinstance(item.body, list) else [ExpressionStatement(expr=item.body)]
            for statement in body[:-1]:
                if isinstance(statement, (VarDecl, VaripDecl, Assignment)) and isinstance(
                        getattr(statement, 'target', ''), str) and (
                        declared_type(statement) == 'bool'
                        or _is_bool(statement.value, names.__contains__, found, bound)):
                    names.add(statement.name if isinstance(statement, (VarDecl, VaripDecl)) else statement.target)
            last = body[-1] if body else None
            if isinstance(last, ExpressionStatement) and _is_bool(last.expr, names.__contains__, found, bound):
                found.add(item.name)
        return found

    def tuple_width(self, value, seen: frozenset = frozenset()) -> Optional[int]:
        """How many values `value` gives a tuple declaration, when that is known: a tuple literal (the third argument
        of `request.security`), a call of a function declared once whose last expression is a tuple or such a
        call, or a `request.security` of one of these; None otherwise."""
        if isinstance(value, ArrayLiteral):
            return len(value.elements)
        if not isinstance(value, FunctionCall) or not isinstance(value.func, str):
            return None
        if value.func in SECURITY_FUNCTIONS:
            expression = value.args[2] if len(value.args) > 2 else value.kwargs.get('expression')
            return self.tuple_width(expression, seen) if expression is not None else None
        declaration = self.single_functions.get(value.func)
        if declaration is None or value.func in seen:
            return None
        body = declaration.body if isinstance(declaration.body, list) else [ExpressionStatement(expr=declaration.body)]
        last = body[-1] if body else None
        if not isinstance(last, ExpressionStatement) or not isinstance(last.expr, (ArrayLiteral, FunctionCall)) \
                or (isinstance(last.expr, FunctionCall) and last.expr.func in SECURITY_FUNCTIONS):
            return None
        return self.tuple_width(last.expr, seen | {value.func})

    def is_loop_counter(self, node, stack: List[_Scope]) -> bool:
        """`node` reads the counter of a `for i = a to b` loop that is open here: the innermost binding of
        its name is the counter, so a name a nested block declares again is that declaration's, not the counter's.
        A name the loop body declares again at the counter's own level keeps the counter's binding (a scope's
        first declaration wins) and is refused: UNMEASURED on TradingView, which refuses the redeclaration itself."""
        depth = self.depth_of(stack, node.name) if isinstance(node, Identifier) else None
        return depth is not None and stack[depth].names[node.name] == LOOP_COUNTER

    def loop_counter_in(self, node, stack: List[_Scope]) -> Optional[Identifier]:
        """The counter of an open `for` loop that `node` reads as an operand: the counter itself (`i[1]`) or
        an operand of an arithmetic expression (`(close * close * i)[1]` -- Pine v4, v5 and v6 refuse the history
        of either, `Variable 'i' doesn't have history values`). An argument of a call is not looked into: a
        function given the counter (`f(close * i)`, its body reading `x[1]`) compiles."""
        if isinstance(node, Identifier):
            return node if self.is_loop_counter(node, stack) else None
        if isinstance(node, BinaryOp):
            return self.loop_counter_in(node.left, stack) or self.loop_counter_in(node.right, stack)
        if isinstance(node, UnaryOp):
            return self.loop_counter_in(node.operand, stack)
        if isinstance(node, TernaryOp):
            return (self.loop_counter_in(node.condition, stack) or self.loop_counter_in(node.true_expr, stack)
                    or self.loop_counter_in(node.false_expr, stack))
        return None

    def binding_in_scope(self, name: str, stack: List[_Scope]) -> Optional[HeadBinding]:
        """The variable `name` names here, with its declared type: the binding in scope, or a
        global (declared above or further down, typed as the scope renamer types it, by its
        first global declaration); None when `name` is no variable here."""
        depth = self.depth_of(stack, name)
        if depth is not None and depth > 0:  # a global (depth 0) is typed below, in the renamer's order
            return HeadBinding(stack[depth].types.get(name))
        return HeadBinding(self.global_types.get(name)) if name in self.global_names else None

    def variable_type(self, name: str, stack: List[_Scope]) -> Optional[str]:
        """The type of the variable `name` names here (`binding_in_scope`), None when unknown."""
        binding = self.binding_in_scope(name, stack)
        return binding.type_name if binding else None

    @staticmethod
    def field_history_message(site: IndexAccess) -> str:
        """The advice indexes the object, then walks the field path: `o.i.v[1]` -> `(o[1]).i.v`
        (`(o.i[1]).v` would index a field again)."""
        index = site.index.value if isinstance(site.index, Literal) else 'n'
        spelled = site.object.name if isinstance(site.object, Identifier) else ''
        if isinstance(site.object, MemberAccess) and isinstance(site.object.object, str):
            spelled = f'{site.object.object}.{site.object.member}'
        head, _, path = spelled.partition('.')
        object_text, field_text = (head, path) if path else ('object', 'field')
        return (f"the [] history operator cannot be used on a field of a user-defined type from Pine v6; "
                f"write ({object_text}[{index}]).{field_text} for the field of the object {index} bar(s) back, "
                f"or assign the field to a variable and index that variable")

    # ------------------------------------------------------------ recursion

    def recursion(self) -> None:
        """A call that closes a cycle in the call graph of the script's functions,
        refused at the call. An overloaded name resolves by signature, so it is
        left out of the graph (expds_097 calls one overload from another); an
        overload that passes its own parameters, unchanged, to its own name
        resolves to itself and is refused."""
        declared = [item for item in self.top_items if isinstance(item, FuncDecl) and not item.is_method]
        # a call by a method's name may select the method (`f(P.new(x))` beside `method f(P this)`)
        method_names = {item.name for item in self.top_items if isinstance(item, FuncDecl) and item.is_method}
        single = {function.name: function for function in declared
                  if function.name not in self.overloaded and function.name not in method_names}
        calls = {name: [node for node in walk(function.body)
                        if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in single]
                 for name, function in single.items()}
        callees = {name: {call.func for call in found} for name, found in calls.items()}
        for name, function in single.items():
            closing = next((call for call in calls[name]
                            if call.func == name or self.reaches(call.func, name, callees)), None)
            if closing is not None:
                self.report_recursion(name, closing)
        for function in declared:
            if function.name in self.overloaded:
                own = next((node for node in walk(function.body) if _passes_own_parameters(node, function)), None)
                if own is not None:
                    self.report_recursion(function.name, own)

    def report_recursion(self, name: str, call: FunctionCall) -> None:
        self.report(REFUSE, 'recursion', name, f"function '{name}' calls itself (directly or through another "
                    f'function); TradingView does not allow recursion', call)

    @staticmethod
    def reaches(start: str, goal: str, callees: Dict[str, set]) -> bool:
        seen, pending = set(), list(callees[start])
        while pending:
            name = pending.pop()
            if name == goal:
                return True
            if name not in seen:
                seen.add(name)
                pending.extend(callees.get(name, ()))
        return False


#: built-in variables whose value is a number on every bar
#: Pine v6 removed these parameters (the migration guide): a call that still names one is refused, as the compiler
#: refuses it ("The "plot" function does not have an argument with the name "transp"")
REMOVED_PARAMETERS_FROM_VERSION = 6
REMOVED_PARAMETERS = {
    'transp': frozenset({'plot', 'plotshape', 'plotchar', 'plotarrow', 'bgcolor', 'fill'}),
    'when': frozenset({'strategy.entry', 'strategy.order', 'strategy.exit', 'strategy.close', 'strategy.close_all',
                       'strategy.cancel', 'strategy.cancel_all'}),
}
#: the plot-family functions whose `offset` must be simple and whose `linewidth` must be at least 1 (Pine v6)
PLOT_FAMILY = frozenset({'plot', 'plotshape', 'plotchar', 'plotarrow', 'plotcandle', 'plotbar', 'bgcolor', 'barcolor'})
#: the farthest history max_bars_back() can ask for: a larger literal is refused ("It must be between 0 and 5000")
MAX_BARS_BACK_LIMIT = 5000
#: Longest string a script may build; MEASURED on TradingView 2026-11-19 (Pine v6)
STRING_LENGTH_LIMIT = 40960
#: A BLOCK STATEMENT has a size budget its body's folded constants spend, and CE10205 is what overflowing
#: it costs. BISECTED on TradingView 2026-11-19 (BINANCE:BTCUSDT 1D, Pine v6) on a body of one
#: `array.push(heap, str.repeat("x", N))`: N = 36969 compiles inside an `if`, N = 36970 does not, and the
#: pair was re-read at the end of the session. The compiler folds `str.repeat` of constants -- that is how
#: CE10146 knows the length of a string that is too long -- so the body is measured by its folded length.
#:
#: What the same session measured about the shape of the budget:
#:
#: * it is NOT an allowance per constant but spent by the whole body, and each statement costs some of it
#:   too: two pushes of 18484 each (36968 in all, BELOW the 36969 one push was allowed) are refused;
#: * every block kind carries it, and the message names the kind: `for` and `for ... in` say "The for
#:   statement is too long", `while` says "while", `else` and `else if` answer under the `if`, and a
#:   `switch` is reported as an **if** (TradingView compiles it into if-chains);
#: * each kind's budget is its own: a ONE-statement `while` is already refused at 36969, where the same
#:   `if` body compiles;
#: * a FUNCTION body carries no such budget -- it compiles at 36970 and at 40960, the longest string Pine
#:   allows at all, so there is nothing left to refuse it with.
#:
#: 36970 is the smallest refusal measured for ANY kind, and 36969 the largest acceptance (in an `if`), so
#: refusing from 36970 up never refuses a body TradingView accepted. A `while` or `switch` body between
#: the two is refused by TradingView and passes here: that miss is deliberate, because modelling the
#: per-statement cost would risk refusing a script TradingView accepts.
BLOCK_STATEMENT_CONSTANT_LIMIT = 36970
#: the keyword CE10205 names for each block statement (a `switch` answers as an `if`)
BLOCK_STATEMENT_KEYWORDS = {'if': 'if', 'else': 'if', 'switch': 'if', 'for': 'for', 'while': 'while'}
BLOCK_STATEMENT_TOO_LONG_MESSAGE = 'The {keyword} statement is too long'
#: the drawing counts a script declaration may raise above which TradingView refuses it (measured 2026-11-15:
#: labels and lines are refused above 500, while max_boxes_count = 600 compiles)
DRAWING_COUNT_LIMITS = {'max_labels_count': 500, 'max_lines_count': 500}
DRAWING_COUNT_MESSAGE = 'The script {parameter} value ({value}) is greater than the maximum possible value ({limit})'
#: the declaration parameters whose value must fall inside a range, and that range (measured 2026-11-15)
DECLARATION_RANGES = {'max_bars_back': (0, MAX_BARS_BACK_LIMIT), 'max_polylines_count': (1, 100)}
#: a call argument whose literal value must fall inside a range, by (function, parameter, position of the
#: positional form): MEASURED 2026-11-18 (BINANCE:BTCUSDT 1D, Pine v6), `math.round(1234.5, -1)` is CE10041
#: «Invalid value "-1" for "precision" parameter of the "math.round()" function. It must be between 0 and 100»
CALL_ARGUMENT_RANGES = {('math.round', 'precision', 1): (0, 100)}
DECLARATION_RANGE_MESSAGE = ('Invalid value "{value}" for "{parameter}" parameter of the "{call}()" function. '
                             'It must be between {low} and {high}')
IMPORT_TWICE_MESSAGE = 'Cannot import a library more than once'
NAMESPACE_MEMBER_MESSAGE = "Could not find function or function reference '{name}'"
NEGATIVE_PRECISION_MESSAGE = ('The "{call}()" function cannot have a negative "precision" value. Use a positive '
                              'value in the function call.')
#: the block kinds a request.*() call may not sit in when dynamic requests are off
CONDITIONAL_SCOPE_KINDS = frozenset({'if', 'for', 'while', 'switch'})
# The output and declaration statements Pine allows at global scope only: inside an if, switch, loop or
# function body the compiler refuses them (an input.*() call is allowed anywhere).
GLOBAL_ONLY_CALLS = PLOT_FAMILY | frozenset({'hline', 'fill', 'alertcondition', 'indicator', 'strategy', 'library'})

#: the built-in types whose values are references (a drawing or a chart point): a method must read such a receiver
REFERENCE_BUILTIN_TYPES = frozenset({'line', 'label', 'box', 'table', 'linefill', 'polyline', 'chart.point'})

_NUMBER_BUILTINS = frozenset({'open', 'high', 'low', 'close', 'volume', 'hl2', 'hlc3', 'ohlc4', 'bar_index',
                              'last_bar_index', 'time', 'time_close'})
_LITERAL_FAMILY = {'int': 'float', 'float': 'float', 'string': 'string', 'bool': 'bool', 'color': 'color'}
#: the number built-ins that carry whole numbers: the refusal names them "series int", not "series float"
_WHOLE_NUMBER_BUILTINS = frozenset({'bar_index', 'last_bar_index', 'time', 'time_close'})


def _value_arms(value) -> list:
    """The expression each arm of an if/switch value ends with (a chained `else if` contributes its own arms)."""
    if isinstance(value, IfExpression):
        arms = [value.true_expr] + ([value.false_expr] if value.false_expr is not None else [])
    elif isinstance(value, SwitchExpression):
        arms = [body for _, body in value.cases] + ([value.default] if value.default is not None else [])
    else:
        return []
    found = []
    for arm in arms:
        if isinstance(arm, list):
            if not arm:
                continue
            arm = arm[-1]
        if isinstance(arm, (ExpressionStatement, ReturnStatement)):
            arm = arm.expr
        if isinstance(arm, (IfExpression, SwitchExpression)):
            found.extend(_value_arms(arm))
        else:
            found.append(arm)
    return found


def _value_family(expression) -> Optional[str]:
    """'float', 'string', 'bool' or 'color' when the expression's type is certain, else None."""
    if isinstance(expression, Literal):
        return _LITERAL_FAMILY.get(expression.literal_type)
    if isinstance(expression, Identifier) and expression.name in _NUMBER_BUILTINS:
        return 'float'
    return None


def _shown_argument(argument, actual: str) -> Optional[str]:
    """How the refusal writes the argument that was given: a name by its spelling, a call by its own form.

    MEASURED 2026-11-17 (BINANCE:BTCUSDT 1D, Pine v6): `alertcondition(title = str.tostring(close))` is
    `argument "title"="call \"str.tostring\" (series string)"`, so a call names itself and carries the type it
    produced. Anything else is left to the caller, which falls back to the parameter name.
    """
    if isinstance(argument, Identifier):
        return argument.name
    if isinstance(argument, FunctionCall) and isinstance(argument.func, str):
        return f'call "{argument.func}" ({actual})'
    # MEASURED 2026-11-19 (BINANCE:BTCUSDT 1D, Pine v6): an operator is written as a call of itself, the
    # ternary under the name "operator ?:", and a field path by its own spelling (four measurements)
    if isinstance(argument, BinaryOp):
        return f'call "operator {argument.op}" ({actual})'
    if isinstance(argument, UnaryOp):
        return f'call "operator {argument.op}" ({actual})'
    if isinstance(argument, TernaryOp):
        return f'call "operator ?:" ({actual})'
    if isinstance(argument, MemberAccess) and isinstance(argument.object, str):
        return f'{argument.object}.{argument.member}'
    return None


def _value_shown(expression) -> str:
    """How an arm of an if/switch value is named in the refusal: its qualifier and its own type.

    MEASURED 2026-11-17 (BINANCE:BTCUSDT 1D, Pine v6), four arms against a string literal: an int literal is
    named "literal int" and a float literal "literal float" (the widening to one numeric family happens for the
    comparison, not in the wording), a price is "series float" and a whole-number chart value "series int".
    """
    if isinstance(expression, Literal):
        return f'literal {expression.literal_type}'
    if isinstance(expression, Identifier) and expression.name in _WHOLE_NUMBER_BUILTINS:
        return 'series int'
    return 'series float'


def _is_na(expression) -> bool:
    return isinstance(expression, NaLiteral) or (isinstance(expression, Identifier) and expression.name == 'na')


def _is_bool(expression, named, bool_functions, bound=lambda name: False) -> bool:
    """`expression` is known to be a bool; `named(name)` says whether a name holds one, `bound(name)` whether the
    script declares the name (then `session.ismarket` reads a field of it, not the builtin)."""
    def builtin(name: str) -> bool:
        return (name in BOOL_BUILTINS or name == 'input.bool') and not bound(name.partition('.')[0])
    if isinstance(expression, Literal):
        return expression.literal_type == 'bool'
    if isinstance(expression, Identifier):
        return expression.name in ('true', 'false') or builtin(expression.name) or named(expression.name)
    if isinstance(expression, BinaryOp):
        return expression.op in BOOL_OPERATORS
    if isinstance(expression, UnaryOp):
        return expression.op == 'not'
    if isinstance(expression, IndexAccess):
        return _is_bool(expression.object, named, bool_functions, bound)
    if isinstance(expression, TernaryOp):
        arms = [arm for arm in (expression.true_expr, expression.false_expr) if not _is_na(arm)]
        return bool(arms) and all(_is_bool(arm, named, bool_functions, bound) for arm in arms)
    if isinstance(expression, FunctionCall):
        if isinstance(expression.func, NaLiteral):
            return True
        return isinstance(expression.func, str) and (builtin(expression.func) or expression.func in bool_functions)
    return False


def _is_a_bool_input(node: InputDecl) -> bool:
    """`input.bool(...)`, or `input(true)` whose default is a bool literal."""
    default = node.args[0] if node.args else node.kwargs.get('defval')
    return input_type(node) == 'bool' or (
        node.func == 'input' and isinstance(default, Literal) and default.literal_type == 'bool')


def _positioned(node):
    """The part of `node` written first that has a position (a call written `na(...)` has none of its own, a
    history read `b[1]` has the position of its `[`)."""
    parts = [part for part in walk(node) if getattr(part, 'line', 0)]
    return min(parts, key=lambda part: (part.line, part.column)) if parts else None


def _declares_static_requests(node) -> bool:
    """Whether a script declaration writes `dynamic_requests = false`, which confines request.*() calls to the
    global scope."""
    if not isinstance(node, (IndicatorDecl, StrategyDecl, LibraryDecl)):
        return False
    value = node.kwargs.get('dynamic_requests')
    if isinstance(value, Literal):
        return value.value is False or value.value == 'false'
    return isinstance(value, Identifier) and value.name == 'false'


def _block_conditions(value) -> list:
    """The conditions an if/switch value tests: the if's own and its chained else-ifs', a switch's subject and
    case expressions."""
    if isinstance(value, SwitchExpression):
        return ([value.expr] if value.expr is not None else []) + [case for case, _ in value.cases]
    if isinstance(value, IfExpression):
        tail = value.false_expr
        if isinstance(tail, list) and len(tail) == 1:
            tail = tail[0]
        if isinstance(tail, ExpressionStatement):
            tail = tail.expr
        return [value.condition] + (_block_conditions(tail) if isinstance(tail, IfExpression) else [])
    return []


def _has_default(value) -> bool:
    """Whether an if/switch value ends in an else/default block (a chained `else if` is followed to its end)."""
    if isinstance(value, SwitchExpression):
        return value.default is not None
    if isinstance(value, IfExpression):
        if value.false_expr is None:
            return False
        tail = value.false_expr
        if isinstance(tail, list) and len(tail) == 1:
            tail = tail[0]
        if isinstance(tail, ExpressionStatement):
            tail = tail.expr
        return _has_default(tail) if isinstance(tail, IfExpression) else True
    return True


def _unique_constant(arm) -> bool:
    """Whether `arm` is a built-in constant of a unique type (the Pine Script Reference Manual's `const <type>`
    entries whose type is not a plain string, number, bool or colour: a plot style, a display mode, a barmerge
    option, ...)."""
    if isinstance(arm, Identifier):
        name = arm.name
    elif isinstance(arm, MemberAccess) and isinstance(arm.object, str):
        name = f'{arm.object}.{arm.member}'
    else:
        return False
    return name in UNIQUE_CONSTANTS


def _names_read(nodes) -> set:
    """Every name read in `nodes`: an identifier, the object of a member access, the namespace of a call."""
    names = set()
    for node in walk(nodes):
        if isinstance(node, Identifier):
            names.add(node.name.split('.', 1)[0])  # a field read parses as the dotted name `this.v`
        elif isinstance(node, FunctionCall) and isinstance(node.func, str):
            names.add(node.func.split('.', 1)[0])
        elif isinstance(getattr(node, 'object', None), str):  # a member or field access, `this.v`
            names.add(node.object)
        if isinstance(getattr(node, 'target', None), str):  # a write to a field, `this.v := x`, reads the object
            names.add(node.target.split('.', 1)[0])
    return names


def _names_declared_in(nodes: list) -> set:
    """Every name a declaration, loop or parameter in `nodes` binds."""
    names = {name for node in nodes for name in _declared_names(node)}
    for node in nodes:
        if isinstance(node, ForLoop):
            names.add(node.var)
        elif isinstance(node, ForInLoop):
            names.update(node.vars)
        elif isinstance(node, FuncDecl):
            names.update(parameter.name for parameter in node.params)
    return names


def _passes_own_parameters(node, function: FuncDecl) -> bool:
    """`f(x)` inside `f(float x)`: the call's arguments are the function's own
    parameters, in order and unchanged (no local of their names), so it
    selects this very overload."""
    if not (isinstance(node, FunctionCall) and node.func == function.name and not node.kwargs):
        return False
    parameters = [parameter.name for parameter in function.params]
    arguments = [argument.name if isinstance(argument, Identifier) else None for argument in node.args]
    redeclared = _names_declared_in(list(walk(function.body)))
    return arguments == parameters and not redeclared.intersection(parameters)
