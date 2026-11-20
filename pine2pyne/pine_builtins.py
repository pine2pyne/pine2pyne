"""
Mapping of Pine Script built-in functions, variables, and constants to PyneCore equivalents.
"""
from typing import Dict, FrozenSet, Set, Tuple

from .namespace_members import NAMESPACE_MEMBERS


# Functions that need to be renamed
FUNCTION_RENAMES: Dict[str, str] = {
    'array.from': 'array.from_items',  # 'from' is Python keyword
}

# Module renames
MODULE_RENAMES: Dict[str, str] = {
    'str': 'string',  # str is Python built-in
}

# Type renames (Pine Script -> Python/PyneCore)
TYPE_RENAMES: Dict[str, str] = {
    # Drawing types
    'line': 'Line',
    'label': 'Label',
    'box': 'Box',
    'table': 'Table',
    'linefill': 'LineFill',
    'polyline': 'Polyline',
    'color': 'Color',
    'chart.point': 'ChartPoint',

    # Data structure types
    'matrix': 'Matrix',

    # Basic types (in type hints)
    'string': 'str',
    'int': 'int',
    'float': 'float',
    'bool': 'bool',
}

# Built-in variables that are always Series
SERIES_VARIABLES: Set[str] = {
    'open', 'high', 'low', 'close', 'volume', 'aggressor_side',
    'hl2', 'hlc3', 'ohlc4', 'hlcc4',
    'bar_index', 'last_bar_index', 'last_bar_time',
    'time', 'time_close', 'time_tradingday', 'timenow',
    'dayofmonth', 'dayofweek', 'hour', 'minute', 'second', 'month', 'weekofyear', 'year',
}
# The builtin series variables that are also builtin functions, `time("D")` and `hour(t)`
# (the names both the functions and the variables of pine_reference_types.json list)
CALLABLE_SERIES_VARIABLES: Set[str] = {
    'time', 'time_close',
    'dayofmonth', 'dayofweek', 'hour', 'minute', 'second', 'month', 'weekofyear', 'year',
}

# Constants that should be preserved as-is
CONSTANTS: Set[str] = {
    # Strategy constants
    'strategy.long',
    'strategy.short',
    'strategy.commission.percent',
    'strategy.commission.cash_per_contract',
    'strategy.commission.cash_per_order',
    'strategy.percent',
    'strategy.fixed',
    'strategy.cash',
    'strategy.percent_of_equity',
    'strategy.closedtrades',
    'strategy.opentrades',

    # Color constants
    'color.red', 'color.green', 'color.blue', 'color.white', 'color.black',
    'color.yellow', 'color.orange', 'color.purple', 'color.gray',
    'color.silver', 'color.maroon', 'color.fuchsia', 'color.lime',
    'color.olive', 'color.navy', 'color.teal', 'color.aqua',

    # Plot styles (need remapping to plot_style module)
    'plot.style_line',
    'plot.style_stepline',
    'plot.style_histogram',
    'plot.style_cross',
    'plot.style_area',
    'plot.style_columns',
    'plot.style_circles',
    'plot.style_linebr',
    'plot.style_steplinebr',

    # Barstate
    'barstate.isfirst',
    'barstate.islast',
    'barstate.ishistory',
    'barstate.isrealtime',
    'barstate.isnew',
    'barstate.isconfirmed',
}

# Plot style constants need special handling
PLOT_STYLE_REMAP: Dict[str, str] = {
    'plot.style_line': 'plot_style.style_line',
    'plot.style_stepline': 'plot_style.style_stepline',
    'plot.style_histogram': 'plot_style.style_histogram',
    'plot.style_cross': 'plot_style.style_cross',
    'plot.style_area': 'plot_style.style_area',
    'plot.style_columns': 'plot_style.style_columns',
    'plot.style_circles': 'plot_style.style_circles',
    'plot.style_linebr': 'plot_style.style_linebr',
    'plot.style_steplinebr': 'plot_style.style_steplinebr',
}

# Modules that should be imported from pynecore.lib
# Synced with the pynecore repository, `src/pynecore/lib`
PYNECORE_LIB_MODULES: Set[str] = {
    # Core modules (submodules as .py files in pynecore/lib/)
    'script', 'input', 'plot', 'color', 'ta', 'math', 'strategy',
    'array', 'matrix', 'map', 'string',
    'timeframe', 'runtime', 'alert', 'label', 'line', 'box', 'table',
    'linefill', 'polyline', 'log', 'barstate', 'session', 'syminfo',
    'adjustment', 'barmerge', 'chart', 'dividends', 'earnings', 'aggressor', 'backadjustment', 'settlement_as_close',
    # Higher-timeframe / other-symbol data access. Present in upstream PyneCore
    # (lib/request.py + lib/ticker.py); without them here the transpiler emits
    # request.*/ticker.* calls it never imports -> NameError at runtime.
    'request', 'ticker',
    # Other upstream lib modules that were likewise never imported. Found by
    # diffing pynecore/lib/*.py against this set — 'splits' is used by the
    # request.splits samples, 'dayofweek' by session/date logic.
    'dayofweek', 'splits', 'footprint', 'pivotpointtype', 'volume_row',
    # Standalone functions (defined in pynecore/lib/__init__.py)
    'currency', 'timestamp', 'max_bars_back',
    # Plotting and coloring functions
    'bgcolor', 'barcolor', 'alertcondition', 'plotarrow', 'plotbar', 'plotcandle',
    'plotchar', 'plotshape', 'fill',
    # Utility functions
    'na', 'nz', 'fixnan',
    # Constant namespaces
    'xloc', 'yloc', 'position', 'size', 'shape', 'location',
    'extend', 'font', 'text', 'display', 'scale', 'hline', 'format',
    'order',
}

# Drawing and structural types that should be imported from pynecore.types
DRAWING_TYPES: Set[str] = {
    'Line', 'Label', 'Box', 'Table', 'LineFill', 'Polyline', 'Color', 'Matrix',
}

# Names the PyneCore runtime defines that a generated module may read: a user binding of one of them is
# renamed `<name>__ren__` (every such binding, whether or not the script reads the builtin). pynecore.lib's
# modules and functions, its types, and the decorators and namespaces the module itself uses.
PYNECORE_RESERVED_NAMES: FrozenSet[str] = frozenset(
    PYNECORE_LIB_MODULES | DRAWING_TYPES | SERIES_VARIABLES
    | {'method', 'str', 'float', 'int', 'bool', 'Series', 'Persistent', 'PersistentSeries', 'NA', 'ask', 'bid',
       'updatetime', 'dayofweek', 'pivotpointtype', 'volume_row'})

#: the built-in series that Pine also calls as functions (`hour`, `hour(t)`)
SERIES_FUNCTIONS: FrozenSet[str] = frozenset({'time', 'time_close', 'hour', 'minute', 'second', 'year', 'month',
                                              'weekofyear', 'dayofmonth', 'dayofweek'})

#: names given a numbered spelling wherever the script binds them, in source order
#: (`range` -> `range_1`, `text` -> `text_1`), before any other rename of the name
NUMBERED_RESERVED_NAMES: FrozenSet[str] = frozenset({'range', 'text', 'in', 'return', 'as'})
#: and in Pine v4, whose `study()` is declared as `indicator()`
NUMBERED_RESERVED_NAMES_V4: FrozenSet[str] = NUMBERED_RESERVED_NAMES | {'indicator', 'do'}

# Pine's type-cast functions of the drawing types (`box(na)`); each shares its name
# with a pynecore namespace module, which is not callable
DRAWING_CAST_FUNCTIONS: Set[str] = {'box', 'label', 'line', 'linefill', 'table'}

# Chart types that should be imported from pynecore.types.chart
CHART_TYPES: Set[str] = {
    'ChartPoint',
}

# The namespaces whose functions a Pine method call can reach: pynecore's
# `method_call('m', receiver, ...)` picks one of them from the receiver's runtime
# class (core/pine_method.py). Which of them defines a method is read from the
# GENERATED namespace_members table, never from a hand-kept list.
METHOD_NAMESPACES = ('array', 'matrix', 'map', 'line', 'box', 'label', 'table', 'linefill', 'polyline', 'footprint',
                     'volume_row')

# Members of those namespaces that are not methods of an object: the constructors
# (`new`, `new_*`, `from_items` for Pine's array.from), `X.all`, the typed na value,
# a module's `annotations` import, and the `style_*` constants.
NON_METHOD_MEMBERS: FrozenSet[str] = frozenset({'new', 'from_items', 'all', 'na_float', 'annotations'})
NON_METHOD_PREFIXES = ('new_', 'style_')


def _namespaces_by_method() -> Dict[str, Tuple[str, ...]]:
    """Method name -> the METHOD_NAMESPACES that define it, in that order."""
    namespaces_by_method: Dict[str, Tuple[str, ...]] = {}
    for namespace in METHOD_NAMESPACES:
        for member in NAMESPACE_MEMBERS.get(namespace, ()):
            if member not in NON_METHOD_MEMBERS and not member.startswith(NON_METHOD_PREFIXES):
                namespaces_by_method[member] = namespaces_by_method.get(member, ()) + (namespace,)
    return namespaces_by_method


NAMESPACES_BY_METHOD: Dict[str, Tuple[str, ...]] = _namespaces_by_method()


def method_namespaces(method_name: str) -> Tuple[str, ...]:
    """Every method namespace that defines `method_name`; () when none does."""
    return NAMESPACES_BY_METHOD.get(method_name, ())


def get_function_name(pine_func: str) -> str:
    """Get the PyneCore function name for a Pine Script function."""
    return FUNCTION_RENAMES.get(pine_func, pine_func)


def get_module_name(pine_module: str) -> str:
    """Get the PyneCore module name for a Pine Script module."""
    return MODULE_RENAMES.get(pine_module, pine_module)


def get_type_name(pine_type: str) -> str:
    """Get the Python/PyneCore type name for a Pine Script type."""
    # Strip .new suffix from UDT types (e.g., TradeStats.new -> TradeStats)
    if '.new' in pine_type and pine_type and pine_type[0].isupper():
        pine_type = pine_type.replace('.new', '')
    return TYPE_RENAMES.get(pine_type, pine_type)


def is_series_variable(name: str) -> bool:
    """Check if a variable is a built-in Series variable."""
    return name in SERIES_VARIABLES


def is_constant(name: str) -> bool:
    """Check if a name is a built-in constant."""
    return name in CONSTANTS


def needs_plot_style_remap(name: str) -> bool:
    """Check if a name needs plot_style module remapping."""
    return name in PLOT_STYLE_REMAP


def get_plot_style_remap(name: str) -> str:
    """Get the remapped plot_style constant name."""
    return PLOT_STYLE_REMAP.get(name, name)


# Input functions that extract to main() parameters
INPUT_FUNCTIONS: Set[str] = {
    'input.int',
    'input.float',
    'input.bool',
    'input.string',
    'input.color',
    'input.source',
    'input.timeframe',
    'input.session',
    'input.symbol',
    'input.price',
    'input.time',
    'input.enum',
}


def is_input_function(func_name: str) -> bool:
    """Check if function is an input.* function."""
    return func_name in INPUT_FUNCTIONS


# Pine Script input.* positional parameter names (after defval, title).
# Pine Script allows these as positional; PyneCore requires keyword-only.
INPUT_POSITIONAL_PARAMS: Dict[str, list[str]] = {
    'input.int':       ['minval', 'maxval', 'step', 'tooltip', 'inline', 'group', 'confirm', 'display', 'active'],
    'input.float':     ['minval', 'maxval', 'step', 'tooltip', 'inline', 'group', 'confirm', 'display', 'active'],
    'input.bool':      ['tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.string':    ['options', 'tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.color':     ['tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.timeframe': ['options', 'tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.session':   ['options', 'tooltip', 'inline', 'group', 'confirm', 'display'],
    # v6 reference: input.symbol has no options; input.source puts confirm last
    'input.symbol':    ['tooltip', 'inline', 'group', 'confirm', 'display', 'active'],
    'input.source':    ['tooltip', 'inline', 'group', 'display', 'active', 'confirm'],
    'input.price':     ['tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.time':      ['tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.enum':      ['options', 'tooltip', 'inline', 'group', 'confirm', 'display', 'active'],
}

# Pine v2-v4 `input(defval, title, type=input.X)`: the keyword names the kind the v5+ `input.X` functions
# carry in their names. The key is the spelling after an optional `input.` prefix.
LEGACY_INPUT_TYPE_ARGUMENT_FUNCTIONS: Dict[str, str] = {
    'integer': 'input.int', 'float': 'input.float', 'bool': 'input.bool', 'string': 'input.string',
    'symbol': 'input.symbol', 'resolution': 'input.timeframe', 'source': 'input.source',
    'session': 'input.session', 'time': 'input.time', 'color': 'input.color', 'price': 'input.price',
}

# input.int/input.float have a second positional form after `title` (v6 reference): the options form,
# selected by a list third argument (pynecore's _bind_numeric_tail, _NUMERIC_OPTIONS_TAIL).
NUMERIC_INPUT_FUNCTIONS: FrozenSet[str] = frozenset({'input.int', 'input.float'})
NUMERIC_INPUT_OPTIONS_POSITIONAL_PARAMS: list[str] = ['options', 'tooltip', 'inline', 'group', 'confirm',
                                                      'display', 'active']


def input_positional_param_names(func_name: str, positional_args: list) -> list[str]:
    """The names Pine gives the positional arguments of an input.* call after `defval` and `title`
    (empty for a function this table does not list). The one place that decides which positional
    form a call uses, for the transformer and the semantic pass alike."""
    from .ast_nodes import ArrayLiteral
    if func_name in NUMERIC_INPUT_FUNCTIONS and len(positional_args) > 2 \
            and isinstance(positional_args[2], ArrayLiteral):
        return NUMERIC_INPUT_OPTIONS_POSITIONAL_PARAMS
    return INPUT_POSITIONAL_PARAMS.get(func_name, [])


def bind_input_positional_args(func_name: str, positional_args: list) -> Tuple[list, dict]:
    """Split an input.* call's positional arguments into `defval, title` and the rest bound by
    name: PyneCore takes everything after `title` as a keyword. A positional past the last name
    is dropped."""
    names = input_positional_param_names(func_name, positional_args)
    if not names or len(positional_args) <= 2:
        return list(positional_args), {}
    return list(positional_args[:2]), dict(zip(names, positional_args[2:]))


# Pine's positional parameter order of the script declarations (Pine v6 reference).
# The parser binds positionals to these names and codegen emits keywords, so pynecore's own
# order (its strategy inserts calc_on_every_history_tick at 9) does not matter. v4's study()
# binds as 'indicator'.
SCRIPT_DECLARATION_POSITIONAL_PARAMS: Dict[str, list[str]] = {
    'indicator': ['title', 'shorttitle', 'overlay', 'format', 'precision', 'scale', 'max_bars_back',
                  'timeframe', 'timeframe_gaps', 'explicit_plot_zorder', 'max_lines_count',
                  'max_labels_count', 'max_boxes_count', 'calc_bars_count', 'max_polylines_count',
                  'dynamic_requests', 'behind_chart'],
    'strategy': ['title', 'shorttitle', 'overlay', 'format', 'precision', 'scale', 'pyramiding',
                 'calc_on_order_fills', 'calc_on_every_tick', 'max_bars_back',
                 'backtest_fill_limits_assumption', 'default_qty_type', 'default_qty_value',
                 'initial_capital', 'currency', 'slippage', 'commission_type', 'commission_value',
                 'process_orders_on_close', 'close_entries_rule', 'margin_long', 'margin_short',
                 'explicit_plot_zorder', 'max_lines_count', 'max_labels_count', 'max_boxes_count',
                 'calc_bars_count', 'risk_free_rate', 'use_bar_magnifier',
                 'fill_orders_on_standard_ohlc', 'max_polylines_count', 'dynamic_requests',
                 'behind_chart'],
    'library': ['title', 'overlay', 'dynamic_requests'],
}

# How many leading SCRIPT_DECLARATION_POSITIONAL_PARAMS names every Pine version shares.
# A positional past them is bound only under //@version>=5 (v4 has no margin/zorder slots).
VERSION_INVARIANT_POSITIONAL_COUNT: Dict[str, int] = {'indicator': 7, 'strategy': 9, 'library': 2}

# v4 study() keywords that indicator() renamed (Pine v5 migration guide).
STUDY_KEYWORD_RENAMES: Dict[str, str] = {'resolution': 'timeframe', 'resolution_gaps': 'timeframe_gaps'}

# Pine functions whose result is a constant when their arguments are: a script declaration
# argument may be built from them (`pyramiding = BASE + int(50 / QTY)`) and still be hoisted.
DECLARATION_CONSTANT_FUNCTIONS: Set[str] = {
    'int', 'float', 'bool',
    # a color built from constants (`GREEN_80 = color.new(GREEN, 80)` is hoisted)
    'color.new', 'color.rgb', 'color.r', 'color.g', 'color.b', 'color.t',
}
DECLARATION_CONSTANT_FUNCTION_NAMESPACES = ('math.',)
# A call that is a constant only in some spellings: `timestamp("01 Jan 2025 00:00 +0000")` names its time zone in
# the string or as its first argument; `timestamp(2025, 1, 1)` reads the symbol's time zone, which is not known
# when the module loads
DECLARATION_TIMESTAMP = 'timestamp'
# ... except the members of those namespaces that keep state between bars or draw a random number:
# hoisted, they would run once at import.
DECLARATION_STATEFUL_FUNCTIONS: Set[str] = {'math.sum', 'math.random'}
# Builtin namespaces whose members a script declaration argument may name (`strategy.fixed`,
# `format.price`); a member of any other builtin namespace (`syminfo.ticker`) is not a constant.
DECLARATION_CONSTANT_NAMESPACES = ('format.', 'scale.', 'currency.', 'strategy.', 'display.', 'math.', 'color.',
                                   # the constants of the drawing and plotting namespaces (`position.top_right`)
                                   'position.', 'size.', 'shape.', 'location.', 'extend.', 'xloc.', 'yloc.', 'text.',
                                   'font.', 'order.', 'barmerge.', 'line.style_', 'label.style_', 'plot.style_',
                                   'hline.style_')


# Python reserved words that cannot be used as function/variable names.
PYTHON_RESERVED_WORDS: Set[str] = {
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
    'try', 'while', 'with', 'yield',
}


#: The suffix given a Pine name that would clash in Python: a reserved word
#: (`from` -> `from__ren__`) or a name pynecore defines (`color` -> `color__ren__`)
RENAMED_SUFFIX = '__ren__'
#: the suffix of a global named like a function, a type or a namespace the script uses (`thma__global__`)
GLOBAL_SUFFIX = '__global__'


def rename_reserved_word(name: str) -> str:
    """Rename a Python reserved word used as an identifier (`lambda__ren__`); any other name, a
    dotted one too (`ta.sma`), stays as it is."""
    if name in PYTHON_RESERVED_WORDS:
        return f'{name}{RENAMED_SUFFIX}'
    return name


def sanitize_identifier(name: str) -> str:
    """A valid Python identifier for `name`: a Python reserved word is renamed (`lambda__ren__`);
    a name that is no identifier (a leading digit, a dot, a space, an empty string) has each character that may
    not stand there replaced by `_` (`123abc` is `_123abc`, `my.name` is `my_name`). Pine's names are identifiers
    already, so they come out unchanged; a value that is no string is returned as it is."""
    if not isinstance(name, str):
        return name
    if name in PYTHON_RESERVED_WORDS:
        return f'{name}{RENAMED_SUFFIX}'
    if name.isidentifier():
        return name
    spelling = ''.join(character if (character.isalnum() or character == '_') else '_' for character in name)
    if not spelling or not (spelling[0].isalpha() or spelling[0] == '_'):
        spelling = '_' + spelling
    return spelling if spelling.isidentifier() else '_'


# Pine Script built-in "variable-functions": used without () in Pine Script
# but require () in PyneCore because they're implemented as functions.
# Values are the argument text of the call (all empty: each bare form takes none).
TA_VARIABLE_FUNCTIONS: Dict[str, str] = {
    # No argument: pynecore's bare form reads hlc3 itself, and only a call with no
    # arguments is hoisted out of a condition
    'ta.vwap': '',
    'ta.tr': '',           # no args needed
    'ta.accdist': '',
    'ta.obv': '',
    'ta.iii': '',
    'ta.nvi': '',
    'ta.pvi': '',
    'ta.pvt': '',
    'ta.wad': '',
    'ta.wvad': '',
}

# Pine variables that pynecore implements as a function returning an array: as a
# method receiver (`box.all.size()`) they become `array.size(box.all)`
DRAWING_ALL_VARIABLES: Set[str] = {
    f'{namespace}.all' for namespace in ('box', 'label', 'line', 'linefill', 'polyline', 'table')
}
