"""
UDT field defaults.

Pine initializes a field with its default "each time that the script creates a
new instance of the UDT"; a field without a default is na, or false for a bool
in Pine v6 (v5 gives a bool field na too).
"The default value cannot be the result of a function call or any other
expression; only a literal value or a compatible built-in variable is allowed"
(Pine v6 manual, User-defined types).

So a default is one of:
- a literal (`1.5`, `-1.5`, `"a"`, `#ff0000`, `true`, `na`) or a namespace or
  enum constant (`color.red`, `xloc.bar_time`, `syminfo.mintick`, `Mode.a`):
  the same on every `.new()`, so it stays a class attribute, as before;
- an undotted series built-in variable (`open`, `bar_index`, `time`, ...,
  pine_builtins.SERIES_VARIABLES): it changes from bar to bar, so it is read
  per instance, by `field(default_factory=...)`; so is a dotted variable on the
  DYNAMIC_DOTTED_DEFAULTS allowlist (`barstate.isfirst`, `strategy.position_size`). A dotted constant, a function and a variable-function (`ta.tr`) are
  not on it: they stay class attributes.
Anything else is refused here, as TradingView refuses it: a SemanticError
('udt-field-default') at the default's Pine line and column, instead
of failing at load.
"""
from typing import AbstractSet, FrozenSet, List, Optional, Tuple

from .ast_nodes import EnumDecl, Identifier, Literal, NaLiteral, Script, TypeDecl, UnaryOp
from .ast_walk import walk
from .errors import SemanticError
from .pine_builtins import SERIES_VARIABLES
from .scope_renamer import NAMESPACE_HEADS

#: MEASURED 2026-11-17 (BINANCE:BTCUSDT 1D, Pine v6): a field default that calls `array.new<float>()` and one
#: that calls `Inner.new()` are both CE10133 with this sentence. Its wording is wider than the rule it
#: states - a field default may read a built-in series, measured earlier - but it is what the compiler says.
DEFAULT_RULE = 'The default value cannot be a function, variable or calculation.'
# dataclasses.field under a name no UDT field, type or enum is likely to take:
# a class-body name `field` would shadow a bare import
FIELD_FACTORY = '_dataclass_field'
FIELD_FACTORY_IMPORT = f'from dataclasses import field as {FIELD_FACTORY}'

# Dotted built-in variables that change while the script runs, so a UDT field default that
# names one is read on every `.new()`. A curated list, not a namespace prefix: a
# namespace also holds constants (`session.regular`, `strategy.fixed`) and functions
# (`strategy.entry`, `ta.sma`), and a variable-function such as `ta.tr` is a call, not a read.
DYNAMIC_DOTTED_DEFAULTS: FrozenSet[str] = frozenset(
    {f'barstate.{member}' for member in (
        'isconfirmed', 'isfirst', 'ishistory', 'islast', 'islastconfirmedhistory', 'isnew', 'isrealtime')}
    | {f'session.{member}' for member in (
        'isfirstbar', 'isfirstbar_regular', 'islastbar', 'islastbar_regular', 'ismarket', 'ispostmarket',
        'ispremarket')}
    | {f'strategy.{member}' for member in (
        'avg_losing_trade', 'avg_losing_trade_percent', 'avg_trade', 'avg_trade_percent', 'avg_winning_trade',
        'avg_winning_trade_percent', 'closedtrades', 'equity', 'eventrades', 'grossloss', 'grossloss_percent',
        'grossprofit', 'grossprofit_percent', 'losstrades', 'margin_liquidation_price', 'max_contracts_held_all',
        'max_contracts_held_long', 'max_contracts_held_short', 'max_drawdown', 'max_drawdown_percent',
        'max_runup', 'max_runup_percent', 'netprofit', 'netprofit_percent', 'openprofit', 'openprofit_percent',
        'opentrades', 'position_avg_price', 'position_entry_name', 'position_size', 'wintrades')}
)


def is_series_default(default) -> bool:
    """A default that is a series built-in variable, undotted or on the dotted allowlist:
    read on every `.new()`. The one predicate field emission and import discovery share."""
    return isinstance(default, Identifier) and (default.name in SERIES_VARIABLES
                                                 or default.name in DYNAMIC_DOTTED_DEFAULTS)


def constant_heads(script: Script) -> FrozenSet[str]:
    """The heads a dotted constant default may have: the built-in namespaces,
    the script's enums and its imported libraries."""
    return frozenset(NAMESPACE_HEADS | {decl.name for decl in script.declarations if isinstance(decl, EnumDecl)}
                     | {library.alias or library.library for library in script.imports})


def invalid_defaults(type_decl: TypeDecl, heads: AbstractSet[str]) -> List[Tuple[str, object]]:
    """(field name, default) of each default that is neither a literal, a
    constant nor a built-in variable. Read by pine_semantics too, so
    `--validate` refuses what transpile() refuses."""
    return [(field_name, default) for field_name, _, default in type_decl.fields
            if default is not None and not _is_allowed_default(default, heads)]


def default_message(type_decl: TypeDecl, field_name: str) -> str:
    return DEFAULT_RULE


def positioned_part(default):
    """The node whose position a refused default reports: the default, else its
    first part that has a column (an operator carries only its field's line)."""
    return next((node for node in walk(default) if getattr(node, 'column', 0)), default)


def refuse_invalid_defaults(type_decl: TypeDecl, heads: AbstractSet[str]) -> None:
    """Raise SemanticError('udt-field-default') for the first default Pine forbids."""
    for field_name, default in invalid_defaults(type_decl, heads)[:1]:
        at = positioned_part(default)
        raise SemanticError('udt-field-default', default_message(type_decl, field_name),
                            getattr(at, 'line', 0), getattr(at, 'column', 0))


def _is_allowed_default(default, constant_heads: AbstractSet[str]) -> bool:
    if isinstance(default, (Literal, NaLiteral)):
        return True
    if isinstance(default, UnaryOp):
        return default.op in ('-', '+') and isinstance(default.operand, Literal)
    if not isinstance(default, Identifier):
        return False
    head, dot, _ = default.name.partition('.')
    return head in constant_heads if dot else default.name in SERIES_VARIABLES


def emitted_default_requires_na(py_type: str, default, bool_is_never_na: bool) -> bool:
    """True when `field_value` declares the field with `na` or `NA(...)`, so the
    script must import them. The one predicate the import resolver and the
    emitter share. A bool field with no default is false in Pine v6
    (`bool_is_never_na`) and na before it."""
    if isinstance(default, NaLiteral):
        return True
    return default is None and not (py_type == 'bool' and bool_is_never_na)


def field_value(py_type: str, na_type: str, default, generated_default: Optional[str],
                of_its_own_class: bool, bool_is_never_na: bool = True) -> str:
    """The Python value a UDT field is declared with. `of_its_own_class`: the
    field holds an instance of a UDT (its own class or another), made per instance.
    A series built-in or any dotted constant (`color.black`, `xloc.bar_index`, an
    enum member) is read through a factory too."""
    if default is None or isinstance(default, NaLiteral):
        return _na_value(py_type, na_type, default is None, of_its_own_class, bool_is_never_na)
    if is_series_default(default) or is_dotted_default(default):
        return f'{FIELD_FACTORY}(default_factory=lambda: {generated_default})'
    return generated_default


def is_dotted_default(default) -> bool:
    """`color.black`, `xloc.bar_index`, `Mode.a`: a member of a namespace or an enum."""
    from .ast_nodes import MemberAccess
    return (isinstance(default, Identifier) and '.' in default.name) or isinstance(default, MemberAccess)


def _na_value(py_type: str, na_type: str, no_default: bool, of_its_own_class: bool,
              bool_is_never_na: bool) -> str:
    """A field with no default, or `= na`, is an na of its type, `na(T)`:
    `na(float)`, `na(bool)`, `na(Line)`, `na(list)` for an array. A field holding
    its own class is made per instance, since the class is not defined yet."""
    base = na_type.strip("'").split('[')[0]
    if of_its_own_class:
        return f'{FIELD_FACTORY}(default_factory=lambda: na({base}))'
    return f'na({base})'
