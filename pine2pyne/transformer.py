"""
AST Transformer: Pine Script AST → PyneCore-compatible AST.

Implements all 54 transformation rules from the transpiler specification.
"""
import dataclasses
import re
from typing import List, Dict, Any, Union, Optional
from .legacy_sessions import SESSION_HELPER
from .ast_nodes import *
from .symbol_table import SymbolTable, Symbol, VariableKind
from .type_inference import TypeInference, receiver_path
from .ast_walk import walk
from .const_int_division import const_int_names, is_const_int
from .block_value_lowering import (
    bool_is_never_na, keeps_three_state_bool, lower_block_values, refuse_tuple_tails_of_unknown_width,
)
from .bool_value_typing import (
    has_untaken_bool_switch, reads_na_of_bool, stamp_bool_values, stamp_bool_values_as_renamed,
)
from .condition_typing import NUMERIC_SERIES, PIVOTS, ConditionTyping
from .grouped_history_lowering import lower_grouped_history
from .for_loop_end_bound import mark_dynamic_end_bounds
from .emitted_names import EMITTED_HELPER_NAMES, free_spelling, identifier_names
from .scope_renamer import collect_names, fresh_name, rename_shadowing_locals
from .strict_boolean import evaluates_and_or_strictly
from .overload_resolution import OverloadArgumentCoercion, mark_overloaded_declarations
from .udt_field_defaults import constant_heads, refuse_invalid_defaults
from .import_resolver import ImportResolver
from .pine_builtins import (
    get_function_name, get_module_name, get_type_name,
    get_plot_style_remap, needs_plot_style_remap, is_series_variable,
    is_input_function, TYPE_RENAMES, PYNECORE_LIB_MODULES, DRAWING_ALL_VARIABLES, DRAWING_CAST_FUNCTIONS,
    method_namespaces, bind_input_positional_args, SERIES_VARIABLES,
)

# Known module names - used to guard against transforming module.function() calls
# as if they were chained member access (e.g. array.size != obj.size)
KNOWN_MODULES = PYNECORE_LIB_MODULES | {
    'bar_index', 'close', 'open', 'high', 'low', 'volume', 'time',
    'hl2', 'hlc3', 'ohlc4', 'hlcc4',
    'dayofmonth', 'dayofweek', 'hour', 'minute', 'month', 'second',
    'weekofyear', 'year', 'timenow',
    'adjustment', 'earnings', 'dividends', 'splits',
    'order', 'str',
}

# builtin series that are values only, with no namespace members of their own: `close.f()` calls a method on the
# value (KNOWN_MODULES keeps them from being read as chained member access)
_SERIES_VALUES = frozenset({'bar_index', 'close', 'open', 'high', 'low', 'volume', 'hl2', 'hlc3', 'ohlc4', 'hlcc4',
                            'timenow'})

def _library_parameter_spelling(name: str) -> str:
    """The spelling of an exported function's parameter, for a keyword call from an importer: a library renames a
    parameter named like a pynecore name or a Python keyword `name__ren__`, so `m.f(scale = 3.0)` passes
    `scale__ren__=3.0`. A builtin series keeps its name."""
    from .pine_builtins import PYNECORE_RESERVED_NAMES, PYTHON_RESERVED_WORDS, RENAMED_SUFFIX
    if (name in PYNECORE_RESERVED_NAMES and name not in SERIES_VARIABLES) or name in PYTHON_RESERVED_WORDS:
        return f'{name}{RENAMED_SUFFIX}'
    return name


RANGE_HELPER = 'pine_range'  # the runtime's fixed-bound loop helper

# The Python type hints of a Pine map (`map<K, V>` -> `dict[K, V]`): `for [k, v]`
# over one reads its key/value pairs, not index/element
MAP_TYPE_PREFIXES = ('dict',)
# The type hints of every collection a for-in can iterate: array, matrix, map
COLLECTION_TYPE_PREFIXES = ('list', 'Matrix') + MAP_TYPE_PREFIXES

# pynecore's runtime method dispatcher: `method_call('m', receiver, ...)` runs the
# namespace function of the receiver's class
RUNTIME_METHOD_DISPATCH = 'method_call'
# array.new<T>(...) element types with a dedicated pynecore factory
ARRAY_FACTORIES = {
    'string': 'array.new_string',
    'int': 'array.new_int',
    'float': 'array.new_float',
    'bool': 'array.new_bool',
    'color': 'array.new_color',
    'line': 'array.new_line',
    'label': 'array.new_label',
    'box': 'array.new_box',
    'table': 'array.new_table',
    'linefill': 'array.new_linefill',  # `array.new<linefill>()` is `array.new_linefill()`
}
ARRAY_FACTORY_ELEMENT_TYPES = frozenset(ARRAY_FACTORIES)
# Built-in object types without a factory: array.new<T>(size, value) goes through array.new
BUILTIN_OBJECT_ELEMENT_TYPES = frozenset({'chart.point', 'polyline'})
#: Pine v6 built-in types a request returns, which pynecore has no class for
BUILTIN_TYPES_WITHOUT_A_CLASS = frozenset({'footprint', 'volume_row'})
# How a method call reaches a user method (Transformer._user_method_route)
USER_METHOD = 'user method'
RUNTIME_CHOICE = 'user method or builtin, by the receiver class'
from .errors import SemanticError, TransformerError, UnsupportedFeatureError
from .value_blocks import lower_value_blocks


# Builtins whose first parameter is a condition -> its keyword name
_CONDITION_PARAMETERS = {'ta.valuewhen': 'condition', 'ta.barssince': 'condition', 'alertcondition': 'condition'}

# Pine type casts of na -> the Python type of the typed na, NA(T)
_NA_CAST_TYPES = {
    'int': 'int', 'float': 'float', 'bool': 'bool', 'string': 'str', 'color': 'Color',
}


def _spelled_type(type_name: Optional[str]) -> str:
    """A Pine or Python type in one spelling: `array<float>` and `list[float]` are the same."""
    return convert_pine_generic_to_python(type_name).replace(' ', '') if type_name else ''


class PyneTransformedScript:
    """Holds the transformed script ready for code generation."""

    def __init__(self):
        self.imports: List[str] = []
        self.type_declarations: List[TypeDecl] = []  # Custom type definitions
        self.enum_declarations: List[EnumDecl] = []  # Enum definitions
        self.helper_functions: List[FuncDecl] = []
        self.main_decorator: Optional[Union[IndicatorDecl, StrategyDecl]] = None
        self.main_params: List[Parameter] = []
        self.main_body: List[Statement] = []
        self.global_vars: List[Union[VarDecl, Assignment]] = []
        self.module_constants: List[Assignment] = []  # Pre-main literal constants
        self.strict_boolean_operators = False  # Pine v5 and earlier: and/or evaluate both operands
        self.bool_is_never_na = True  # False for Pine v5 and earlier, where a UDT bool field without a default is na
        self.shadowed_namespaces: Dict[str, str] = {}  # built-in namespace alias of a library -> its module name
        self.line_blocks: Dict[int, int] = {}  # Pine statement line -> the id of its block
        self.reads_na_of_bool = False  # Pine v5: a named bool is read with history or by na() (the script's na_bool)
        self.pine_version = 6  # the script's //@version
        # names whose user method's result is read with history (`x.src(s)[1]`): declared a series
        self.method_history_receivers: set = set()


def convert_pine_generic_to_python(pine_type: str) -> str:
    """
    Convert Pine Script generic types to Python type hints.

    Examples:
        map<string, float> -> dict[str, float]
        array<int> -> list[int]
    """
    # Check if it contains generic syntax
    if '<' not in pine_type or '>' not in pine_type:
        return get_type_name(pine_type)

    # Extract base type and parameters
    if '<' in pine_type:
        base_type = pine_type[:pine_type.index('<')]
        params_str = pine_type[pine_type.index('<')+1:pine_type.rindex('>')]
        params = [p.strip() for p in params_str.split(',')]

        # Convert base type
        if base_type == 'map':
            base_type = 'dict'
        elif base_type == 'array':
            base_type = 'list'
        else:
            # For other types like matrix, line, etc., use type name mapping
            base_type = get_type_name(base_type)

        # Convert parameter types
        converted_params = [get_type_name(p) for p in params]

        return f"{base_type}[{', '.join(converted_params)}]"

    return get_type_name(pine_type)


def _pine_spelling(name: str) -> str:
    """The Pine name an emitted name spells: `close__ren__` -> `close`, `x__global__` -> `x`."""
    return re.sub(r'(__ren__|__global__|__[0-9a-f]{8}__)+$', '', name)




def _declared_names(body) -> set:
    """The names a function body declares: assignments, var/varip, tuple elements and loop counters."""
    names = set()
    for node in walk(body):
        if isinstance(node, Assignment):
            names.update(node.target.names if isinstance(node.target, TupleDestructure) else [node.target])
        elif isinstance(node, (VarDecl, VaripDecl)):
            names.add(node.name)
        elif isinstance(node, ForLoop):
            names.add(node.var)
        elif isinstance(node, ForInLoop):
            names.update(node.vars)
    return names


#: the helper a bool expression that may be na is plotted through, defined first in main
BOOL_MARKER = '__pc_bool_marker'


def _bool_marker_declaration() -> FuncDecl:
    """`__pc_bool_marker(bool b) => na(b) ? na : b ? 1 : 0`"""
    b = Identifier(name='b')
    return FuncDecl(name=BOOL_MARKER, params=[Parameter(name='b', type_hint='bool')],
                    body=TernaryOp(condition=FunctionCall(func='na', args=[b], kwargs={}), true_expr=NaLiteral(),
                                   false_expr=TernaryOp(condition=b, true_expr=Literal(value=1, literal_type='int'),
                                                        false_expr=Literal(value=0, literal_type='int'))))


def _unread_history_declarations(script: Script) -> set[int]:
    """The ids of the declarations in a nested block (of main() or a function) whose Pine binding no read
    `x[n]` resolves to, though another binding of the same name in that scope is read so."""
    unread: set[int] = set()

    def scope(statements) -> None:
        read: set[int] = set()
        nested: dict[int, str] = {}
        history_names: set[str] = set()

        def resolve(name, stack):
            for names in reversed(stack):
                if name in names:
                    return names[name]
            return None

        def expressions(node, stack) -> None:
            for found in walk(node):
                if isinstance(found, IndexAccess) and isinstance(found.object, Identifier):
                    history_names.add(found.object.name)
                    declaration = resolve(found.object.name, stack)
                    if declaration is not None:
                        read.add(declaration)

        def block(body, stack) -> None:
            stack = stack + [{}]
            for statement in body or []:
                one(statement, stack)

        def one(statement, stack) -> None:
            if isinstance(statement, FuncDecl):
                return
            if isinstance(statement, IfStatement):
                expressions(statement.condition, stack)
                block(statement.body, stack)
                for condition, body in statement.elseifs:
                    expressions(condition, stack)
                    block(body, stack)
                block(statement.else_body, stack)
            elif isinstance(statement, (ForLoop, ForInLoop, WhileLoop)):
                expressions(dataclasses.replace(statement, body=[]), stack)
                block(statement.body, stack)
            elif isinstance(statement, SwitchStatement):
                expressions(statement.expr, stack)
                for case, body in statement.cases:
                    expressions(case, stack)
                    block(body if isinstance(body, list) else [body], stack)
                block(statement.default if isinstance(statement.default, list) else
                      [statement.default] if statement.default is not None else [], stack)
            elif isinstance(statement, (Assignment, VarDecl, VaripDecl)) and isinstance(
                    statement.target if isinstance(statement, Assignment) else statement.name, str):
                name = statement.target if isinstance(statement, Assignment) else statement.name
                expressions(statement.value, stack)
                stack[-1][name] = id(statement)
                if len(stack) > 1:
                    nested[id(statement)] = name
            else:
                expressions(statement, stack)

        block(statements, [])
        unread.update(declaration for declaration, name in nested.items()
                      if declaration not in read and name in history_names)

    scope([item for item in list(script.declarations) + list(script.body)])
    for function in walk(script):
        if isinstance(function, FuncDecl) and isinstance(function.body, list):
            scope(function.body)
    return unread


def _casts_alike(operand: Expression) -> bool:
    """Whether Pine v5 `bool(operand)` and `cast_bool(operand)` agree whatever the bar: a bool literal, a number
    literal that is 0 or beyond cast_bool's 1e-10 tolerance, a comparison (false, never na, where an operand is na)."""
    if isinstance(operand, Identifier):
        return operand.name in ('true', 'false')
    if isinstance(operand, Literal):
        if operand.literal_type == 'bool':
            return True
        if operand.literal_type in ('int', 'float') and isinstance(operand.value, (int, float)):
            return operand.value == 0 or abs(operand.value) > 1e-10
        return False
    return isinstance(operand, BinaryOp) and operand.op in ('==', '!=', '<', '>', '<=', '>=')


#: the built-in collections of every drawing of a kind, and that kind
_ALL_COLLECTIONS = {f'{kind}.all': kind for kind in ('line', 'label', 'box', 'linefill', 'table', 'polyline')}


def _ternary_tree(ternary: TernaryOp):
    """The ids of the ternaries `ternary` holds as arms, at any depth, and the names among its bare arms."""
    ids, names, pending = set(), set(), [ternary]
    while pending:
        node = pending.pop()
        if isinstance(node, TernaryOp):
            ids.add(id(node))
            pending.extend([node.true_expr, node.false_expr])
        elif isinstance(node, Identifier):
            names.add(node.name)
    return ids, names


_POSITION_FIELDS = re.compile(r'\b(line|column|source_line|end_line|value_line|question_line)=\d+')


def _expression_text(expression) -> str:
    """`expression` as text, the same wherever it is written."""
    return _POSITION_FIELDS.sub('', repr(expression))


def _call_signature(call: FunctionCall) -> str:
    """The text of a call with literal and name arguments (`ta.pivotlow(3, 3)`), the same for every copy of it."""
    def text(node) -> str:
        if isinstance(node, Literal):
            return repr(node.value)
        if isinstance(node, Identifier):
            return node.name
        return f'<{id(node)}>'
    arguments = [text(argument) for argument in call.args] + [f'{k}={text(v)}' for k, v in call.kwargs.items()]
    return f"{call.func}({', '.join(arguments)})"


_CHART_COLOURS = frozenset({'chart.fg_color', 'chart.bg_color'})
#: Pine's named colour constants (`color.red`)
COLOUR_CONSTANT_NAMES = frozenset({
    'aqua', 'black', 'blue', 'fuchsia', 'gray', 'green', 'lime', 'maroon', 'navy', 'olive', 'orange',
    'purple', 'red', 'silver', 'teal', 'white', 'yellow'})


def _and_or_chain(chain: BinaryOp):
    """The `and`/`or` nodes of the chain `chain` heads (of its operator, not in parentheses) and its operands."""
    nodes, operands, pending = [], [], [chain]
    while pending:
        node = pending.pop()
        if isinstance(node, BinaryOp) and node.op == chain.op and (node is chain or not node.parenthesized):
            nodes.append(node)
            pending.extend([node.right, node.left])
        else:
            operands.append(node)
    return nodes, operands


def _leftmost_operand(chain: Expression) -> Expression:
    """The first operand of an `and`/`or` chain (`a` of `a and b or c`)."""
    while isinstance(chain, BinaryOp) and chain.op in ('and', 'or'):
        chain = chain.left
    return chain


#: the calls whose number a typed bool declaration keeps (`bool a = time(tf, s)`)
_BOOL_DECLARATION_KEPT_CALLS = frozenset({'time', 'time_close', 'ta.dev'})


def _is_kept_as_a_bool_value(value) -> bool:
    """A typed bool declaration's value written as it is: a name, or a direct call of `time`, `time_close` or
    `ta.dev` (neither is cast; `ta.change(...)` there is)."""
    if isinstance(value, Identifier):
        return '.' not in value.name
    return isinstance(value, FunctionCall) and isinstance(value.func, str) and value.func in _BOOL_DECLARATION_KEPT_CALLS


def _has_a_na_arm_in_its_else_chain(ternary: TernaryOp) -> bool:
    """`c ? x : na`, `c ? na : x`, or a chain whose last ternary has a na arm (`c ? x : d ? y : na`)."""
    if sum(isinstance(arm, NaLiteral) for arm in (ternary.true_expr, ternary.false_expr)) == 1:
        return True
    last = ternary
    while isinstance(last.false_expr, TernaryOp):
        last = last.false_expr
    return last is not ternary and isinstance(last.false_expr, NaLiteral)


def _else_chain(ternary: Optional[Expression]) -> set[int]:
    """The ids of `ternary` and of the ternaries its else arm is, at any depth (`a ? x : b ? y : na`)."""
    found: set[int] = set()
    while isinstance(ternary, TernaryOp):
        found.add(id(ternary))
        ternary = ternary.false_expr
    return found


def _arm_ternaries(ternary: Optional[Expression]) -> set[int]:
    """The ids of `ternary` and of the ternaries its arms are, at any depth: each gives the value it is part of."""
    found: set[int] = set()
    pending = [ternary]
    while pending:
        node = pending.pop()
        if isinstance(node, TernaryOp):
            found.add(id(node))
            pending += [node.true_expr, node.false_expr]
    return found


def _pivot_names(script: Script) -> set:
    """The names given a `ta.pivothigh`/`ta.pivotlow` value anywhere, a declaration or a `:=`."""
    names = set()
    for node in walk(script):
        if isinstance(node, (Assignment, Reassignment)) and isinstance(node.target, str):
            name, value = node.target, node.value
        elif isinstance(node, (VarDecl, VaripDecl)):
            name, value = node.name, node.value
        else:
            continue
        if isinstance(value, FunctionCall) and isinstance(value.func, str) and value.func in PIVOTS:
            names.add(name)
    return names


def _condition_names(expression) -> set:
    """The names an expression reads as a condition: itself, or through `and`, `or` and `not`."""
    if isinstance(expression, Identifier):
        return {expression.name}
    if isinstance(expression, BinaryOp) and expression.op in ('and', 'or'):
        return _condition_names(expression.left) | _condition_names(expression.right)
    if isinstance(expression, UnaryOp) and expression.op == 'not':
        return _condition_names(expression.operand)
    return set()


def _has_a_comparison(condition, in_its_chain: bool) -> bool:
    """`condition` is a comparison (`a > b`), or (`in_its_chain`) an `and`/`or` with one among its operands."""
    if isinstance(condition, BinaryOp) and condition.op in ('==', '!=', '<', '>', '<=', '>='):
        return True
    return in_its_chain and isinstance(condition, BinaryOp) and condition.op in ('and', 'or') and (
        _has_a_comparison(condition.left, True) or _has_a_comparison(condition.right, True))


def _names_read_as_conditions(script: Script) -> set:
    """The names an if/while header or a call argument reads as a condition (`if x`, `plotshape(c or x)`)."""
    names = set()
    for node in walk(script):
        if isinstance(node, IfStatement):
            names |= _condition_names(node.condition)
            for condition, _ in node.elseifs:
                names |= _condition_names(condition)
        elif isinstance(node, WhileLoop):
            names |= _condition_names(node.condition)
        elif isinstance(node, FunctionCall):
            for argument in [*node.args, *node.kwargs.values()]:
                names |= _condition_names(argument)
    return names


def _function_result_ternaries(script: Script) -> set:
    """The ids of the ternaries a user function returns (its tail)."""
    ids = set()
    for node in walk(script):
        if isinstance(node, FuncDecl):
            tail = node.body[-1] if isinstance(node.body, list) and node.body else node.body
            tail = getattr(tail, 'expr', tail)
            for inner in walk(tail):
                if isinstance(inner, TernaryOp):
                    ids.add(id(inner))
    return ids

def _line_leading_types(script: Script) -> Dict[int, str]:
    """line -> the type of its first typed `= na` declaration, for each line that declares more than one name:
    every `= na` of the line takes it (`var bool d = false, var line l = na, var label m = na`: `m` is `na(Line)`)."""
    by_line: Dict[int, list] = {}
    for node in walk(script):
        if isinstance(node, (VarDecl, VaripDecl, Assignment)) and getattr(node, 'type_hint', None) and node.line:
            by_line.setdefault(node.line, []).append(node)
    leading = {line: min((node for node in nodes if isinstance(node.value, NaLiteral)), key=lambda node: node.column)
               for line, nodes in by_line.items()
               if len(nodes) > 1 and any(isinstance(node.value, NaLiteral) for node in nodes)}
    return {line: node.type_hint for line, node in leading.items()}

class Transformer:
    """Transforms Pine Script AST to PyneCore-compatible form.

    One instance transforms ONE script: inputs, symbols, import state and method
    maps live on the instance, so a second transform() raises TransformerError. Build a fresh
    Transformer per script, as transpile() does.
    """

    def __init__(self):
        self._transform_started = False  # set by transform() before any work; see transform()
        self.symbol_table = SymbolTable()
        self.type_inference = TypeInference(self.symbol_table)
        self.import_resolver = ImportResolver()
        self.input_declarations: List[InputDecl] = []
        self.current_function_params: Dict[str, str] = {}  # Track current function parameters
        self.current_function_shadowed_names: set = set()  # Untyped parameters that also name a global
        self.bool_is_never_na = True  # Pine v6 and later; set per script by transform()
        self.pine_version = 6  # set per script by transform()
        self.three_state_bool = False  # Pine v4 and v5: the decorator sets na_bool; set per script by transform()
        self.type_inference.current_scope = lambda: (self.current_function_params, self.current_function_shadowed_names)
        # Each @method function name -> the Python types of its `this` parameter
        self.method_functions: Dict[str, set] = {}
        self._renamed_functions: Dict[str, str] = {}  # Pine name -> sanitized Python name
        self._pine_keywords: Dict[str, Dict[str, str]] = {}  # set by transform()
        # A switch-subject temporary avoids every Pine name of the script; the
        # names are collected on first need
        self._names_source: Optional[Script] = None
        self._object_element_types: Optional[set] = None  # array.new<T> element types
        self._used_names: Optional[set] = None
        self._range_helper: Optional[str] = None  # the name pine_range is imported under
        # Identifiers of the script and the names the module defines itself: what a generated
        # spelling (the tuple temporary, a hidden source parameter) must not take
        self._identifier_names: Optional[set] = None
        # float(...) for a float-typed argument where int and float overloads meet
        self._overload_arguments = OverloadArgumentCoercion(
            Script(), self._visit_ast_nodes, self._receiver_type, self.symbol_table.udt_field_types)

        # PRIORITY 1 FIX: Track Series variables in user-defined functions
        self.in_user_function = False  # Are we inside a user-defined function?
        self.series_variables: set = set()  # Variables that need to be Series objects
        self.function_local_vars: set = set()  # All local variables in current function
        # Names the scope being emitted (main, or the current function) reads with `[n]`
        self.names_read_with_history: set = set()

    def _visit_ast_nodes(self, nodes: list, visitor_fn) -> None:
        """Call visitor_fn on every AST node, parents first. The shared walk also
        enters else-if arms and switch cases, which are tuples."""
        for node in walk(nodes):
            visitor_fn(node)

    def _function_shadowed_names(self, func: FuncDecl) -> set:
        """Untyped parameters, and locals, that also name a global.

        The transformer types such a name from the GLOBAL symbol, so a field path
        on it must not be typed from there. A function local keeps the name of a global
        it hides (it is not renamed): it is typed from its own value only.
        """
        names = {param.name for param in func.params
                 if not param.type_hint and self.symbol_table.lookup_global(param.name)}
        body = func.body if isinstance(func.body, list) else []
        for node in walk(body):
            if isinstance(node, Assignment):
                bound = node.target.names if isinstance(node.target, TupleDestructure) else [node.target]
            elif isinstance(node, (VarDecl, VaripDecl)):
                bound = [node.name]
            elif isinstance(node, ForLoop):
                bound = [node.var]
            elif isinstance(node, ForInLoop):
                bound = list(node.vars)
            else:
                continue
            names.update(name for name in bound if isinstance(name, str) and self.symbol_table.lookup_global(name))
        return names

    def _collect_reassignment_vars(self, script: Script) -> set:
        """Collect variable names that are reassigned with := operator."""
        reassign_vars = set()

        def visitor(node):
            if isinstance(node, Reassignment):
                reassign_vars.add(node.target)

        self._visit_ast_nodes(list(script.declarations) + list(script.body), visitor)
        return reassign_vars

    def _find_series_variables_in_function(self, func_body: List[Statement]) -> set:
        """Find variables in a function that need to be Series objects.
        A variable needs to be Series if it's reassigned with := or indexed with [n].
        """
        series_vars = set()

        def visitor(node):
            if isinstance(node, Reassignment):
                series_vars.add(node.target)
            if isinstance(node, IndexAccess):
                if isinstance(node.object, Identifier):
                    series_vars.add(node.object.name)

        self._visit_ast_nodes(func_body, visitor)
        return series_vars

    def transform(self, script: Script) -> PyneTransformedScript:
        """Transform entire script. Single use: a second call raises."""
        if self._transform_started:
            raise TransformerError(
                'a Transformer transforms one script and keeps its state; create a fresh Transformer '
                'for the next script (pine2pyne.transpile() does)')
        self._transform_started = True  # before any work, so a failed transform consumes the instance too
        output = PyneTransformedScript()
        output.strict_boolean_operators = evaluates_and_or_strictly(script)
        version_text = getattr(script.version, 'version', '') if script.version else ''
        self.pine_version = int(version_text) if str(version_text).isdigit() else 6
        output.bool_is_never_na = bool_is_never_na(script)

        # A Pine local that shadows an outer name gets its own name,
        # so every stage below sees one binding per name
        # Pine v6: an untaken if/switch of bool type is false, not na
        self.bool_is_never_na = bool_is_never_na(script)
        self.three_state_bool = keeps_three_state_bool(script)
        # A binding named like a pynecore.lib name the script also uses gets its
        # own name; a namespace under an input or export that keeps it gets an alias
        # if/switch values become statements around __block_result__ temporaries
        # Pine v5: a switch without a default that gives a bool may be na (before the switches are lowered)
        untaken_bool_switch = self.pine_version == 5 and has_untaken_bool_switch(script)
        # Pine v6: an untaken if/switch of type bool is false; typed before the values are lowered
        if self.bool_is_never_na:
            script = stamp_bool_values_as_renamed(script, rename_shadowing_locals(script).script)
        refuse_tuple_tails_of_unknown_width(script)
        script = lower_value_blocks(script)
        renamed = rename_shadowing_locals(script)
        script = renamed.script
        # the block id of each Pine statement line: a temporary the emitter hoists into a nested block takes it
        output.line_blocks = renamed.line_blocks
        # Pine v5: `na_bool=True` is marked only where a named bool is read with history or by na()
        output.pine_version = self.pine_version
        methods = {node.name for node in walk(script) if isinstance(node, FuncDecl) and node.is_method}
        # the receiver of a method whose result is read with history (`x.src(s)[1]`, `a.sum()[1]`; not `src(x, s)[1]`
        # nor `array.sum(a)[1]`)
        namespaces_and_types = set(KNOWN_MODULES) | {imported.alias or imported.library for imported in script.imports} \
            | {node.name for node in walk(script) if isinstance(node, (TypeDecl, EnumDecl))}
        output.method_history_receivers = {
            node.object.func.split('.')[0] for node in walk(script)
            if isinstance(node, IndexAccess) and isinstance(node.object, FunctionCall)
            and isinstance(node.object.func, str) and node.object.func.count('.') == 1
            and (node.object.func.split('.')[1] in methods
                 or node.object.func.split('.')[0] not in namespaces_and_types)}
        output.reads_na_of_bool = (untaken_bool_switch or reads_na_of_bool(script)) if self.pine_version >= 5 else False
        # whether a bool may be na: always in Pine v4, in v5 where the script reads one's na (`na_bool=True`)
        self._bool_may_be_na = self.pine_version <= 4 or (self.pine_version == 5 and output.reads_na_of_bool)
        # which values are read as numbers in a condition (cast_bool)
        self._condition_typing = ConditionTyping(script)
        # Pine v6 `var float a = na, var int b = na`: every na of the line is typed like its first `= na`
        # declaration
        self._line_na_types = _line_leading_types(script) if self.pine_version >= 6 else {}
        self._pivot_names = _pivot_names(script)
        self._na_observed_names = _names_read_as_conditions(script)
        self._na_keeping_ternaries = _function_result_ternaries(script)
        # Pine v4/v5: the division of two const ints is an int (`cast_int(9 / 2)`)
        self._const_ints = const_int_names(script) if self.pine_version <= 5 else None
        self.import_resolver.aliased_modules = dict(renamed.namespace_aliases)
        # user function or method name -> {emitted parameter name: its Pine name}:
        # a builtin method of the same name is called with the Pine keyword
        self._pine_keywords = {name: {emitted: pine for pine, emitted in renames.items()}
                               for name, renames in renamed.parameter_names.items()}
        # Pine v6: a for loop whose end may change in its body re-reads it on
        # every pass; marked before the lowering, which refuses a history site there
        script = mark_dynamic_end_bounds(script)
        # A history index on an expression reads a named local. It
        # runs first: a branch it turns into a block is lowered by the next pass
        # the history of an expression is read with inline_series(expr, n): no lowering
        # (`_transform_index_access`); the grouped-history lowering is not run
        # Pine v6: the bool type of every if/switch/loop value and function,
        # decided once before the lowering rewrites them
        if self.bool_is_never_na:
            script = stamp_bool_values(script)
        # A multi-statement branch of an if/switch used as a value becomes
        # statements that assign the value
        script = lower_block_values(script)
        self._names_source, self._used_names, self._range_helper = script, None, None
        self._read_name_set = None
        self._holds_maps = None
        self._object_element_types = None  # array.new<T> element types, read on first use
        self._identifier_names = None

        # Build symbol table
        self._build_symbol_table(script)

        # Infer types
        self.type_inference.infer_types(script)
        self._overload_arguments = OverloadArgumentCoercion(
            script, self._visit_ast_nodes, self._receiver_type, self.symbol_table.udt_field_types)

        # (Removed: _history_ref_vars replaced by symbol.is_indexed from type inference)

        # Collect variables that use := (reassignment)
        self._reassignment_vars = self._collect_reassignment_vars(script)
        self._history_method_names = frozenset(node.name for node in walk(script)
                                               if isinstance(node, FuncDecl) and node.is_method)
        self.names_read_with_history = self._main_scope_history_reads(script)
        # a declaration in a block whose own binding nothing reads with history (`x` declared in two `if`
        # blocks, `x[1]` read in one): not a series there
        self._unread_history_declarations = _unread_history_declarations(script)

        # Analyze imports
        self.import_resolver.analyze(script)

        # Transform script declaration (indicator/strategy)
        if script.script_decl:
            output.main_decorator = self._transform_script_declaration(script.script_decl)

        # Extract input declarations to main() parameters
        self._extract_input_declarations(script)
        self._source_defaults = self._series_source_defaults(script)
        output.main_params = self._create_main_parameters()

        # Transform type and enum declarations - stay at module level
        heads = constant_heads(script)
        for decl in script.declarations:
            if isinstance(decl, TypeDecl):
                refuse_invalid_defaults(decl, heads)  # only literals and built-ins
                output.type_declarations.append(decl)  # Keep as-is for codegen
            elif isinstance(decl, EnumDecl):
                output.enum_declarations.append(decl)  # Keep as-is for codegen

        # Identify body assignments that must be module-level constants.
        # In Pine Script, declarations before the first function call (like
        # `const string G_ENTRY = '...'` or `DEFAULT_PYRAMIDING = 6`) can be
        # referenced by input.* group= params and strategy() kwargs, which
        # become main() parameter defaults evaluated at function definition time.
        # These must be emitted at module level, not inside main().
        module_const_names = self._collect_module_constant_names(script)

        # Merge all non-type, non-input declarations with body statements
        # sorted by source line number to preserve original Pine Script ordering
        all_items: List[tuple] = []
        for decl in script.declarations:
            if isinstance(decl, (VarDecl, VaripDecl)):
                all_items.append(('var', decl))
            elif isinstance(decl, FuncDecl):
                all_items.append(('func', decl))
            # TypeDecl and InputDecl are handled separately above

        for stmt in script.body:
            all_items.append(('body', stmt))
        for decl in self.input_declarations:
            if decl.name in self._source_defaults:
                all_items.append(('source_default', decl))

        # Sort by line number to preserve source order
        all_items.sort(key=lambda x: (x[1].line, getattr(x[1], 'column', 0) or 0))  # `a = 1, var b = 2`: a first

        # Process all items in source order into main_body
        # Propagate source line numbers for blank line detection in codegen
        for item_type, item in all_items:
            source_line = item.line
            if item_type == 'var':
                # Check if this var declaration should be a module-level constant
                # (module_const_names holds only names whose value may be hoisted)
                if isinstance(item, (VarDecl, VaripDecl)) and item.name in module_const_names:
                    # Emit as a plain module-level assignment with only its declared
                    # type: pynecore allows a Persistent or Series only inside a function
                    transformed = Assignment(
                        target=item.name, value=self._transform_expression(item.value),
                        type_hint=convert_pine_generic_to_python(item.type_hint) if item.type_hint else None)
                    transformed.line = source_line
                    output.module_constants.append(transformed)
                    continue
                result = self._transform_var_declaration(item)
                result.line = source_line
                output.main_body.append(result)
            elif item_type == 'func':
                result = self._transform_function(item)
                result.line = source_line
                output.main_body.append(result)
            elif item_type == 'source_default':
                result = self._source_default_resolution(item.name)
                result.line = source_line
                output.main_body.append(result)
            else:  # body
                # Check if this is a literal assignment that should be module-level
                if (isinstance(item, Assignment) and
                    isinstance(item.target, str) and
                    item.target in module_const_names):
                    # Emit as module-level constant instead of inside main()
                    transformed = self._transform_statement(item)
                    if transformed and not isinstance(transformed, list):
                        transformed.line = source_line
                        output.module_constants.append(transformed)
                    continue

                transformed = self._transform_statement(item)
                if transformed:
                    if isinstance(transformed, list):
                        for t in transformed:
                            t.line = source_line
                        output.main_body.extend(transformed)
                    else:
                        transformed.line = source_line
                        output.main_body.append(transformed)

        # Same-name declarations become pynecore @overload implementations
        if getattr(self, '_uses_bool_marker', False):
            # after the Pine v4 session helper, which main() defines first
            first = 1 if output.main_body and getattr(output.main_body[0], 'name', None) == SESSION_HELPER else 0
            output.main_body.insert(first, self._transform_function(_bool_marker_declaration()))
        output.main_body, self.import_resolver.uses_overload = mark_overloaded_declarations(
            output.main_body, {decl.name for decl in output.enum_declarations},
            self.type_inference.declared_type_name)

        # Analyze transformed AST for type names in type hints (Matrix, Line, etc.)
        self._analyze_transformed_types(output)

        # Generate imports
        output.imports = self.import_resolver.generate_imports()
        output.shadowed_namespaces = dict(self.import_resolver.shadowed_namespaces or {})

        return output

    def _get_func_name_from_call(self, call: FunctionCall) -> Optional[str]:
        """Extract function name string from a FunctionCall node."""
        if isinstance(call.func, str):
            return call.func
        elif isinstance(call.func, MemberAccess):
            if isinstance(call.func.object, str):
                return f"{call.func.object}.{call.func.member}"
            elif isinstance(call.func.object, Identifier):
                return f"{call.func.object.name}.{call.func.member}"
        return None

    def _analyze_transformed_types(self, output: PyneTransformedScript) -> None:
        """Analyze transformed AST to extract type names from type hints."""
        import re

        # Track if Series is actually used
        series_used = False
        persistent_used = False

        def extract_types_from_hint(type_hint: str) -> None:
            """Extract type names like Line, Matrix from type hints."""
            nonlocal series_used, persistent_used

            if not type_hint:
                return

            # Check for Series and Persistent usage
            # A typeless na initialiser is annotated bare `Series`;
            # `PersistentSeries` alone needs no `Series` import
            if re.search(r'\bSeries\b', type_hint):
                series_used = True
            if re.search(r'\bPersistent\b', type_hint):
                persistent_used = True

            # Match capital letter type names that might be drawing/structural types
            for match in re.finditer(r'\b([A-Z][a-zA-Z]*)\b', type_hint):
                type_name = match.group(1)
                # Add to import resolver if it's a known type
                self.import_resolver.add_drawing_type(type_name)

        def scan_statement(stmt: Statement) -> None:
            """Recursively scan a statement for type hints."""
            # Handle lists (flattened statement lists from hoisting)
            if isinstance(stmt, list):
                for s in stmt:
                    scan_statement(s)
                return
            if stmt is None:
                return

            # Check if statement has type hint
            if hasattr(stmt, 'type_hint') and stmt.type_hint:
                extract_types_from_hint(stmt.type_hint)

            # Recursively scan nested structures
            if isinstance(stmt, FuncDecl) and hasattr(stmt, 'params'):
                for param in stmt.params:
                    if hasattr(param, 'type_hint') and param.type_hint:
                        extract_types_from_hint(param.type_hint)
                if hasattr(stmt, 'body') and isinstance(stmt.body, list):
                    for body_stmt in stmt.body:
                        scan_statement(body_stmt)

            # Scan all compound statement bodies
            elif isinstance(stmt, IfStatement):
                for body_stmt in (stmt.body or []):
                    scan_statement(body_stmt)
                for _, elif_body in (stmt.elseifs or []):
                    for body_stmt in elif_body:
                        scan_statement(body_stmt)
                for body_stmt in (stmt.else_body or []):
                    scan_statement(body_stmt)
            elif isinstance(stmt, (ForLoop, ForInLoop, WhileLoop)):
                for body_stmt in (stmt.body or []):
                    scan_statement(body_stmt)

        # Scan global vars for type hints (var/varip declarations)
        for var in output.global_vars:
            if hasattr(var, 'type_hint') and var.type_hint:
                extract_types_from_hint(var.type_hint)

        # Scan module constants and the main body for type hints
        for stmt in output.module_constants + output.main_body:
            scan_statement(stmt)

        # Scan main parameters for type hints
        for param in output.main_params:
            if hasattr(param, 'type_hint') and param.type_hint:
                extract_types_from_hint(param.type_hint)

        # Update import resolver based on actual usage in transformed code
        self.import_resolver.uses_series = series_used
        self.import_resolver.uses_persistent = persistent_used

    # ========================================================================
    # Symbol table building
    # ========================================================================

    def _build_symbol_table(self, script: Script) -> None:
        """Build symbol table from script declarations."""
        # Record declared UDT field types first: typing a var below (e.g.
        # `var b = t.bx`) and every later method call on a field need them.
        for decl in script.declarations:
            if isinstance(decl, TypeDecl):
                self.symbol_table.udt_field_types[decl.name] = {
                    field_name: convert_pine_generic_to_python(field_type)
                    for field_name, field_type, *_ in decl.fields if field_type
                }

        # Register input declarations
        for decl in script.declarations:
            if isinstance(decl, InputDecl):
                symbol = Symbol(
                    decl.name,
                    VariableKind.INPUT,
                    is_global=True
                )
                self.symbol_table.define(symbol)

        # Register var/varip declarations
        for decl in script.declarations:
            if isinstance(decl, VarDecl):
                type_hint = self._infer_type_from_declaration(decl)
                symbol = Symbol(
                    decl.name,
                    VariableKind.VAR,
                    type_hint=type_hint,
                    is_global=True
                )
                self.symbol_table.define(symbol)
            elif isinstance(decl, VaripDecl):
                type_hint = self._infer_type_from_declaration(decl)
                symbol = Symbol(
                    decl.name,
                    VariableKind.VARIP,
                    type_hint=type_hint,
                    is_global=True
                )
                self.symbol_table.define(symbol)

        # Register type declarations (UDTs)
        for decl in script.declarations:
            if isinstance(decl, TypeDecl):
                symbol = Symbol(
                    decl.name,
                    VariableKind.TYPE,
                    is_global=True
                )
                self.symbol_table.define(symbol)

        # Register function declarations
        for decl in script.declarations:
            if isinstance(decl, FuncDecl):
                symbol = Symbol(
                    decl.name,
                    VariableKind.FUNCTION,
                    is_global=True
                )
                self.symbol_table.define(symbol)

        # Register assignments as variables (including nested scopes)
        self._register_assignments_recursive(script.body)

    def _register_assignments_recursive(self, stmts: List[Statement]) -> None:
        """Recursively register assignments in the symbol table, including nested scopes."""
        for stmt in stmts:
            if isinstance(stmt, Assignment) and isinstance(stmt.target, str):
                if not self.symbol_table.lookup_global(stmt.target):
                    if hasattr(stmt, 'type_hint') and stmt.type_hint:
                        type_hint = convert_pine_generic_to_python(stmt.type_hint)
                    else:
                        type_hint = self.type_inference.infer_type_hint(stmt.value)
                    symbol = Symbol(
                        stmt.target,
                        VariableKind.SERIES,
                        type_hint=type_hint,
                        is_global=True
                    )
                    symbol.is_series = True
                    self.symbol_table.define(symbol)
            elif isinstance(stmt, (VarDecl, VaripDecl)) and not self.symbol_table.lookup_global(stmt.name):
                # a `var` of a block (`if c` / `var table t = table.new(...)`): its methods route by its type
                self.symbol_table.define(Symbol(
                    stmt.name, VariableKind.VAR if isinstance(stmt, VarDecl) else VariableKind.VARIP,
                    type_hint=self._infer_type_from_declaration(stmt), is_global=True))
            # Recurse into nested scopes
            if isinstance(stmt, IfStatement):
                self._register_assignments_recursive(stmt.body)
                for _, elif_body in stmt.elseifs:
                    self._register_assignments_recursive(elif_body)
                if stmt.else_body:
                    self._register_assignments_recursive(stmt.else_body)
            elif isinstance(stmt, SwitchStatement):
                for _, case_body in stmt.cases:
                    self._register_assignments_recursive(case_body)
                if stmt.default:
                    self._register_assignments_recursive(stmt.default)
            elif isinstance(stmt, (ForLoop, ForInLoop, WhileLoop)):
                if isinstance(stmt.body, list):
                    self._register_assignments_recursive(stmt.body)

    def _infer_type_from_declaration(self, decl: Union[VarDecl, VaripDecl]) -> str:
        """Infer type from var/varip declaration."""
        if decl.type_hint:
            return convert_pine_generic_to_python(decl.type_hint)
        return self.type_inference.infer_type_hint(decl.value)

    # ========================================================================
    # Module-level constant detection
    # ========================================================================

    def _collect_identifier_names(self, node: Any) -> set:
        """Recursively collect all Identifier names referenced in an AST node."""
        names = set()
        if isinstance(node, Identifier):
            names.add(node.name)
        elif isinstance(node, (FunctionCall, MethodCall)):
            for arg in getattr(node, 'args', []):
                names.update(self._collect_identifier_names(arg))
            for val in getattr(node, 'kwargs', {}).values():
                names.update(self._collect_identifier_names(val))
        elif isinstance(node, (BinaryOp,)):
            names.update(self._collect_identifier_names(node.left))
            names.update(self._collect_identifier_names(node.right))
        elif isinstance(node, (UnaryOp,)):
            names.update(self._collect_identifier_names(node.operand))
        elif isinstance(node, (TernaryOp,)):
            names.update(self._collect_identifier_names(node.condition))
            names.update(self._collect_identifier_names(node.true_expr))
            names.update(self._collect_identifier_names(node.false_expr))
        elif isinstance(node, MemberAccess):
            names.update(self._collect_identifier_names(node.object))
        elif isinstance(node, IndexAccess):
            names.update(self._collect_identifier_names(node.object))
            names.update(self._collect_identifier_names(node.index))
        elif isinstance(node, ArrayLiteral):
            for element in node.elements:
                names.update(self._collect_identifier_names(element))
        return names

    def _collect_module_constant_names(self, script: Script) -> set:
        """Collect names of body assignments that must be module-level constants.

        In Pine Script, literal assignments before the first function call
        (e.g., `const string G_ENTRY = '...'`) can be referenced by input.*()
        group= params and strategy/indicator kwargs. Since these become main()
        parameter defaults (evaluated at definition time), the referenced
        variables must exist at module level.
        """
        # Collect all identifier names referenced in input declarations
        referenced_names = set()
        for decl in script.declarations:
            if isinstance(decl, InputDecl):
                for arg in decl.args:
                    referenced_names.update(self._collect_identifier_names(arg))
                for val in decl.kwargs.values():
                    referenced_names.update(self._collect_identifier_names(val))

        # Find body assignments that are literal values AND referenced by inputs
        module_const_names = set()
        for stmt in script.body:
            if (isinstance(stmt, Assignment) and
                isinstance(stmt.target, str) and
                isinstance(stmt.value, Literal) and
                stmt.target in referenced_names):
                module_const_names.add(stmt.target)

        # Also check var/varip declarations with literal values referenced by inputs
        # (e.g., `var string G_STRATEGY = 'Strategy Settings'` used in group= params)
        for decl in script.declarations:
            if (isinstance(decl, (VarDecl, VaripDecl)) and
                isinstance(decl.value, Literal) and
                decl.name in referenced_names):
                module_const_names.add(decl.name)

        # a top-level `const` declaration is a module constant, with the top-level constants its value
        # reads (`const int A1 = 30, A2 = 40` declares A2 without the keyword, and `const color C = color.new(c, A2)`
        # reads it when the module loads); one that reads anything else stays in main()
        bindings = self._top_level_bindings(script)
        for stmt in script.body:
            if not (isinstance(stmt, Assignment) and stmt.is_const and isinstance(stmt.target, str)
                    and self._is_constant_expression(stmt.value)):
                continue
            try:
                read = self._constant_closure(
                    script, {name for name in self._read_names(stmt.value) if name in bindings}, stmt.line,
                    'const declaration', 'a module constant is read when the module loads', unbound_roots_allowed=True)
            except SemanticError:
                continue
            module_const_names.add(stmt.target)
            module_const_names.update(read)
        # strategy/indicator arguments and input arguments, through every constant they are built from
        return (module_const_names | self._declaration_constant_names(script)
                | self._input_constant_names(script))

    def _declaration_constant_names(self, script: Script) -> set:
        """Names the script declaration's arguments read, with the constants they are built from.

        The decorator runs at import, before main(), so each name in its arguments must be a
        module constant: a top-level binding whose value holds only literals, constant
        namespace members (`strategy.fixed`), enum or imported-library members, constant
        functions and other such constants. Anything else is refused here, by name, instead
        of failing or reading an unset value at import.
        """
        if not script.script_decl:
            return set()
        roots = {name for value in script.script_decl.kwargs.values() for name in self._read_names(value)}
        return self._constant_closure(script, roots, script.script_decl.line,
                                      'script declaration argument', 'the declaration needs its value before the script runs',
                                      unbound_roots_allowed=False)

    def _input_constant_names(self, script: Script) -> set:
        """Names the inputs' arguments read, with the constants they are built from.

        An input call is a default of main()'s signature, evaluated when the module loads, so a
        top-level constant it reads (`const int LENGTH = BASE * 2`) must be a module constant
        too: the same rule as the declaration's. A name that is no top-level binding (`close`,
        `syminfo.tickerid`, another input) is not read from a constant and is left alone.
        """
        needed = set()
        bindings = self._top_level_bindings(script)
        for decl in script.declarations:
            if not isinstance(decl, InputDecl):
                continue
            arguments = [*decl.args, *decl.kwargs.values()]
            if decl.name in getattr(self, '_source_defaults', {}):
                arguments = arguments[1:] if decl.args else [value for key, value in decl.kwargs.items()
                                                             if key != 'defval']
            roots = {name for argument in arguments for name in self._read_names(argument)}
            needed |= self._constant_closure(
                script, {name for name in roots if name in bindings}, decl.line,
                f"default of input '{decl.name}'", 'an input default is read when the module loads',
                unbound_roots_allowed=True, code='input-argument-not-constant')
        return needed

    def _constant_closure(self, script: Script, roots: set, line: int, subject: str, reason: str,
                          unbound_roots_allowed: bool, code: str = 'declaration-argument-not-constant') -> set:
        """The top-level constants `roots` are built from, each checked to be a constant.

        A name reached through a constant's value must be a constant, a constant namespace
        member or an enum or library member; a root need not be bound (`unbound_roots_allowed`).
        """
        bindings = self._top_level_bindings(script)
        namespaces = {decl.name for decl in script.declarations if isinstance(decl, EnumDecl)}
        namespaces.update(imported.alias for imported in script.imports if imported.alias)
        pending = [(name, True) for name in sorted(roots)]
        needed = set()
        while pending:
            name, is_root = pending.pop()
            if name in needed or self._is_namespace_constant(name, namespaces):
                continue
            value = bindings.get(name)
            if value is None and is_root and unbound_roots_allowed:
                continue
            if value is None or not self._is_constant_expression(value):
                # TradingView takes a declaration's and an input's arguments as constants (`const int`)
                raise SemanticError(code,
                    f"{subject} uses '{name}', which is not a constant declared "
                    f"at the top level; {reason}",
                    line)
            needed.add(name)
            pending.extend((read, False) for read in sorted(self._read_names(value)))
        return needed

    def _top_level_bindings(self, script: Script) -> Dict[str, Any]:
        """Each top-level `name = value` and var/varip declaration's value, by name."""
        bindings = {decl.name: decl.value for decl in script.declarations
                    if isinstance(decl, (VarDecl, VaripDecl))}
        bindings.update({stmt.target: stmt.value for stmt in script.body
                         if isinstance(stmt, Assignment) and isinstance(stmt.target, str)})
        return bindings

    def _read_names(self, node: Any) -> set:
        """The names `node` reads, except the true/false/na literals."""
        return self._collect_identifier_names(node) - {'true', 'false', 'na'}

    def _is_namespace_constant(self, name: str, namespaces: set) -> bool:
        """A member of a constant builtin namespace, of an enum, or of an imported library."""
        from .pine_builtins import DECLARATION_CONSTANT_NAMESPACES
        return '.' in name and (name.startswith(DECLARATION_CONSTANT_NAMESPACES)
                                or name.split('.', 1)[0] in namespaces)

    def _is_constant_expression(self, node: Any) -> bool:
        """Built only from literals, names, operators and constant functions (the caller checks names)."""
        from .pine_builtins import (DECLARATION_CONSTANT_FUNCTIONS, DECLARATION_CONSTANT_FUNCTION_NAMESPACES,
                                    DECLARATION_STATEFUL_FUNCTIONS, DECLARATION_TIMESTAMP)
        if isinstance(node, (Literal, Identifier)):
            return True
        if isinstance(node, ArrayLiteral):
            return all(self._is_constant_expression(element) for element in node.elements)
        if isinstance(node, FunctionCall):
            constant_function = isinstance(node.func, str) and (
                node.func in DECLARATION_CONSTANT_FUNCTIONS
                or (node.func.startswith(DECLARATION_CONSTANT_FUNCTION_NAMESPACES)
                    and node.func not in DECLARATION_STATEFUL_FUNCTIONS)
                or (node.func == DECLARATION_TIMESTAMP and not node.kwargs and node.args
                    and isinstance(node.args[0], Literal) and node.args[0].literal_type == 'string'))
            return constant_function and all(
                self._is_constant_expression(argument) for argument in [*node.args, *node.kwargs.values()])
        if isinstance(node, BinaryOp):
            return self._is_constant_expression(node.left) and self._is_constant_expression(node.right)
        if isinstance(node, UnaryOp):
            return self._is_constant_expression(node.operand)
        if isinstance(node, TernaryOp):
            return all(self._is_constant_expression(part)
                       for part in (node.condition, node.true_expr, node.false_expr))
        return False

    # ========================================================================
    # Top-level transformations
    # ========================================================================

    def _transform_script_declaration(self, decl: Union[IndicatorDecl, StrategyDecl, LibraryDecl]) -> Union[IndicatorDecl, StrategyDecl, LibraryDecl]:
        """Transform indicator/strategy/library declaration (Rules 2-3, 17, 54)."""
        # Transform kwargs: convert true/false to True/False (Rule 17, 54)
        transformed_kwargs = {}
        for key, value in decl.kwargs.items():
            transformed_kwargs[key] = self._transform_decorator_value(value)
        if self.three_state_bool:
            # Pine v4 and v5: `bool b = na` is a real na (the runtime's NA(bool) is False without this)
            transformed_kwargs['na_bool'] = True

        return type(decl)(title=decl.title, kwargs=transformed_kwargs, line=decl.line,
                          positional_count=decl.positional_count, title_double_quoted=decl.title_double_quoted,
                          keyword_order=decl.keyword_order)

    def _transform_decorator_value(self, value: Any) -> Any:
        """Transform decorator keyword argument values (Rule 17, 54)."""
        if isinstance(value, Literal):
            if value.literal_type == 'bool':
                # Convert true/false to True/False (Rule 17)
                return value.value  # Will be True or False in Python
            return value
        elif isinstance(value, Identifier):
            # Handle true/false identifiers
            if value.name == 'true':
                return True
            elif value.name == 'false':
                return False
        return value

    def _extract_input_declarations(self, script: Script) -> None:
        """Extract input declarations for main() parameters (Rules 8-9)."""
        for decl in script.declarations:
            if isinstance(decl, InputDecl):
                self.input_declarations.append(decl)

    def _create_main_parameters(self) -> List[Parameter]:
        """Create main() function parameters from input declarations (Rule 8)."""
        params = []
        input_names = {input_decl.name for input_decl in self.input_declarations}

        for input_decl in self.input_declarations:
            args = list(input_decl.args)
            kwargs = dict(input_decl.kwargs)

            # Convert excess positional args to keyword args.
            # Pine Script allows input.int(defval, title, minval, maxval, ...)
            # but PyneCore requires minval=, maxval=, ... as keyword-only.
            args, positional_keywords = bind_input_positional_args(input_decl.func, args)
            # the positionals bound to names come first, in their order, then the source's keywords
            kwargs = {**positional_keywords, **kwargs}
            if input_decl.name in self._source_defaults:
                # a series default is read in main(), where the input stands (`_source_default_resolution`)
                if args:
                    args[0] = Literal(value='', literal_type='string', is_double_quoted=True)
                else:
                    kwargs['defval'] = Literal(value='', literal_type='string', is_double_quoted=True)
            kwargs = self._input_keywords_without_input_reads(input_decl, kwargs, input_names)

            func_call = FunctionCall(
                func=input_decl.func,
                args=args,
                kwargs=kwargs
            )

            # A source input (input.source, or input() of a price) is a series: Series[float]
            if self._is_source_input(input_decl.func, args, kwargs):
                params.append(Parameter(name=input_decl.name, type_hint='Series[float]', default=func_call,
                                        line=input_decl.line))
                self.import_resolver.uses_series = True
                continue

            param = Parameter(
                name=input_decl.name,
                type_hint=self._input_parameter_annotation(input_decl.name, func_call),
                default=func_call,
                line=input_decl.line
            )
            params.append(param)

        return params

    @staticmethod
    def _is_source_input(function: str, args: list, kwargs: dict) -> bool:
        if function == 'input.source':
            return True
        if function != 'input':
            return False
        default = args[0] if args else kwargs.get('defval')
        return isinstance(default, Identifier) and default.name in (
            'close', 'open', 'high', 'low', 'volume', 'hl2', 'hlc3', 'ohlc4', 'hlcc4')

    @staticmethod
    def _input_keywords_without_input_reads(input_decl: InputDecl, keywords: dict, input_names: set) -> dict:
        """The input's keyword arguments (`keywords`, its positional ones bound by name), for a
        main() parameter default, which the module evaluates before any input exists. Pine lets two arguments read another
        input; the semantic pass refuses the rest. `active` (an input bool) only greys out the
        field in TradingView, and pynecore never reads it: it is dropped. input.source's `defval` (a series float)
        is read in main() instead (`_series_source_defaults`)."""
        kwargs = dict(keywords)
        # `active` only greys the field out in TradingView: it is dropped, a constant one too
        kwargs.pop('active', None)
        return kwargs

    def _series_source_defaults(self, script: Script) -> Dict[str, Any]:
        """The input.source inputs whose default is a series of the script, by name, with that default.

        Pine reads input.source's `defval` as a series float: another input (`input.source(a, "B")`),
        a variable or an expression of the script. The runtime resolves a source input by name, from a
        builtin series (`close`, `hl2`), so a main() parameter default can hold only such a name. A
        series default is written as the empty name instead and read in main(), where the input
        stands: the input keeps that series while its setting is left alone, and takes the source the
        setting names once it is set."""
        user_names = {decl.name for decl in self.input_declarations} | set(self._top_level_bindings(script))
        defaults = {}
        for decl in self.input_declarations:
            if decl.func != 'input.source':
                continue
            default = decl.args[0] if decl.args else decl.kwargs.get('defval')
            if default is None or (isinstance(default, Literal) and default.literal_type == 'string'):
                continue
            if isinstance(default, Identifier) and default.name not in user_names:
                continue  # a builtin series, resolved by name
            defaults[decl.name] = default
        return defaults

    #: the sources a source input's setting can name, in the order the resolution tests them
    SOURCE_SETTINGS = ('open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4', 'hlcc4', 'volume')

    def _source_default_resolution(self, name: str) -> Assignment:
        """`b = <default> if b == "" else open if b == "open" else ... else na` for a source input with a
        series default: the empty name is the setting left alone; a set one names a builtin source,
        which the chain picks as the runtime's own lookup on `lib` would (that lookup is no `@pyne edge`
        code, so it is spelled out)."""
        def named(source: str) -> BinaryOp:
            return BinaryOp(left=Identifier(name=name), op='==',
                            right=Literal(value=source, literal_type='string', is_double_quoted=True))
        value: Any = NaLiteral()
        for source in reversed(self.SOURCE_SETTINGS):
            value = TernaryOp(condition=named(source), true_expr=Identifier(name=source), false_expr=value)
        self.import_resolver.builtin_variables.update(self.SOURCE_SETTINGS)
        self.import_resolver.uses_na = True
        default = self._transform_expression(self._source_defaults[name])
        return Assignment(target=name, value=TernaryOp(condition=named(''), true_expr=default, false_expr=value))

    def _input_parameter_annotation(self, name: str, input_call: FunctionCall) -> Optional[str]:
        """`Series[T]` for an input some scope reads with history (`length[1]`), else none.

        An input is a main() parameter holding a plain value; the runtime keeps a
        history only for a name annotated as a series. The parameter keeps its name,
        which is the input's settings id."""
        if name not in self.names_read_with_history:
            return None
        element_type = self.type_inference.infer_type_hint(input_call)
        default = input_call.args[0] if input_call.args else input_call.kwargs.get('defval')
        if input_call.func == 'input' and isinstance(default, Literal) and default.literal_type in ('int', 'float'):
            element_type = default.literal_type  # Pine v4 `input(5)`: `Series[int]`
        return self._series_annotation(name, None if element_type == 'Any' else element_type)

    # ========================================================================
    # Function transformations
    # ========================================================================

    def _history_reads(self, tree) -> set:
        """Names read with history (`source[i]`) anywhere in `tree`, including
        else-if arms, if/switch-expression branches and switch cases; the receiver of a user method whose result
        is read with history counts too (`x.src('oc2')[1]` reads `x` so; `src(x, 'oc2')[1]` does not), and
        so does the receiver of a builtin method (`a.sum()[1]`; `array.sum(a)[1]` does not)."""
        methods = getattr(self, '_history_method_names', frozenset())
        names = set()
        for node in walk(tree):
            if not isinstance(node, IndexAccess):
                continue
            if isinstance(node.object, Identifier):
                names.add(node.object.name)
            elif isinstance(node.object, FunctionCall) and isinstance(node.object.func, str) \
                    and node.object.func.count('.') == 1:
                receiver, method_name = node.object.func.split('.')
                if method_name in methods or not (self._names_a_namespace(receiver) or self._names_a_type(receiver)):
                    # a builtin method's result too (`a.sum()[1]` reads `a` with history)
                    names.add(receiver)
        return names

    def _main_scope_history_reads(self, script: Script) -> set:
        """Names the main scope reads with history: its own `x[n]` sites, and a
        function's sites on a name that is neither its parameter nor its own local,
        which is a global (a function local that shadows a global was renamed; one
        declared before a global of its name keeps the name)."""
        names = set()
        for item in list(script.declarations) + list(script.body):
            if isinstance(item, FuncDecl):
                names |= (self._history_reads(item.body) - {param.name for param in item.params}
                          - _declared_names(item.body))
            else:
                names |= self._history_reads(item)
        return names

    def _transform_function(self, func: FuncDecl) -> FuncDecl:
        """Transform function declaration (Rules 20-22, 46-48)."""
        # Save current function params context
        old_params = self.current_function_params
        self.current_function_params = {}
        # the parameters of this function without a declared type: their methods are routed at run time
        old_untyped_parameters = getattr(self, '_untyped_parameters', frozenset())
        self._untyped_parameters = frozenset(param.name for param in func.params if not param.type_hint)
        old_shadowed_names = self.current_function_shadowed_names
        self.current_function_shadowed_names = self._function_shadowed_names(func)
        typing = getattr(self, '_condition_typing', None)
        old_typing_scope = getattr(self, '_typing_scope', None)
        self._typing_scope = func
        if typing is not None:
            typing.enter(func)

        # PRIORITY 1 FIX: Save and set function context
        old_in_function = self.in_user_function
        old_series_vars = self.series_variables
        old_local_vars = self.function_local_vars

        self.in_user_function = True
        self.series_variables = self._find_series_variables_in_function(
            func.body if isinstance(func.body, list) else [func.body]
        )
        self.function_local_vars = set()

        old_history_reads = self.names_read_with_history
        self.names_read_with_history = self._history_reads(func.body)

        # Find which parameters are indexed in the function body
        param_names = {p.name for p in func.params}
        indexed_params = self.names_read_with_history & param_names

        # Transform parameters
        # For methods, the first parameter has special syntax
        transformed_params = []
        for i, param in enumerate(func.params):
            # Convert Pine generic types to Python types
            if param.type_hint:
                type_hint = convert_pine_generic_to_python(param.type_hint)
                # Wrap in Series[T] if parameter is indexed with [n] history access.
                # The runtime SeriesTransformer will create SeriesImpl and register it.
                if param.name in indexed_params or param.is_series:
                    type_hint = f'Series[{type_hint}]'
            else:
                # If parameter is indexed and has no type hint, add bare Series
                type_hint = 'Series' if param.name in indexed_params else None

            # Track parameter types for method call transformation
            if type_hint:
                self.current_function_params[param.name] = type_hint

            default_value = self._transform_expression(param.default) if param.default else None
            if param.type_hint == 'bool' and isinstance(param.default, NaLiteral):
                default_value = Literal(value=False, literal_type='bool')  # `bool b = na` is `b: bool = False`
            if default_value is None and any(earlier.default for earlier in func.params[:i]):
                # Python needs a default after a defaulted parameter: it is given na
                default_value = Identifier(name='na')

            transformed_params.append(Parameter(
                name=param.name,
                type_hint=type_hint,
                default=default_value
            ))

        # Transform body
        if isinstance(func.body, Expression):
            # Single-line arrow function (Rule 21)
            transformed_body = ReturnStatement(expr=self._transform_expression(func.body))
        else:
            # Multi-line function (Rule 22)
            transformed_body = [self._transform_statement(stmt) for stmt in func.body]
            # The value of a Pine function is the value of its last statement (Rule 20);
            # this also flattens statement lists and drops None statements
            transformed_body = self._return_tail_value(transformed_body, untaken_is_false=func.value_is_bool)

        # Restore previous params context
        self.current_function_params = old_params
        self._untyped_parameters = old_untyped_parameters
        self.current_function_shadowed_names = old_shadowed_names

        # PRIORITY 1 FIX: Restore function context
        self._typing_scope = old_typing_scope
        if typing is not None:
            typing.enter(old_typing_scope)
        self.in_user_function = old_in_function
        self.series_variables = old_series_vars
        self.function_local_vars = old_local_vars
        self.names_read_with_history = old_history_reads

        # Track method functions, with their `this` types, for call transformation
        if func.is_method:
            this_hint = func.params[0].type_hint if func.params else None
            this_type = convert_pine_generic_to_python(this_hint) if this_hint else None
            self.method_functions.setdefault(func.name, set()).add(this_type)

        # Sanitize function name if it clashes with Python reserved words
        from .pine_builtins import rename_reserved_word
        sanitized_name = rename_reserved_word(func.name)
        if sanitized_name != func.name:
            self._renamed_functions[func.name] = sanitized_name

        return FuncDecl(
            name=sanitized_name,
            params=transformed_params,
            body=transformed_body,
            is_method=func.is_method,
            is_export=func.is_export,  # codegen emits `@export` for it
            indexed_params=indexed_params,
        )

    def _return_tail_value(self, body: List[Statement], is_function_body: bool = True,
                           untaken_is_false: bool = False) -> List[Statement]:
        """Return a copy of a function body (or of one arm of its tail if) that returns its value.

        An expression tail becomes ``return expr``; an assignment tail is followed by
        ``return target``. An if tail is rewritten arm by arm (if, every else-if, else),
        recursively, so the arm that runs returns its own value. An arm that yields no
        value (an untaken if without else, a loop) falls through to one ``return na``
        after the function's tail if, or ``return False`` when `untaken_is_false`
        (Pine v6: the function's value is of type bool). Pine has no early return, so returning from inside
        the LAST statement skips nothing that would have run after it.
        """
        statements = self._flatten_statements(body)
        if not statements:
            return statements
        last = statements[-1]
        if isinstance(last, ExpressionStatement):
            return statements[:-1] + [self._keep_source_line(last, ReturnStatement(expr=last.expr))]
        if isinstance(last, Assignment):
            return statements + [ReturnStatement(expr=self._assigned_value(last.target))]
        if not isinstance(last, IfStatement):
            return statements
        returning_if = self._keep_source_line(last, IfStatement(
            condition=last.condition,
            body=self._return_tail_value(last.body, is_function_body=False),
            elseifs=[(condition, self._return_tail_value(arm, is_function_body=False))
                     for condition, arm in last.elseifs],
            else_body=(self._return_tail_value(last.else_body, is_function_body=False)
                       if last.else_body else None),
        ))
        # No trailing `return na` when every arm returns: it could never run
        if not is_function_body or self._always_returns([returning_if]):
            return statements[:-1] + [returning_if]
        if untaken_is_false:
            return statements[:-1] + [returning_if, ReturnStatement(expr=Literal(value=False, literal_type='bool'))]
        # Pine's value for an untaken if, or for an arm without a value, is na
        self.import_resolver.lib_modules.add('na')
        return statements[:-1] + [returning_if, ReturnStatement(expr=Identifier(name='na'))]

    @staticmethod
    def _assigned_value(target) -> Expression:
        """The value an assignment leaves behind: the variable, or the tuple of destructured names."""
        if isinstance(target, TupleDestructure):
            return ArrayLiteral(elements=[Identifier(name=name) for name in target.names])
        return Identifier(name=target) if isinstance(target, str) else target

    @staticmethod
    def _always_returns(statements: List[Statement]) -> bool:
        """True when every path through the statements ends in a return."""
        if not statements:
            return False
        last = statements[-1]
        if isinstance(last, ReturnStatement):
            return True
        if isinstance(last, IfStatement) and last.else_body:
            arms = [last.body] + [arm for _, arm in last.elseifs] + [last.else_body]
            return all(Transformer._always_returns(arm) for arm in arms)
        return False

    @staticmethod
    def _flatten_statements(statements: List[Statement]) -> List[Statement]:
        """Branch bodies keep statement lists (a split tuple, a loop used as a value) and dropped
        (None) statements; flatten them."""
        flat = []
        for statement in statements:
            if statement is None:
                continue
            if isinstance(statement, list):
                flat.extend(Transformer._flatten_statements(statement))
            else:
                flat.append(statement)
        return flat

    # ========================================================================
    # Statement transformations
    # ========================================================================

    def _transform_statement(self, stmt: Statement) -> Union[Statement, List[Statement], None]:
        """Transform statement."""
        if isinstance(stmt, Assignment):
            result = self._transform_assignment(stmt)
        elif isinstance(stmt, Reassignment):
            result = self._transform_reassignment(stmt)
        elif isinstance(stmt, (VarDecl, VaripDecl)):
            result = self._transform_var_declaration(stmt)
        elif isinstance(stmt, IfStatement):
            result = self._transform_if_statement(stmt)
        elif isinstance(stmt, ForLoop):
            result = self._transform_for_loop(stmt)
        elif isinstance(stmt, ForInLoop):
            result = self._transform_for_in_loop(stmt)
        elif isinstance(stmt, WhileLoop):
            result = self._transform_while_loop(stmt)
        elif isinstance(stmt, SwitchStatement):
            result = self._transform_switch_statement(stmt)
        elif isinstance(stmt, ExpressionStatement):
            # Skip stray declaration calls (indicator/strategy/library) in body
            if isinstance(stmt.expr, FunctionCall) and isinstance(stmt.expr.func, str):
                if stmt.expr.func in ('indicator', 'strategy', 'library'):
                    return None
            result = ExpressionStatement(expr=self._transform_expression(stmt.expr))
        elif isinstance(stmt, (BreakStatement, ContinueStatement)):
            result = stmt
        else:
            result = stmt
        return self._keep_source_line(stmt, result)

    @staticmethod
    def _keep_source_line(source: Statement, result):
        """A statement built for a nested Pine statement remembers its Pine line in
        `source_line`, so a layout printer can place blank lines where the source
        has them. `line` stays unset: refusals keep reporting the enclosing
        top-level line. Only the first statement of an expansion takes it."""
        line = getattr(source, 'line', 0) or getattr(source, 'source_line', 0)
        first = result[0] if isinstance(result, list) and result else result
        if line and first is not None and not isinstance(first, list) and not getattr(first, 'line', 0):
            try:
                first.source_line = line
            except AttributeError:
                pass
        return result

    def _transform_assignment(self, stmt: Assignment) -> Union[Assignment, List[Statement]]:
        """Transform assignment (Rules 5-7)."""
        return self._without_unread_history(stmt, self._transform_assignment_statement)

    def _without_unread_history(self, stmt, transform):
        """`transform(stmt)`, where a declaration in a block whose own binding no scope reads with history is
        not a series even though another binding of its name is."""
        name = stmt.target if isinstance(stmt, Assignment) else getattr(stmt, 'name', None)
        if id(stmt) not in getattr(self, '_unread_history_declarations', ()) or not isinstance(name, str):
            return transform(stmt)
        saved = self.names_read_with_history
        self.names_read_with_history = saved - {name}
        try:
            return transform(stmt)
        finally:
            self.names_read_with_history = saved

    def _transform_assignment_statement(self, stmt: Assignment) -> Union[Assignment, List[Statement]]:
        # `[a, b] = g()` stays one destructure; a name this scope reads with history is
        # declared first, bare: `p: Series`
        if isinstance(stmt.target, TupleDestructure) and not getattr(self, '_declaring_tuple', False):
            series_names = [name for name in stmt.target.names if name in self.names_read_with_history]
            if series_names:
                self.import_resolver.uses_series = True
                self._declaring_tuple = True
                try:
                    rest = self._transform_assignment(stmt)
                finally:
                    self._declaring_tuple = False
                declarations = [RawCode(code=f'{name}: Series') for name in series_names]
                return declarations + (rest if isinstance(rest, list) else [rest])

        # PRIORITY 1 FIX: Track first assignment to Series variables in functions
        is_first_assignment_to_series = False
        if self.in_user_function and isinstance(stmt.target, str):
            if stmt.target in self.series_variables and stmt.target not in self.function_local_vars:
                is_first_assignment_to_series = True
                self.function_local_vars.add(stmt.target)

        # Determine type annotation first (we need it for na value transformation)
        type_hint = None
        converted_type = None
        if isinstance(stmt.target, str):
            symbol = self.symbol_table.lookup(stmt.target)
            if self.in_user_function and stmt.target in self.current_function_shadowed_names:
                symbol = None  # a function's own local, not the global it hides
            # CRITICAL: In Pine Script, series is the default type qualifier
            if hasattr(stmt, 'type_hint') and stmt.type_hint:
                # Use the explicit type hint from parser (e.g., map<int, float> mapExample = na)
                converted_type = convert_pine_generic_to_python(stmt.type_hint)
                # If variable is used with historical reference [], wrap in Series[type]; a method's result read with
                # history reads its receiver so too (`a.sum()[1]`)
                if (symbol and symbol.is_indexed) or (isinstance(stmt.target, str) and symbol is not None
                                                      and stmt.target in self.names_read_with_history):
                    type_hint = self._series_annotation(stmt.target, converted_type)
                else:
                    type_hint = converted_type
            elif symbol:
                # No explicit type in Pine source — only infer for na value transformation
                # and var/varip persistence (the same rule as a var declaration)
                if symbol.kind in (VariableKind.VAR, VariableKind.VARIP):
                    base_type = symbol.type_hint or self.type_inference.infer_type_hint(stmt.value)
                    converted_type = base_type
                    type_hint = self._series_annotation(
                        stmt.target, self._series_element_type(base_type, stmt.value), persistent=True)
                elif isinstance(stmt.value, NaLiteral):
                    # Need type for na() conversion even without explicit annotation
                    base_type = symbol.type_hint or self.type_inference.infer_type_hint(stmt.value)
                    converted_type = base_type
                    if symbol.is_indexed:
                        type_hint = self._series_annotation(
                            stmt.target, self._series_element_type(base_type, stmt.value))
                elif isinstance(stmt.target, str) and stmt.target in getattr(self, '_reassignment_vars', set()) and symbol.is_indexed:
                    # Variable uses both := (reassignment) and [] (historical reference)
                    # WITHOUT 'var' keyword - this is a Series in Pine Script semantics
                    base_type = self._series_element_type(
                        symbol.type_hint or self.type_inference.infer_type_hint(stmt.value), stmt.value)
                    converted_type = base_type
                    type_hint = self._series_annotation(stmt.target, base_type)
                elif isinstance(stmt.target, str) and symbol.is_indexed:
                    # Variable is used with [] - in Pine Script, series is the default type qualifier
                    base_type = self._series_element_type(
                        symbol.type_hint or self.type_inference.infer_type_hint(stmt.value), stmt.value)
                    converted_type = base_type
                    type_hint = self._series_annotation(stmt.target, base_type)
                # Otherwise: no type annotation for body assignments without explicit Pine types

        # Transform value - special case for na with type hint
        if isinstance(stmt.value, NaLiteral) and converted_type:
            value = self._na_initialiser(self._line_na_type(stmt, converted_type))
        elif getattr(stmt, 'type_hint', None) == 'bool' and not getattr(stmt, 'is_reassignment', False) \
                and self._ends_in_false_over_lines(stmt, stmt.value):
            value = self._transform_assigned(stmt.value, stmt.target if isinstance(stmt.target, str) else None, stmt)
        elif getattr(stmt, 'type_hint', None) == 'bool' and not getattr(stmt, 'is_reassignment', False) \
                and isinstance(stmt.value, TernaryOp) and _has_a_na_arm_in_its_else_chain(stmt.value) \
                and self._bool_may_be_na and self.pine_version <= 5 and self._only_bool_arms(stmt.value):
            # where a bool may be na, `bool p = c ? true : d ? false : na` keeps its value, na arm included, uncast
            # since a cast would read the na as false
            value = self._transform_assigned(stmt.value, stmt.target if isinstance(stmt.target, str) else None, stmt)
        elif getattr(stmt, 'type_hint', None) == 'bool' and not getattr(stmt, 'is_reassignment', False) \
                and isinstance(stmt.value, TernaryOp) and _has_a_na_arm_in_its_else_chain(stmt.value):
            # `bool b = c ? x > y : na` is the cast of the value, its na arm read as for `b = c ? x > y : na`; so is
            # `bool b = c ? x : d ? y : na`, a chain ending in na
            self.import_resolver.uses_cast_bool = True
            value = FunctionCall(func='cast_bool', args=[self._transform_assigned(
                stmt.value, stmt.target if isinstance(stmt.target, str) else None, stmt)], kwargs={})
        elif getattr(stmt, 'type_hint', None) == 'bool' and not getattr(stmt, 'is_reassignment', False) \
                and self.pine_version <= 5 and _is_kept_as_a_bool_value(stmt.value):
            # Pine v4/v5 `bool a = time(tf, s)`, `bool c = ta.dev(close, 3)`, `bool g = u`: the value is kept, not
            # cast_bool(...): Pine's bool of an na number is na (`na(c)` holds while ta.dev warms up), the cast would
            # read it false; a condition reads the kept value as the cast does
            value = self._transform_assigned(stmt.value, stmt.target if isinstance(stmt.target, str) else None, stmt)
        elif getattr(stmt, 'type_hint', None) == 'bool' and not getattr(stmt, 'is_reassignment', False):
            value = self._condition(stmt.value)  # `bool b = ta.change(time('D'))` is cast
        else:
            value = self._transform_assigned(stmt.value, stmt.target if isinstance(stmt.target, str) else None, stmt)

        # Suppress 'Any' type annotations - omit annotation when type is unknown
        if type_hint == 'Any':
            type_hint = None

        # PRIORITY 1 FIX: For Series variables in functions, add Series type hint
        if is_first_assignment_to_series:
            base_type = self._series_element_type(
                converted_type or self.type_inference.infer_type_hint(stmt.value), stmt.value)
            type_hint = self._series_annotation(stmt.target, base_type)

        # Record this local's type in the current-function scope map so later
        # method calls on it resolve type-aware, e.g. `thisT = m.transpose()`
        # (thisT: Matrix[int]) then `thisT.sort(...)` → matrix.sort(...). Function
        # locals are not in the global symbol table, so this is the only place
        # their type is known; the map is per-function (saved/reset around each
        # function), and body statements transform in source order so the type is
        # in place before the call. Only concrete types are recorded.
        if self.in_user_function and isinstance(stmt.target, str):
            # Prefer the concrete converted type; otherwise infer from the
            # TRANSFORMED value (e.g. m.transpose() is now matrix.transpose(m),
            # whose return type the inference knows), not the pre-transform AST.
            local_type = converted_type or self.type_inference.infer_type_hint(value)
            if (not local_type or local_type == 'Any') and isinstance(stmt.value, FunctionCall) \
                    and isinstance(stmt.value.func, str) and stmt.value.func.endswith('.new') \
                    and stmt.value.func[:-4] in self.symbol_table.udt_field_types:
                local_type = stmt.value.func[:-4]  # `t = U.new(...)`: a U
            if local_type and local_type != 'Any':
                self.current_function_params[stmt.target] = local_type

        return Assignment(target=stmt.target, value=value, type_hint=type_hint)

    def _series_element_type(self, inferred_type: Optional[str], value: Expression) -> Optional[str]:
        """The element type of a Series annotation: the inferred type when it is
        known, else Color for a colour value, else None. A guessed
        float breaks a colour or a string held in an untyped variable."""
        if inferred_type and inferred_type != 'Any':
            return inferred_type
        return 'Color' if self.type_inference.is_colour_value(value) else None

    def _series_annotation(self, name: str, element_type: Optional[str],
                           persistent: bool = False) -> Optional[str]:
        """The annotation of a name that keeps a history.

        `Series[T]`, or for a var `PersistentSeries[T]`, which the runtime splits
        into `Persistent[T]` + `Series[T]` so that `x[n]` works on any var. The
        runtime drops a `Series[T]` half that is never indexed. An element type
        the inference cannot name (None) gives the bare form: the runtime types
        it from the values assigned, as for a typeless na. The
        runtime never drops a bare Series half, so the bare form is emitted only
        for a name this scope reads with history; any other name gets no Series
        half: a var is a bare `Persistent`, anything else has no annotation."""
        if not element_type and name not in self.names_read_with_history:
            return 'Persistent' if persistent else None
        if persistent:
            kind = 'PersistentSeries'
            self.import_resolver.uses_persistent_series = True
        else:
            kind = 'Series'
            self.import_resolver.uses_series = True
        return f"{kind}[{element_type}]" if element_type else kind

    def _transform_reassignment(self, stmt: Reassignment) -> Assignment:
        """
        Transform reassignment := to = (Rule 11).
        """
        # Default behavior: regular assignment
        # PRIORITY 1 FIX: Series type hint will be added during assignment transformation
        value = self._transform_assigned(stmt.value, stmt.target, stmt)
        declared_type = self._same_line_declared_type(stmt)
        if declared_type and isinstance(value, TernaryOp):
            # `float x = na , x := c ? na : y`: the reassignment shares the typed declaration's line, and its na
            # arms take the type, `na(float) if c else y`; on a line of its own they stay `na`
            value = self._typed_na_arms(value, declared_type)
        typing = getattr(self, '_condition_typing', None)
        if typing is not None and isinstance(stmt.target, str) and typing.is_number_value(stmt.value) \
                and typing.is_bool(Identifier(name=stmt.target)) and not typing.is_declared_as_a_name(stmt.target):
            # a number given to a bool variable (`islong := 0`, `shown := c ? 1 : 0`) is cast
            self.import_resolver.uses_cast_bool = True
            value = FunctionCall(func='cast_bool', args=[value], kwargs={})
        return Assignment(target=stmt.target, value=value)

    def _same_line_declared_type(self, stmt: Reassignment) -> Optional[str]:
        """The Python type of the typed declaration of `stmt`'s target written on the same line, or None."""
        if not isinstance(stmt.target, str) or not getattr(stmt, 'line', 0):
            return None
        found = getattr(self, '_typed_declaration_lines', None)
        if found is None:
            found = {}
            for node in walk(self._names_source):
                if isinstance(node, (VarDecl, VaripDecl)) and node.type_hint:
                    found[(node.name, node.line)] = node.type_hint
                elif isinstance(node, Assignment) and isinstance(node.target, str) and getattr(node, 'type_hint', None):
                    found[(node.target, node.line)] = node.type_hint
            self._typed_declaration_lines = found
        pine_type = found.get((stmt.target, stmt.line))
        return get_type_name(pine_type) if pine_type in ('float', 'int', 'bool', 'string', 'color') else None

    def _typed_na_arms(self, ternary: TernaryOp, type_name: str) -> TernaryOp:
        """`ternary` with its na arms, down its else chain, written `na(type_name)`."""
        def typed(arm):
            if isinstance(arm, NaLiteral) or (isinstance(arm, Identifier) and arm.name == 'na'):
                return FunctionCall(func=NaLiteral(), args=[Identifier(name=type_name)], kwargs={})
            if isinstance(arm, TernaryOp):
                return self._typed_na_arms(arm, type_name)
            return arm
        return dataclasses.replace(ternary, true_expr=typed(ternary.true_expr), false_expr=typed(ternary.false_expr))

    def _transform_var_declaration(self, decl: Union[VarDecl, VaripDecl]) -> Assignment:
        return self._without_unread_history(decl, self._transform_var_statement)

    def _transform_var_statement(self, decl: Union[VarDecl, VaripDecl]) -> Assignment:
        """Transform var/varip declaration to a PersistentSeries assignment (Rule 5-6).

        Every type, scalar or not, keeps a Series half, so `x[n]` reads the
        value (for an array, map, matrix, UDT or drawing: the object itself) from n
        bars ago, as in Pine. The exception is a var of unknown type
        that no scope reads with history: a bare `Persistent` (`_series_annotation`).
        """
        base_type = decl.type_hint or self.type_inference.infer_type_hint(decl.value)
        # Convert generic types properly
        converted_type = convert_pine_generic_to_python(base_type)

        # Special case: if value is na, convert to na(Type) function call
        value = decl.value
        if isinstance(value, NaLiteral):
            value = self._na_initialiser(self._line_na_type(decl, converted_type))
        else:
            value = self._transform_assigned(value, decl.name, decl)

        type_hint = self._series_annotation(
            decl.name, self._series_element_type(converted_type, decl.value), persistent=True)
        if isinstance(decl, VaripDecl) and type_hint:
            # varip persists across a bar's intrabar updates too: pynecore's IBPersistent
            type_hint = 'IB' + type_hint

        # a var local of a function types its method calls, as an assignment's local does
        if self.in_user_function and converted_type and converted_type != 'Any':
            self.current_function_params[decl.name] = converted_type

        return Assignment(
            target=decl.name,
            value=value,
            type_hint=type_hint
        )

    _NUMERIC_TYPES = frozenset({'int', 'float'})

    def _condition(self, pine_expr: Expression, header: bool = False, negated: bool = False,
                   ternary: Optional[TernaryOp] = None, operand: bool = False, argument: bool = False,
                   assigned: bool = False) -> Expression:
        """A Pine expression read as a condition: a number there is cast_bool(...) (see
        condition_typing.py). `header`: an if/while condition, or (`operand`) an `and`/`or` operand of
        one; `negated`: the operand of `not`; `assigned`: the condition of `ternary`, the whole value
        of an assignment or an arm of it."""
        previous = (getattr(self, '_header_bool_op', False), getattr(self, '_header_ternary', None),
                    getattr(self, '_uncast_bool_op', False), getattr(self, '_value_bool_op', False))
        # an `and`/`or` read as the condition of a ternary or of a builtin (`alertcondition(c and x)`), not as a
        # header: its operands keep the mark
        value_bool_op = previous[3] if operand else (
            (ternary is not None or argument) and not header and isinstance(pine_expr, BinaryOp)
            and pine_expr.op in ('and', 'or'))
        self._value_bool_op = value_bool_op
        previous_first = getattr(self, '_value_chain_first', None)
        if not operand:
            self._value_chain_first = _leftmost_operand(pine_expr) if value_bool_op else None
        # the operands of an `and`/`or` that is the header are read as the header is
        self._header_bool_op = header and isinstance(pine_expr, BinaryOp) and pine_expr.op in ('and', 'or')
        # a ternary that is the whole header (`if c ? 1 : na`)
        self._header_ternary = pine_expr if header and not operand and isinstance(pine_expr, TernaryOp) else None
        # the operands of an `and`/`or` that is the condition of a ternary in an argument of a call, written as the
        # condition of a ternary in an earlier argument of the same call, are not cast
        # (`plot(s and p ? p : na, color = s and p ? color.lime : na)`)
        seen = getattr(self, '_call_ternary_conditions', None)
        repeated = False
        if ternary is not None and seen is not None and not operand:
            key = _expression_text(pine_expr)
            repeated = key in seen
            seen.add(key)
        self._uncast_bool_op = repeated and isinstance(pine_expr, BinaryOp) and pine_expr.op in ('and', 'or')
        # nor those of a ternary's `and`/`or` condition (Pine v4/v5) with a later operand that is itself a stateful
        # call, which is computed first into a temporary without a type (`c1 or ta.change(x) ? 1 : 0`)
        if ternary is not None and not header and self.pine_version <= 5 and isinstance(pine_expr, BinaryOp) \
                and pine_expr.op in ('and', 'or') and getattr(self, '_condition_typing', None) is not None \
                and not isinstance(_leftmost_operand(pine_expr), Identifier) \
                and self._condition_typing.has_a_later_stateful_operand(pine_expr):
            self._uncast_bool_op = True  # a chain starting with a name casts as usual (`pc and ta.change(pc)`)
        # the `and`/`or` that is the condition of a ternary choosing a colour or na (`d and w ? color.red : na`)
        previous_colour_test = getattr(self, '_colour_test', False)
        if not operand:
            self._colour_test = ternary is not None and isinstance(pine_expr, BinaryOp) \
                and pine_expr.op in ('and', 'or') and any(
                    self.type_inference.is_colour_value(arm) for arm in (ternary.true_expr, ternary.false_expr))
        try:
            transformed = self._transform_expression(pine_expr)
        finally:
            self._header_bool_op, self._header_ternary, self._uncast_bool_op, self._value_bool_op = previous
            value_chain_first, self._value_chain_first = getattr(self, '_value_chain_first', None), previous_first
            self._colour_test = previous_colour_test
        typing = getattr(self, '_condition_typing', None)
        if typing is not None and self.pine_version <= 5 and typing.is_a_bool_nz(pine_expr) \
                and isinstance(transformed, FunctionCall) and typing.is_number_value(pine_expr.args[0]):
            # `nz(x, false)` read as a condition: a number `x` is cast; assigned whole it is not
            self.import_resolver.uses_cast_bool = True
            transformed = dataclasses.replace(transformed, args=[
                FunctionCall(func='cast_bool', args=[transformed.args[0]], kwargs={})] + list(transformed.args[1:]))
        if typing is None:
            numeric = self.type_inference.infer_type_hint(pine_expr) in self._NUMERIC_TYPES
        else:
            colours = ternary is not None and (self.type_inference.is_colour_value(ternary.true_expr)
                                               or self.type_inference.is_colour_value(ternary.false_expr))
            numeric = (typing.is_number(pine_expr, header, colour_ternary=colours, ternary=ternary is not None,
                                        argument=argument)
                       and (not negated or typing.is_pivot(pine_expr))
                       and not (operand and header and ternary is None
                                and id(pine_expr) in getattr(self, '_later_operands', ())
                                and typing.is_a_hoisted_local_change(pine_expr)))
        if numeric and operand and value_bool_op and pine_expr is previous_first \
                and isinstance(pine_expr, FunctionCall) and pine_expr.func == 'ta.change' \
                and pine_expr.args and isinstance(pine_expr.args[0], Identifier) and typing is not None \
                and pine_expr.args[0].name in typing._values:
            # `c ? ... : ...` with `ta.change(x) and d`, `x` a variable of the script, first in its chain: not cast
            # whereas `ta.change(close)` there is, and so is either in an assignment or a header
            numeric = False
        if not numeric and header and not operand and typing is not None \
                and typing.is_the_na_of_a_pivot_in_a_function(pine_expr):
            numeric = True  # in a function `p = na(ta.pivothigh(...)) ? false : true`, then `if p`
        if numeric and header and not operand and isinstance(pine_expr, (IndexAccess, TernaryOp)):
            # `if x[1]` / `if c ? 1 : na` is not cast; `if x[1] and c` / `if (c ? up : na) and d` is
            numeric = False
        if numeric and ((header and not operand) or ternary is not None) and isinstance(pine_expr, FunctionCall) \
                and isinstance(pine_expr.func, str) and pine_expr.func in PIVOTS:
            numeric = False  # `if ta.pivothigh(2, 2)` is not cast; `if ta.pivothigh(2, 2) and c` is
        if numeric and ternary is not None and isinstance(pine_expr, FunctionCall) and pine_expr.func == 'ta.change' \
                and pine_expr is ternary.condition and getattr(pine_expr, 'line', 0) \
                and getattr(ternary, 'question_line', 0) > pine_expr.line:
            numeric = False  # `ta.change(x)` ending its line, the `?` on the next: not cast
        tree = getattr(self, '_ternary_tree', None)
        if numeric and operand and tree is not None and isinstance(pine_expr, Identifier) and pine_expr.name in tree[1] \
                and getattr(self, '_ternary_arm_depth', 0) >= 1:
            numeric = False  # `c ? ses : ses and d`: the name is a bare arm of the same ternary
        if numeric and operand and isinstance(pine_expr, FunctionCall) and isinstance(pine_expr.func, str) \
                and pine_expr.func in PIVOTS:
            signature = _call_signature(pine_expr)
            uncast = getattr(self, '_uncast_pivot_calls', set())
            if getattr(self, '_ternary_arm_depth', 0) >= 2:
                # an `and`/`or` operand in a ternary that is the arm of another ternary (`a ? na : c ? ph and x : na`)
                # is not cast, nor, after it, the same call written anywhere
                self._uncast_pivot_calls = uncast | {signature}
                numeric = False
            elif signature in uncast:
                numeric = False
        if numeric and ternary is not None and ternary is getattr(self, '_assigned_ternary', None) \
                and isinstance(pine_expr, Identifier) and isinstance(ternary.true_expr, Identifier) \
                and ternary.true_expr.name == pine_expr.name and not (typing and typing.is_pivot(pine_expr))\
                and pine_expr.name not in getattr(self, '_pivot_names', ()):
            numeric = False  # `y = x ? x : na` keeps its number; `plot(x ? x : na)` casts it
        if numeric and assigned and typing is not None \
                and any(typing.reads_an_untyped_result(arm) for arm in (ternary.true_expr, ternary.false_expr)):
            # `y = f(x)` of a function whose result has no type, then `c = x ? y : 0`: not cast;
            # `plot(x ? y : 0)` and `c = x ? f(x) : 0` are
            numeric = False
        if numeric:
            self.import_resolver.uses_cast_bool = True
            return FunctionCall(func='cast_bool', args=[transformed], kwargs={})
        return transformed

    def _is_a_non_int_value(self, expression, depth: int = 0) -> bool:
        """Pine v4: whether `expression` is known not to be an int, so `rsi(x, expression)` is the two-series form:
        a float literal, a built-in series (`volume`), a user function's result, arithmetic over one of these or a
        division, or a name declared from one. An int literal, an int input, a rounding and a name of no known
        value (a parameter) are a length."""
        if depth > 8:
            return False
        if isinstance(expression, Literal):
            return expression.literal_type == 'float'
        if isinstance(expression, BinaryOp) and expression.op in ('+', '-', '*', '/', '%'):
            return expression.op == '/' or self._is_a_non_int_value(expression.left, depth + 1) \
                or self._is_a_non_int_value(expression.right, depth + 1)
        if isinstance(expression, UnaryOp) and expression.op == '-':
            return self._is_a_non_int_value(expression.operand, depth + 1)
        if isinstance(expression, TernaryOp):
            return self._is_a_non_int_value(expression.true_expr, depth + 1) or self._is_a_non_int_value(
                expression.false_expr, depth + 1)
        if isinstance(expression, FunctionCall) and isinstance(expression.func, str):
            if expression.func in ('round', 'floor', 'ceil', 'int', 'math.round', 'math.floor', 'math.ceil'):
                return False
            user_functions = {node.name for node in self._names_source.declarations if isinstance(node, FuncDecl)}
            return expression.func in user_functions or expression.func.startswith(('ta.', 'math.'))
        if not isinstance(expression, Identifier):
            return False
        if expression.name in SERIES_VARIABLES and expression.name not in ('bar_index', 'last_bar_index', 'time'):
            return True
        for node in self._names_source.declarations:
            if isinstance(node, InputDecl) and node.name == expression.name:
                default = node.args[0] if node.args else node.kwargs.get('defval')
                return node.func == 'input.float' or (isinstance(default, Literal) and default.literal_type == 'float')
        typing = getattr(self, '_condition_typing', None)
        declared = typing._values.get(expression.name) if typing is not None else None
        if isinstance(declared, tuple):
            type_hint, value = declared
            if type_hint:
                return type_hint == 'float'
            return value is not None and self._is_a_non_int_value(value, depth + 1)
        return False

    def _is_float_series(self, expression) -> bool:
        """A sum (`math.sum(x, n)`), or a name declared as one or as a `float`: no length."""
        if isinstance(expression, FunctionCall):
            return expression.func == 'math.sum'
        typing = getattr(self, '_condition_typing', None)
        declared = typing._values.get(expression.name) if typing is not None and isinstance(expression, Identifier) \
            else None
        if isinstance(declared, tuple):
            type_hint, value = declared
            return type_hint == 'float' or (isinstance(value, FunctionCall) and value.func == 'math.sum')
        return False

    def _plotshape_bool_series(self, call: FunctionCall) -> FunctionCall:
        """`plotshape(c ? b : na)` with `b` a bool: where a bool may be na (Pine v4, v5 with `na_bool`)
        the bool arm is plotted as a number that keeps its na, `(na if na(b) else 1 if b else 0)`;
        where it may not (v5 without `na_bool`, v6), the na arm is `False`."""
        series = call.args[0] if call.args else call.kwargs['series']
        if isinstance(series, BinaryOp) and series.op in ('and', 'or'):
            self._plotshape_chains = getattr(self, '_plotshape_chains', set()) | {id(series)}
        if not isinstance(series, TernaryOp):
            return call
        self._plotshape_series = getattr(self, '_plotshape_series', set()) | {id(series)}
        arms = [series.true_expr, series.false_expr]
        if sum(isinstance(arm, NaLiteral) for arm in arms) != 1:
            return call
        index = 0 if isinstance(arms[1], NaLiteral) else 1
        arm = arms[index]
        typing = getattr(self, '_condition_typing', None)
        if typing is None or not (typing.is_plain_bool(arm) or (
                getattr(self, '_bool_may_be_na', True) and typing.is_a_typed_element(arm))):
            return call
        literal = arm.value if isinstance(arm, Literal) and arm.literal_type == 'bool' else (
            arm.name == 'true' if isinstance(arm, Identifier) and arm.name in ('true', 'false') else None)
        if literal is not None and getattr(self, '_bool_may_be_na', True):
            # a bool literal is plotted as the number it is, `1 if c else na`
            arms[index] = Literal(value=1 if literal else 0, literal_type='int')
        elif not isinstance(arm, Identifier):
            if not getattr(self, '_bool_may_be_na', True):
                return call  # the na arm is False (_na_arm_is_false)
            # an expression is plotted through a helper that keeps its na (`__pc_bool_marker(a or b)`)
            self._uses_bool_marker = True
            arms[index] = FunctionCall(func=BOOL_MARKER, args=[arm], kwargs={})
        elif getattr(self, '_bool_may_be_na', True):
            arms[index] = TernaryOp(condition=FunctionCall(func='na', args=[arm], kwargs={}), true_expr=NaLiteral(),
                                    false_expr=TernaryOp(condition=arm, true_expr=Literal(value=1, literal_type='int'),
                                                         false_expr=Literal(value=0, literal_type='int')))
        else:
            arms[1 - index] = Literal(value=False, literal_type='bool')
        series = dataclasses.replace(series, true_expr=arms[0], false_expr=arms[1])
        self._plotshape_series |= {id(series)}
        if call.args:
            return dataclasses.replace(call, args=[series, *call.args[1:]])
        return dataclasses.replace(call, kwargs={**call.kwargs, 'series': series})

    def _transform_assigned(self, value: Expression, target: Optional[str] = None, statement=None) -> Expression:
        """The value of an assignment, a ternary there marked as the whole value, `target` the name it binds,
        `statement` the assignment (where its value stands decides the na arms of a ternary, _na_arm_is_false)."""
        previous = (getattr(self, '_assigned_ternary', None), getattr(self, '_assigned_target', None),
                    getattr(self, '_assigned_over_lines', False), getattr(self, '_assigned_again', False))
        self._assigned_ternary = value if isinstance(value, TernaryOp) else None
        self._assigned_target = target
        self._assigned_over_lines = self._ternary_over_lines(statement, value)
        self._assigned_again = isinstance(statement, Reassignment) or getattr(statement, 'is_reassignment', False)
        try:
            return self._transform_expression(value)
        finally:
            (self._assigned_ternary, self._assigned_target, self._assigned_over_lines,
             self._assigned_again) = previous

    def _ternary_over_lines(self, statement, value: Expression) -> bool:
        """Pine v5, in main: `value` is a ternary, not in parentheses, that `statement` writes over several lines,
        the value below its name or continued on the next line."""
        if self.pine_version != 5 or self.in_user_function or statement is None:
            return False
        if not isinstance(value, TernaryOp) or value.parenthesized or statement.line <= 0:
            return False
        return statement.value_line > statement.line or statement.end_line > statement.line

    def _ends_in_false_over_lines(self, statement, value: Expression) -> bool:
        """Whether the else chain of `value`, a ternary written over several lines, ends in a na beside a bool,
        written `False` (`bool b = ...` is then not cast)."""
        if not self._ternary_over_lines(statement, value):
            return False
        last = value
        while isinstance(last.false_expr, TernaryOp):
            last = last.false_expr
        typing = getattr(self, '_condition_typing', None)
        return isinstance(last.false_expr, NaLiteral) and typing is not None and typing.is_plain_bool(last.true_expr)

    def _transform_if_statement(self, stmt: IfStatement) -> IfStatement:
        """Transform if statement (Rule 15). Pine v4/v5: a name that is the whole condition of an assigned if
        expression (`x = if n`) is not cast, a number or a pivot alike; one in an `and`/`or` there or in an
        `else if` is."""
        if stmt.of_an_assigned_value and self.pine_version <= 5 and isinstance(stmt.condition, Identifier):
            condition = self._transform_expression(stmt.condition)
        elif self.pine_version == 5 and not stmt.switch_arms and isinstance(stmt.condition, Identifier) \
                and stmt.condition.parenthesized:
            # Pine v5: a name written in parentheses as the whole condition (`if (n)`) is not cast; `if n` is
            condition = self._transform_expression(stmt.condition)
        else:
            condition = self._header(stmt.condition, stmt.switch_arms)
        body = [self._transform_statement(s) for s in stmt.body]

        elseifs = []
        for elif_cond, elif_body in stmt.elseifs:
            transformed_cond = self._header(elif_cond, stmt.switch_arms)
            transformed_body = [self._transform_statement(s) for s in elif_body]
            elseifs.append((transformed_cond, transformed_body))

        else_body = None
        if stmt.else_body:
            else_body = [self._transform_statement(s) for s in stmt.else_body]

        return IfStatement(
            condition=condition,
            body=body,
            elseifs=elseifs,
            else_body=else_body
        )

    def _header(self, condition: Expression, switch_arms: str = '') -> Expression:
        """The condition of an if or else-if, or (`switch_arms`) a case of a switch without a subject: the
        `and`/`or` operands of a case are cast as those of an if header, the whole case only where it is a name
        declared without a type from a value holding `c ? x : na`, and only in a switch statement (Pine v5)."""
        if not switch_arms:
            return self._condition(condition, header=True)
        if isinstance(condition, BinaryOp) and condition.op in ('and', 'or'):
            return self._condition(condition, header=True)
        transformed = self._transform_expression(condition)
        typing = getattr(self, '_condition_typing', None)
        if switch_arms == 'statement' and typing is not None and typing.is_cast_as_a_switch_case(condition):
            self.import_resolver.uses_cast_bool = True
            return FunctionCall(func='cast_bool', args=[transformed], kwargs={})
        return transformed

    def _transform_for_loop(self, stmt: ForLoop) -> ForLoop:
        """Transform for loop to range() (Rules 12-13)."""
        from_val = self._transform_expression(stmt.from_val)
        to_val = self._transform_expression(stmt.to_val)
        step = self._transform_expression(stmt.step) if stmt.step else None
        if step is None and self.pine_version < 6 and not stmt.is_assigned_value:
            # a Pine v4/v5 loop's default step is spelled out, not
            # for a loop given as the value of an assignment
            step = Literal(value=1, literal_type='int')
        body = [self._transform_statement(s) for s in stmt.body]
        # a loop whose end may change in its body reads it before every pass: pine_loop (for_loop_end_bound.py)
        return ForLoop(
            var=stmt.var,
            from_val=from_val,
            to_val=to_val,
            step=step,
            body=body,
            counter_name=stmt.counter_name,
            counter_helper=stmt.counter_helper,
            counter_entry=stmt.counter_entry,
            range_helper=self._range_helper_name()
        )

    def _range_helper_name(self) -> str:
        """The name `pine_range` is imported under: its own spelling unless the script uses that name
        (`pine_range = 2`), then a fresh one, as for `pine_loop`."""
        if self._range_helper is None:
            if self._used_names is None:
                self._used_names = collect_names(self._names_source)
            self._range_helper = (fresh_name(RANGE_HELPER, self._used_names)
                                  if RANGE_HELPER in self._used_names else RANGE_HELPER)
            self.import_resolver.pine_range_name = self._range_helper
        return self._range_helper

    def _transform_for_in_loop(self, stmt: ForInLoop) -> Union[ForInLoop, List[Statement]]:
        """Transform for-in loop (Rule 14)."""
        iterable = self._transform_expression(stmt.iterable)
        self._active_for_in = getattr(self, '_active_for_in', ()) + (stmt,)
        try:
            body = [self._transform_statement(s) for s in stmt.body]
        finally:
            self._active_for_in = self._active_for_in[:-1]
        if len(stmt.vars) > 1:
            return self._key_value_loop(stmt, iterable, body)
        return ForInLoop(
            vars=stmt.vars,
            iterable=iterable,
            body=body
        )

    def _key_value_loop(self, stmt: ForInLoop, iterable: Expression,
                        body: List[Statement]) -> Union[ForInLoop, List[Statement]]:
        """`for [k, v] in x`: an array or a matrix gives (index, element)
        through enumerate(); a map gives (key, value) through `.items()`;
        a value of unknown type is read as an array."""
        iterable_type = self._iterable_type(stmt.iterable)
        if iterable_type and not iterable_type.startswith(MAP_TYPE_PREFIXES):
            return ForInLoop(vars=stmt.vars, iterable=FunctionCall(func='enumerate', args=[iterable], kwargs={}),
                             body=body)
        if iterable_type:  # a map: `for k, v in m.items():`
            return ForInLoop(vars=stmt.vars, iterable=MethodCall(object=iterable, method='items', args=[], kwargs={}),
                             body=body)
        pine_iterable = stmt.iterable
        member = pine_iterable.method if isinstance(pine_iterable, MethodCall) else (
            pine_iterable.func.rpartition('.')[2] if isinstance(pine_iterable, FunctionCall)
            and isinstance(pine_iterable.func, str) and '.' in pine_iterable.func else None)
        if member in ('keys', 'values') and not (isinstance(pine_iterable, (MethodCall, FunctionCall))
                                                 and pine_iterable.args):
            # a map's `keys()` / `values()` is an array: (index, element)
            return ForInLoop(vars=stmt.vars, iterable=FunctionCall(func='enumerate', args=[iterable], kwargs={}),
                             body=body)
        if not self._script_holds_maps():
            # no map in the script: an array's (index, element) (Trendoscope/Zigzag
            # `_indicatorNames`, a field of its own type)
            return ForInLoop(vars=stmt.vars, iterable=FunctionCall(func='enumerate', args=[iterable], kwargs={}),
                             body=body)
        # a map the transpiler cannot type (a function result, a ternary, a copy): enumerate() would give
        # (index, key) for a dict; its class decides at run time instead, the
        # iterable evaluated once
        binding = []
        if not (isinstance(iterable, Identifier) and '.' not in iterable.name):
            temporary = self._fresh_temporary_name('iterable')
            binding = [Assignment(target=temporary, value=iterable)]
            iterable = Identifier(name=temporary)
        loop = ForInLoop(vars=stmt.vars, iterable=TernaryOp(
            condition=FunctionCall(func='isinstance', args=[iterable, Identifier(name='dict')], kwargs={}),
            true_expr=MethodCall(object=iterable, method='items', args=[], kwargs={}),
            false_expr=FunctionCall(func='enumerate', args=[iterable], kwargs={})), body=body)
        return binding + [loop] if binding else loop

    def _script_holds_maps(self) -> bool:
        """Whether the script names a map anywhere: a `map.*` call or a `map<K, V>` type. Without one no
        value it iterates is a map of its own making."""
        if self._holds_maps is None:
            script = self._names_source
            self._holds_maps = any(
                isinstance(value, str) and (value.startswith('map.') or 'map<' in value)
                for node in walk(script) if dataclasses.is_dataclass(node)
                for value in (getattr(node, field.name) for field in dataclasses.fields(node)))
        return self._holds_maps

    def _iterable_type(self, pine_iterable: Expression) -> Optional[str]:
        """The static type of a for-in iterable, or None. A name or field path has
        its declared type; `m.keys()`/`m.values()` of a typed map is an array; any
        other expression counts only when the inference gives a collection
        (`array.from(...)`, `map.new<K, V>()`)."""
        if receiver_path(pine_iterable):
            return self._receiver_type(pine_iterable)
        if isinstance(pine_iterable, FunctionCall) and isinstance(pine_iterable.func, str):
            receiver, _, member = pine_iterable.func.rpartition('.')
            if member in ('keys', 'values') and receiver \
                    and (self._receiver_type(receiver) or '').startswith(MAP_TYPE_PREFIXES):
                return 'list[Any]'
        inferred = self.type_inference.declared_type_name(self.type_inference.infer_type_hint(pine_iterable))
        return inferred if inferred and inferred.startswith(COLLECTION_TYPE_PREFIXES) else None

    def _transform_while_loop(self, stmt: WhileLoop) -> WhileLoop:
        """Transform while loop."""
        condition = self._condition(stmt.condition, header=True)
        body = [self._transform_statement(s) for s in stmt.body]
        return WhileLoop(condition=condition, body=body)

    def _transform_switch_statement(self, stmt: SwitchStatement) -> Union[IfStatement, List[Statement], None]:
        """Transform switch to if/elif/else chain (Rules 41-42)."""
        if not stmt.cases:
            # Default-only switch: the default always runs; a
            # subject is still evaluated, as Pine evaluates it
            subject = [ExpressionStatement(expr=self._transform_expression(stmt.expr))] if stmt.expr else []
            default_body = [self._transform_statement(s) for s in (stmt.default or [])]
            return self._flatten_statements(subject + default_body) or None

        # Every case condition first, the first one binding the subject
        first_condition, *elif_conditions = self._switch_case_conditions(stmt.expr, stmt.cases)

        transformed_body = [self._transform_statement(s) for s in stmt.cases[0][1]]
        elseifs = [(condition, [self._transform_statement(s) for s in case_body])
                   for condition, (_, case_body) in zip(elif_conditions, stmt.cases[1:])]

        # Default case becomes else
        else_body = None
        if stmt.default:
            else_body = [self._transform_statement(s) for s in stmt.default]

        return IfStatement(
            condition=first_condition,
            body=transformed_body,
            elseifs=elseifs,
            else_body=else_body
        )

    def _switch_case_conditions(self, subject: Optional[Expression], cases) -> List[Expression]:
        """The condition of each case, in case order.

        With a subject, the FIRST condition binds it, `(s := subject) == c1`, and
        the others read it, `s == c2`: the subject runs once, inside the first case
        test, which runs exactly when the switch does. A plain name or a
        literal is read again as is. Without a subject a case is its own condition.
        Every read is its own node.

        Statement-position twin: `block_value_lowering._BlockValueLowering.switch_into`
        binds the subject of a switch lowered to statements with an assignment
        instead; a change to how one binds the subject belongs in the other too.
        """
        if subject is None:
            return [self._header(case, 'statement') for case, _ in cases]
        subject_value = self._transform_expression(subject)
        read_again_as_is = isinstance(subject_value, Literal) or (
            isinstance(subject_value, Identifier) and '.' not in subject_value.name)
        if read_again_as_is:
            reads = [subject_value] + [dataclasses.replace(subject_value) for _ in cases[1:]]
        else:
            name = self._fresh_temporary_name('switch_subject')
            reads = [NamedExpression(name=name, value=subject_value)] + [Identifier(name=name) for _ in cases[1:]]
        return [BinaryOp(left=read, op='==', right=self._transform_expression(case))
                for read, (case, _) in zip(reads, cases)]

    def _legacy_bool_cast(self, operand: Expression) -> Expression:
        """Pine v5 `bool(x)`: `na(bool) if na(x) else bool(x)`. MEASURED on TradingView
        (2026-10-05, v5): `bool(1e-10)`, `bool(-1e-10)` and `bool(1.000001e-10)` are
        true, `bool(0.0)` is false, and `f(bool(2))` selects a bool overload; v5 has no
        1e-10 tolerance, unlike v6's `cast_bool`. MEASURED again (2026-10-06):
        `bool()` of an na float or an na bool is na, typed or untyped, so the na branch is
        `na(bool)`, not False (it reads as False wherever a condition reads it); that is
        Pine v5's three-state bool, which the decorator's `na_bool=True` switches on. An
        earlier reading `na=false` for `b_na = bool(naF)` is superseded (its script shape
        differed; the cause is UNPROVEN). Python's `bool` is true for nan, hence the `na`
        test. An operand that is not a name or a literal is evaluated once, into a temporary.

        A bool literal, a number literal that is 0 or beyond the tolerance, and a comparison (never na) cast the
        same either way: there it is `cast_bool(x)`."""
        value = self._transform_expression(operand)
        if _casts_alike(operand):
            self.import_resolver.uses_cast_bool = True
            return FunctionCall(func='cast_bool', args=[value], kwargs={})
        if isinstance(value, Literal) or (isinstance(value, Identifier) and '.' not in value.name):
            tested, read = value, dataclasses.replace(value)
        else:
            temporary = self._fresh_temporary_name('bool_operand')
            tested, read = NamedExpression(name=temporary, value=value), Identifier(name=temporary)
        self.import_resolver.add_module('na')
        return TernaryOp(condition=FunctionCall(func='na', args=[tested], kwargs={}),
                         true_expr=self._na_initialiser('bool'),
                         false_expr=FunctionCall(func='bool', args=[read], kwargs={}))

    def _free_spelling(self, preferred: str) -> str:
        """`preferred` when no name of the script or the module's own imports holds it, else the
        lowest free suffix of it; every spelling this returns is reserved for the rest of the run."""
        if self._identifier_names is None:
            self._identifier_names = (identifier_names(self._names_source) | EMITTED_HELPER_NAMES
                                      | set(self.import_resolver.aliased_modules.values()))
        return free_spelling(preferred, self._identifier_names)

    def _fresh_temporary_name(self, stem: str) -> str:
        """A temporary (a switch subject, a method receiver) that no Pine name of the
        script uses. The names are collected on the first call, so a script without
        such a temporary pays nothing for them."""
        if self._used_names is None:
            self._used_names = collect_names(self._names_source)
        return fresh_name(stem, self._used_names)

    # ========================================================================
    # Expression transformations
    # ========================================================================

    def _transform_expression(self, expr: Expression) -> Expression:
        """Transform expression."""
        if isinstance(expr, BinaryOp):
            if expr.op in ('and', 'or'):
                # an operand is cast only beside one that has a type: `s and x` with `x` an untyped
                # parameter casts neither; nor where the `and`/`or` chooses colours
                typing = getattr(self, '_condition_typing', None)
                header = getattr(self, '_header_bool_op', False)
                colours = getattr(self, '_uncast_bool_op', False)
                untyped_chains = getattr(self, '_untyped_chain_nodes', set())
                if typing is not None and id(expr) not in untyped_chains:
                    nodes, operands = _and_or_chain(expr)
                    if len(operands) > 2:
                        # beside a parameter only the operand of a two-operand `and`/`or` stays uncast: in a longer
                        # chain every operand is cast as anywhere else (`x and s and y > 0` casts `s`)
                        self._long_chain_nodes = getattr(self, '_long_chain_nodes', set()) | {id(node) for node in nodes}
                    plotted = id(expr) in getattr(self, '_plotshape_chains', ())
                    if any(typing.has_no_type_in_a_chain(operand, header) or (plotted and typing.is_reassigned_a_tuple_value(
                            operand)) for operand in operands):
                        # beside an operand without a type, no operand of the chain is cast
                        self._untyped_chain_nodes = untyped_chains | {id(node) for node in nodes}
                untyped_chain = id(expr) in getattr(self, '_untyped_chain_nodes', set())
                long_chain = id(expr) in getattr(self, '_long_chain_nodes', set())
                self._header_bool_op = False
                try:
                    # a name declared from `time(...)` is cast there too (`session and show ? color.red : na`)
                    left = (self._condition(expr.left, header=header, operand=True)
                            if (not colours or (typing is not None and typing.is_declared_from_a_direct_cast(expr.left)))
                            and (typing is None or long_chain or typing.has_a_type(expr.right, header, getattr(self, '_colour_test', False))) and not untyped_chain
                            else self._transform_expression(expr.left))
                    self._later_operands = getattr(self, '_later_operands', set()) | {id(expr.right)}
                    # a later `ta.change(x)` is computed first into a temporary, cast even beside a parameter
                    # (`t and ta.change(x)`: `t and cast_bool(change_1)`)
                    hoisted_change = isinstance(expr.right, FunctionCall) and expr.right.func == 'ta.change' \
                        and bool(expr.right.args) and self.pine_version <= 5
                    right = (self._condition(expr.right, header=header, operand=True)
                             if (not colours or (typing is not None and typing.is_declared_from_a_direct_cast(expr.right)))
                             and (typing is None or long_chain or hoisted_change or typing.has_a_type(expr.left, header, getattr(self, '_colour_test', False)))
                             and not untyped_chain
                             else self._transform_expression(expr.right))
                finally:
                    self._header_bool_op = header
            elif expr.op in ('==', '!=') and isinstance(expr.right, NaLiteral) and self._is_a_string(expr.left):
                # `s == na` of a string is `na(s)`, `s != na` `not na(s)`; a number or a bool, its
                # history and `na == s` stay compared
                test = FunctionCall(func='na', args=[self._transform_expression(expr.left)], kwargs={})
                return test if expr.op == '==' else UnaryOp(op='not', operand=test)
            elif expr.op in ('==', '!=') and self._compares_a_bool(expr):
                # `flag == 1`: the number compared with a bool is cast
                left, right = self._condition(expr.left), self._condition(expr.right)
            else:
                left, right = self._transform_expression(expr.left), self._transform_expression(expr.right)
            division = BinaryOp(
                left=left,
                op=expr.op,
                right=right,
                # Pine v4/v5 `x += y` is printed `x = x + y`
                # but `x op= <if/switch value>`, read into a block result, keeps its operator
                compound=expr.compound and (self.pine_version >= 6 or expr.of_a_block_value),
            )
            if expr.op == '/' and self._divides_const_ints(expr):
                self.import_resolver.uses_cast_int = True
                if isinstance(left, UnaryOp) and left.op == '-':
                    # `-9 / 2` is `-cast_int(9 / 2)`: the sign is read outside the division
                    division.left = left.operand
                    return UnaryOp(op='-', operand=FunctionCall(func='cast_int', args=[division], kwargs={}))
                return FunctionCall(func='cast_int', args=[division], kwargs={})
            return division
        elif isinstance(expr, UnaryOp):
            return UnaryOp(
                op=expr.op,
                operand=(self._condition(expr.operand, negated=True) if expr.op == 'not'
                         else self._transform_expression(expr.operand))
            )
        elif isinstance(expr, TernaryOp):
            return self._transform_ternary(expr)
        elif isinstance(expr, FunctionCall):
            return self._transform_function_call(expr)
        elif isinstance(expr, MethodCall):
            return self._transform_method_call(expr)
        elif isinstance(expr, MemberAccess):
            direct = self._directly_called_method(expr.object)
            if direct is not None:
                return MemberAccess(object=direct, member=expr.member)
            return self._transform_member_access(expr)
        elif isinstance(expr, IndexAccess):
            return self._transform_index_access(expr)
        elif isinstance(expr, Identifier):
            return self._transform_identifier(expr)
        elif isinstance(expr, Literal):
            return self._transform_literal(expr)
        elif isinstance(expr, NaLiteral):
            return self._transform_na_literal(expr)
        elif isinstance(expr, ArrayLiteral):
            return ArrayLiteral(elements=[self._transform_expression(e) for e in expr.elements])
        elif isinstance(expr, IfExpression):
            return self._transform_if_expression(expr)
        elif isinstance(expr, SwitchExpression):
            return self._transform_switch_expression(expr)
        else:
            return expr

    def _is_a_string(self, expression, depth: int = 8) -> bool:
        """`expression` is a string: typed so, `array.get` of a string array, or a name declared as one."""
        if depth <= 0:
            return False
        if self.type_inference.infer_type_hint(expression) == 'str':
            return True
        if isinstance(expression, FunctionCall) and expression.func == 'array.get' and expression.args:
            return self.type_inference.infer_type_hint(expression.args[0]) == 'list[str]'
        typing = getattr(self, '_condition_typing', None)
        declared = typing._values.get(expression.name) if typing is not None and isinstance(expression, Identifier) \
            else None
        if isinstance(declared, tuple):
            return declared[0] == 'string' or (declared[1] is not None and self._is_a_string(declared[1], depth - 1))
        return False

    def _divides_const_ints(self, expr: BinaryOp) -> bool:
        """Pine v4/v5 `a / b` of two const ints, an int division."""
        names = getattr(self, '_const_ints', None)
        return names is not None and is_const_int(expr.left, names) and is_const_int(expr.right, names)

    #: bool builtins not read as a bool where they are compared: `timeframe.change(tf) != 0`
    #: keeps its number
    _COMPARED_AS_NUMBERS = frozenset({'timeframe.change'})

    def _compares_a_bool(self, expr: BinaryOp) -> bool:
        """`flag == 1`: one side of an equality is a bool."""
        sides = [side for side in (expr.left, expr.right)
                 if not (isinstance(side, FunctionCall) and isinstance(side.func, str)
                         and side.func in self._COMPARED_AS_NUMBERS)]
        typing = getattr(self, '_condition_typing', None)
        if typing is not None and any(typing.is_bool(side) for side in sides):
            return True
        # a switch subject's temporary is bound again by every switch (`__switch__ = c`, later `__switch__ = n`):
        # its type is the subject's, which the condition typing reads, not the name's first binding
        return 'bool' in [self.type_inference.infer_type_hint(side) for side in sides
                          if (typing is None or not typing.reads_an_untyped_value(side))
                          and not (typing is not None and isinstance(side, Identifier)
                                   and side.name.startswith('__switch'))]

    def _only_bool_arms(self, ternary: TernaryOp) -> bool:
        """Every arm of `ternary`'s else chain but its na is a plain bool (a literal, a comparison, a bool name...)."""
        typing = getattr(self, '_condition_typing', None)
        node = ternary
        while isinstance(node, TernaryOp):
            arms = [node.true_expr] + ([] if isinstance(node.false_expr, TernaryOp) else [node.false_expr])
            if typing is None or any(not isinstance(arm, NaLiteral) and not typing.is_plain_bool(arm) for arm in arms):
                return False
            node = node.false_expr
        return True

    def _na_arm_is_false(self, expr: TernaryOp, other: Expression, typing, assigned_whole: bool) -> bool:
        """Whether the na arm of `expr` is written `False`: by where the ternary stands and what
        its other arm is, in Pine v4 and v5 whether or not a bool may be na there.

        * the condition of an if: the na of its else arm (`if c ? 1 : na`), not of its first (`if c ? na : 1`);
        * the whole value of an assignment (`x = c ? b : na`): a bool `b`, but for a parenthesised `(a and b)`
          and as _keeps_the_na_read_as_a_condition tells for a builtin call, a name or a literal;
        * elsewhere (an arm of another ternary, an `and`/`or` operand, an argument): a bool builtin call or a
          name declared a bool, and from v5 on a bool literal; a comparison or an `and`/`or` keeps the na, as
          does a v4 name in the arm of an outer ternary that is not its else chain.

        Pine v5: in main, the na ending the else chain of an assigned ternary written over several lines (the
        value below its name, or continued on the next line) is False beside any bool, unless the ternary is
        written in parentheses.

        A function's result keeps its na. The `plotshape` series is False for any bool where a bool cannot be
        na; Pine v6 keeps the rule of v5 without `na_bool` before these positions were told apart."""
        if expr is getattr(self, '_header_ternary', None):
            return isinstance(expr.false_expr, NaLiteral)  # `if c ? na : 1` keeps its na
        if not typing.is_plain_bool(other):
            return False
        if self.pine_version <= 5 and getattr(self, '_bool_may_be_na', False):
            # where a bool may be na (Pine v4, Pine v5 with na_bool) the na arm is na: Pine's bool is three-state there
            # (`bool p = c ? true : d ? false : na` is na on a bar where neither holds)
            return False
        if self.pine_version <= 5 and isinstance(other, TernaryOp) \
                and any(isinstance(arm, NaLiteral) for arm in (other.true_expr, other.false_expr)):
            return False  # beside a ternary that keeps its own na (`c1 ? (c2 ? a > b : na) : na`)
        if id(expr) in getattr(self, '_plotshape_series', ()):
            return not getattr(self, '_bool_may_be_na', True)
        assigned = getattr(self, '_assigned_ternary', None)
        if self.pine_version <= 5 and assigned is not None and assigned.parenthesized \
                and str(getattr(self, '_assigned_target', '') or '').startswith('__block_result') \
                and id(expr) in _else_chain(assigned):
            return False  # an if/switch branch valued `(c ? a : d ? b : na)` in parentheses keeps its na
        if getattr(self, '_assigned_over_lines', False) and isinstance(expr.false_expr, NaLiteral) \
                and id(expr) in _else_chain(getattr(self, '_assigned_ternary', None)):
            return True
        assigned = getattr(self, '_assigned_ternary', None)
        if self.pine_version <= 5 and assigned is not None and isinstance(expr.false_expr, NaLiteral) \
                and id(expr) in _else_chain(assigned) and typing.is_led_by_a_cast(assigned.true_expr):
            return True  # `x = c ? n and p : d ? p and q : na`, `n` a number: its first arm is a bool
        if self.pine_version >= 6:
            return not (isinstance(other, BinaryOp) and other.op in ('and', 'or')
                        and not isinstance(other.left, Identifier)) \
                and not (assigned_whole and getattr(self, '_assigned_target', None)
                         in getattr(self, '_na_observed_names', ()))
        if self.pine_version == 5 and isinstance(other, Identifier) and isinstance(expr.false_expr, NaLiteral) \
                and assigned is not None and expr is not assigned and id(expr) in _else_chain(assigned) \
                and getattr(self, '_assigned_target', None) in self._bare_argument_names():
            # `x = c ? a : d ? b : na` with `x` passed to a call as it is (`ta.vwap(close, x)`): the na ending the
            # else chain stays; read in a ternary's condition it is False
            return False
        call = isinstance(other, FunctionCall)
        if expr is getattr(self, '_assigned_ternary', None):
            if call or isinstance(other, (Identifier, Literal)):
                return not self._keeps_the_na_read_as_a_condition(expr, typing)
            if isinstance(other, BinaryOp) and other.op in ('and', 'or') and isinstance(other.left, Identifier) \
                    and getattr(self, '_assigned_target', None) not in self._script_read_names():
                # `x = c ? a and b : na` with `x` read nowhere keeps its na where a name leads the chain; one led by
                # `not na(s)` is `False`
                return False
            return not (isinstance(other, BinaryOp) and other.parenthesized and self.pine_version >= 5)
        if call:
            return True
        if isinstance(other, Literal) or (isinstance(other, Identifier) and other.name in ('true', 'false')):
            return self.pine_version >= 5
        if isinstance(other, Identifier):
            return self.pine_version >= 5 or not (assigned_whole and id(expr) not in _else_chain(
                getattr(self, '_assigned_ternary', None)))
        if isinstance(other, UnaryOp):
            return self.pine_version >= 5  # not measured: as before
        return False

    def _bare_argument_names(self) -> set:
        """The names the script passes to a call as they are (`f(x)`, `ta.vwap(close, x)`), collected once."""
        if getattr(self, '_bare_argument_set', None) is None:
            self._bare_argument_set = {argument.name for node in walk(self._names_source)
                                       if isinstance(node, FunctionCall)
                                       for argument in [*node.args, *node.kwargs.values()]
                                       if isinstance(argument, Identifier)} if self._names_source is not None else set()
        return self._bare_argument_set

    def _script_read_names(self) -> set:
        """The names the script reads somewhere (an identifier), collected once."""
        if getattr(self, '_read_name_set', None) is None:
            self._read_name_set = {node.name for node in walk(self._names_source) if isinstance(node, Identifier)} \
                if self._names_source is not None else set()
        return self._read_name_set

    def _keeps_the_na_read_as_a_condition(self, expr: TernaryOp, typing) -> bool:
        """Pine v4/v5, `x = c ? b : na` with `b` a bool builtin call, name or literal: the na is kept where an
        if header or a call argument reads `x` as a condition (`if x`, `alertcondition(x)`), or nothing reads `x`, unless `c` is a
        comparison (Pine v4: or an `and`/`or` with a comparison operand) and, for `x := c ? b : na`, unless `x` is
        declared with another value than `bool(na)` (`x = false`, `var bool x = na`)."""
        target = getattr(self, '_assigned_target', None)
        # a name read nowhere counts as one read as a condition (`CB_a = c and s ? long_c : na`, unread)
        observed = target in getattr(self, '_na_observed_names', ()) or target not in self._script_read_names()
        if not observed or _has_a_comparison(expr.condition, self.pine_version <= 4):
            return False
        if not getattr(self, '_assigned_again', False):
            return True
        declared = typing._values.get(target)
        value = declared[1] if isinstance(declared, tuple) else None
        return isinstance(value, FunctionCall) and value.func == 'bool' and len(value.args) == 1 \
            and isinstance(value.args[0], NaLiteral)

    def _transform_ternary(self, expr: TernaryOp) -> TernaryOp:
        """Transform ternary operator (Rule 10): cond ? a : b => a if cond else b. A name that is a bare arm of a
        ternary, at any depth of its arms, is not cast as an `and`/`or` operand elsewhere in those arms
        (`c ? ses : ses and d`)."""
        tree = getattr(self, '_ternary_tree', None)
        if tree is not None and id(expr) in tree[0]:
            return self._transform_ternary_in_tree(expr)
        self._ternary_tree = _ternary_tree(expr)
        try:
            return self._transform_ternary_in_tree(expr)
        finally:
            self._ternary_tree = tree

    def _transform_ternary_in_tree(self, expr: TernaryOp) -> TernaryOp:
        arms = (expr.true_expr, expr.false_expr)
        typing = getattr(self, '_condition_typing', None)
        other = next((arm for arm in arms if not isinstance(arm, NaLiteral)), None)
        assigned_whole = id(expr) in _arm_ternaries(getattr(self, '_assigned_ternary', None))
        if typing is not None and sum(isinstance(arm, NaLiteral) for arm in arms) == 1 \
                and id(expr) not in getattr(self, '_na_keeping_ternaries', ()) \
                and self._na_arm_is_false(expr, other, typing, assigned_whole):
            # `c ? ta.crossover(a, b) : na` is `... else False`; see _na_arm_is_false
            false = Literal(value=False, literal_type='bool')
            expr = dataclasses.replace(expr, true_expr=false if isinstance(expr.true_expr, NaLiteral) else expr.true_expr,
                                       false_expr=false if isinstance(expr.false_expr, NaLiteral) else expr.false_expr)
        # Note: We keep it as TernaryOp, codegen will convert to Python syntax
        return TernaryOp(
            condition=self._condition(expr.condition, ternary=expr, assigned=assigned_whole),
            true_expr=self._ternary_arm(expr.true_expr, expr.false_expr, typing),
            false_expr=self._ternary_arm(expr.false_expr, expr.true_expr, typing)
        )

    def _ternary_arm(self, arm: Expression, other: Expression, typing) -> Expression:
        """An arm of a ternary; a number name beside a bool literal is cast (`c ? n : false` is
        `cast_bool(n) if c else False`)."""
        depth = getattr(self, '_ternary_arm_depth', 0)
        self._ternary_arm_depth = depth + 1
        try:
            transformed = self._transform_expression(arm)
        finally:
            self._ternary_arm_depth = depth
        other_is_bool_literal = (isinstance(other, Literal) and other.literal_type == 'bool') or (
            isinstance(other, Identifier) and other.name in ('true', 'false'))
        if typing is not None and other_is_bool_literal and isinstance(arm, Identifier) and typing.is_number(arm):
            self.import_resolver.uses_cast_bool = True
            return FunctionCall(func='cast_bool', args=[transformed], kwargs={})
        if typing is not None and self.pine_version <= 5 and isinstance(arm, Identifier) and arm.name in NUMERIC_SERIES \
                and isinstance(other, BinaryOp) and other.op in ('==', '!=', '<', '>', '<=', '>=', 'and', 'or') \
                and not (other.parenthesized and self.pine_version >= 5):
            # a built-in series beside a comparison (`c ? k > d : close`) is cast; a name of the script is not,
            # nor (v5) one beside a comparison in parentheses
            self.import_resolver.uses_cast_bool = True
            return FunctionCall(func='cast_bool', args=[transformed], kwargs={})
        return transformed

    def _is_generic_array_of_objects(self, func_name: str, call: FunctionCall) -> bool:
        """`array.new<T>(...)` with arguments, for an element type that has no
        `array.new_*` factory (a UDT, an enum, `chart.point`, ...). A type the script
        neither declares nor imports is refused at its line. Without
        arguments it is `array.new(0, NA(T))`."""
        if not func_name.startswith('array.new<'):
            return False
        element_type = func_name[func_name.index('<') + 1:func_name.rindex('>')].strip()
        if element_type in ARRAY_FACTORY_ELEMENT_TYPES:
            return False
        if element_type in BUILTIN_TYPES_WITHOUT_A_CLASS and not (call.args or call.kwargs):
            return False  # `array.new<footprint>()`: an empty array, `list()`; pynecore has no class to type its na
        known = self._array_object_element_types()
        library_type = '.' in element_type and element_type[:element_type.index('.') + 1] in known
        if element_type not in known and not library_type:
            raise SemanticError('unknown-type', f"array.new<{element_type}>: '{element_type}' is not a type the "
                                'script declares or imports, nor a built-in type', call.line, call.column)
        return True

    def _array_object_element_types(self) -> set:
        """The script's UDTs and enums, the built-in object types without an
        `array.new_*` factory, and each imported library alias (its `alias.Type`)."""
        if self._object_element_types is None:
            self._object_element_types = (
                {declaration.name for declaration in self._names_source.declarations
                 if isinstance(declaration, (TypeDecl, EnumDecl))}
                | BUILTIN_OBJECT_ELEMENT_TYPES
                | {f'{library.alias or library.library}.' for library in self._names_source.imports})
        return self._object_element_types

    def _generic_array_new(self, call: FunctionCall, func_name: str = '') -> FunctionCall:
        """`array.new(size, value)`: the runtime repeats the one value it is given, so
        every element is the same object (MEASURED on TradingView, 2026-10-05: after
        `arr.get(0).v := 99` every element reads 99). The value is evaluated once,
        at any size; an omitted value is `na`."""
        args = [self._transform_expression(argument) for argument in call.args]
        kwargs = {name: self._transform_expression(value) for name, value in call.kwargs.items()}
        if '<' in func_name and any(isinstance(argument, NaLiteral) for argument in call.args):
            # a bare na given to `array.new<T>` (as its size or its value) is the na of the array's type,
            # `na(list[T])`
            element = func_name[func_name.index('<') + 1:func_name.rindex('>')].strip()
            python_type = convert_pine_generic_to_python(element)
            if python_type == 'ChartPoint':
                self.import_resolver.chart_types.add('ChartPoint')
            self.import_resolver.add_module('na')
            args = [FunctionCall(func='na', args=[Identifier(name=f'list[{python_type}]')], kwargs={})
                    if isinstance(original, NaLiteral) else argument for original, argument in zip(call.args, args)]
        if not args and not kwargs:
            args = [Literal(value=0, literal_type='int')]  # `array.new<T>()`: `array.new(0, NA(T))`
        if len(args) + len(kwargs) == 1 and 'initial_value' not in kwargs:  # a lone size gets T's na
            value = self._typed_na(func_name)
            if args:
                args.append(value)
            else:
                kwargs['initial_value'] = value
        self.import_resolver.add_module('array')
        return FunctionCall(func='array.new', args=args, kwargs=kwargs)

    def _typed_na(self, func_name: str) -> Expression:
        """`NA(T)` for the element type of `array.new<T>`/`matrix.new<T>`; `na` when it has none."""
        if '<' not in func_name:
            self.import_resolver.add_module('na')
            return Identifier(name='na')
        element = func_name[func_name.index('<') + 1:func_name.rindex('>')].strip()
        python_type = convert_pine_generic_to_python(element)
        self.import_resolver.uses_na = True
        if python_type == 'ChartPoint':
            self.import_resolver.chart_types.add('ChartPoint')
        return FunctionCall(func='NA', args=[Identifier(name=python_type)], kwargs={})

    def _generic_matrix_new(self, call: FunctionCall, func_name: str) -> FunctionCall:
        """`matrix.new<T>(rows, columns[, value])`: without a value it is passed
        `initial_value=NA(T)`; an integer value of a float matrix is written as a float, any other value that is not
        a literal is cast to float."""
        element = func_name[func_name.index('<') + 1:func_name.rindex('>')].strip()
        args = [self._transform_expression(argument) for argument in call.args]
        kwargs = {name: self._transform_expression(value) for name, value in call.kwargs.items()}
        if len(args) < 3 and 'initial_value' not in kwargs:
            kwargs['initial_value'] = self._typed_na(func_name)
        elif len(args) >= 3 and isinstance(call.args[2], NaLiteral):
            args[2] = self._typed_na(func_name)  # `matrix.new<string>(r, c, na)`: NA(str)
        elif element == 'float' and len(args) >= 3 and isinstance(args[2], Literal) \
                and args[2].literal_type == 'int':
            args[2] = Literal(value=float(args[2].value), literal_type='float')
        elif element == 'float' and len(args) >= 3 and not isinstance(args[2], Literal):
            # a value that is not a literal is cast to the element type: `matrix.new(1, n, cast_float(source))`
            self.import_resolver.uses_cast_float = True
            args[2] = FunctionCall(func='cast_float', args=[args[2]], kwargs={})
        self.import_resolver.add_module('matrix')
        return FunctionCall(func='matrix.new', args=args, kwargs=kwargs)

    def _transform_generic_function_name(self, func_name: str) -> str:
        """Transform function name with generics."""
        if '<' not in func_name:
            return func_name

        # For map.new<K,V>() -> map.new() (just strip generics)
        if 'map.new<' in func_name or func_name.startswith('map.new<'):
            return 'map.new'

        # For array.new<T>() -> array.new_T()
        if 'array.new<' in func_name or func_name.startswith('array.new<'):
            # Extract the type from array.new<type>
            start = func_name.index('<') + 1
            end = func_name.index('>')
            type_param = func_name[start:end].strip()

            # Map Pine Script types to PyneCore array.new_* functions; any other
            # declared type with arguments is routed to array.new() before this
            return ARRAY_FACTORIES.get(type_param, 'list')

        # For matrix.new<T>() -> matrix.new() (just strip generics)
        if 'matrix.new<' in func_name or func_name.startswith('matrix.new<'):
            return 'matrix.new'

        # For other generic functions, strip generics for now
        base_name = func_name[:func_name.index('<')]
        return base_name

    def _parse_dotted_path(self, path: str) -> Expression:
        """Parse a dotted path string like 'obj.lbl' into nested MemberAccess expressions."""
        parts = path.split('.')
        if len(parts) == 1:
            return Identifier(name=parts[0])

        # Build nested MemberAccess from left to right
        expr = Identifier(name=parts[0])
        for part in parts[1:]:
            expr = MemberAccess(object=expr, member=part)
        return expr

    @staticmethod
    def _module_of_type(type_str: Optional[str]) -> Optional[str]:
        """The builtin namespace of a declared receiver type (`list[float]` -> 'array'),
        or None for an unknown, UDT or scalar type. Callers pass the type already
        unwrapped (TypeInference.declared_type_name)."""
        if not type_str:
            return None
        if type_str.startswith('list'):
            return 'array'
        if type_str.startswith('dict'):
            return 'map'
        if type_str in ('line', 'label', 'box', 'table', 'linefill', 'polyline',
                        'Line', 'Label', 'Box', 'Table', 'LineFill', 'Polyline'):
            return type_str.lower()
        if type_str.startswith('matrix') or type_str.startswith('Matrix'):
            return 'matrix'
        if type_str in ('footprint', 'volume_row'):
            return type_str
        if '<' in type_str:
            return type_str[:type_str.index('<')]
        return None

    def _resolve_module_for_method(self, method_name: str, receiver_type: Optional[str]) -> Optional[str]:
        """The one routing rule for a builtin method, whatever the receiver's shape:
        the namespace of the receiver's declared type when it defines the method; else the
        one method namespace that defines it, statically; else RUNTIME_METHOD_DISPATCH when
        several do (pynecore picks by the receiver's class); None when none does."""
        namespaces = method_namespaces(method_name)
        typed_module = self._module_of_type(receiver_type)
        if typed_module in namespaces:
            return typed_module
        if len(namespaces) == 1:
            return namespaces[0]
        return RUNTIME_METHOD_DISPATCH if namespaces else None

    def _receiver_type(self, receiver) -> Optional[str]:
        """Declared type of a method-call receiver in the current scope, or None."""
        return self.type_inference.infer_receiver_type(
            receiver, self.current_function_params, self.current_function_shadowed_names)

    def _resolve_module_for_receiver(self, method_name: str, receiver) -> Optional[str]:
        """Module for `receiver.method_name()` by the routing rule, from the receiver's type. A receiver
        without a declared type goes through `method_call`, even where one namespace
        defines the method (`l = line.new(...)`, `l.set_x2(x)`)."""
        receiver_type = self._routing_receiver_type(receiver)
        namespaces = method_namespaces(method_name)
        if receiver_type is None and namespaces:
            return RUNTIME_METHOD_DISPATCH
        return self._resolve_module_for_method(method_name, receiver_type)

    _COLLECTION_CONSTRUCTORS = ('array.new', 'array.from', 'map.new', 'matrix.new')

    def _routing_receiver_type(self, receiver) -> Optional[str]:
        """The receiver type `receiver.method()` is routed by: only a type read off a
        declaration. A name declared with a type, a typed
        parameter, a name whose value is a collection constructor (`array.new<line>()`,
        `array.from(...)`, `map.new<K, V>()`) or a UDT's `T.new()`, a field of such a value and a
        for-in variable over a typed collection have it; a drawing (`l = line.new(...)`), a copy
        (`l6 = l1`) and a function's result do not: those go through `method_call`."""
        if isinstance(receiver, IndexAccess):
            return None  # a history read (`a[1].get(i)`) has no declared type (chrono_utils)
        path = receiver_path(receiver)
        if not path:
            return None
        loop_element = self._loop_element_type(path[0])
        typed_by_field = getattr(self, '_field_typed_calls', {}).get(getattr(self, '_routing_call_id', None))
        if loop_element is None and len(path) == 1 and typed_by_field and typed_by_field[0] == path[0]:
            return typed_by_field[1]  # `e.f` read or written before `e.m()`: e is typed by its field
        scope = getattr(self, '_typing_scope', None)
        self._declared_receiver_names()
        if loop_element is None and scope is not None and path[0] in getattr(self, '_untyped_function_locals', {}).get(
                (scope.name, scope.line), ()):
            return None  # `m = f(x)` in this function: untyped here, whatever another function's `m` is
        declared = path[0] in self._declared_receiver_names() and not (
            path[0] in self._parameter_only_receivers and path[0] not in (self.current_function_params or {}))
        if loop_element is None and (not declared or path[0] in getattr(self, '_untyped_parameters', ())):
            return None
        receiver_type = None if loop_element is not None else self._receiver_type(receiver)
        if receiver_type is None:
            # `var w = T.new().init()` (chrono_utils, biased_price_target), or a for-in variable
            scope = getattr(self, '_typing_scope', None)
            owner = (scope.name, scope.line) if scope is not None else None
            constructed = getattr(self, '_constructed_types', {})
            local = path[0] in self.current_function_shadowed_names or path[0] in (self.current_function_params or {})
            element = loop_element or constructed.get((owner, path[0])) or (
                None if local else constructed.get((None, path[0])))
            if element is not None and len(path) == 1:
                return element
            for field in path[1:] if element is not None else ():
                # a field of a for-in variable (`zigzagObject.zigzagLine`, Trendoscope/Zigzag)
                element = self.symbol_table.udt_field_types.get(
                    self.type_inference.declared_type_name(element) or element, {}).get(field)
                if element is None:
                    break
            else:
                return element
        return receiver_type

    def _loop_element_type(self, name: str) -> Optional[str]:
        """The element type of the collection the innermost for-in loop being transformed that binds `name`
        walks: a for-in variable is typed inside its own loop only (`for l in labels` does not type a later
        `for l in lines`, nor a later `b = box.new(...)` a `for b in boxes` did)."""
        for loop in reversed(getattr(self, '_active_for_in', ())):
            if name not in loop.vars:
                continue
            if len(loop.vars) > 1 and name != loop.vars[1]:
                return None  # `for [k, v] in m`: only the value is an element
            iterable = loop.iterable
            typing = getattr(self, '_condition_typing', None)
            declared = typing._values.get(iterable.name) if typing is not None and isinstance(iterable, Identifier) \
                else None
            if isinstance(declared, tuple) and isinstance(declared[1], Identifier):
                iterable = declared[1]  # `lines = line.all`, then `for l in lines`
            if isinstance(iterable, Identifier) and iterable.name in _ALL_COLLECTIONS:
                return _ALL_COLLECTIONS[iterable.name]  # `for l in line.all`: a line
            container = self._receiver_type(loop.iterable) or ''
            path = receiver_path(loop.iterable)
            if not container and path and len(path) == 2 and path[0] not in loop.vars:
                # `for l in p.lines` with `p` an outer for-in variable over `array<Profile>`: the field's type
                owner = self._loop_element_type(path[0])
                container = self.symbol_table.udt_field_types.get(owner, {}).get(path[1], '') if owner else ''
            if container.startswith('list[') and container.endswith(']'):
                return container[len('list['):-1]  # a for-in variable: its collection's element type
            if container.startswith('dict[') and container.endswith(']') and ',' in container:
                return container[container.index(',') + 1:-1].strip()  # `for [k, v] in m`: the map's values
            return None
        return None

    def _declared_receiver_names(self) -> frozenset:
        names = getattr(self, '_declared_receivers', None)
        if names is not None:
            return names
        type_names = {node.name for node in self._names_source.declarations if isinstance(node, (TypeDecl, EnumDecl))}

        aliases = set(self._library_aliases())

        def constructed_type(value) -> Optional[str]:
            """`T.new()`, `lib.T.new()` or a method chain on one (`T.new().init()`): T."""
            while isinstance(value, MethodCall):
                value = value.object
            if isinstance(value, FunctionCall) and isinstance(value.func, str):
                head, _, member = value.func.partition('<')[0].rpartition('.')
                if member == 'new' and (head in type_names or (head.partition('.')[0] in aliases
                                                                and head.count('.') == 1)):
                    return head
            return None

        drawings = {'line': 'Line', 'label': 'Label', 'box': 'Box', 'table': 'Table', 'linefill': 'LineFill',
                    'polyline': 'Polyline'}

        def all_drawings(value) -> Optional[str]:
            """`line.all`: list[Line] (toodegrees/forex_factory_utility)."""
            dotted = value.name if isinstance(value, Identifier) else (
                f'{value.object}.{value.member}' if isinstance(value, MemberAccess) and isinstance(value.object, str)
                else None)
            if dotted and dotted.endswith('.all') and dotted[:-len('.all')] in drawings:
                return f'list[{drawings[dotted[:-len(".all")]]}]'
            return None

        typed_names = {node.name for node in walk(self._names_source) if isinstance(node, InputDecl)} | {
            node.target if isinstance(node, Assignment) else node.name for node in walk(self._names_source)
            if isinstance(node, (Assignment, VarDecl, VaripDecl))
            and (getattr(node, 'type_hint', None) or isinstance(node.value, Literal))  # `b = 0.0` is a float
            and isinstance(getattr(node, 'target', ''), str)}

        def typed_element(value) -> bool:
            """An `array.from` element that types the array: a literal, an input or typed name, a string or colour
            call; not a `T.new()` (`array.from(G.new(...), ...)` routes its methods at run time)."""
            if isinstance(value, UnaryOp) and value.op == '-':
                value = value.operand
            if isinstance(value, NaLiteral):
                return False  # a bare na is the na of the others' type: it types nothing
            if isinstance(value, Literal):
                return True
            if isinstance(value, Identifier):
                # an inline input, hoisted into a parameter (`__input_3__`), types nothing
                return (value.name in typed_names and not re.fullmatch(r'__input_\d+__', value.name)) \
                    or value.name.startswith('color.')
            return isinstance(value, FunctionCall) and isinstance(value.func, str) \
                and value.func.startswith(('str.', 'color.'))

        def typed_value(value) -> bool:
            if isinstance(value, FunctionCall) and isinstance(value.func, str) \
                    and value.func.startswith(self._COLLECTION_CONSTRUCTORS):
                # `array.from(a, b)` of untyped values has no declared type (`EXlist.sum()` is
                # routed at run time)
                # (typed by any typed element: `array.from(close, a)` of a `var float a`, `array.from(na, 1.0)`;
                # a bare na, a ternary or a price series types nothing)
                return value.func != 'array.from' or any(typed_element(argument) for argument in value.args)
            return constructed_type(value) is not None or all_drawings(value) is not None

        input_kinds = {node.name: {'input.float': 'float', 'input.int': 'int', 'input.bool': 'bool',
                                   'input.string': 'string', 'input.color': 'color'}.get(node.func)
                       for node in walk(self._names_source) if isinstance(node, InputDecl)}
        self._field_typed_calls = self._calls_after_a_field_access()
        declared = set()
        parameters = set()
        # the type a name is constructed with, per function (`var a = array.new<float>()`, `var w = T.new()`)
        self._constructed_types = {}
        owners = {}
        for function in walk(self._names_source):
            if isinstance(function, FuncDecl):
                for inner in walk(function.body):
                    owners.setdefault(id(inner), (function.name, function.line))
        for node in walk(self._names_source):
            if isinstance(node, (Assignment, VarDecl, VaripDecl)):
                name = node.target if isinstance(node, Assignment) else node.name
                if isinstance(name, str) and (getattr(node, 'type_hint', None) or typed_value(node.value)):
                    declared.add(name)
                    key = (owners.get(id(node)), name)
                    if constructed_type(node.value):
                        self._constructed_types.setdefault(key, constructed_type(node.value))
                    elif all_drawings(node.value):
                        self._constructed_types.setdefault(key, all_drawings(node.value))
                    elif isinstance(node.value, FunctionCall) and isinstance(node.value.func, str) \
                            and re.fullmatch(r'(array|map|matrix)\.new<.+>', node.value.func):
                        # `var a = array.new<float>()` in a block (series_collection)
                        generic = node.value.func.replace('.new', '', 1)
                        self._constructed_types.setdefault(key, convert_pine_generic_to_python(generic))
                    elif isinstance(node.value, FunctionCall) and node.value.func == 'array.from' and node.value.args:
                        element = self.type_inference.infer_type_hint(node.value.args[0])  # `array.from(0., 1.)`
                        if element == 'str' or (element not in ('int', 'float', 'bool', 'string', 'color') and any(
                                isinstance(argument, Literal) and argument.literal_type == 'string'
                                for argument in node.value.args)):
                            element = 'string'  # `array.from("a", "b")`, `array.from(str.format(...), "b")`: list[str]
                        first = node.value.args[0]
                        if element not in ('int', 'float', 'bool', 'string', 'color') and isinstance(first, Identifier):
                            element = input_kinds.get(first.name)  # `array.from(fib0, fib1)` of `input.float`s
                        if element in ('int', 'float', 'bool', 'string', 'color'):
                            self._constructed_types.setdefault(key, convert_pine_generic_to_python(f'array<{element}>'))
            elif isinstance(node, InputDecl):
                declared.add(node.name)
            elif isinstance(node, FuncDecl):
                parameters.update(parameter.name for parameter in node.params if parameter.type_hint)
        # a name only a function's typed parameter declares is typed in that function only
        self._parameter_only_receivers = frozenset(parameters - declared)
        declared |= parameters
        # a for-in variable is typed inside its own loop only (_loop_element_type)
        self._declared_receivers = frozenset(declared)
        # per function, the locals every binding of which in that function is untyped (`m = f(x)`): a typed local of
        # that name in another function does not type them
        self._untyped_function_locals = {}
        for function in walk(self._names_source):
            if not isinstance(function, FuncDecl):
                continue
            typed, untyped = set(), set()
            for node in walk(function.body):
                if isinstance(node, (Assignment, VarDecl, VaripDecl)):
                    name = node.target if isinstance(node, Assignment) else node.name
                    if isinstance(name, str):
                        (typed if getattr(node, 'type_hint', None) or typed_value(node.value) else untyped).add(name)
            self._untyped_function_locals[(function.name, function.line)] = frozenset(
                untyped - typed - {parameter.name for parameter in function.params})
        return self._declared_receivers

    def _calls_after_a_field_access(self) -> dict:
        """id of a `name.m(...)` call -> (name, the type declaring a field of `name` read or written before the call
        in the same function, in source order and in any block). A name is typed by the fields read off it:
        `e.m()` is `method_call(m, e)` until `e.f` has been read or written (`x = e.f`, `if e.f > 0`, `e.f := 1`),
        and `m(e)` after. A for-in variable is not typed so in its loop (`for sp in m.values()`: `if sp.price > 0`
        leaves `sp.mark()` a `method_call`)."""
        owners: dict = {}
        for type_name, fields in self.symbol_table.udt_field_types.items():
            for field_name in fields:
                owners.setdefault(field_name, type_name)
        found: dict = {}
        loop_variables: List[str] = []  # the for-in variables of the loops being scanned

        def accessed(text, seen: dict) -> None:
            parts = text.split('.') if isinstance(text, str) else []
            if len(parts) >= 2 and parts[1] in owners and parts[0] not in KNOWN_MODULES \
                    and parts[0] not in loop_variables:
                seen.setdefault(parts[0], owners[parts[1]])

        def scan(part, seen: dict) -> None:
            nodes = list(walk(part))
            for node in nodes:
                if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func.count('.') == 1:
                    name = node.func.split('.')[0]
                    if name in seen:
                        found[id(node)] = (name, seen[name])
            for node in nodes:
                if isinstance(node, Identifier):
                    accessed(node.name, seen)
                elif isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func.count('.') >= 2:
                    accessed(node.func, seen)
                elif isinstance(node, (Assignment, Reassignment)) and isinstance(node.target, str):
                    accessed(node.target, seen)

        def statement(node, seen: dict) -> None:
            if isinstance(node, FuncDecl):
                block(node.body if isinstance(node.body, list) else [node.body], {})
                return
            if isinstance(node, ForInLoop):
                scan(node.iterable, seen)
                names = [name for name in (node.vars or []) if isinstance(name, str)]
                for name in names:
                    seen.pop(name, None)  # the loop binds it anew
                loop_variables.extend(names)
                try:
                    block(node.body, seen)
                finally:
                    del loop_variables[len(loop_variables) - len(names):]
                return
            if not dataclasses.is_dataclass(node) or isinstance(node, type):
                scan(node, seen)
                return
            blocks, parts = [], []
            for field in dataclasses.fields(node):
                value = getattr(node, field.name)
                if isinstance(value, list) and value and all(isinstance(item, Statement) for item in value):
                    blocks.append(value)
                elif isinstance(value, list) and any(isinstance(item, tuple) for item in value):
                    blocks.append(value)  # `elseifs`, `cases`: (condition, body) pairs
                else:
                    parts.append(value)
            if not blocks:
                scan(node, seen)  # a simple statement, its target too (`e.f := 1`)
                return
            for value in parts:
                scan(value, seen)
            for value in blocks:
                if all(isinstance(item, tuple) for item in value):
                    for pair in value:
                        for part in pair:
                            if isinstance(part, list):
                                block(part, seen)
                            else:
                                scan(part, seen)
                else:
                    block(value, seen)

        def block(statements, seen: dict) -> None:
            for item in statements:
                statement(item, seen)

        items = sorted([*self._names_source.declarations, *self._names_source.body],
                       key=lambda item: (getattr(item, 'line', 0) or 0, getattr(item, 'column', 0) or 0))
        block(items, {})
        return found

    def _method_signatures(self, method_name: str) -> List[tuple]:
        """(`this` type, required, total arguments after `this`) of every user method named `method_name`."""
        return [(parameters[0].type_hint, sum(1 for parameter in parameters[1:] if parameter.default is None),
                 len(parameters) - 1)
                for node in walk(self._names_source)
                if isinstance(node, FuncDecl) and node.is_method and node.name == method_name
                for parameters in [node.params] if parameters]

    def _user_method_route(self, method_name: str, receiver, call=None) -> Optional[str]:
        """How `receiver.method_name()` reaches a user method of that name:
        USER_METHOD, RUNTIME_CHOICE, or None for the builtin route (also when no user
        method has the name). A typed receiver takes the user method unless its builtin
        namespace defines the name and no user method is declared on that namespace's
        type. An untyped receiver whose method name a UDT method shares with a builtin
        namespace is decided at run time by its class."""
        this_types = self.method_functions.get(method_name)
        if this_types is None:
            return None
        namespaces = method_namespaces(method_name)
        receiver_type = self._routing_receiver_type(receiver)
        receiver_module = self._module_of_type(receiver_type)
        if not namespaces or (receiver_type and receiver_module not in namespaces):
            return USER_METHOD
        if receiver_type:
            # a builtin namespace has the name too: the user method only for its own `this` type and a call it
            # takes (Trendoscope/Zigzag: `values.push(x)` on array<float> is array.push, `this.clear()` on
            # array<ZigzagDrawing> the user method)
            wanted = _spelled_type(receiver_type)
            arguments = None if call is None else len(call.args) + len(call.kwargs)
            return USER_METHOD if any(
                _spelled_type(this_type) == wanted and (arguments is None or required <= arguments <= total)
                for this_type, required, total in self._method_signatures(method_name)) else None
        udt_names = self.symbol_table.udt_field_types
        return RUNTIME_CHOICE if all(this_type in udt_names for this_type in this_types) else USER_METHOD

    def _route_method_call(self, call, method_name: str, receiver, make_receiver) -> Optional[Expression]:
        """`receiver.method_name(...)` as a function call, or None when no route applies.
        `receiver` is the Pine receiver (a node or dotted path); `make_receiver()` builds
        a fresh transformed receiver expression each time it is called."""
        user_methods = self._user_methods()
        if self._has_a_library_type(receiver) and not self._is_a_script_method_of(method_name, receiver):
            # a receiver of a type an imported library declares (`dr.Line l`): the library's method, chosen at run
            # time by name, `method_call('draw', this.l)`, whatever the script or a builtin namespace calls so
            args = [make_receiver()] + [self._transform_expression(arg) for arg in call.args]
            kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
            self.import_resolver.uses_method_call = True
            return FunctionCall(func=RUNTIME_METHOD_DISPATCH,
                                args=[Literal(value=method_name, literal_type='string')] + args, kwargs=kwargs)
        # on a value that is no name (`T.new().m()`, `matrix.row(m, r).nonZero()`, `(a * b).m()`) the method
        # is called directly only when every declaration of its name is exported
        typed_arm = isinstance(receiver, TernaryOp) and any(
            receiver_path(arm) is not None and self._has_a_known_type(arm)
            for arm in (receiver.true_expr, receiver.false_expr))  # `(c ? T.new() : this.t).m()` (chrono_utils)
        if typed_arm and method_name in user_methods:
            return self._user_method_call(method_name, receiver, make_receiver(), call)
        if method_name in user_methods and isinstance(receiver, MemberAccess) \
                and isinstance(receiver.object, (FunctionCall, MethodCall)) and self._names_a_plain_field(receiver.member):
            # a field of a call's result (`f().a.m()`, `arr.get(0).a.m()`) that every type declares as a plain value
            # is typed by its name: the method is called directly; a field of an object type, a field of a field
            # (`arr.get(0).p.a.m()`) or of an untyped name is chosen at run time
            return self._user_method_call(method_name, receiver, make_receiver(), call)
        # an inline input hoisted into a parameter (`input.enum(...).to_string()`) is still a call's result
        hoisted_input = isinstance(receiver, Identifier) and re.fullmatch(r'\w+__input|__input_\d+__', receiver.name)
        if (receiver_path(receiver) is None and not isinstance(receiver, Literal)) or hoisted_input:
            dispatched = method_name in user_methods and not user_methods[method_name]
            if method_name in user_methods and not dispatched:  # `clear(delete(this))` (Trendoscope/Drawing)
                return self._user_method_call(method_name, receiver, make_receiver(), call)
        else:
            # a literal receiver dispatches a method of the script, but an exported one is called directly
            # (`"Hello".announce()` of `export method announce`)
            dispatched = method_name in user_methods and not self._has_a_known_type(receiver) and not (
                isinstance(receiver, Literal) and user_methods[method_name])
        renamed_keywords = set(call.kwargs) & set(self._pine_keywords.get(method_name, {}))
        if dispatched and renamed_keywords and method_namespaces(method_name) and method_name in self.method_functions:
            # a keyword the user method spells otherwise than the builtin of its name: one call cannot pass both,
            # so the receiver's class chooses the arm, each with its own keyword
            return self._runtime_chosen_method_call(method_name, receiver, make_receiver, call)
        if dispatched and not user_methods[method_name] and self._method_keeps_state(method_name):
            # a method with state of its own (`var`, `ta.*`): `method_call(m, x)` shares one state across every call
            # site, Pine keeps one per site; a direct call keeps it
            if method_namespaces(method_name) and method_name in self.method_functions:
                return self._runtime_chosen_method_call(method_name, receiver, make_receiver, call)
            return self._user_method_call(method_name, receiver, make_receiver(), call)
        if dispatched:
            # a user method on a receiver without a declared type is chosen at run time,
            # `method_call(m, x)`; an exported method by its name, `method_call('m', x)`
            args = [make_receiver()] + [self._transform_expression(arg) for arg in call.args]
            kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
            # an overload set chosen by value: a float-typed receiver or argument holding an int is passed as a float
            pine_receiver = self._parse_dotted_path(receiver) if isinstance(receiver, str) else receiver
            args, kwargs = self._overload_arguments.coerce(
                call, method_name, [pine_receiver, *call.args], args, call.kwargs, kwargs)
            callee = (Literal(value=method_name, literal_type='string') if user_methods[method_name]
                      else Identifier(name=method_name))
            self.import_resolver.uses_method_call = True
            return FunctionCall(func=RUNTIME_METHOD_DISPATCH, args=[callee] + args, kwargs=kwargs)
        user_route = self._user_method_route(method_name, receiver, call)
        if user_route == USER_METHOD:
            return self._user_method_call(method_name, receiver, make_receiver(), call)
        if user_route == RUNTIME_CHOICE:
            return self._runtime_chosen_method_call(method_name, receiver, make_receiver, call)
        module_name = self._resolve_module_for_receiver(method_name, receiver)
        if module_name is None and method_name not in user_methods and not method_namespaces(method_name) \
                and self._library_aliases():
            # a method an imported library exports (`labels.flush()`): chosen at run time (Trendoscope/Zigzag)
            module_name = RUNTIME_METHOD_DISPATCH
        if module_name is None:
            return None
        return self._builtin_method_call(module_name, method_name, make_receiver(), call)

    def _names_a_plain_field(self, field_name: str) -> bool:
        """Every type declaring a field `field_name` declares it `int`, `float`, `bool` or `string`."""
        declared = [fields[field_name] for fields in self.symbol_table.udt_field_types.values() if field_name in fields]
        return bool(declared) and all(field_type in ('int', 'float', 'bool', 'str') for field_type in declared)

    def _method_keeps_state(self, method_name: str) -> bool:
        """Whether a declaration of the user method keeps state that changes from bar to bar: a `var`/`varip` it
        reassigns (`c += 1`), or a `ta.*` call. A `var` set once is the same at every call site."""
        stateful = getattr(self, '_stateful_methods', None)
        if stateful is None:
            stateful = set()
            for node in walk(self._names_source):
                if not (isinstance(node, FuncDecl) and node.is_method):
                    continue
                inner_nodes = list(walk(node.body))
                persistent = {inner.name for inner in inner_nodes if isinstance(inner, (VarDecl, VaripDecl))}
                if any((isinstance(inner, Reassignment) and inner.target in persistent) or (
                        isinstance(inner, FunctionCall) and isinstance(inner.func, str) and inner.func.startswith('ta.'))
                       for inner in inner_nodes):
                    stateful.add(node.name)
            self._stateful_methods = stateful
        return method_name in stateful

    def _has_a_library_type(self, receiver) -> bool:
        """The receiver's declared type is one an imported library exports (`dr.Line`)."""
        aliases = self._library_aliases()
        if not aliases or isinstance(receiver, (Literal, TernaryOp)):
            return False
        receiver_type = self._routing_receiver_type(receiver)
        return bool(receiver_type) and receiver_type.split('.', 1)[0] in aliases and '.' in receiver_type

    def _is_a_script_method_of(self, method_name: str, receiver) -> bool:
        """The script declares `method_name` for the receiver's type (`method divergence(zg.Zigzag this)`)."""
        wanted = _spelled_type(self._routing_receiver_type(receiver) or '')
        return any(_spelled_type(this_type) == wanted for this_type in self.method_functions.get(method_name, ()))

    def _has_a_known_type(self, receiver) -> bool:
        """A receiver whose type is known for a user method: a declared one, an input, a
        built-in series (`close.f()`) or a name declared from a literal; a literal itself is not
        (`"America/New_York".ts(s)` is `method_call(ts, 'America/New_York', s)`)."""
        if isinstance(receiver, Literal):
            return False
        if self._routing_receiver_type(receiver) is not None:
            return True
        path = receiver_path(receiver)
        if path and '.'.join(path) in _CHART_COLOURS:
            return True  # the chart's colours are colours (`chart.fg_color.transp(50)`); a `color.*` constant is not
        if not path or len(path) != 1:
            return False
        name = path[0]
        if any(isinstance(node, (Assignment, VarDecl)) and getattr(node, 'target', getattr(node, 'name', None)) == name
               and isinstance(node.value, Identifier) and node.value.name in _CHART_COLOURS
               for node in walk(self._names_source)):
            return True  # `col = chart.fg_color`, then `col.transp(50)`
        if any(isinstance(node, (Assignment, VarDecl)) and getattr(node, 'target', getattr(node, 'name', None)) == name
               and (isinstance(node.value, Literal)
                    or (isinstance(node.value, Identifier) and node.value.name in ('true', 'false')))
               for node in walk(self._names_source)):
            return True  # a name declared from a literal (`msg = '{...}'`, then `msg.parse(...)`)
        if name in SERIES_VARIABLES:
            return True
        return any(isinstance(node, InputDecl) and node.name == name for node in self._names_source.declarations)

    def _user_methods(self) -> Dict[str, bool]:
        """Every user method's name -> whether all its declarations are exported (a dispatched call then names
        it `method_call('m', x)`, else `method_call(m, x)`)."""
        methods = getattr(self, '_user_method_names', None)
        if methods is None:
            methods = {}
            for node in walk(self._names_source):
                if isinstance(node, FuncDecl) and node.is_method:
                    methods[node.name] = methods.get(node.name, True) and node.is_export
            self._user_method_names = methods
        return methods

    def _user_method_call(self, method_name: str, receiver, receiver_expr, call) -> FunctionCall:
        """`method_name(receiver, ...)`, a call of the user method."""
        pine_receiver = self._parse_dotted_path(receiver) if isinstance(receiver, str) else receiver
        args = [receiver_expr] + [self._transform_expression(arg) for arg in call.args]
        kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
        args, kwargs = self._overload_arguments.coerce(
            call, method_name, [pine_receiver, *call.args], args, call.kwargs, kwargs)
        return FunctionCall(func=method_name, args=args, kwargs=kwargs)

    def _builtin_method_call(self, module_name: str, method_name: str, receiver_expr, call) -> FunctionCall:
        """`module.method(receiver, ...)`, or `method_call('method', receiver, ...)` to let
        pynecore pick the namespace from the receiver's class. A keyword the scope
        renamer renamed for a user method of the same name keeps its Pine name."""
        args = [receiver_expr] + [self._transform_expression(arg) for arg in call.args]
        pine_keywords = self._pine_keywords.get(method_name, {})
        kwargs = {pine_keywords.get(k, k): self._transform_expression(v) for k, v in call.kwargs.items()}
        if module_name == RUNTIME_METHOD_DISPATCH:
            self.import_resolver.uses_method_call = True
            method_literal = Literal(value=method_name, literal_type='string')
            return FunctionCall(func=RUNTIME_METHOD_DISPATCH, args=[method_literal] + args, kwargs=kwargs)
        self.import_resolver.add_module(module_name)
        return FunctionCall(func=f'{module_name}.{method_name}', args=args, kwargs=kwargs)

    def _runtime_chosen_method_call(self, method_name: str, receiver, make_receiver, call) -> TernaryOp:
        """`m(x) if isinstance(x, Z) else <builtin route>(x)`: an untyped receiver runs the
        user method when it is an instance of the method's UDT, else the builtin. A direct
        call keeps the user method's per-call-site state, which pynecore's callable
        `method_call(m, x)` would share across call sites. A receiver that is not a name
        or field path (a call result) is evaluated once, into a temporary."""
        classes = [Identifier(name=this_type) for this_type in sorted(self.method_functions[method_name])]
        udt_classes = classes[0] if len(classes) == 1 else ArrayLiteral(elements=classes)
        builtin_module = self._resolve_module_for_method(method_name, None)
        if receiver_path(receiver):
            tested_receiver, read_receiver = make_receiver(), make_receiver
        else:
            temporary = self._fresh_temporary_name('method_receiver')
            tested_receiver = NamedExpression(name=temporary, value=make_receiver())
            read_receiver = lambda: Identifier(name=temporary)  # noqa: E731 - the arms read the temporary
        return TernaryOp(
            condition=FunctionCall(func='isinstance', args=[tested_receiver, udt_classes], kwargs={}),
            true_expr=self._user_method_call(method_name, receiver, read_receiver(), call),
            false_expr=self._builtin_method_call(builtin_module, method_name, read_receiver(), call))

    def _transform_dotted_string_method_call(self, call: FunctionCall, func: str) -> Optional[Expression]:
        """Handle var.method() → module.method(var, ...) for dotted string function names.

        Routes a user method, or a receiver typed as a builtin namespace; returns None
        for the rest, which the chained handler takes (it skips module names: `math.max`).
        """
        var_name, _, method_name = func.partition('.')
        if not method_name or '.' in method_name or self._names_a_type(var_name):  # Not a variable method call
            return None
        self._routing_call_id = id(call)
        if self._names_a_namespace(var_name):  # `array.push(a, x)` is a namespace function
            return None
        if method_name == 'copy' and not call.args and not call.kwargs and 'copy' not in self._user_methods():
            self.import_resolver.uses_udt_copy = True  # `d.copy()` is `udt_copy(d)`, an array too
            return FunctionCall(func='udt_copy', args=[self._transform_expression(Identifier(name=var_name))],
                                kwargs={})
        if method_name not in self.method_functions and not self._module_of_type(self._receiver_type(var_name)):
            return None
        return self._route_method_call(call, method_name, Identifier(name=var_name), lambda: Identifier(name=var_name))

    def _transform_chained_dotted_method_call(self, call: FunctionCall, func: str) -> Optional[Expression]:
        """Handle obj.field.method() → module.method(obj.field, ...) for chained access.

        Returns transformed FunctionCall or None if not a chained method pattern.
        """
        all_variable, _, all_method_name = func.rpartition('.')
        if all_variable in DRAWING_ALL_VARIABLES and all_method_name in self._user_methods() \
                and 'array' not in method_namespaces(all_method_name):
            # a user method on the drawings (`method clean(array<line> a)`, `line.all.clean()`): called directly
            return self._user_method_call(all_method_name, Identifier(name=all_variable),
                                          Identifier(name=all_variable), call)
        if all_variable in DRAWING_ALL_VARIABLES:  # box.all.size() -> array.size(box.all)
            return self._builtin_method_call('array', all_method_name, Identifier(name=all_variable), call)

        prefix, _, last = func.rpartition('.')
        if last in self._user_methods() and (prefix in _CHART_COLOURS or (
                prefix.startswith('color.') and prefix[len('color.'):] in COLOUR_CONSTANT_NAMES)):
            # a user method on a built-in value (`chart.fg_color.transp(50)`, `color.red.transp(50)`)
            return self._route_method_call(call, last, prefix, lambda: self._parse_dotted_path(prefix))
        first_component = func.split('.')[0]
        if first_component in _SERIES_VALUES and func.count('.') == 1 and last not in self._user_methods() \
                and self._library_aliases() and not method_namespaces(last):
            # a method an imported library exports, called on a builtin series (`close.f(3.0)`): `close` is a value,
            # not a namespace; the method is chosen at run time by its name, as on any other receiver
            return self._route_method_call(call, last, Identifier(name=first_component),
                                           lambda: Identifier(name=first_component))
        if (first_component in KNOWN_MODULES or self._names_a_type(first_component)
                or first_component in self.import_resolver.aliased_modules.values()
                or first_component in self._library_aliases()):
            return None

        obj_path, _, method_name = func.rpartition('.')  # "obj.lbl", "set_x"

        # Special case: .copy() on object fields -> udt_copy()
        # Must be checked BEFORE set matching (Location 1 behavior)
        if method_name == 'copy' and self._user_method_route(method_name, obj_path) is None:
            obj_expr = self._parse_dotted_path(obj_path)
            transformed_args = [obj_expr] + [self._transform_expression(arg) for arg in call.args]
            transformed_kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
            self.import_resolver.uses_udt_copy = True
            return FunctionCall(func='udt_copy', args=transformed_args, kwargs=transformed_kwargs)

        return self._route_method_call(call, method_name, obj_path, lambda: self._parse_dotted_path(obj_path))

    def _directly_called_method(self, value) -> Optional[FunctionCall]:
        """`x.m(...)` whose result is read for a field (`buf.rotate(n).data`): the user method is called
        directly, whatever is known of `x` (jason5480/series_collection)."""
        if not (isinstance(value, FunctionCall) and isinstance(value.func, str) and '.' in value.func):
            return None
        receiver, _, method_name = value.func.rpartition('.')
        if method_name not in self._user_methods() or self._names_a_namespace(receiver.split('.')[0]) \
                or self._names_a_type(receiver.split('.')[0]):
            return None
        return self._user_method_call(method_name, receiver, self._parse_dotted_path(receiver), value)

    def _names_a_user_type(self, name: str) -> bool:
        return any(isinstance(node, TypeDecl) and node.name == name for node in self._names_source.declarations)

    def _names_a_type(self, name: str) -> bool:
        """`T.new()`, `E.member`: a capitalised name that is no variable of the script (`EMSP.repeat()`
        on `const string EMSP` is a method call)."""
        if any(isinstance(node, (TypeDecl, EnumDecl)) and node.name == name for node in self._names_source.declarations):
            return True
        return name[:1].isupper() and name not in self._declared_receiver_names() \
            and self.symbol_table.lookup(name) is None and not self._is_a_parameter(name)

    def _is_a_parameter(self, name: str) -> bool:
        """`name` is a parameter of the function being transformed (`kernel(X1, X2) => X1.size()`)."""
        scope = getattr(self, '_typing_scope', None)
        return scope is not None and any(parameter.name == name for parameter in getattr(scope, 'params', ()))

    def _names_a_namespace(self, name: str) -> bool:
        """A pynecore namespace or an imported library alias, not a series variable:
        `close.m()` is a method call on the series, `math.abs(x)` is not."""
        return ((name in KNOWN_MODULES and not is_series_variable(name))
                or name in self.import_resolver.aliased_modules.values()
                or name in self._library_aliases())

    def _library_aliases(self) -> set:
        """The names the script's `import user/lib/version as alias` lines bind (an unaliased import binds the library name):
        `lib.max(a, b)` calls the library's `max`, it is not a method call on `lib`."""
        return {imported.alias or imported.library for imported in self._names_source.imports}

    def _transform_member_access_method_call(self, call: FunctionCall, func: MemberAccess) -> Optional[Expression]:
        """Handle MemberAccess-based method calls like obj.lbl.set_x(...).

        Returns transformed FunctionCall or None if not a recognized method.
        """
        return self._route_method_call(call, func.member, func.object,
                                       lambda: self._transform_expression(func.object))

    def _transform_function_call(self, call: FunctionCall) -> Expression:
        """Transform function call (Rules 23-25, 49-51)."""
        pine_call = call  # the node as parsed; `call` may be rebuilt below
        if call.func in ('plotshape', 'plotchar') and (call.args or 'series' in call.kwargs):
            call = self._plotshape_bool_series(call)  # `plotchar` plots its series as `plotshape` does
        if call.func == 'ta.rsi' and len(call.args) == 2 and not call.kwargs and (
                self._is_float_series(call.args[1])
                or (self.pine_version <= 4 and self._is_a_non_int_value(call.args[1]))):
            # `rsi(upper, lower)` of two series is the RSI of their ratio: `100 - 100 / (1 + upper / lower)`
            # (Pine's two-series form; the runtime's ta.rsi takes a length there)
            return self._transform_expression(BinaryOp(
                left=Literal(value=100.0, literal_type='float'), op='-',
                right=BinaryOp(left=Literal(value=100.0, literal_type='float'), op='/',
                               right=BinaryOp(left=Literal(value=1.0, literal_type='float'), op='+',
                                              right=BinaryOp(left=call.args[0], op='/', right=call.args[1])))))
        func = call.func
        pine_name = func if isinstance(func, str) else None
        if isinstance(func, str) and call.kwargs and func.count('.') == 1 \
                and func.partition('.')[0] in self._library_aliases():
            call = FunctionCall(func=func, args=call.args,
                                kwargs={_library_parameter_spelling(name): value for name, value in call.kwargs.items()})
        if isinstance(func, str):
            # Apply renamed function mapping (e.g., pass -> pass_)
            if func in self._renamed_functions:
                func = self._renamed_functions[func]
                call = FunctionCall(func=func, args=call.args, kwargs=call.kwargs)

            # Convert excess positional args to keyword args for input.* calls.
            # Pine Script allows input.int(defval, title, minval, maxval, ...)
            # but PyneCore requires minval=, maxval=, ... as keyword-only.
            if is_input_function(func) and len(call.args) > 2:
                new_args, positional_keywords = bind_input_positional_args(func, call.args)
                call = FunctionCall(func=func, args=new_args, kwargs={**call.kwargs, **positional_keywords})

            # Handle method calls disguised as dotted function calls
            if '.' in func and '<' not in func and '.' not in func.split('.')[0]:
                # Try var.method() → module.method(var, ...)
                result = self._transform_dotted_string_method_call(call, func)
                if result:
                    return result

                # Try chained member access: obj.lbl.set_x → label.set_x(obj.lbl, ...)
                if '.' in func:
                    result = self._transform_chained_dotted_method_call(call, func)
                    if result:
                        return result

            # `array.max(a, nth)` / `array.min(a, nth)` stay as written (jordanfray library)
            # Handle generic syntax (map.new<K,V>, array.new<T>, etc.)
            if '<' in func and self._is_generic_array_of_objects(func, call):
                return self._generic_array_new(call, func)
            if func.startswith('matrix.new<'):
                return self._generic_matrix_new(call, func)
            if '<' in func:
                func = self._transform_generic_function_name(func)
            elif func.endswith('.copy') and self._names_a_user_type(func[:-len('.copy')]) and len(call.args) == 1:
                # `T.copy(x)` is `udt_copy(x)` (Trendoscope/Zigzag)
                self.import_resolver.uses_udt_copy = True
                return FunctionCall(func='udt_copy', args=[self._transform_expression(call.args[0])], kwargs={})
            elif '.new' in func:
                base_name = func.replace('.new', '')
                # a UDT constructor, `T.new(...)` -> `T(...)`; a library's type keeps `lib.T.new(...)`
                if base_name and '.' not in base_name and (
                        base_name[0].isupper() or base_name in self.symbol_table.udt_field_types):
                    func = base_name
            elif func in _NA_CAST_TYPES and len(call.args) == 1 and not call.kwargs \
                    and isinstance(call.args[0], NaLiteral):
                # `float(na)`, `color(na)`, `box(na)`: a typed na, NA(T)
                return FunctionCall(func='NA', args=[Identifier(name=_NA_CAST_TYPES[func])], kwargs={})
            elif func == 'int':
                func = 'cast_int'
                self.import_resolver.uses_cast_int = True
            elif func == 'float':
                func = 'cast_float'
                self.import_resolver.uses_cast_float = True
            elif func == 'bool' and self.bool_is_never_na and len(call.args) + len(call.kwargs) == 1 \
                    and not isinstance([*call.args, *call.kwargs.values()][0], NaLiteral):
                # Pine v6: na, nan and |x| <= 1e-10 are false; Python's bool(nan) is true
                func = 'cast_bool'
                self.import_resolver.uses_cast_bool = True
            elif func == 'bool' and not self.bool_is_never_na and len(call.args) + len(call.kwargs) == 1 \
                    and not isinstance([*call.args, *call.kwargs.values()][0], NaLiteral):
                # Pine v5 and earlier: any nonzero number is true, 0 and na false
                return self._legacy_bool_cast([*call.args, *call.kwargs.values()][0])
            elif func == 'string' and len(call.args) == 1 and not call.kwargs \
                    and not isinstance(call.args[0], NaLiteral):
                func = 'cast_string'  # `string(x)` is `cast_string(x)`
                self.import_resolver.uses_cast_string = True
            elif func in ('color', 'bool', 'string'):
                # Type cast: color(na) → na, color(x) → x
                # When arg is na, return plain na (Pine na is untyped at runtime)
                if len(call.args) == 1 and isinstance(call.args[0], NaLiteral):
                    self.import_resolver.add_module('na')
                    return NaLiteral()
                # For non-na args, pass through as identity (type casts are no-ops in Python)
                if len(call.args) == 1:
                    return self._transform_expression(call.args[0])
                func = func  # keep as-is for module usage
            elif func in DRAWING_CAST_FUNCTIONS and (len(call.args), set(call.kwargs)) in ((1, set()), (0, {'x'})):
                return self._drawing_type_cast(func, call.args[0] if call.args else call.kwargs['x'])
            else:
                func = get_function_name(func)
                if '.' in func:
                    parts = func.split('.')
                    parts[0] = get_module_name(parts[0])
                    func = '.'.join(parts)

        elif isinstance(func, MemberAccess):
            result = self._transform_member_access_method_call(call, func)
            if result:
                return result

            # Handle generics in member access
            if '<' in func.member:
                func.member = self._transform_generic_function_name(f"{func.object if isinstance(func.object, str) else 'obj'}.{func.member}")
                if func.member in ('dict', 'list'):
                    func = func.member
            else:
                func = self._transform_member_access(func)

        # Transform arguments; a condition argument holding a number is cast_bool(...)
        condition_parameter = _CONDITION_PARAMETERS.get(func) if isinstance(func, str) else None
        enclosing_conditions, self._call_ternary_conditions = getattr(self, '_call_ternary_conditions', None), set()
        try:
            return self._call_with_arguments(call, pine_call, pine_name, func, condition_parameter)
        finally:
            self._call_ternary_conditions = enclosing_conditions

    def _call_with_arguments(self, call: FunctionCall, pine_call, pine_name, func, condition_parameter) -> Expression:
        transformed_args = [self._condition(arg, argument=True) if condition_parameter and index == 0
                            else self._transform_expression(arg)
                            for index, arg in enumerate(call.args)]
        if func == 'iff' and len(call.args) == 3 and not call.kwargs:
            # Pine v4 `iff(c, a, b)` becomes `a if c else b`: a number name `c` is cast, an expression written there
            # is not (`iff(close * 1.2, ...)`), but a stateful call, which is computed first into a name of its
            # own (`lowest_1 = ta.lowest(low, 58)`, `1 if cast_bool(lowest_1) else 0`)
            condition = call.args[0]
            typing = getattr(self, '_condition_typing', None)
            if isinstance(condition, FunctionCall) and isinstance(condition.func, str) \
                    and condition.func.startswith(('ta.', 'request.')) and typing is not None \
                    and typing._kind(condition, False, 32) == 'number':
                self.import_resolver.uses_cast_bool = True
                transformed_args[0] = FunctionCall(func='cast_bool', args=[self._transform_expression(condition)],
                                                   kwargs={})
            else:
                transformed_args[0] = self._condition(condition, argument=True)
        transformed_kwargs = {k: self._condition(v, argument=True) if k == condition_parameter
                              else self._transform_expression(v)
                              for k, v in call.kwargs.items()}
        if func in ('ta.valuewhen', 'alertcondition'):
            # a number-or-na ternary given as the condition of `ta.valuewhen` or `alertcondition`
            # (`ta.valuewhen(d == 1 ? x : na, close, 0)`) is passed as it is, not cast_bool(...): a nonzero number
            # is true and na false either way (`ta.barssince` casts it)
            pine_condition = call.args[0] if call.args else call.kwargs.get(condition_parameter)
            if isinstance(pine_condition, TernaryOp) and (isinstance(pine_condition.true_expr, NaLiteral)
                                                          or isinstance(pine_condition.false_expr, NaLiteral)):
                for container, key in ((transformed_args, 0), (transformed_kwargs, condition_parameter)):
                    if (key in container if isinstance(container, dict) else len(container) > key):
                        cast = container[key]
                        if isinstance(cast, FunctionCall) and cast.func == 'cast_bool' and len(cast.args) == 1 \
                                and isinstance(cast.args[0], TernaryOp):
                            container[key] = cast.args[0]
        transformed_args, transformed_kwargs = self._overload_arguments.coerce(
            pine_call, pine_name, call.args, transformed_args, call.kwargs, transformed_kwargs)
        if func == 'strategy.exit' and self.pine_version <= 5:
            # Pine v4/v5 `strategy.exit`: a price parameter given beside its tick twin is used instead of it (`stop`
            # instead of `loss`) when it is not na; the runtime takes the level nearer the entry (the v6 rule), so the
            # tick value is na where the price is set
            for ticks, price in (('profit', 'limit'), ('loss', 'stop'), ('trail_points', 'trail_price')):
                if ticks in transformed_kwargs and price in transformed_kwargs:
                    unless_price = TernaryOp(
                        condition=FunctionCall(func='na', args=[dataclasses.replace(transformed_kwargs[price])],
                                               kwargs={}),
                        true_expr=Literal(value=0.0, literal_type='float'), false_expr=NaLiteral())
                    transformed_kwargs[ticks] = BinaryOp(left=transformed_kwargs[ticks], op='+', right=unless_price)
        return FunctionCall(func=func, args=transformed_args, kwargs=transformed_kwargs)

    def _transform_method_call(self, call: MethodCall) -> Union[MethodCall, FunctionCall, TernaryOp]:
        """Transform method call (Rule 48)."""
        if '.' in call.method:  # zs.get(0).bx.delete() parses as method 'bx.delete'
            *field_names, method_name = call.method.split('.')
            receiver = call.object
            for field_name in field_names:
                receiver = MemberAccess(object=receiver, member=field_name)
            return self._transform_method_call(dataclasses.replace(call, object=receiver, method=method_name))

        def transformed_receiver():
            return self._transform_expression(call.object)

        # A user-defined @method, also on an expression: (expr).even_decimal() -> even_decimal(expr).
        # First, so a user method named like the special cases below still runs.
        if call.method in self.method_functions:
            routed = self._route_method_call(call, call.method, call.object, transformed_receiver)
            if routed is not None:
                return routed

        # Special case: UDT.new() should become UDT() (remove .new())
        if call.method == 'new' and isinstance(call.object, Identifier):
            # Check if this is a UDT type (starts with uppercase)
            if call.object.name[0].isupper():
                # Convert to function call: Statistics.new() -> Statistics()
                args = [self._transform_expression(arg) for arg in call.args]
                kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
                return FunctionCall(func=call.object.name, args=args, kwargs=kwargs)

        # Special case: .copy() on UDT object fields should become udt_copy()
        # Example: this.lbl.copy() -> udt_copy(this.lbl); and on a call's result (`a.get(j).copy()`)
        # `x.copy()` is `udt_copy(x)` whatever `x` is, an array included (never array.copy for the method
        # form: mlmi `distances.copy()`, forex_factory_utility `N.copy()`)
        if call.method == 'copy' and not call.args and 'copy' not in self._user_methods():
            obj_expr = self._transform_expression(call.object)
            args = [obj_expr] + [self._transform_expression(arg) for arg in call.args]
            kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
            self.import_resolver.uses_udt_copy = True
            return FunctionCall(func='udt_copy', args=args, kwargs=kwargs)

        # A builtin method: id.push(value) -> array.push(id, value)
        routed = self._route_method_call(call, call.method, call.object, transformed_receiver)
        if routed is not None:
            return routed

        obj = self._transform_expression(call.object)
        args = [self._transform_expression(arg) for arg in call.args]
        kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}

        return MethodCall(object=obj, method=call.method, args=args, kwargs=kwargs)

    def _transform_member_access(self, expr: MemberAccess) -> MemberAccess:
        """Transform member access (Rules 23-25)."""
        # Handle module renames
        if isinstance(expr.object, str):
            obj = get_module_name(expr.object)
        else:
            obj = self._transform_expression(expr.object)

        # Handle plot style remapping (Rule 25)
        if isinstance(expr.object, str):
            full_name = f"{expr.object}.{expr.member}"
            if needs_plot_style_remap(full_name):
                remapped = get_plot_style_remap(full_name)
                parts = remapped.split('.')
                return MemberAccess(object=parts[0], member=parts[1])

        return MemberAccess(object=obj, member=expr.member)

    def _transform_index_access(self, expr: IndexAccess) -> Expression:
        """Transform index access (Rule 52: warn about negative indices).

        A group or a call as the object of a history index was already bound
        to a named local by grouped_history_lowering.
        """
        obj = self._transform_expression(expr.object)
        index = self._transform_expression(expr.index)
        if not isinstance(expr.object, (Identifier, MemberAccess)) and not (
                isinstance(expr.object, str)):
            # the history of an expression or a call: pynecore's inline_series(expr, n)
            return FunctionCall(func='inline_series', args=[obj, index], kwargs={})
        return IndexAccess(object=obj, index=index)

    def _transform_identifier(self, expr: Identifier) -> Identifier:
        """Transform identifier (Rule 23 for module prefixes)."""
        # Handle true/false conversion (Rule 17)
        if expr.name == 'true':
            return Literal(value=True, literal_type='bool')
        elif expr.name == 'false':
            return Literal(value=False, literal_type='bool')

        return Identifier(name=expr.name)

    def _transform_literal(self, expr: Literal) -> Literal:
        """Transform literal (Rule 17: true/false to True/False)."""
        if expr.literal_type == 'bool':
            # Ensure Python boolean values
            return Literal(value=bool(expr.value), literal_type='bool')
        return expr

    def _line_na_type(self, declaration, converted_type: Optional[str]) -> Optional[str]:
        """The type of an `= na` initialiser: the line's first declaration's, where one line declares several."""
        leading = getattr(self, '_line_na_types', {}).get(getattr(declaration, 'line', 0))
        return convert_pine_generic_to_python(leading) if leading and converted_type else converted_type

    def _na_initialiser(self, converted_type: Optional[str]) -> Expression:
        """The value of an `= na` initialiser: `na(T)` for a known type, else bare `na`.

        Pine's na is typeless until a later assignment gives it one; PyneCore
        has the same value (bare `na` -> `lib._na_none`), which the transpiler
        already emits for `color(na)` and for-loop expression targets. Emitting
        `na(Any)` would reference a name nothing imports.
        """
        if not converted_type or converted_type == 'Any':
            return NaLiteral()
        return FunctionCall(func='na', args=[Identifier(name=converted_type)], kwargs={})

    def _drawing_type_cast(self, cast_name: str, argument: Expression) -> Expression:
        """`box(na)` -> `NA(Box)`, an na of the declared type; `box(x)` -> `x`.
        Emitted as is, the cast would call the namespace module."""
        if not isinstance(argument, NaLiteral):
            return self._transform_expression(argument)
        type_name = get_type_name(cast_name)
        self.import_resolver.add_drawing_type(type_name)
        return FunctionCall(func='NA', args=[Identifier(name=type_name)], kwargs={})

    def _transform_na_literal(self, expr: NaLiteral) -> Identifier:
        """Transform na literal to na identifier (Rule 19)."""
        # In PyneCore, 'na' is used directly, not NA()
        return Identifier(name='na')

    def _extract_last_expression_from_block(self, block):
        """Extract the last expression from a block of statements.
        In Pine Script, the last evaluated expression in a block is the return value.
        """
        if isinstance(block, Expression):
            return self._transform_expression(block)
        last = block[-1] if isinstance(block, list) and block else block
        # If the last item is an expression statement (ExpressionStatement or raw Expression)
        if isinstance(last, Expression):
            return self._transform_expression(last)
        # lower_block_values rewrites every block it can; emitting only the tail
        # here would drop the block's statements or the binding of a
        # sole-binding branch, and a bare loop or an empty branch would be
        # emitted as None
        raise UnsupportedFeatureError(
            f'{type(last).__name__} as the value of an if/switch expression branch', getattr(last, 'line', None))

    def _transform_if_expression(self, expr: IfExpression) -> TernaryOp:
        """Transform if expression to Python inline if/else (Rule 16)."""
        # if cond\n    val1\nelse\n    val2  =>  val1 if cond else val2
        condition = self._transform_expression(expr.condition)

        # Handle single expression vs block
        true_expr = self._extract_last_expression_from_block(expr.true_expr)

        if expr.false_expr:
            false_expr = self._extract_last_expression_from_block(expr.false_expr)
        elif expr.value_is_bool:
            # Pine v6: an untaken if of bool type is false, not na
            false_expr = Literal(value=False, literal_type='bool')
        else:
            false_expr = self._untaken_na()

        return TernaryOp(condition=condition, true_expr=true_expr, false_expr=false_expr)

    def _untaken_na(self) -> NaLiteral:
        """The value of an if/switch expression whose arms do not run: Pine's na.
        Python's None is not na to pynecore (`na(None)` is falsy)."""
        self.import_resolver.uses_na = True
        return NaLiteral()

    def _transform_switch_expression(self, expr: SwitchExpression) -> Expression:
        """Transform switch expression to nested ternary (if-elif-else chain)."""
        # Transform the switch expression to a chain of ternary operators
        # switch expr
        #     val1 => result1
        #     val2 => result2
        #     => default
        # Becomes: result1 if (expr == val1) else (result2 if (expr == val2) else default)

        # Or for switch without expr (condition-based):
        # switch
        #     cond1 => result1
        #     cond2 => result2
        #     => default
        # Becomes: result1 if cond1 else (result2 if cond2 else default)

        if not expr.cases:
            # No cases, just return default
            if expr.default:
                return self._extract_last_expression_from_block(expr.default)
            return self._untaken_na()

        # Conditions in case order: the first one binds the subject
        conditions = self._switch_case_conditions(expr.expr, expr.cases)

        # Build nested ternary from right to left
        result = None
        if expr.default:
            result = self._extract_last_expression_from_block(expr.default)
        elif expr.value_is_bool:
            # Pine v6: an untaken switch of bool type is false, not na
            result = Literal(value=False, literal_type='bool')
        else:
            result = self._untaken_na()

        # Process cases in reverse order to build nested ternary
        for condition, (_, case_value) in reversed(list(zip(conditions, expr.cases))):
            # Transform case value
            value_expr = self._extract_last_expression_from_block(case_value)

            # Build ternary: value if condition else result
            result = TernaryOp(
                condition=condition,
                true_expr=value_expr,
                false_expr=result
            )

        return result
