"""
Type inference for Pine Script variables.

Determines whether variables should be Series[T] or Persistent[T] in PyneCore.
"""
from typing import AbstractSet, Dict, List, Optional, Set
from .ast_nodes import *
from .symbol_table import SymbolTable, Symbol, VariableKind
from .bool_builtins import BOOL_BUILTINS
from .pine_builtins import get_type_name

TYPE_WRAPPERS = ('Persistent', 'Series', 'PersistentSeries')
# The color functions that return a colour; color.r/g/b/t return numbers
COLOUR_CONSTRUCTORS = frozenset({'color.new', 'color.rgb', 'color.from_gradient'})


def receiver_path(receiver) -> Optional[List[str]]:
    """Names along a receiver: `t.inner.bx` -> ['t', 'inner', 'bx'].

    A history read keeps its base's type (`b[1]` is still a box), so it
    reduces to the path of its object. Any other expression has no path.
    """
    if isinstance(receiver, str):
        return receiver.split('.')
    if isinstance(receiver, Identifier):
        return receiver.name.split('.')
    if isinstance(receiver, IndexAccess):
        return receiver_path(receiver.object)
    if isinstance(receiver, MemberAccess):
        object_path = receiver_path(receiver.object)
        return object_path + [receiver.member] if object_path else None
    return None


class TypeInference:
    """Analyzes AST to infer variable types (Series vs Persistent vs simple)."""

    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table
        self.ta_functions = self._init_ta_functions()
        # (local types, shadowed names) of the scope being transformed; the
        # transformer replaces this so a dotted value is typed where it appears
        self.current_scope = lambda: ({}, frozenset())

    def _init_ta_functions(self) -> Set[str]:
        """Initialize complete set of ta.* functions/variables (Pine v5/v6)."""
        return {
            # Moving averages
            'ta.sma', 'ta.ema', 'ta.wma', 'ta.vwma', 'ta.swma', 'ta.alma',
            'ta.rma', 'ta.hma', 'ta.linreg',
            # Oscillators & indicators
            'ta.rsi', 'ta.macd', 'ta.stoch', 'ta.cci', 'ta.cmo', 'ta.mfi',
            'ta.mom', 'ta.roc', 'ta.tsi', 'ta.wpr',
            # Volatility
            'ta.atr', 'ta.tr', 'ta.bb', 'ta.bbw', 'ta.kc', 'ta.kcw',
            # Trend
            'ta.dmi', 'ta.supertrend', 'ta.sar', 'ta.cog',
            # Volume-based (functions and built-in variables)
            'ta.obv', 'ta.nvi', 'ta.pvi', 'ta.pvt', 'ta.vwap',
            'ta.accdist', 'ta.iii', 'ta.wad', 'ta.wvad',
            # Statistical
            'ta.dev', 'ta.stdev', 'ta.variance', 'ta.correlation',
            'ta.median', 'ta.mode', 'ta.range', 'ta.cum',
            'ta.percentile_linear_interpolation', 'ta.percentile_nearest_rank',
            'ta.percentrank',
            # Lookback
            'ta.highest', 'ta.lowest', 'ta.highestbars', 'ta.lowestbars',
            'ta.valuewhen', 'ta.barssince', 'ta.change',
            # Pivot
            'ta.pivothigh', 'ta.pivotlow',
            # Boolean series
            'ta.rising', 'ta.falling', 'ta.cross', 'ta.crossover', 'ta.crossunder',
        }

    def infer_types(self, script: Script) -> None:
        """Infer types for all variables in the script."""
        # First pass: mark variables with var/varip as Persistent
        self._mark_persistent_variables(script)

        # Second pass: mark variables from ta.* functions as Series
        self._mark_ta_series_variables(script)

        # Third pass: mark indexed variables as Series
        self._mark_indexed_variables(script)

        # Fourth pass: mark global non-var variables as Series (PRIMARY RULE)
        self._mark_global_series_variables(script)

        # Fifth pass: propagate Series type from assignments
        self._propagate_series_types(script)

    def _mark_persistent_variables(self, script: Script) -> None:
        """Mark all var/varip variables as Persistent."""
        for decl in script.declarations:
            if isinstance(decl, (VarDecl, VaripDecl)):
                symbol = self.symbol_table.lookup(decl.name)
                if symbol:
                    # These are already marked as VAR/VARIP in symbol table
                    symbol.is_series = False  # Not Series, will be Persistent

    def _mark_ta_series_variables(self, script: Script) -> None:
        """Mark variables assigned from ta.* functions as Series."""
        # Check global assignments
        for stmt in script.body:
            if isinstance(stmt, Assignment):
                if self._is_ta_function_call(stmt.value):
                    if isinstance(stmt.target, str):
                        self.symbol_table.mark_from_ta_function(stmt.target)

    def _mark_indexed_variables(self, script: Script) -> None:
        """Mark variables that are indexed (e.g., close[1]) as Series."""
        self.indexed_var_names: set[str] = set()
        self._find_indexed_in_node(script)

    def _find_indexed_in_node(self, node: Any) -> None:
        """Recursively find indexed access patterns."""
        if isinstance(node, IndexAccess):
            if isinstance(node.object, Identifier):
                self.indexed_var_names.add(node.object.name)
                self.symbol_table.mark_as_indexed(node.object.name)

        # Recursively check all child nodes. dict must be traversed too:
        # function-call keyword arguments are stored as a dict, so a history
        # access like `x[1]` inside a kwarg (e.g. plot(color=(x >= x[1] ? ...)))
        # would otherwise be invisible and x would not be promoted to Series.
        if isinstance(node, dict):
            for value in node.values():
                self._find_indexed_in_node(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                self._find_indexed_in_node(item)
        elif hasattr(node, '__dict__'):
            for attr_value in node.__dict__.values():
                if isinstance(attr_value, (ASTNode, list, tuple, dict)):
                    self._find_indexed_in_node(attr_value)

    def _mark_global_series_variables(self, script: Script) -> None:
        """Mark all global non-var, non-input variables as Series (PRIMARY RULE)."""
        for symbol in self.symbol_table.get_all_globals():
            # Skip var/varip (they're Persistent)
            if symbol.kind in (VariableKind.VAR, VariableKind.VARIP):
                continue

            # Skip inputs (they become function parameters)
            if symbol.kind == VariableKind.INPUT:
                continue

            # Skip functions
            if symbol.kind == VariableKind.FUNCTION:
                continue

            # Everything else is Series by default
            if symbol.is_global:
                symbol.is_series = True

    def _propagate_series_types(self, script: Script) -> None:
        """Propagate Series type from one variable to another in assignments."""
        for stmt in script.body:
            if isinstance(stmt, Assignment):
                if isinstance(stmt.target, str):
                    # Check if value is a Series variable
                    if isinstance(stmt.value, Identifier):
                        source_symbol = self.symbol_table.lookup(stmt.value.name)
                        if source_symbol and source_symbol.is_series:
                            self.symbol_table.mark_as_series(stmt.target)

    def _is_ta_function_call(self, expr: Expression) -> bool:
        """Check if expression is a ta.* function call."""
        if isinstance(expr, FunctionCall):
            if isinstance(expr.func, str):
                return expr.func in self.ta_functions
            elif isinstance(expr.func, MemberAccess):
                func_name = f"{expr.func.object}.{expr.func.member}"
                return func_name in self.ta_functions
        return False

    def infer_type_hint(self, value: Expression) -> str:
        """Infer the Python type hint for an expression."""
        if isinstance(value, Literal):
            type_map = {
                'int': 'int',
                'float': 'float',
                'string': 'str',
                'bool': 'bool',
                'color': 'Color',
            }
            return type_map.get(value.literal_type, 'Any')

        if isinstance(value, NaLiteral):
            return 'Any'  # NA can be any type

        if isinstance(value, UnaryOp):
            if value.op == 'not':
                return 'bool'
            # Unary +/- preserves operand type
            return self.infer_type_hint(value.operand)

        if isinstance(value, BinaryOp):
            # Boolean operations always return bool
            if value.op in ('and', 'or', '==', '!=', '>', '<', '>=', '<='):
                return 'bool'

            # Infer from operands
            left_type = self.infer_type_hint(value.left)
            right_type = self.infer_type_hint(value.right)

            # If either is float, result is float
            if left_type == 'float' or right_type == 'float':
                return 'float'

            # If both are int, result is int (except for division)
            if left_type == 'int' and right_type == 'int':
                if value.op == '/':
                    return 'float'
                return 'int'

            return 'Any'

        if isinstance(value, TernaryOp):
            # Type is union of true and false branches
            true_type = self.infer_type_hint(value.true_expr)
            false_type = self.infer_type_hint(value.false_expr)
            if true_type == false_type:
                return true_type
            # If mixed int/float, return float
            if {true_type, false_type} == {'int', 'float'}:
                return 'float'
            return 'Any'

        if isinstance(value, FunctionCall):
            # Check if it's a ta.* function
            func_name = None
            if isinstance(value.func, str):
                func_name = value.func
            elif isinstance(value.func, MemberAccess):
                func_name = f"{value.func.object}.{value.func.member}"

            if func_name in BOOL_BUILTINS:
                return 'bool'  # ta.crossover, str.contains, array.includes, ...
            if func_name and func_name in self.ta_functions:
                return 'float'  # Most ta.* functions return float series

            # input.* functions
            if func_name and func_name.startswith('input.'):
                input_type_map = {
                    'input.int': 'int',
                    'input.float': 'float',
                    'input.bool': 'bool',
                    'input.string': 'str',
                    'input.color': 'Color',
                    'input.source': 'float',  # Sources are float series
                }
                return input_type_map.get(func_name, 'Any')

            # PyneCore collection constructors
            if func_name:
                # matrix.new<float>() -> Matrix[float]
                if func_name.startswith('matrix.new'):
                    # Check for generic type in func_name: matrix.new<float>
                    if '<' in func_name and '>' in func_name:
                        generic = func_name[func_name.index('<')+1:func_name.index('>')]
                        return f'Matrix[{get_type_name(generic.strip())}]'  # `matrix.new<line>()`: Matrix[Line]
                    return 'Matrix[float]'  # Default to float

                # matrix.* methods that return a NEW Matrix (so a variable assigned
                # from them, e.g. `t = m.transpose()`, is typed as a Matrix and its
                # own method calls resolve to the matrix module). Two spellings
                # reach here: the module form `matrix.transpose(m)` (after method
                # resolution), and the pre-resolution method form `m.transpose()`
                # which the parser stores as FunctionCall(func='m.transpose') — for
                # the latter the object must be a known Matrix so the same method
                # names on arrays/maps are not mistyped.
                _MATRIX_RETURNING = ('transpose', 'copy', 'submatrix', 'mult', 'kron',
                                     'pow', 'inv', 'pinv', 'reshape', 'diff')
                if func_name.startswith('matrix.') and func_name.split('.', 1)[1] in _MATRIX_RETURNING:
                    return 'Matrix[float]'
                if '.' in func_name:
                    _obj_name, _, _meth = func_name.rpartition('.')
                    if _meth in _MATRIX_RETURNING and '.' not in _obj_name:
                        _obj_symbol = self.symbol_table.lookup(_obj_name)
                        if _obj_symbol and _obj_symbol.type_hint and 'Matrix' in _obj_symbol.type_hint:
                            return 'Matrix[float]'

                # array.new<T>() -> list[T]; array.from(...) -> list
                if func_name.startswith('array.new<') and func_name.endswith('>'):
                    return f"list[{get_type_name(func_name[len('array.new<'):-1].strip())}]"
                if func_name == 'array.from':
                    # the arguments' common type (`array.from(1.0, 2.0)` is list[float])
                    elements = {self.infer_type_hint(argument) for argument in value.args}
                    if elements == {'int', 'float'}:
                        elements = {'float'}
                    if len(elements) == 1 and 'Any' not in elements and None not in elements:
                        return f'list[{elements.pop()}]'
                    return 'list[Any]'

                # array.new_*() -> list[type]
                if func_name.startswith('array.new_'):
                    array_type_map = {
                        'array.new_int': 'list[int]',
                        'array.new_float': 'list[float]',
                        'array.new_bool': 'list[bool]',
                        'array.new_string': 'list[str]',
                        'array.new_color': 'list[Color]',
                        'array.new_line': 'list[Line]',
                        'array.new_label': 'list[Label]',
                        'array.new_box': 'list[Box]',
                        'array.new_table': 'list[Table]',
                        'array.new_linefill': 'list[LineFill]',
                        'array.new_polyline': 'list[Polyline]',
                    }
                    return array_type_map.get(func_name, 'list[Any]')

                # map.new<K,V>() -> dict[K, V]
                if func_name.startswith('map.new'):
                    if '<' in func_name and '>' in func_name:
                        generic = func_name[func_name.index('<')+1:func_name.index('>')]
                        parts = [p.strip() for p in generic.split(',')]
                        if len(parts) == 2:
                            k = get_type_name(parts[0])
                            v = get_type_name(parts[1])
                            return f'dict[{k}, {v}]'
                    return 'dict[str, float]'  # Default generic

                # map.get -> float (most common map value type in Pine strategies)
                if func_name == 'map.get':
                    return 'float'

                # map.copy(m) / map.keys(m) / map.values(m) -> same type
                if func_name == 'map.copy':
                    return 'dict[str, float]'  # Copy returns a map
                if func_name in ('map.keys', 'array.copy'):
                    return 'list[Any]'
                if func_name == 'map.values':
                    return 'list[Any]'

                # label.new(), line.new(), box.new(), table.new()
                if func_name in ('label.new', 'line.new', 'box.new', 'table.new'):
                    type_name = func_name.split('.')[0].capitalize()
                    return type_name

            # Check if it's a UDT constructor call (e.g., TradeStats())
            if isinstance(value.func, str):
                # Check if the function name is a known UDT type
                symbol = self.symbol_table.lookup(value.func)
                if symbol and symbol.kind == VariableKind.TYPE:
                    return value.func  # Return the UDT type name
                # `info.new()` of a type the script declares, whatever the case of its name
                if value.func.endswith('.new') and '.' not in value.func[:-len('.new')]:
                    owner = self.symbol_table.lookup(value.func[:-len('.new')])
                    if owner and owner.kind == VariableKind.TYPE:
                        return value.func[:-len('.new')]
                # Fallback: if function name starts with uppercase, assume it's a UDT
                elif value.func and value.func[0].isupper():
                    return value.func

        # Handle MethodCall (e.g., TradeStats.new() before transformation)
        if isinstance(value, MethodCall):
            # Check if it's UDT.new() pattern
            if value.method == 'new' and isinstance(value.object, Identifier):
                # Check if the object is a UDT type
                if value.object.name[0].isupper():
                    return value.object.name  # Return just the UDT name without .new suffix

        # Handle FunctionCall for UDT constructors (after transformation: TradeStats())
        if isinstance(value, FunctionCall):
            if isinstance(value.func, str) and value.func and value.func[0].isupper():
                # Already a constructor call - return the type name directly
                return value.func

        if isinstance(value, Identifier):
            if value.name in ('chart.fg_color', 'chart.bg_color'):
                return 'Color'  # the chart's colours
            symbol = self.symbol_table.lookup(value.name)
            if symbol and symbol.type_hint:
                return symbol.type_hint

        # A UDT field (`b = t.bx`) has the field's declared type
        if isinstance(value, (Identifier, MemberAccess)):
            return self.infer_receiver_type(value, *self.current_scope()) or 'Any'

        return 'Any'

    def is_colour_value(self, value: Expression) -> bool:
        """True for a colour: a colour literal or `color.*` constant, a colour
        constructor, or a ternary whose arms are colours (an arm may be na)."""
        if isinstance(value, Literal):
            return value.literal_type == 'color'
        if isinstance(value, Identifier):
            return value.name.startswith('color.')
        if isinstance(value, FunctionCall):
            return isinstance(value.func, str) and value.func in COLOUR_CONSTRUCTORS
        if isinstance(value, TernaryOp):
            arms = [value.true_expr, value.false_expr]
            return (any(self.is_colour_value(arm) for arm in arms)
                    and all(isinstance(arm, NaLiteral) or self.is_colour_value(arm) for arm in arms))
        return False

    def infer_receiver_type(self, receiver, local_types: Dict[str, str],
                            shadowed_names: AbstractSet[str] = frozenset()) -> Optional[str]:
        """Declared type of a method-call receiver, or None when it is not known.

        The base name is typed from `local_types` (the current function's
        parameters and locals), then the symbol table; each further name is a
        UDT field typed from its `type` declaration, to any depth. A field path
        whose base is a function local shadowing a global is left unknown: the
        transformer records the global's type for such a local; so is the base itself
        unless the current scope types it (N1).
        """
        path = receiver_path(receiver)
        if not path:
            return None
        base_name, *field_names = path
        if base_name in shadowed_names and base_name not in local_types:
            return None
        receiver_type = local_types.get(base_name)
        if receiver_type is None:
            symbol = self.symbol_table.lookup(base_name)
            receiver_type = symbol.type_hint if symbol else None
        for field_name in field_names:
            field_types = self.symbol_table.udt_field_types.get(self.declared_type_name(receiver_type))
            receiver_type = field_types.get(field_name) if field_types else None
        return self.declared_type_name(receiver_type)

    def declared_type_name(self, type_hint: Optional[str]) -> Optional[str]:
        """`Persistent[Box]` -> 'Box'; a UDT constructor type `T.new` -> 'T';
        None for an unknown or untyped ('Any', bare 'Series') value."""
        while type_hint and type_hint.endswith(']') and type_hint.split('[', 1)[0] in TYPE_WRAPPERS:
            type_hint = type_hint[type_hint.index('[') + 1:-1]
        if type_hint and type_hint.endswith('.new') and type_hint[:-len('.new')] in self.symbol_table.udt_field_types:
            type_hint = type_hint[:-len('.new')]
        return None if type_hint in (None, '', 'Any', *TYPE_WRAPPERS) else type_hint
