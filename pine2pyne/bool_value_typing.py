"""
Pine v6 bool typing of the values an untaken arm can default.

In Pine v6 an if/switch whose value is of TYPE bool gives false for an arm
that does not run, not na (migration guide to v6). The type comes from the
tail's syntax, a declared or inferred name, a builtin, a UDT field, the
element of an `array<bool>`, or a user function's own tail. This pass decides
it once, on the renamed script and before the block-value lowering rewrites it, and
stamps `value_is_bool` on each if/switch expression, loop and function whose
value is bool. Every later emit site (the block-value pre-pass, the ternary
emitter, the function-tail return) reads the stamp. Unknown is not bool: it
stays na.

One walk records the scope of every value (Pine's scopes, as the overload
resolution builds them: `rename_shadowing_locals` runs first). Each question
("is this value, function or binding bool?") is then a formula over other
questions, built once. The formulas are solved together as a least fixpoint,
so a chain of untyped names of any length is followed without recursion and
without a bound, and a cycle (recursion is not valid Pine) is not bool.
An if/switch/loop in the tail of another value or of a function has the
same type as it: Pine gives one if/switch one type.
"""
import dataclasses
from typing import Dict, List, Optional, Tuple

from .ast_nodes import (
    Assignment, BinaryOp, ExpressionStatement, ForInLoop, ForLoop, FuncDecl, FunctionCall, Identifier,
    IfExpression, IfStatement, IndexAccess, InputDecl, Literal, MemberAccess, MethodCall, NaLiteral, Reassignment,
    Script, SwitchExpression, SwitchStatement, TernaryOp, TupleDestructure, TypeDecl, UnaryOp, VarDecl,
    VaripDecl, WhileLoop,
)
from .ast_walk import CONTAINER_TYPES, LEAF_TYPES, child_field_names, map_children
from .block_value_lowering import BOOL_OPERATORS, LOOPS
from .bool_builtins import BOOL_BUILTINS, BUILTIN_SERIES_TYPES
from .namespace_members import NAMESPACE_MEMBERS
from .overload_resolution import (
    Binding, BlockScope, DeclaredTypeResolver, bind_declaration, nested_blocks, overloads_taking,
)
from .type_inference import receiver_path

VALUE_NODES = (IfExpression, SwitchExpression) + LOOPS
BLOCK_STATEMENTS = (FuncDecl, IfStatement, SwitchStatement) + LOOPS
BINDINGS = (Assignment, VarDecl, VaripDecl, Reassignment)
# A Pine cast and an input of type bool; pynecore does not annotate them
BOOL_CONSTRUCTS = frozenset({'bool', 'input.bool'})
# The Pine namespace of each builtin receiver kind: `a.includes(1)` is `array.includes`
RECEIVER_NAMESPACES = {'array': 'array', 'map': 'map', 'matrix': 'matrix', 'string': 'str'}
# The array methods that return an element: bool on an `array<bool>`
ARRAY_ELEMENT_METHODS = frozenset({'get', 'first', 'last', 'pop'})

# A formula is True, False, ('ref', key), ('any', items) or ('all', items); a key
# is ('value' | 'function' | 'binding', id of the node or binding)
Key = Tuple[str, int]


def stamp_bool_values(script: Script) -> Script:
    """A copy of `script` with `value_is_bool=True` on every bool-valued
    if/switch expression, loop and function; `script` itself when none is."""
    typing = _BoolTyping(script)
    stamped = {node_id for kind, node_id in typing.bool_keys() if kind != 'binding'}
    if not stamped:
        return script
    return _Restamp(stamped, typing.ancestors_of(stamped)).rebuilt(script)


#: the nodes a bool stamp goes on
STAMPED_NODES = (IfExpression, SwitchExpression, FuncDecl) + LOOPS


def stamp_bool_values_as_renamed(script: Script, renamed: Script) -> Script:
    """`script` with the stamps `stamp_bool_values` puts on `renamed`, its copy with shadowing locals
    renamed (scope_renamer.rename_shadowing_locals): the typing needs one binding per name, and the
    lowering of the values runs before the renaming. The renaming keeps every node in its
    place, so a stamp moves to the node at the same place in `script`."""
    stamped = stamp_bool_values(renamed)
    if stamped is renamed:
        return script
    flags = [node.value_is_bool for node in walk_nodes(stamped) if isinstance(node, STAMPED_NODES)]
    nodes = [node for node in walk_nodes(script) if isinstance(node, STAMPED_NODES)]
    if len(flags) != len(nodes):  # not the same tree: typed as it stands
        return stamp_bool_values(script)
    ids = {id(node) for node, flag in zip(nodes, flags) if flag and not node.value_is_bool}
    if not ids:
        return script
    parents: dict = {}

    def record(parent):
        def child_of(child):
            parents[id(child)] = parent
            return child
        return child_of
    for node in walk_nodes(script):
        for name in child_field_names(type(node)):
            map_children(getattr(node, name), record(node))
    ancestors: set = set()
    for node_id in ids:
        parent = parents.get(node_id)
        while parent is not None and id(parent) not in ancestors:
            ancestors.add(id(parent))
            parent = parents.get(id(parent))
    return _Restamp(ids, ancestors).rebuilt(script)


def reads_na_of_bool(script: Script) -> bool:
    """Whether a named bool value is read where it may be na: with history (`b[1]`) or
    through `na(b)`, matched by the name (a bool of that name in another scope counts). Pine v5 marks
    only such a script `na_bool=True`; the history of an expression (`(a > b)[1]`) or of an untyped
    parameter does not count."""
    typing = _BoolTyping(script)
    for name, scope in typing.na_calls:
        binding = scope.lookup(name)
        if binding is not None and _is_bool_collection(binding.declared_type):
            return True  # `na(a)` of an `array<bool>`, not its history: any Pine version
    if _reads_na_of_a_bool_collection_argument(script):
        return True
    if _declared_version(script) >= 6:
        return _reads_na_of_a_bool_name(script)
    if _declared_version(script) != 5:
        return False
    # a read in a global statement whose value reaches nothing shows no na (`x = b[1]`, `y = x and c`, `y` read
    # nowhere: the script stays unmarked)
    script = _without_unread_globals(script)
    typing = _BoolTyping(script)
    if typing.reads_na_of_a_cast or _tuple_under_a_bool_condition(script, typing):
        return True
    # a bool whose value nothing reads but its own assignments (`e := e[1] or c`, `e` read nowhere else) shows no
    # na: its history and na() reads do not count
    read = _names_read_elsewhere(script)
    typing.na_reads = [(name, scope) for name, scope in typing.na_reads if name in read]
    typing.na_calls = [(name, scope) for name, scope in typing.na_calls if name in read]
    # a history or na() read of a name some type declares a bool field of, matched by the name alone (`x[1]` of a
    # float parameter `x` beside `type T` with `bool x`)
    bool_fields = {field[0] for node in walk_nodes(script) if isinstance(node, TypeDecl) for field in node.fields
                   if str(field[1]) == 'bool'}
    if any(name in bool_fields for name, _ in typing.na_reads):
        return True
    if any(name in _collections_declared_na_and_reassigned(script) for name, _ in typing.na_calls):
        return True  # `var array<T> rows = na`, `if not na(rows)`, `rows := array.new<T>(n)`
    # a printed bool shows its na only where it may be na: not one every value of which is a comparison, an
    # `and`/`or`/`not`, a literal or a bool input (no script is marked for printing such a bool)
    never_na = _names_never_na(script)
    # nor a bool parameter (`method f(bool b) => str.tostring(b)`): no script is marked for printing one
    parameters = {parameter.name for node in walk_nodes(script) if isinstance(node, FuncDecl)
                  for parameter in node.params}
    string_reads = [(name, scope) for name, scope in typing.string_reads if name not in never_na and not (
        name in parameters and getattr(scope.lookup(name), 'initialiser', True) is None)]
    # a bool that may be na compared with `==` / `!=` shows its na too: TradingView compares an na bool as no
    # false (`var bool b = na`, `b == false` is false on every bar, where the two-state bool reads it false), so
    # such a read marks the script as a history or na() read does
    maybe_na = _names_given_na(script)
    equality_reads = [(name, scope) for name, scope in typing.equality_reads
                      if name in read and name in maybe_na and name not in parameters]
    formulas = [typing._name(name, scope) for name, scope in typing.na_reads + string_reads + equality_reads]
    # a history or na() read is matched by the name: a bool of that name in any other scope counts too
    # (`x[1]` of a function's float local beside a global bool `x`)
    scopes = list({id(scope): scope for scope in typing.scope_of.values()}.values())
    formulas += [typing._name(name, other) for name, scope in typing.na_reads
                 for other in scopes if other is not scope and name in other.bindings]
    if any(formula is True for formula in formulas):
        return True
    references = [reference for formula in formulas for reference in _refs(formula)]
    if not references:
        return False
    true_keys = typing.bool_keys(tuple(references))
    return any(_holds(formula, true_keys) for formula in formulas)


# calls that do something besides giving a value: a statement holding one is kept whether or not its name is read
_EFFECT_NAMESPACES = frozenset({
    'plot', 'plotshape', 'plotchar', 'plotarrow', 'plotbar', 'plotcandle', 'hline', 'fill', 'bgcolor', 'barcolor',
    'label', 'line', 'box', 'table', 'polyline', 'linefill', 'chart', 'strategy', 'alert', 'alertcondition',
    'runtime', 'log', 'array', 'map', 'matrix', 'library', 'indicator', 'study',
})


def _without_unread_globals(script: Script) -> Script:
    """`script` without its global assignments whose values reach nothing: a name no statement reads, or only
    statements that are dropped themselves (`x = b[1]`, `y = x and c`, `y` read nowhere). A statement whose value
    draws, trades, alerts, changes a collection or calls a method stays, and so does every block and function."""
    statements = list(script.declarations) + list(script.body)
    functions = {node.name: node for node in statements if isinstance(node, FuncDecl)}

    def targets(statement) -> Optional[set]:
        if isinstance(statement, Assignment):
            if isinstance(statement.target, str):
                return {statement.target}
            return set(statement.target.names) if isinstance(statement.target, TupleDestructure) else None
        if isinstance(statement, Reassignment):
            return {statement.target} if isinstance(statement.target, str) else None
        if isinstance(statement, (VarDecl, VaripDecl)):
            return {statement.name}
        return None

    effects: Dict[str, bool] = {}

    def has_effect(node, seen: frozenset = frozenset()) -> bool:
        for child in walk_nodes(node):
            if isinstance(child, MethodCall):
                return True
            if isinstance(child, FunctionCall) and isinstance(child.func, str):
                if child.func.split('.')[0] in _EFFECT_NAMESPACES:
                    return True
                if child.func in functions and child.func not in seen:
                    if child.func not in effects:
                        effects[child.func] = has_effect(functions[child.func].body, seen | {child.func})
                    if effects[child.func]:
                        return True
        return False

    def reads(node) -> set:
        """The names `node` reads; a name assigned inside a block counts as read there (`if c` / `e := true`)."""
        names = set()
        for child in walk_nodes(node):
            if isinstance(child, (Assignment, Reassignment)) and isinstance(child.target, str):
                names.add(child.target)
            elif isinstance(child, (VarDecl, VaripDecl)):
                names.add(child.name)
            if isinstance(child, Identifier):
                names.add(child.name.split('.')[0])
            elif isinstance(child, FunctionCall) and isinstance(child.func, str):
                names.add(child.func.split('.')[0])
                names.add(child.func)
        return names

    candidates = [(statement, names) for statement in statements
                  if (names := targets(statement)) and not has_effect(statement.value)]
    candidate_ids = {id(statement) for statement, _ in candidates}
    live = set()
    for statement in statements:
        if id(statement) not in candidate_ids:
            live |= reads(statement)
    kept = set()
    changed = True
    while changed:
        changed = False
        for statement, names in candidates:
            if id(statement) not in kept and names & live:
                kept.add(id(statement))
                live |= reads(statement.value)
                changed = True
    dropped = candidate_ids - kept
    if not dropped:
        return script
    return dataclasses.replace(
        script, declarations=[node for node in script.declarations if id(node) not in dropped],
        body=[node for node in script.body if id(node) not in dropped])


#: builtins whose value has the type of one argument (position, keyword), each with a `series bool` overload
_SOURCE_TYPED_ARGUMENT = {'ta.change': (0, 'source'), 'change': (0, 'source'),
                          'ta.valuewhen': (1, 'source'), 'valuewhen': (1, 'source')}
_NEVER_NA_CALLS = frozenset({'na', 'ta.crossover', 'ta.crossunder', 'ta.cross', 'ta.rising', 'ta.falling',
                             'crossover', 'crossunder', 'cross', 'rising', 'falling'})


def _names_read_elsewhere(script: Script) -> set:
    """The names whose value something reads: a read outside the values assigned to the name itself
    (`e := e[1] or c` reads `e` for `e` only), an assignment inside an if, loop or switch block, or one that is a
    function's last statement (its result)."""
    read = set()

    def assigned(node):
        if isinstance(node, (Assignment, Reassignment)):
            return node.target if isinstance(node.target, str) else None
        return node.name if isinstance(node, (VarDecl, VaripDecl)) else None

    def visit(node, own, in_block: bool) -> None:
        if isinstance(node, Identifier) and node.name != own:
            read.add(node.name)
        if isinstance(node, (Assignment, Reassignment, VarDecl, VaripDecl)):
            target = assigned(node)
            if in_block and target:
                read.add(target)
            visit(node.value, target, in_block)
            return
        if isinstance(node, FuncDecl):
            body = node.body if isinstance(node.body, list) else [node.body]
            if body and assigned(body[-1]):
                read.add(assigned(body[-1]))
        block = isinstance(node, (IfStatement, ForLoop, ForInLoop, WhileLoop, SwitchStatement))
        for child in _children(node):
            visit(child, own, in_block or block)

    visit(script, None, False)
    return read


def _children(node):
    if isinstance(node, (list, tuple)):
        return list(node)
    if isinstance(node, dict):
        return list(node.values())
    if not dataclasses.is_dataclass(node):
        return []
    return [getattr(node, field.name) for field in dataclasses.fields(node)
            if not isinstance(getattr(node, field.name), (str, int, float, bool, type(None)))]


def _collections_declared_na_and_reassigned(script: Script) -> set:
    """The names declared as an array, a map or a matrix with the value na and reassigned a new one later
    (`rows := array.new<T>(n)`, not `array.from(...)`): Pine v5's `na(x)` of them marks the script
    `na_bool=True`."""
    def collection(hint) -> bool:
        hint = str(hint or '').replace(' ', '')
        return hint.startswith(('array<', 'map<', 'matrix<')) or hint.endswith('[]')

    declared = set()
    for node in walk_nodes(script):
        if isinstance(node, (VarDecl, VaripDecl, Assignment)) and isinstance(node.value, NaLiteral):
            name = node.name if isinstance(node, (VarDecl, VaripDecl)) else node.target
            if isinstance(name, str) and collection(getattr(node, 'type_hint', None)):
                declared.add(name)
        elif isinstance(node, TypeDecl):
            # a field so declared (`FVGInfo[] FVGInfoList = na`), matched by its name
            declared.update(field[0] for field in node.fields if collection(field[1])
                            and (len(field) < 3 or field[2] is None or isinstance(field[2], NaLiteral)))
    # reassigned a new one, the variable or a field of that name (`info.FVGInfoList := array.new<FVGInfo>(0)`)
    return {node.target.rpartition('.')[2] for node in walk_nodes(script) if isinstance(node, Reassignment)
            and isinstance(node.target, str) and node.target.rpartition('.')[2] in declared
            and isinstance(node.value, FunctionCall) and isinstance(node.value.func, str)
            and node.value.func.split('<')[0] in ('array.new', 'map.new', 'matrix.new')}


def _names_given_na(script: Script) -> set:
    """The names some value of which is an explicit na: `na` itself (`var bool b = na`, `b := na`) or a ternary
    with an `na` arm (`c ? x : na`). A value computed by comparisons or calls is not na in Pine even where an
    operand is (a comparison with na is false), so only these show an na bool."""
    from .ast_walk import walk

    def gives_na(value) -> bool:
        if isinstance(value, NaLiteral) or (isinstance(value, Identifier) and value.name == 'na'):
            return True
        if isinstance(value, TernaryOp):
            return gives_na(value.true_expr) or gives_na(value.false_expr)
        return False

    found: set = set()
    for node in walk(script):
        if isinstance(node, (VarDecl, VaripDecl)) and node.value is not None and gives_na(node.value):
            found.add(node.name)
        elif isinstance(node, (Assignment, Reassignment)) and isinstance(node.target, str) and gives_na(node.value):
            found.add(node.target)
    return found


def _names_never_na(script: Script) -> set:
    """The names every value of which is a bool that is never na: a comparison (Pine compares na as false), an
    `and`/`or`/`not`, `true`/`false`, a bool input or a cross test."""
    from .ast_walk import walk

    def never_na(value) -> bool:
        if isinstance(value, Literal):
            return value.literal_type == 'bool'
        if isinstance(value, Identifier):
            return value.name in ('true', 'false')
        if isinstance(value, BinaryOp):
            return value.op in ('<', '>', '<=', '>=', '==', '!=', 'and', 'or')
        if isinstance(value, UnaryOp):
            return value.op == 'not'
        if isinstance(value, FunctionCall):
            return isinstance(value.func, NaLiteral) or (isinstance(value.func, str) and value.func in _NEVER_NA_CALLS)
        return False

    values: dict = {}
    for node in walk(script):
        if isinstance(node, (VarDecl, VaripDecl)):
            values.setdefault(node.name, []).append(never_na(node.value))
        elif isinstance(node, (Assignment, Reassignment)) and isinstance(node.target, str):
            values.setdefault(node.target, []).append(never_na(node.value))
        elif isinstance(node, InputDecl):
            values.setdefault(node.name, []).append(node.func == 'input.bool' or (
                node.func == 'input' and bool(node.args) and never_na(node.args[0])))
        elif isinstance(node, TupleDestructure):
            for name in node.names:
                values.setdefault(name, []).append(False)
    return {name for name, found in values.items() if found and all(found)}


def _reads_na_of_a_bool_name(script: Script) -> bool:
    """Pine v6: `na(x)` of a name some declaration of the script types bool (a variable, a parameter, a field, a
    bool input), matched by the name alone: a string local named like a bool field counts too.
    A field read (`na(t.flag)`) or a history read does not, nor a name ever given na: Pine v6 reads its na as false,
    which the switch would change."""
    from .ast_walk import walk
    from .ast_nodes import Parameter
    nodes = list(walk(script))
    names = set()
    for node in nodes:
        if isinstance(node, (VarDecl, VaripDecl)) and node.type_hint == 'bool':
            names.add(node.name)
        elif isinstance(node, Assignment) and isinstance(node.target, str) and node.type_hint == 'bool':
            names.add(node.target)
        elif isinstance(node, Parameter) and node.type_hint == 'bool':
            names.add(node.name)
        elif isinstance(node, TypeDecl):
            names.update(field[0] for field in node.fields if str(field[1]) == 'bool')
        elif isinstance(node, InputDecl) and node.func == 'input.bool':
            names.add(node.name)
    # a bool ever given na keeps Pine v6's reading (a bool is never na: `na(b)` is false), so it does not count;
    # an int or a float of the same name given na (`int result = na` beside `bool result = false`) does
    bool_names = set(names)
    given_na = {node.name if isinstance(node, (VarDecl, VaripDecl)) else node.target for node in nodes
                if isinstance(node, (VarDecl, VaripDecl, Assignment, Reassignment)) and isinstance(node.value, NaLiteral)
                and isinstance(node.name if isinstance(node, (VarDecl, VaripDecl)) else node.target, str)
                and (getattr(node, 'type_hint', None) == 'bool'
                     or (isinstance(node, Reassignment) and node.target in bool_names)
                     or (not getattr(node, 'type_hint', None) and not isinstance(node, Reassignment)))}
    given_na |= {field[0] for node in nodes if isinstance(node, TypeDecl) for field in node.fields
                 if str(field[1]) == 'bool' and (len(field) < 3 or field[2] is None or isinstance(field[2], NaLiteral))}
    names -= given_na
    # a name reassigned the result of a user function whose value spans several lines (`live := newLeg(d, a)` with
    # `Leg.new(d, a,` and its arguments below): that result cannot be typed, so it may be a bool
    type_names = {node.name for node in nodes if isinstance(node, TypeDecl)}
    untyped_functions = {node.name for node in nodes
                         if isinstance(node, FuncDecl) and _tail_spans_lines(node, type_names)}
    names |= {node.target for node in nodes if isinstance(node, Reassignment) and isinstance(node.target, str)
              and isinstance(node.value, FunctionCall) and isinstance(node.value.func, str)
              and node.value.func in untyped_functions}
    return any(isinstance(node, FunctionCall) and (isinstance(node.func, NaLiteral) or node.func == 'na')
               and node.args and isinstance(node.args[0], Identifier) and '.' not in node.args[0].name
               and node.args[0].name in names for node in nodes)


def _tail_spans_lines(function: FuncDecl, type_names: set) -> bool:
    """Whether the last statement of `function` is a user type's constructor going on over several lines
    (`Leg.new(d, a,` and its arguments below); a drawing's (`line.new(...)`) is typed whatever its layout."""
    from .ast_walk import walk
    body = function.body if isinstance(function.body, list) else [function.body]
    last = body[-1] if body else None
    value = getattr(last, 'expr', last)
    start = getattr(last, 'line', 0) or 0
    constructs = isinstance(value, (FunctionCall, MethodCall)) and (
        (isinstance(value, FunctionCall) and isinstance(value.func, str) and value.func.endswith('.new')
         and value.func[:-len('.new')] in type_names)
        or (isinstance(value, MethodCall) and value.method == 'new' and isinstance(value.object, Identifier)
            and value.object.name in type_names))
    return constructs and bool(start) and max((getattr(node, 'line', 0) or 0) for node in walk(last)) > start


def _typed_by_layout_only(statement) -> bool:
    """An untyped declaration whose value starts below its name, or a ternary that continues on the
    next line: a declaration is typed by its first line, so these have no type."""
    if not isinstance(statement, (Assignment, VarDecl, VaripDecl)) or getattr(statement, 'type_hint', None):
        return False
    if isinstance(statement, Assignment) and not isinstance(statement.target, str):
        return False
    value = statement.value
    first_arm_on_first_line = isinstance(value, TernaryOp) and value.question_line == statement.line \
        and value.colon_line == statement.line  # `x = c ? a > b :`, the else arm below: typed by its first arm
    return statement.line > 0 and (statement.value_line > statement.line or (
        statement.end_line > statement.line and isinstance(value, TernaryOp) and not first_arm_on_first_line))


def _is_bool_collection(declared_type) -> bool:
    """`array<bool>`, `matrix<bool>`, `map<K, bool>` (or `bool[]`)."""
    if not isinstance(declared_type, str):
        return False
    compact = declared_type.replace(' ', '')
    return compact in ('array<bool>', 'matrix<bool>', 'bool[]') or (compact.startswith('map<')
                                                                    and compact.endswith(',bool>'))


def _reads_na_of_a_bool_collection_argument(script: Script) -> bool:
    """`is_empty(arr) => na(arr) or ...` called with an `array<bool>`: the untyped parameter is a bool
    collection where it is called with one."""
    from .ast_walk import walk
    from .ast_nodes import MethodCall, Parameter
    nodes = list(walk(script))
    collections = {node.name for node in nodes if isinstance(node, Parameter) and _is_bool_collection(node.type_hint)}
    collections |= {getattr(node, 'name', None) or node.target for node in nodes
                    if isinstance(node, (VarDecl, VaripDecl, Assignment))
                    and _is_bool_collection(getattr(node, 'type_hint', None))
                    and isinstance(getattr(node, 'name', None) or node.target, str)}
    if not collections:
        return False
    functions = {node.name: node for node in nodes if isinstance(node, FuncDecl)}
    flagged = set()
    for node in nodes:
        if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in functions:
            for position, argument in enumerate(node.args):
                parameters = functions[node.func].params
                if isinstance(argument, Identifier) and argument.name in collections and position < len(parameters) \
                        and not parameters[position].type_hint:
                    flagged.add((node.func, parameters[position].name))
    for function_name, parameter in flagged:
        for node in walk(functions[function_name].body):
            if isinstance(node, FunctionCall) and (isinstance(node.func, NaLiteral) or node.func == 'na') \
                    and node.args and isinstance(node.args[0], Identifier) and node.args[0].name == parameter:
                return True
    return False


def _is_bool_literal(value) -> bool:
    return (isinstance(value, Literal) and value.literal_type == 'bool') or (
        isinstance(value, Identifier) and value.name in ('true', 'false'))


def _tuple_under_a_bool_condition(script: Script, typing: '_BoolTyping') -> bool:
    """Pine v5: a tuple declared (`[a, b] = f()`, `[m, s, h] = ta.macd(...)`) directly in the body of an `if`
    whose condition is made of bool names only (`if s`, `if not s`, `if s or t`) makes the script `na_bool=True`;
    a condition in parentheses (`if(s)`) or with a comparison (`if s and close > open`), the
    else branch or a plain declaration does not."""
    def names(condition) -> Optional[list]:
        if getattr(condition, 'parenthesized', False):
            return None  # `if(s)` does not count
        if isinstance(condition, Identifier) and '.' not in condition.name:
            return [condition.name]
        if isinstance(condition, UnaryOp) and condition.op == 'not':
            return names(condition.operand)
        if isinstance(condition, BinaryOp) and condition.op in ('and', 'or'):
            left, right = names(condition.left), names(condition.right)
            return left + right if left is not None and right is not None else None
        return None

    for node in walk_nodes(script):
        tuples = [statement for statement in node.body if isinstance(statement, Assignment)
                  and isinstance(statement.target, TupleDestructure)] if isinstance(node, IfStatement) else []
        if not tuples:
            continue
        condition_names = names(node.condition)
        # the if's own block, whose lookups reach the scope the condition is read in
        scope = typing.block_scopes.get(id(node.body))
        if not condition_names or scope is None:
            continue
        # an untyped parameter is typed by the name: a global bool of its name makes it one (`f(src, repaint)` beside
        # `repaint = input.bool(...)`), whatever the call passes
        root = scope
        while root.parent is not None:
            root = root.parent
        formulas = [typing._name(name, root if _is_untyped_parameter(scope.lookup(name)) and name in root.bindings
                                 else scope) for name in condition_names]
        keys = typing.bool_keys(tuple(reference for formula in formulas for reference in _refs(formula)))
        if all(formula is True or (formula and _holds(formula, keys)) for formula in formulas):
            return True
    return False


def _is_untyped_parameter(binding) -> bool:
    """A parameter declared without a type: a binding with neither a type nor a value."""
    return binding is not None and binding.declared_type is None and binding.initialiser is None


def _declared_version(script: Script) -> int:
    text = getattr(script.version, 'version', '') if script.version else ''
    return int(text) if str(text).isdigit() else 6


def has_untaken_bool_switch(script: Script) -> bool:
    """Whether a `switch` without a default arm gives a bool: an untaken switch is na, so a Pine v5
    script that has one is `na_bool=True`. A switch used as
    a value or as a statement whose arms end with a bool (`x := true`) counts, and so does one whose arms are
    arrays; an `if` does not."""
    switches = [node for node in walk_nodes(script)
                if isinstance(node, (SwitchStatement, SwitchExpression)) and node.default is None and node.cases]
    if not switches:
        return False
    typing = _BoolTyping(script)
    # a name declared as an array (`var a = array.new<string>(0)`): a switch of arrays counts too (one
    # of user objects, drawings, colours, strings or numbers does not)
    arrays = {node.target if isinstance(node, Assignment) else node.name for node in walk_nodes(script)
              if isinstance(node, (Assignment, VarDecl, VaripDecl)) and isinstance(node.value, FunctionCall)
              and isinstance(node.value.func, str) and node.value.func.startswith(('array.new', 'array.from'))}
    for switch in switches:
        scope = typing.scope_of.get(id(switch))
        for _, arm in switch.cases:
            tail_value = arm[-1].expr if isinstance(arm, list) and arm and isinstance(arm[-1], ExpressionStatement) \
                else arm
            if isinstance(tail_value, Identifier) and tail_value.name in arrays:
                return True
            arm_scope = typing.block_scopes.get(id(arm), scope) if isinstance(arm, list) else scope
            if arm_scope is None:
                continue
            tail = arm[-1] if isinstance(arm, list) and arm else arm
            formula = typing._tail(tail, arm_scope, ('value', id(switch))) if isinstance(arm, list) \
                else typing._expression(arm, arm_scope)
            if formula is True or (formula and _holds(formula, typing.bool_keys(tuple(_refs(formula))))):
                return True
    return False


def walk_nodes(tree):
    from .ast_walk import walk
    return walk(tree)


def _any(items) -> object:
    items = tuple(item for item in items if item is not False)
    if True in items:
        return True
    return (items[0] if len(items) == 1 else ('any', items)) if items else False


def _all(items) -> object:
    items = tuple(item for item in items if item is not True)
    if False in items:
        return False
    return (items[0] if len(items) == 1 else ('all', items)) if items else True


def _refs(formula) -> list:
    if formula is True or formula is False:
        return []
    if formula[0] == 'ref':
        return [formula[1]]
    return [key for item in formula[1] for key in _refs(item)]


def _holds(formula, true_keys: set) -> bool:
    if formula is True or formula is False:
        return formula
    kind, content = formula
    if kind == 'ref':
        return content in true_keys
    combine = any if kind == 'any' else all
    return combine(_holds(item, true_keys) for item in content)


def _statement_arms(statement) -> list:
    """The bodies whose tails give a tail if/switch statement its value."""
    if isinstance(statement, IfStatement):
        return [statement.body] + [body for _, body in statement.elseifs] + [statement.else_body]
    return [body for _, body in statement.cases] + [statement.default]


def _value_arms(value) -> list:
    if isinstance(value, IfExpression):
        return [value.true_expr, value.false_expr]
    if isinstance(value, SwitchExpression):
        return [body for _, body in value.cases] + [value.default]
    return [value.body]


def _statement_headers(statement) -> list:
    """The expressions of a block statement that run in the enclosing scope."""
    if isinstance(statement, FuncDecl):
        return list(statement.params)
    if isinstance(statement, IfStatement):
        return [statement.condition] + [condition for condition, _ in statement.elseifs]
    if isinstance(statement, SwitchStatement):
        return [statement.expr] + [case for case, _ in statement.cases]
    if isinstance(statement, ForLoop):
        return [statement.from_val, statement.to_val, statement.step]
    if isinstance(statement, ForInLoop):
        return [statement.iterable]
    return [statement.condition]


def _declared_container(type_name: Optional[str]):
    """(kind, element is bool) for a declared builtin container or string type."""
    if type_name == 'string':
        return 'string', False
    kind, bracket, element = (type_name or '').partition('<')
    if bracket and kind in ('array', 'map', 'matrix'):
        return kind, kind == 'array' and element.rstrip('>').strip() == 'bool'
    return None


class _BoolTyping:
    """The scopes of one script and the bool answer of each of its values."""

    def __init__(self, script: Script):
        self.functions: Dict[str, List[FuncDecl]] = {}
        self.nodes: Dict[int, object] = {}               # id -> value node, function or binding
        self.scope_of: Dict[int, BlockScope] = {}         # id(value node) -> the scope it is in
        self.block_scopes: Dict[int, BlockScope] = {}     # id(block list) -> its scope
        self.function_scopes: Dict[int, BlockScope] = {}  # id(function) -> its body's scope
        self.contexts: Dict[int, list] = {}               # id(value node) -> what its target says
        self.tuple_elements: Dict[int, tuple] = {}        # id(binding) -> (binding, destructured value, position)
        self.implied_by: Dict[Key, List[Key]] = {}        # a tail value -> the values it is the tail of
        self.parents: Dict[int, list] = {}
        self.na_reads: List[tuple] = []                   # (name, scope): `x[n]` and `na(x)` of a named value
        self.na_calls: List[tuple] = []                   # (name, scope): `na(x)` only
        self.string_reads: List[tuple] = []               # (name, scope): `str.tostring(x)`, which prints na
        self.equality_reads: List[tuple] = []             # (name, scope): `x == y` / `x != y`, where na is no false
        self.reads_na_of_a_cast = False                   # `na(bool(x))`, `str.tostring(bool(x))`, `na(t.flag)`
        udt_field_types = {declaration.name: {name: field_type for name, field_type, *_ in declaration.fields}
                           for declaration in script.declarations if isinstance(declaration, TypeDecl)}
        # the bool fields of the script's types: `na(t.flag)` reads a field that may be na (Pine v5)
        self.bool_fields = {name for fields in udt_field_types.values() for name, field_type in fields.items()
                            if field_type == 'bool'}
        self.resolver = DeclaredTypeResolver(lambda node: None, udt_field_types)
        self._walk(script)

    # ------------------------------------------------------------ the walk
    def _walk(self, script: Script) -> None:
        """Visit every node once, giving each the scope it is in. Every file is
        walked, so the loop makes no call per plain node."""
        pending = [(script, BlockScope(), None)]
        fields_by_type: Dict[type, tuple] = {}
        while pending:
            value, scope, parent = pending.pop()
            value_type = type(value)
            if value_type in CONTAINER_TYPES or value_type is dict:
                items = value.values() if value_type is dict else value
                pending.extend([(item, scope, parent) for item in items if type(item) not in LEAF_TYPES])
                continue
            fields = fields_by_type.get(value_type)
            if fields is None:
                is_node_type = hasattr(value_type, '__dataclass_fields__')
                fields = fields_by_type[value_type] = child_field_names(value_type) if is_node_type else False
            if fields is False:
                continue
            self.parents.setdefault(id(value), []).append(parent)
            if isinstance(value, BLOCK_STATEMENTS + (IfExpression, SwitchExpression)):
                self._enter_block(value, scope, pending)
                continue
            if isinstance(value, BINDINGS + (InputDecl,)):
                self._bind(value, scope)
            elif value_type is IndexAccess or value_type is FunctionCall:
                self._record_na_read(value, scope)
            elif value_type is BinaryOp and value.op in ('==', '!='):
                # an na bool compared is no false (`b == false` is false on TradingView where `b` is na)
                self.equality_reads.extend((operand.name, scope) for operand in (value.left, value.right)
                                           if isinstance(operand, Identifier) and '.' not in operand.name
                                           and operand.name not in ('true', 'false'))
            elif value_type is Identifier and value.read_at_zero:
                # `b[0]` reads the current value, but marks a v5 script `na_bool` as a history read does
                self._record_na_read(IndexAccess(object=value, index=Literal(value=0, literal_type='int')), scope)
            pending.extend([(child, scope, value) for child in [getattr(value, name) for name in fields]
                            if type(child) not in LEAF_TYPES])

    def _enter_block(self, value, scope: BlockScope, pending: list) -> None:
        """A function, if, switch or loop: its value and the scopes of its blocks."""
        if not isinstance(value, (IfStatement, SwitchStatement)):
            self.nodes[id(value)] = value
            self.scope_of[id(value)] = scope
        if isinstance(value, BLOCK_STATEMENTS):
            self._enter_block_statement(value, scope, pending)
        else:
            self._enter_value_expression(value, scope, pending)

    def _enter_block_statement(self, statement, scope: BlockScope, pending: list) -> None:
        blocks = nested_blocks(statement, scope)
        for body, block_scope in blocks:
            self.block_scopes[id(body)] = block_scope
            pending.append((body, block_scope, statement))
        if isinstance(statement, FuncDecl):
            self.functions.setdefault(statement.name, []).append(statement)
            self.function_scopes[id(statement)] = blocks[0][1]
        pending.extend((header, scope, statement) for header in _statement_headers(statement) if header is not None)

    def _enter_value_expression(self, value, scope: BlockScope, pending: list) -> None:
        """An arm that is a block is a Pine local scope; any other child is in `scope`."""
        headers = [value.condition] if isinstance(value, IfExpression) else (
            [value.expr] + [case for case, _ in value.cases])
        pending.extend((header, scope, value) for header in headers if header is not None)
        for arm in _value_arms(value):
            arm_scope = scope
            if isinstance(arm, list):
                arm_scope = self.block_scopes[id(arm)] = BlockScope(scope)
            if arm is not None:
                pending.append((arm, arm_scope, value))

    def _bind(self, statement, scope: BlockScope) -> None:
        if isinstance(statement, Reassignment):
            if isinstance(statement.value, VALUE_NODES):
                self.contexts.setdefault(id(statement.value), []).append((statement.target, scope))
            return
        if isinstance(statement, Assignment) and isinstance(statement.target, TupleDestructure):
            for position, name in enumerate(statement.target.names):
                element = Binding(None, None, scope)
                self.tuple_elements[id(element)] = (element, statement.value, position)
                scope.bind(name, element)
            return
        if isinstance(statement, InputDecl) and (statement.func == 'input.bool' or (
                statement.func == 'input' and _is_bool_literal(statement.args[0] if statement.args
                                                               else statement.kwargs.get('defval')))):
            # `input.bool(...)`, or `input(true)` / `input(defval=true)`: a bool input
            scope.bind(statement.name, Binding('bool', None, scope))
            return
        if _typed_by_layout_only(statement):
            value = statement.value
            if isinstance(value, TernaryOp) and statement.value_line == statement.line < value.question_line:
                # `x = p == 2` with `? a : b` on the next lines: the name is typed by its first line,
                # the condition (a bool)
                bind_declaration(dataclasses.replace(statement, value=value.condition), scope)
                return
            # `up =` with its value on the next lines: the name has no type
            scope.bind(statement.target if isinstance(statement, Assignment) else statement.name,
                       Binding(None, None, scope))
            return
        bind_declaration(statement, scope)
        if getattr(statement, 'type_hint', None) == 'bool' and isinstance(statement.value, VALUE_NODES):
            self.contexts.setdefault(id(statement.value), []).append(True)

    def _record_na_read(self, value, scope: BlockScope) -> None:
        """`x[n]` or `na(x)` of a name: where a v5 bool may be na. `str.tostring(x)` prints an na bool
        (`NaN`) and `na(bool(x))` reads the na of a cast (as TradingView does): they are kept apart,
        the bool typing decides."""
        if isinstance(value, FunctionCall) and value.func in ('str.tostring', 'tostring') and value.args:
            target = value.args[0]
            if isinstance(target, FunctionCall) and target.func == 'bool':
                self.reads_na_of_a_cast = True
            elif isinstance(target, Identifier) and '.' not in target.name:
                self.string_reads.append((target.name, scope))
            return
        if isinstance(value, IndexAccess):
            target = value.object
            field = target.name.rpartition('.')[2] if isinstance(target, Identifier) and '.' in target.name \
                else getattr(target, 'member', None)
            if field in self.bool_fields:
                self.reads_na_of_a_cast = True  # `t.flag[1]`: the history of a bool field
                return
        elif (isinstance(value.func, NaLiteral) or value.func == 'na') and len(value.args) == 1:
            target = value.args[0]
            if isinstance(target, FunctionCall) and target.func == 'bool':
                self.reads_na_of_a_cast = True
                return
            field = target.name.rpartition('.')[2] if isinstance(target, Identifier) and '.' in target.name \
                else getattr(target, 'member', None)
            if field in self.bool_fields:
                self.reads_na_of_a_cast = True  # `na(t.flag)` of a bool field
                return
        else:
            return
        if isinstance(target, Identifier) and '.' not in target.name:
            self.na_reads.append((target.name, scope))
            if not isinstance(value, IndexAccess):
                self.na_calls.append((target.name, scope))  # `na(x)` itself

    # ------------------------------------------------------------ solving
    def bool_keys(self, extra_roots: tuple = ()) -> set:
        """The keys whose answer is bool: the least fixpoint of their formulas."""
        roots = [('function' if isinstance(node, FuncDecl) else 'value', node_id)
                 for node_id, node in list(self.nodes.items()) if not isinstance(node, Binding)]
        formulas, dependents = self._formulas(roots + list(extra_roots))
        for key, owners in self.implied_by.items():
            for owner in owners:
                dependents.setdefault(owner, []).append(key)
        true_keys: set = set()
        queue = list(formulas)
        while queue:
            key = queue.pop()
            if key in true_keys:
                continue
            if any(owner in true_keys for owner in self.implied_by.get(key, ())) or _holds(formulas[key], true_keys):
                true_keys.add(key)
                queue.extend(dependents.get(key, ()))
        return true_keys

    def _formulas(self, roots: list) -> tuple:
        formulas: Dict[Key, object] = {}
        dependents: Dict[Key, List[Key]] = {}
        pending = list(roots)
        while pending:
            key = pending.pop()
            if key in formulas:
                continue
            formula = formulas[key] = self._formula_of(key)
            for reference in _refs(formula):
                dependents.setdefault(reference, []).append(key)
                if reference not in formulas:
                    pending.append(reference)
        return formulas, dependents

    def _formula_of(self, key: Key):
        kind, node_id = key
        node = self.nodes[node_id]
        if kind == 'binding':
            return self._binding_formula(node)
        if kind == 'function':
            scope = self.function_scopes[node_id]
            if isinstance(node.body, list):
                return self._tail(node.body[-1], scope, key) if node.body else False
            return self._tail_value(node.body, scope, key)
        scope = self.scope_of[node_id]
        contexts = [context if context is True else self._name(*context)
                    for context in self.contexts.get(node_id, ())]
        return _any(contexts + [self._arm(arm, scope, key) for arm in _value_arms(node)])

    # ------------------------------------------------------------ formulas
    def _arm(self, arm, scope: BlockScope, owner: Key):
        if not arm:
            return False
        if isinstance(arm, list):
            return self._tail(arm[-1], self.block_scopes.get(id(arm), scope), owner)
        return self._tail_value(arm, scope, owner)

    def _tail(self, tail, scope: BlockScope, owner: Key):
        """The formula of the last statement of a block, whose value is the block's."""
        if isinstance(tail, ExpressionStatement):
            return self._tail_value(tail.expr, scope, owner)
        if isinstance(tail, (IfStatement, SwitchStatement)):
            return _any(self._arm(arm, scope, owner) for arm in _statement_arms(tail))
        if isinstance(tail, BINDINGS):
            target = tail.name if isinstance(tail, (VarDecl, VaripDecl)) else tail.target
            return self._name(target, scope) if isinstance(target, str) else False
        return self._tail_value(tail, scope, owner)

    def _tail_value(self, value, scope: BlockScope, owner: Key):
        """A value in the tail of `owner` has its type: one if/switch has one type."""
        if isinstance(value, VALUE_NODES):
            key = ('value', id(value))
            self.implied_by.setdefault(key, []).append(owner)
            return ('ref', key)
        return self._expression(value, scope)

    def _expression(self, value, scope: BlockScope):
        if isinstance(value, Literal):
            return value.literal_type == 'bool'
        if isinstance(value, BinaryOp):
            if value.op in ('and', 'or'):
                return True
            # a comparison is typed by its operands: one that cannot be typed leaves it untyped
            return value.op in BOOL_OPERATORS and not (self._untyped_operand(value.left, scope)
                                                       or self._untyped_operand(value.right, scope))
        if isinstance(value, UnaryOp):
            return value.op == 'not'
        if isinstance(value, TernaryOp):
            return _all((self._expression(value.true_expr, scope), self._expression(value.false_expr, scope)))
        if isinstance(value, VALUE_NODES):
            return ('ref', ('value', id(value)))
        if isinstance(value, IndexAccess):
            return self._expression(value.object, scope)
        if isinstance(value, FunctionCall):
            return self._call(value, scope)
        if isinstance(value, Identifier) and '.' not in value.name:
            return self._name(value.name, scope)
        if isinstance(value, (Identifier, MemberAccess)):
            path = receiver_path(value)
            return ((path is not None and '.'.join(path) in BOOL_BUILTINS)
                    or self.resolver.field_type_in_scope(value, scope) == 'bool')
        return False

    def _untyped_operand(self, operand, scope: BlockScope) -> bool:
        """An operand that has no type: a name bound to an element of a builtin's tuple
        (`[m, s] = ta.macd(...)`) or,
        untyped, to an if/switch value, or a method called on a value of the script (`tV.size()`)."""
        if isinstance(operand, Identifier) and '.' not in operand.name:
            binding = scope.lookup(operand.name)
            if binding is None:
                return False
            if id(binding) in self.tuple_elements:  # of a builtin's tuple; a user function's tail types it
                value = self.tuple_elements[id(binding)][1]
                return not (isinstance(value, FunctionCall) and value.func in self.functions)
            value = binding.initialiser  # an if/switch value, or its lowered `__block_result__`
            return binding.declared_type is None and (isinstance(value, VALUE_NODES) or (
                isinstance(value, Identifier) and value.name.startswith('__block_result')))
        if isinstance(operand, MethodCall):
            return True
        if isinstance(operand, FunctionCall) and isinstance(operand.func, str) and '.' in operand.func:
            return scope.lookup(operand.func.split('.')[0]) is not None
        return False

    def _name(self, name: str, scope: BlockScope):
        if name in ('true', 'false'):
            return True
        binding = scope.lookup(name)
        if binding is None:
            return False
        if binding.initialiser is None and id(binding) not in self.tuple_elements:
            return binding.declared_type == 'bool'
        self.nodes[id(binding)] = binding
        return ('ref', ('binding', id(binding)))

    def _binding_formula(self, binding: Binding):
        if id(binding) in self.tuple_elements:
            _, value, position = self.tuple_elements[id(binding)]
            return self._tuple_element(value, position)
        return self._expression(binding.initialiser, binding.scope)

    def _tuple_element(self, value, position: int):
        """`[p, q] = f()`: p is bool when every f's tail tuple has a bool at p."""
        if not (isinstance(value, FunctionCall) and isinstance(value.func, str) and value.func in self.functions):
            return False
        return _all(self._tail_element(function, position) for function in self.functions[value.func])

    def _tail_element(self, function: FuncDecl, position: int):
        body = function.body
        tail = body[-1].expr if isinstance(body, list) and body and isinstance(body[-1], ExpressionStatement) else body
        elements = getattr(tail, 'elements', None)
        if not elements or position >= len(elements):
            return False
        return self._expression(elements[position], self.function_scopes[id(function)])

    # ------------------------------------------------------------ calls
    def _call(self, call: FunctionCall, scope: BlockScope):
        name = 'na' if isinstance(call.func, NaLiteral) else call.func
        if not isinstance(name, str):
            return False
        if name in self.functions:
            return self._user_call(self.functions[name], call.args, scope)
        if name in BOOL_BUILTINS or name in BOOL_CONSTRUCTS:
            return True
        source = _SOURCE_TYPED_ARGUMENT.get(name)
        if source is not None:
            # a builtin of the type of its source (Pine's `series bool` overloads): `ta.change(flag)`,
            # `ta.valuewhen(c, flag, 0)` are bool where `flag` is
            position, keyword = source
            argument = call.kwargs.get(keyword) if keyword in call.kwargs else (
                call.args[position] if len(call.args) > position else None)
            return False if argument is None else self._expression(argument, scope)
        receiver, dot, method = name.rpartition('.')
        if not dot:
            return False
        if receiver in NAMESPACE_MEMBERS or receiver in RECEIVER_NAMESPACES.values():
            # `array.get(a, 0)`: the function form of a method on its first argument
            is_element_access = receiver == 'array' and method in ARRAY_ELEMENT_METHODS and call.args
            return self._method_on(self._container(call.args[0], scope), method) if is_element_access else False
        methods = [function for function in self.functions.get(method, ()) if function.is_method]
        if methods:
            return self._user_call(methods, [Identifier(name=receiver)] + list(call.args), scope)
        return self._method_on(self._container(Identifier(name=receiver), scope), method)

    def _user_call(self, declarations: list, arguments: list, scope: BlockScope):
        if len(declarations) > 1:
            declarations = overloads_taking(declarations, [self._argument_type(argument, scope)
                                                           for argument in arguments])
        return _all(('ref', ('function', id(function))) for function in declarations) if declarations else False

    def _argument_type(self, argument, scope: BlockScope) -> Optional[str]:
        if isinstance(argument, Identifier) and scope.lookup(argument.name) is None:
            return BUILTIN_SERIES_TYPES.get(argument.name)
        return self.resolver.scalar_type(argument, scope)

    @staticmethod
    def _method_on(container, method: str):
        """A method of a known builtin receiver: bool by its builtin, or an `array<bool>` element."""
        if container is None:
            return False
        kind, element_is_bool = container
        if f'{RECEIVER_NAMESPACES[kind]}.{method}' in BOOL_BUILTINS:
            return True
        return element_is_bool if kind == 'array' and method in ARRAY_ELEMENT_METHODS else False

    def _container(self, expression, scope: BlockScope):
        """(kind, element formula) when `expression` is a builtin array, map,
        matrix or string: declared, constructed, or bound to one; else None."""
        followed = set()
        while True:
            if isinstance(expression, Literal):
                return ('string', False) if expression.literal_type == 'string' else None
            if isinstance(expression, FunctionCall):
                return self._constructed_container(expression, scope)
            if isinstance(expression, Identifier) and '.' not in expression.name:
                binding = scope.lookup(expression.name)
                if binding is None or id(binding) in followed:
                    return None
                if binding.initialiser is None:
                    return _declared_container(binding.declared_type)
                followed.add(id(binding))
                expression, scope = binding.initialiser, binding.scope
                continue
            if isinstance(expression, (Identifier, MemberAccess)):
                return _declared_container(self.resolver.field_type_in_scope(expression, scope))
            return None

    def _constructed_container(self, call: FunctionCall, scope: BlockScope):
        func = call.func if isinstance(call.func, str) else ''
        if func == 'array.from':
            return 'array', (_all(self._expression(argument, scope) for argument in call.args)
                             if call.args else False)
        if func.startswith('array.new'):
            return 'array', func in ('array.new_bool', 'array.new<bool>')
        kind = func.partition('.new')[0] if '.new' in func else None
        return (kind, False) if kind in ('map', 'matrix') else None

    # ------------------------------------------------------------ restamping
    def ancestors_of(self, node_ids: set) -> set:
        """The ids of every node above one of `node_ids`."""
        ancestors: set = set()
        pending = list(node_ids)
        while pending:
            for parent in self.parents.get(pending.pop(), ()):
                if parent is not None and id(parent) not in ancestors:
                    ancestors.add(id(parent))
                    pending.append(id(parent))
        return ancestors


@dataclasses.dataclass(frozen=True)
class _Restamp:
    """Copies only the nodes on the way to a stamped one; the rest is shared."""
    stamped: set
    ancestors: set

    def rebuilt(self, node):
        node_id = id(node)
        if node_id not in self.ancestors and node_id not in self.stamped:
            return node
        changes = {}
        if node_id in self.ancestors:
            changes = {name: map_children(getattr(node, name), self.rebuilt) for name in child_field_names(type(node))}
        if node_id in self.stamped:
            changes['value_is_bool'] = True
        return dataclasses.replace(node, **changes)
