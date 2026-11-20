"""
Overload resolution for same-name Pine functions and methods.

Pine picks an overload from the static types of ALL its arguments. pynecore's
``@overload`` selects the same way from every parameter's annotation (user
types, drawings, containers, typed na, int before float), so each same-name
declaration keeps its name and is emitted under ``@overload``; no dispatcher
is generated. Two declarations the runtime cannot tell apart are refused.

pynecore selects by the VALUE it is given, Pine by the declared TYPE. They
disagree only where one overload takes an int and another a float, and a
float-typed value holds a Python int (``float v = 1``). At such a slot an
argument whose float type comes from declarations is passed as ``float(...)``.
Declarations are: a typed name or parameter, ``input.float`` / ``input.int``,
a literal, an untyped name whose initialiser is itself declared (followed in
the scope that owns the initialiser), a history read or UDT field of such a
value, a sign, and ``+ - * %`` or a ternary / if / switch over such values.
Expression inference (``/``, ``ta.*``, other calls) is never used: it calls
int-valued Pine expressions float, and pynecore's static type pin types them.

Scopes follow Pine: the global scope, one per function and one per nested
block. ``rename_shadowing_locals`` runs first, so a name bound in an
inner scope never hides an outer binding of the same name here. Within ONE
scope, a second binding with a different type or initialiser makes the name
unknown.
"""
import dataclasses
import re
from typing import Callable, Dict, FrozenSet, List, NamedTuple, Optional, Tuple

from .ast_nodes import (
    Assignment, BinaryOp, Expression, ExpressionStatement, ForInLoop, ForLoop, FuncDecl,
    FunctionCall, Identifier, IfExpression, IfStatement, IndexAccess, InputDecl, Literal,
    MemberAccess, MethodCall, Reassignment, Script, SwitchExpression, SwitchStatement, TernaryOp,
    TupleDestructure, UnaryOp, VarDecl, VaripDecl, WhileLoop,
)
from .ast_walk import walk
from .errors import UnsupportedFeatureError
from .type_inference import receiver_path

SCALAR_TYPES = ('int', 'float')
SIGN_OPERATORS = ('+', '-')
# Pine's numeric promotion: int op int is int, any float operand makes it float.
# `/` is left out: Pine v6 gives a fractional result for int / int.
PROMOTING_OPERATORS = ('+', '-', '*', '%')
INPUT_TYPES = {'input.float': 'float', 'input.int': 'int'}
# Untyped names followed per argument before its type is given up as unknown
MAXIMUM_CHAIN_LINKS = 256


# ============================================================================
# @overload declarations
# ============================================================================

def mark_overloaded_declarations(main_body: list, enum_names: set,
                                 runtime_type_name: Callable[[Optional[str]], Optional[str]]) -> Tuple[list, bool]:
    """Return `main_body` with every same-name function declaration marked
    ``is_overload`` (emitted under ``@overload``), and whether there is one.

    An enum parameter keeps its enum's name.
    """
    declarations_by_name: Dict[str, List[FuncDecl]] = {}
    for statement in main_body:
        if isinstance(statement, FuncDecl):
            declarations_by_name.setdefault(statement.name, []).append(statement)
    overloaded_groups = [group for group in declarations_by_name.values() if len(group) > 1]
    for group in overloaded_groups:
        refuse_indistinguishable_overloads(group, enum_names, runtime_type_name)
    overloaded = {id(declaration) for group in overloaded_groups for declaration in group}
    marked_body = [_as_overload(statement, enum_names) if id(statement) in overloaded else statement
                   for statement in main_body]
    return marked_body, bool(overloaded)


def with_enums_as_str(type_hint: Optional[str], enum_names: set) -> Optional[str]:
    """The annotation the runtime sees: every enum name, alone or inside a
    container (`list[E]`, `Matrix[E]`, `dict[str, E]`), becomes ``str``."""
    if not type_hint or not enum_names:
        return type_hint
    enum_pattern = r'\b(?:' + '|'.join(re.escape(name) for name in sorted(enum_names)) + r')\b'
    return re.sub(enum_pattern, 'str', type_hint)


def _as_overload(declaration: FuncDecl, enum_names: set) -> FuncDecl:
    # an enum parameter keeps its enum's name (chrono_utils `timezone: Timezone`)
    return dataclasses.replace(declaration, is_overload=True)


def refuse_indistinguishable_overloads(declarations: List[FuncDecl], enum_names: set,
                                       runtime_type_name: Callable[[Optional[str]], Optional[str]]) -> None:
    """Two overloads with the same runtime parameter types (`simple string` and
    `string`, `f(E e)` next to `f(string s)`) are kept;
    pynecore runs the first that matches."""
    earlier_by_runtime_types: Dict[tuple, FuncDecl] = {}
    for declaration in declarations:
        runtime_types = tuple(runtime_type_name(with_enums_as_str(parameter.type_hint, enum_names))
                              for parameter in declaration.params)
        earlier_by_runtime_types.setdefault(runtime_types, declaration)
        # such overloads are emitted as written (`simple string this` next to `string this`, jason5480's
        # libraries); pynecore runs the first that matches, as the Pine call would


def _collision_message(earlier: FuncDecl, later: FuncDecl, runtime_types: tuple, enum_names: set) -> str:
    def label(type_hint):
        if type_hint in enum_names:
            return f'enum {type_hint}'
        enums_inside = sorted(name for name in enum_names if re.search(rf'\b{re.escape(name)}\b', str(type_hint)))
        return f"{type_hint} (enum {', '.join(enums_inside)})" if enums_inside else str(type_hint)
    differences = [f'{label(first.type_hint)} (line {earlier.line}) and {label(second.type_hint)} (line {later.line})'
                   for first, second in zip(earlier.params, later.params) if first.type_hint != second.type_hint]
    collision = ('; '.join(differences) + ' are the same type at runtime' if differences
                 else 'both declare the same parameter types')
    return (f"overloads of '{later.name}' on lines {earlier.line} and {later.line} cannot be told apart: "
            f"{collision}, {runtime_types}, so a call cannot pick one")


# ============================================================================
# Slots where one overload takes an int and another a float
# ============================================================================

class IntFloatSlots(NamedTuple):
    """The positions and parameter names at which one overload of a name takes
    an int and another a float."""
    positions: FrozenSet[int]
    parameter_names: FrozenSet[str]

    def includes(self, slot) -> bool:
        return slot in (self.positions if isinstance(slot, int) else self.parameter_names)


def int_float_overload_slots(declarations: list) -> Dict[str, IntFloatSlots]:
    """Overloaded Pine name -> its int/float slots, for the names that have any."""
    declarations_by_name: Dict[str, List[FuncDecl]] = {}
    for declaration in declarations:
        if isinstance(declaration, FuncDecl):
            declarations_by_name.setdefault(declaration.name, []).append(declaration)
    slots_by_name: Dict[str, IntFloatSlots] = {}
    for name, group in declarations_by_name.items():
        if len(group) < 2:
            continue
        types_by_position: Dict[int, set] = {}
        types_by_parameter_name: Dict[str, set] = {}
        for declaration in group:
            for position, parameter in enumerate(declaration.params):
                types_by_position.setdefault(position, set()).add(parameter.type_hint)
                types_by_parameter_name.setdefault(parameter.name, set()).add(parameter.type_hint)
        slots = IntFloatSlots(_int_and_float(types_by_position), _int_and_float(types_by_parameter_name))
        if slots.positions or slots.parameter_names:
            slots_by_name[name] = slots
    return slots_by_name


def _int_and_float(types_by_slot: dict) -> frozenset:
    return frozenset(slot for slot, types in types_by_slot.items() if set(SCALAR_TYPES) <= types)


def overloads_taking(declarations: List[FuncDecl], argument_types: List[Optional[str]]) -> List[FuncDecl]:
    """The overloads of one name a call with these argument types can select,
    as Pine resolves them: those the call can reach (a parameter
    for every argument, an argument for every parameter without a default),
    then those whose typed parameters take each argument's type exactly, else
    those that take it once an int is promoted to float. An argument of
    unknown type (None) and an untyped parameter rule nothing out, so every
    overload of the result may be the one that runs."""
    def takes(declaration: FuncDecl, promoting: bool) -> bool:
        return all(parameter.type_hint is None or argument_type is None or parameter.type_hint == argument_type
                   or (promoting and (argument_type, parameter.type_hint) == ('int', 'float'))
                   for parameter, argument_type in zip(declaration.params, argument_types))
    reachable = [declaration for declaration in declarations
                 if sum(parameter.default is None for parameter in declaration.params)
                 <= len(argument_types) <= len(declaration.params)]
    exact = [declaration for declaration in reachable if takes(declaration, promoting=False)]
    return exact or [declaration for declaration in reachable if takes(declaration, promoting=True)]


# ============================================================================
# Scopes and the declarations they hold
# ============================================================================

class Binding(NamedTuple):
    """What one declaration says about a name: its declared Pine type, or the
    initialiser of an untyped declaration together with the scope owning it."""
    declared_type: Optional[str]
    initialiser: Optional[Expression]
    scope: Optional['BlockScope']


UNKNOWN_BINDING = Binding(None, None, None)


class BlockScope:
    """One Pine scope: the global scope, a function body, or a nested block."""

    def __init__(self, parent: Optional['BlockScope'] = None):
        self.parent = parent
        self.bindings: Dict[str, Binding] = {}

    def bind(self, name: str, binding: Binding) -> None:
        """A second binding in this scope that says something else makes the name unknown."""
        earlier = self.bindings.get(name)
        agrees = (earlier is not None and earlier.declared_type == binding.declared_type
                  and earlier.initialiser is binding.initialiser)
        self.bindings[name] = binding if earlier is None or agrees else UNKNOWN_BINDING

    def lookup(self, name: str) -> Optional[Binding]:
        scope = self
        while scope is not None:
            if name in scope.bindings:
                return scope.bindings[name]
            scope = scope.parent
        return None


def collect_call_scopes(script: Script, visit_ast_nodes) -> Dict[int, BlockScope]:
    """id() of every function and method call in `script` -> the scope it is in,
    with every scope's declarations bound. `visit_ast_nodes` is the
    transformer's depth-first walker."""
    call_scopes: Dict[int, BlockScope] = {}

    def note_calls(nodes: list, scope: BlockScope) -> None:
        def note(node):
            if isinstance(node, (FunctionCall, MethodCall)):
                call_scopes[id(node)] = scope
        # The walker enters the (condition, body) pairs of else-if arms and
        # switch cases (ast_walk); walking them again here would
        # revisit each nested branch once per ancestor
        visit_ast_nodes(nodes, note)

    def walk_block(statements: list, scope: BlockScope) -> None:
        # Every call in a statement is first given the statement's scope; the
        # calls inside each nested block are then given that block's scope
        for statement in statements:
            bind_declaration(statement, scope)
            note_calls([statement], scope)
            for body, block_scope in nested_blocks(statement, scope):
                walk_block(body, block_scope)

    walk_block(list(script.declarations) + list(script.body), BlockScope())
    return call_scopes


def bind_declaration(statement, scope: BlockScope) -> None:
    """Bind the name(s) a statement declares in `scope`."""
    if isinstance(statement, Assignment) and isinstance(statement.target, TupleDestructure):
        for name in statement.target.names:
            scope.bind(name, UNKNOWN_BINDING)
    elif isinstance(statement, (Assignment, VarDecl, VaripDecl)):
        name = statement.target if isinstance(statement, Assignment) else statement.name
        if statement.type_hint:
            scope.bind(name, Binding(statement.type_hint, None, scope))
        else:
            scope.bind(name, Binding(None, statement.value, scope))
    elif isinstance(statement, InputDecl):
        scope.bind(statement.name, Binding(INPUT_TYPES.get(statement.func), None, scope))


def nested_blocks(statement, scope: BlockScope) -> List[Tuple[list, BlockScope]]:
    """The blocks a statement opens, each with its own scope under `scope`."""
    def opened(body, bound_names=(), parameters=()) -> Tuple[list, BlockScope]:
        block_scope = BlockScope(scope)
        for name in bound_names:
            block_scope.bind(name, UNKNOWN_BINDING)
        for parameter in parameters:
            block_scope.bind(parameter.name, Binding(parameter.type_hint, None, block_scope))
        return (body if isinstance(body, list) else [body]), block_scope

    if isinstance(statement, FuncDecl):
        return [opened(statement.body, parameters=statement.params)]
    if isinstance(statement, IfStatement):
        bodies = [statement.body] + [body for _, body in statement.elseifs] + [statement.else_body]
    elif isinstance(statement, SwitchStatement):
        bodies = [body for _, body in statement.cases] + [statement.default]
    elif isinstance(statement, ForLoop):
        return [opened(statement.body, bound_names=[statement.var])]
    elif isinstance(statement, ForInLoop):
        return [opened(statement.body, bound_names=statement.vars)]
    elif isinstance(statement, WhileLoop):
        bodies = [statement.body]
    else:
        return []
    return [opened(body) for body in bodies if body]


# ============================================================================
# Declared scalar types
# ============================================================================

class _Pending(NamedTuple):
    """One node on the explicit stack of DeclaredTypeResolver.scalar_type."""
    node: object
    scope: BlockScope
    in_initialiser: bool = False        # reached through a binding, not the call's own argument
    operands: Optional[list] = None     # [(node, scope)] once expanded
    binding: Optional[Binding] = None   # the binding an identifier was followed through


#: the temporary an if/switch value is lowered into (value_blocks)
BLOCK_RESULT = re.compile(r'__block_result(_\d+)?__')


class DeclaredTypeResolver:
    """'int' or 'float' for an expression whose type comes from declarations
    alone, else None.

    A UDT field path (`t.v`) written as the call's own argument is typed by
    `field_type`, the transformer's receiver typing where the call is. One
    inside an initialiser is typed from the scope that owns the initialiser,
    with `udt_field_types` (type name -> field -> Pine type): the call's
    function context (a parameter `t`) must not leak into a remembered type.
    """

    def __init__(self, field_type: Callable[[Expression], Optional[str]], udt_field_types: Dict[str, Dict[str, str]]):
        self.field_type = field_type
        self.udt_field_types = udt_field_types
        self.binding_types: Dict[int, Optional[str]] = {}  # id(binding) -> type, once computed
        self.block_results: Dict[str, list] = {}  # `__block_result_<n>__` -> the values its arms assign

    def scalar_type(self, expression, scope: BlockScope) -> Optional[str]:
        """Iterative, so a long chain of untyped names (`x0 = x1`, `x1 = x2`, ...)
        cannot exhaust the Python stack. Each binding is typed once and
        remembered; the walk gives up (None) after MAXIMUM_CHAIN_LINKS links,
        which also ends a cycle."""
        types: Dict[int, Optional[str]] = {}
        stack = [_Pending(expression, scope)]
        links_followed = 0
        while stack:
            pending = stack.pop()
            if pending.operands is not None:
                types[id(pending.node)] = self._combined_type(pending, types)
                continue
            leaf_type, operands, binding = self._operands(pending.node, pending.scope, pending.in_initialiser)
            if operands is None:
                types[id(pending.node)] = leaf_type
                continue
            if binding is not None:
                links_followed += 1
                if links_followed > MAXIMUM_CHAIN_LINKS:
                    return None
            stack.append(pending._replace(operands=operands, binding=binding))
            in_initialiser = pending.in_initialiser or binding is not None
            stack.extend(_Pending(node, operand_scope, in_initialiser) for node, operand_scope in operands)
        return types[id(expression)]

    def _operands(self, node, scope: BlockScope, in_initialiser: bool = False) -> tuple:
        """(type, None, None) for a leaf; (None, [(operand, scope)], binding) otherwise."""
        leaf = (None, None, None)
        if isinstance(node, Literal):
            return _scalar(node.literal_type), None, None
        if isinstance(node, UnaryOp):
            return (None, [(node.operand, scope)], None) if node.op in SIGN_OPERATORS else leaf
        if isinstance(node, BinaryOp):
            return (None, [(node.left, scope), (node.right, scope)], None) if node.op in PROMOTING_OPERATORS else leaf
        if isinstance(node, (TernaryOp, IfExpression, SwitchExpression)):
            branch_values = _branch_values(node)
            return leaf if None in branch_values else (None, [(value, scope) for value in branch_values], None)
        if isinstance(node, IndexAccess):
            return None, [(node.object, scope)], None
        if isinstance(node, MemberAccess) or (isinstance(node, Identifier) and '.' in node.name):
            field_type = self.field_type_in_scope(node, scope) if in_initialiser else self.field_type(node)
            return _scalar(field_type), None, None
        if isinstance(node, Identifier):
            return self._identifier_operands(node.name, scope)
        return leaf

    def _identifier_operands(self, name: str, scope: BlockScope) -> tuple:
        binding = scope.lookup(name)
        if binding is None or binding.initialiser is None:
            return (_scalar(binding.declared_type) if binding else None), None, None
        if id(binding) in self.binding_types:
            return self.binding_types[id(binding)], None, None
        if name in self.block_results:
            # an if/switch value lowered into `__block_result_<n>__` (value_blocks): the values its arms assign,
            # `1` and `2.5` a float as Pine types the if
            return None, [(value, binding.scope) for value in self.block_results[name]], binding
        # The initialiser is typed in the scope that owns it, not at the call
        return None, [(binding.initialiser, binding.scope)], binding

    def field_type_in_scope(self, field_path, scope: BlockScope) -> Optional[str]:
        """The declared type of `t.v` / `t.inner.v` read in `scope`: the root
        name's declared type, or the UDT its untyped initialiser constructs
        (`t = T.new(...)`, `t = w.inner`), then each field's declared type."""
        pending_fields: List[str] = []
        expression, expression_scope = field_path, scope
        for _ in range(MAXIMUM_CHAIN_LINKS):
            path = receiver_path(expression)
            if not path:
                type_name = _constructed_type(expression, self.udt_field_types)
                break
            root, *fields = path
            pending_fields = fields + pending_fields
            binding = expression_scope.lookup(root)
            if binding is None or binding.initialiser is None:
                type_name = binding.declared_type if binding else None
                break
            expression, expression_scope = binding.initialiser, binding.scope
        else:
            return None
        for field_name in pending_fields:
            type_name = self.udt_field_types.get(type_name, {}).get(field_name)
        return type_name

    def _combined_type(self, pending: _Pending, types: dict) -> Optional[str]:
        operand_types = [types[id(node)] for node, _ in pending.operands]
        if isinstance(pending.node, (UnaryOp, IndexAccess, Identifier)) and len(operand_types) == 1:
            combined = operand_types[0]
        else:
            combined = _promoted(operand_types)
        if pending.binding is not None:
            self.binding_types[id(pending.binding)] = combined
        return combined


def _constructed_type(expression, udt_field_types: dict) -> Optional[str]:
    """'T' for `T.new(...)` when T is a user type, else None."""
    if isinstance(expression, FunctionCall) and isinstance(expression.func, str) and expression.func.endswith('.new'):
        type_name = expression.func[:-len('.new')]
        return type_name if type_name in udt_field_types else None
    return None


def _scalar(type_name: Optional[str]) -> Optional[str]:
    return type_name if type_name in SCALAR_TYPES else None


def _promoted(operand_types: list) -> Optional[str]:
    """Pine's numeric promotion: float if any operand is float, int if all are int."""
    if not operand_types or not all(type_name in SCALAR_TYPES for type_name in operand_types):
        return None
    return 'float' if 'float' in operand_types else 'int'


def _branch_values(node) -> list:
    """The value expression of every branch; None for a branch with no single value."""
    if isinstance(node, TernaryOp):
        return [node.true_expr, node.false_expr]
    if isinstance(node, IfExpression):
        branches = [node.true_expr] + ([node.false_expr] if node.false_expr is not None else [])
    else:
        branches = [body for _, body in node.cases] + ([node.default] if node.default is not None else [])
    return [_branch_value(branch) for branch in branches]


def _branch_value(branch) -> Optional[Expression]:
    last = branch[-1] if isinstance(branch, list) and branch else branch
    if isinstance(last, ExpressionStatement):
        return last.expr
    return last if isinstance(last, Expression) else None


# ============================================================================
# Coercion at the call site
# ============================================================================

class OverloadArgumentCoercion:
    """Passes a float-typed argument as ``float(...)`` at the int/float slots
    of an overloaded name, so pynecore selects the overload Pine does."""

    def __init__(self, script: Script, visit_ast_nodes, field_type: Callable[[Expression], Optional[str]],
                 udt_field_types: Dict[str, Dict[str, str]]):
        self.slots_by_name = int_float_overload_slots(script.declarations)
        self.call_scopes = collect_call_scopes(script, visit_ast_nodes) if self.slots_by_name else {}
        self.resolver = DeclaredTypeResolver(field_type, udt_field_types)
        for node in walk(script):
            if isinstance(node, Reassignment) and isinstance(node.target, str) \
                    and BLOCK_RESULT.fullmatch(node.target):
                self.resolver.block_results.setdefault(node.target, []).append(node.value)

    def coerce(self, call, pine_name, pine_arguments: list, arguments: list,
               pine_keywords: dict, keywords: dict) -> Tuple[list, dict]:
        """`call` is the Pine call node; the pine_* values are its untransformed
        arguments, `arguments` / `keywords` the transformed ones."""
        slots = self.slots_by_name.get(pine_name)
        if slots is None:
            return arguments, keywords
        scope = self.call_scopes.get(id(call))
        if scope is None:
            # Every Pine call was given a scope; a miss means a caller passed a
            # rebuilt node, and the coercion would silently be skipped
            raise RuntimeError(f"internal error: the call to overloaded '{pine_name}' "
                               f"(line {getattr(call, 'line', 0)}) was not seen when scopes were collected")

        def coerced(slot, pine_argument, argument):
            if (slots.includes(slot) and not isinstance(pine_argument, Literal)
                    and self.resolver.scalar_type(pine_argument, scope) == 'float'):
                return FunctionCall(func='float', args=[argument], kwargs={})
            return argument
        return ([coerced(position, pine_argument, argument)
                 for position, (pine_argument, argument) in enumerate(zip(pine_arguments, arguments))],
                {name: coerced(name, pine_keywords[name], argument) for name, argument in keywords.items()})
