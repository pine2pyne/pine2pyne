"""
Library collisions: a user binding named like a pynecore.lib name.

Pine keeps a variable, a parameter or a function apart from a built-in
namespace or function of the same name: `color.new(color, 50)` inside
`f(color = color.red)` calls the namespace on the parameter. Generated Python
has one namespace for both, and pynecore's import normalizer turns every read
of an imported name into the library object. The scope renamer therefore gives
such a binding a fresh name; this module decides which names collide and, for
each `N.m`, whether it reads the binding or the namespace N.
"""
from typing import Callable, Dict, FrozenSet, List, Mapping, NamedTuple, Optional, Set

from .ast_nodes import (
    Assignment, Expression, ExpressionStatement, ForInLoop, ForLoop, FuncDecl, FunctionCall, Identifier,
    IfExpression, InputDecl, Literal, MemberAccess, NaLiteral, Parameter, Script, SwitchExpression,
    TernaryOp, TupleDestructure, TypeDecl, VarDecl, VaripDecl,
)
from .ast_walk import walk
from .namespace_members import NAMESPACE_MEMBERS
from .pine_builtins import PYNECORE_LIB_MODULES
from .tokens import TYPE_KEYWORDS

NO_MEMBERS: FrozenSet[str] = frozenset()
# A constructor is called on the namespace, never as a method of a value:
# `line.new(...)` next to an array named `line`, `array.new_float(...)`, `array.from(...)`
CONSTRUCTORS = frozenset({'new', 'from'})
TYPED_CONSTRUCTOR_PREFIX = 'new_'
# `x.copy()` makes a value of x's type, and is a method of every object value
COPY_MEMBER = 'copy'


def colliding_names(script: Script) -> FrozenSet[str]:
    """Names pynecore defines that the script binds itself (a variable, parameter, loop
    variable, input or function). Every one of them is renamed, whether or not
    the script reads the builtin (`pine_builtins.PYNECORE_RESERVED_NAMES`)."""
    from .pine_builtins import PYNECORE_RESERVED_NAMES
    bound: Set[str] = set()
    for node in walk(script):
        bound.update(_names_bound_by(node))
    # a local named like a computed series (`hl2 = (h + l) / 2`) keeps its name where the script never reads
    # the builtin: nothing imports it, so nothing is hidden
    unread = {name for name in COMPUTED_SERIES & bound if not reads_the_builtin(script, name)}
    return frozenset((bound & PYNECORE_RESERVED_NAMES) - unread)


#: the builtin series pynecore computes from the bar (a local named like one may keep its name)
COMPUTED_SERIES = frozenset({'hl2', 'hlc3', 'ohlc4', 'hlcc4'})


def reads_the_builtin(script: Script, name: str) -> bool:
    """Whether the script reads the builtin `name`: a read where no binding of its scope has the name."""
    def reads(tree) -> bool:
        return any(isinstance(node, Identifier) and node.name == name for node in walk(tree))

    def binds(tree) -> bool:
        return any(name in _names_bound_by(node) for node in walk(tree))

    top_level = [item for item in [*script.declarations, *script.body] if not isinstance(item, FuncDecl)]
    if reads(top_level) and not binds(top_level):
        return True
    return any(reads(function.body) and not (binds(function.body) or name in {p.name for p in function.params})
               for function in [*script.declarations, *script.body] if isinstance(function, FuncDecl))


def _names_bound_by(node):
    if isinstance(node, (VarDecl, VaripDecl, InputDecl, Parameter)):
        return (node.name,)
    if isinstance(node, Assignment):
        return node.target.names if isinstance(node.target, TupleDestructure) else (node.target,)
    if isinstance(node, FuncDecl) and not node.is_method:
        return (node.name,)
    if isinstance(node, ForLoop):
        return (node.var,)
    if isinstance(node, ForInLoop):
        return node.vars
    return ()


def _dotted_head_or_callee(node) -> Optional[str]:
    if isinstance(node, FunctionCall) and isinstance(node.func, str):
        return node.func.split('.')[0]
    if isinstance(node, Identifier) and '.' in node.name:
        return node.name.split('.')[0]
    if isinstance(node, MemberAccess) and isinstance(node.object, str):
        return node.object.split('.')[0]
    return None


class TypeEvidence(NamedTuple):
    """What proves a value's type beyond its own declaration: the script's user type
    names, the user type a user function provably returns, the type a variable holds where the
    value is read (a container keeps its element type: `array<T>`), and the user method names (a
    call by one may select the method, so it proves no element). Empty: the declaration alone."""
    type_names: FrozenSet[str] = NO_MEMBERS
    function_type: Callable[[str], Optional[str]] = lambda name: None
    variable_type: Callable[[str], Optional[str]] = lambda name: None
    method_names: FrozenSet[str] = NO_MEMBERS


NO_EVIDENCE = TypeEvidence()
# What `array.get(a, 0)`, `a.first()`, `matrix.get(m, 0, 0)` or `m.get(k)` returns: an element of the
# container (a map's value)
ELEMENT_ACCESSORS = {'array': frozenset({'get', 'first', 'last', 'pop', 'shift'}),
                     'matrix': frozenset({'get'}), 'map': frozenset({'get'})}


def base_type(type_name: Optional[str]) -> Optional[str]:
    """A type without its qualifier and element types: `array<T>` -> `array`, `series float` -> `float`."""
    return type_name.split('<')[0].split()[-1] if type_name else None


def element_type(type_name: Optional[str]) -> Optional[str]:
    """The type of a container's elements, a map's values: `array<T>`, `map<string, T>` -> `T`."""
    if not type_name or '<' not in type_name or not type_name.endswith('>'):
        return None
    arguments, depth, start = type_name[type_name.index('<') + 1:-1], 0, 0
    for index, character in enumerate(arguments):
        depth += {'<': 1, '>': -1}.get(character, 0)
        if character == ',' and depth == 0:
            start = index + 1
    return arguments[start:].strip() or None


def binding_type(type_hint: Optional[str], value, evidence: TypeEvidence = NO_EVIDENCE) -> Optional[str]:
    """The Pine type a declaration gives its binding, as far as it is proven: its type annotation,
    a constructor call (`P.new()`, `array.new<P>()`, `array.from(...)`), a literal, a call of a user
    function that provably returns a user-defined type (`evidence.function_type`, from
    `user_function_types`), or a value whose user type `evidence` proves (`_proven_user_type`).
    A container keeps its element type (`array<P>`; `base_type` gives `array`). None when nothing
    proves it: never a guess."""
    if isinstance(value, FunctionCall) and isinstance(value.func, str) and '.' not in value.func:
        return _without_qualifier(type_hint) if type_hint else evidence.function_type(value.func)
    if type_hint:
        return _without_qualifier(type_hint)  # `series float`, `map<string, float>`
    if isinstance(value, Literal):
        return value.literal_type or None
    constructed = _constructed_type(value, evidence)
    return constructed if constructed is not None else _proven_user_type(value, evidence)


def _without_qualifier(type_hint: str) -> str:
    prefix, bracket, elements = type_hint.partition('<')
    return prefix.split()[-1] + bracket + elements


def _constructed_type(value, evidence: TypeEvidence) -> Optional[str]:
    """`T.new()` -> T, `array.new<T>()` -> array<T>, `array.from(T.new())` -> array<T> when every
    element is that user type; `p.copy()` copies a variable, its head names no type."""
    if not (isinstance(value, FunctionCall) and isinstance(value.func, str) and '.' in value.func):
        return None
    head, _, member = value.func.rpartition('.')
    makes_a_value = is_constructor(member) or member == COPY_MEMBER
    if not makes_a_value or not (head in TYPE_KEYWORDS or head in PYNECORE_LIB_MODULES or head[:1].isupper()):
        return None
    _, bracket, elements = member.partition('<')
    if bracket:
        return head + bracket + elements
    if head == 'array' and member == 'from' and value.args:
        element_types = {_user_type(element, evidence) for element in value.args}
        if len(element_types) == 1 and None not in element_types:
            return f'array<{element_types.pop()}>'
    return head


def _proven_user_type(value, evidence: TypeEvidence) -> Optional[str]:
    """The user type `evidence` proves a value has: a variable that holds one, an element
    of a container of one, or a ternary, switch or if-expression whose every branch that is not
    `na` is that one type. Only a user type: a builtin type stays unknown here, as before."""
    if isinstance(value, Identifier) and '.' not in value.name:
        return _as_user_type(evidence.variable_type(value.name), evidence)
    if isinstance(value, (TernaryOp, IfExpression, SwitchExpression)):
        branch_types = {_branch_type(branch, evidence) for branch in _branches(value)} - {NA_BRANCH}
        return branch_types.pop() if len(branch_types) == 1 else None
    if isinstance(value, FunctionCall):
        return _as_user_type(element_type(_accessed_container(value, evidence)), evidence)
    return None


NA_BRANCH = 'na'


def _branches(value) -> list:
    if isinstance(value, TernaryOp):
        return [value.true_expr, value.false_expr]
    if isinstance(value, IfExpression):
        return [value.true_expr, value.false_expr]  # no `else`: the value is na
    return [body for _, body in value.cases] + [value.default]


def _branch_type(branch, evidence: TypeEvidence) -> Optional[str]:
    """The user type of one branch, NA_BRANCH for `na`. A block's value is its last expression,
    read with the block's own declarations: a block local hides the outer name it shares."""
    if isinstance(branch, list):
        branch, evidence = _block_value(branch, evidence)
    if branch is None or isinstance(branch, NaLiteral):
        return NA_BRANCH
    return _user_type(branch, evidence)


def _block_value(block: list, evidence: TypeEvidence):
    """A block's last expression (False when the block ends otherwise) and the evidence it is read
    with: each declaration at the block's top typed in order; a name bound anywhere else in the
    block (a nested block, a loop, a tuple) proves nothing."""
    last = block[-1] if block else None
    last = last.expr if isinstance(last, ExpressionStatement) else last
    if not isinstance(last, Expression):
        return False, evidence
    local_types: Dict[str, Optional[str]] = {}
    outer = evidence.variable_type
    for statement in block[:-1]:
        evidence_here = evidence._replace(
            variable_type=lambda name, known=dict(local_types): known[name] if name in known else outer(name))
        if isinstance(statement, (VarDecl, VaripDecl)) or (
                isinstance(statement, Assignment) and not isinstance(statement.target, TupleDestructure)):
            local_types.setdefault(statement.name if isinstance(statement, (VarDecl, VaripDecl))
                                   else statement.target,
                                   binding_type(statement.type_hint, statement.value, evidence_here))
    for node in walk(block):
        for name in _names_bound_by(node):
            local_types.setdefault(name, None)
    return last, evidence._replace(
        variable_type=lambda name: local_types[name] if name in local_types else outer(name))


def _user_type(value, evidence: TypeEvidence) -> Optional[str]:
    return _as_user_type(binding_type(None, value, evidence), evidence)


def _as_user_type(type_name: Optional[str], evidence: TypeEvidence) -> Optional[str]:
    return type_name if type_name in evidence.type_names else None


def _accessed_container(call: FunctionCall, evidence: TypeEvidence) -> Optional[str]:
    """The type of the container whose element `call` returns: `array.get(a, 0)` reads its first
    argument, `a.first()` the variable `a`; None for any other call, or a method form whose
    member a user method shares."""
    if not isinstance(call.func, str) or '.' not in call.func:
        return None
    head, _, member = call.func.rpartition('.')
    if head in ELEMENT_ACCESSORS:
        kind = head
        container = _container_type(call.args[0], evidence) if call.args else None
    elif '.' not in head and member not in evidence.method_names:
        container = evidence.variable_type(head)
        kind = base_type(container)
    else:
        return None
    return container if base_type(container) == kind and member in ELEMENT_ACCESSORS.get(kind, ()) else None


def _container_type(value, evidence: TypeEvidence) -> Optional[str]:
    if isinstance(value, Identifier) and '.' not in value.name:
        return evidence.variable_type(value.name)
    return binding_type(None, value, evidence)


def script_type_evidence(script: Script) -> TypeEvidence:
    """The evidence a whole script gives (no variables): its user types, user function returns
    and user method names. A caller adds `variable_type` from its own scope model."""
    return TypeEvidence(
        frozenset(node.name for node in script.declarations if isinstance(node, TypeDecl)),
        user_function_types(script).get, NO_MEMBERS,
        frozenset(node.name for node in script.declarations if isinstance(node, FuncDecl) and node.is_method))


def user_function_types(script: Script) -> Dict[str, str]:
    """User function name -> the user-defined type every declaration of it provably
    returns: its last expression is a constructor or a copy of that type
    (`mk() => T.new(close)`), a parameter or local declared with it, a call of such
    a function, or a value whose type those prove (a ternary of that type, an
    element of a container of it). A name a method shares is left out: a call by it may
    select the method."""
    type_names = frozenset(node.name for node in script.declarations if isinstance(node, TypeDecl))
    functions: Dict[str, List[FuncDecl]] = {}
    method_names = set()
    for node in script.declarations:
        if isinstance(node, FuncDecl):
            if node.is_method:
                method_names.add(node.name)
            else:
                functions.setdefault(node.name, []).append(node)
    resolved: Dict[str, Optional[str]] = {}

    def returned(name: str, visiting: FrozenSet[str]) -> Optional[str]:
        if name in resolved:
            return resolved[name]
        if name not in functions or name in method_names or name in visiting:
            return None
        evidence = TypeEvidence(type_names, lambda callee: returned(callee, visiting | {name}),
                                method_names=frozenset(method_names))
        returns = {_returned_type(declaration, evidence) for declaration in functions[name]}
        result = returns.pop() if len(returns) == 1 else None
        resolved[name] = result if result in type_names else None
        return resolved[name]

    return {name: type_name for name in functions
            if (type_name := returned(name, frozenset())) is not None}


def _returned_type(function: FuncDecl, evidence: TypeEvidence) -> Optional[str]:
    """The type of a function's last expression, as its declarations prove it: a name reads
    the function's own parameters and locals only."""
    local_types: Dict[str, Optional[str]] = {
        parameter.name: binding_type(parameter.type_hint, None) for parameter in function.params}
    body = function.body if isinstance(function.body, list) else [function.body]
    for statement in body[:-1]:
        if isinstance(statement, (VarDecl, VaripDecl)) or (
                isinstance(statement, Assignment) and not isinstance(statement.target, TupleDestructure)):
            name = statement.name if isinstance(statement, (VarDecl, VaripDecl)) else statement.target
            local_types.setdefault(name, binding_type(statement.type_hint, statement.value,
                                                      evidence._replace(variable_type=dict(local_types).get)))
    last = body[-1] if body else None
    last = last.expr if isinstance(last, ExpressionStatement) else last
    if isinstance(last, Identifier) and '.' not in last.name:
        return local_types.get(last.name)
    return binding_type(None, last, evidence._replace(variable_type=local_types.get))


def binding_types(script: Script) -> Dict[str, Optional[str]]:
    """Every name the script binds, with the Pine type its declarations give it
    (`declared_type`); None when no declaration says or two disagree."""
    evidence = script_type_evidence(script)
    types: Dict[str, Optional[str]] = {}
    for node in walk(script):
        for name in _names_bound_by(node):
            declared = declared_type(node, evidence._replace(variable_type=types.get))
            types[name] = declared if types.get(name, declared) == declared else None
    return types


def global_declarations(script: Script) -> list:
    """The declarations of globals (`var`/`varip`, inputs, top-level assignments) in source order,
    the order the scope renamer binds and types them up front: a value reads the
    globals declared above it, whichever kind each is. The parser keeps `var` declarations and
    inputs apart from the body, so the two lists are merged by line."""
    nodes = [node for node in list(script.declarations) + list(script.body)
             if isinstance(node, (VarDecl, VaripDecl, InputDecl, Assignment))]
    return sorted(nodes, key=lambda node: node.line)


def global_binding_types(script: Script) -> Dict[str, Optional[str]]:
    """Every global name with the type of its first declaration, in the order the scope
    renamer binds globals up front (`global_declarations`); a value reads the globals typed
    before it."""
    evidence = script_type_evidence(script)
    types: Dict[str, Optional[str]] = {}
    for node in global_declarations(script):
        for name in _names_bound_by(node):
            types.setdefault(name, declared_type(node, evidence._replace(variable_type=types.get)))
    return types


def declared_type(node, evidence: TypeEvidence = NO_EVIDENCE) -> Optional[str]:
    """The Pine type a declaring node gives the names it binds: a variable, an input, a
    parameter (`binding_type`); None for a tuple, a loop variable or a function. An
    untyped parameter's type comes from each call's argument, so it is None here."""
    if isinstance(node, (VarDecl, VaripDecl)):
        return binding_type(node.type_hint, node.value, evidence)
    if isinstance(node, Assignment) and not isinstance(node.target, TupleDestructure):
        return binding_type(node.type_hint, node.value, evidence)
    if isinstance(node, Parameter):
        return binding_type(node.type_hint, node.default)
    if isinstance(node, InputDecl):
        return input_type(node)
    return None


def is_constructor(member: str) -> bool:
    """`new`, `new<float>`, `new_float`, `from`: a namespace function that
    makes a value (a field such as `newest` is not one)."""
    name = member.split('<')[0]
    return name in CONSTRUCTORS or name.startswith(TYPED_CONSTRUCTOR_PREFIX)


def input_type(declaration: InputDecl) -> str:
    """An input's value type is named by its function: `input.int` -> int."""
    return declaration.func.rpartition('.')[2]


class MemberRule:
    """Decides, for `N.m` where N names a binding, whether `m` is a member of
    the binding (a field or method of its type) or of the namespace N."""

    def __init__(self, script: Script):
        self.type_fields: Dict[str, FrozenSet[str]] = {
            node.name: frozenset(field[0] for field in node.fields)
            for node in script.declarations if isinstance(node, TypeDecl)}
        # user method name -> the base types of its receiver (None: untyped)
        self.method_receivers: Dict[str, Set[Optional[str]]] = {}
        for node in script.declarations:
            if isinstance(node, FuncDecl) and node.is_method and node.params:
                receiver = base_type(binding_type(node.params[0].type_hint, None))
                self.method_receivers.setdefault(node.name, set()).add(receiver)

    def binding_owns(self, type_name: Optional[str], namespace: str, member: str) -> bool:
        """True when `namespace.member` reads the binding named `namespace`,
        whose type is `type_name` (None when unknown). `member` is Pine's
        spelling (`from`, `new<float>`), so a constructor is told by its name. `type_name` may
        keep its element type (`array<T>`): the decision reads its base."""
        type_name = base_type(type_name)
        receivers = self.method_receivers.get(member, ())
        if receivers and (type_name is None or None in receivers or type_name in receivers):
            return True
        if is_constructor(member):
            return False
        if type_name is None:
            # Unknown type: the binding's own member is what the namespace lacks. A member the
            # namespace has stays the namespace's, never a guess: a binding
            # holding a user type is known by its declaration (`binding_type`)
            return member not in NAMESPACE_MEMBERS.get(namespace, NO_MEMBERS)
        if type_name in self.type_fields:
            return member in self.type_fields[type_name] or member == COPY_MEMBER
        if type_name == namespace or (namespace, type_name) == ('str', 'string') or type_name not in TYPE_KEYWORDS:
            # `matrix.set(matrix, ...)` on a matrix named matrix is the namespace call; so is `str.tostring(x)`
            # where a string is named `str` (facejungle/utils)
            return False
        # A built-in type's methods are its namespace's functions: `a.size()` is array.size(a)
        return member in NAMESPACE_MEMBERS.get(type_name, NO_MEMBERS)


class HeadBinding(NamedTuple):
    """The binding a dotted head names where it is read: its Pine type, None when unknown."""
    type_name: Optional[str]


class FieldReadRule:
    """Decides whether the object of a history index reads a field of a user-defined
    type: a member of an expression (`f().x`, `(t[1]).x`), or a dotted name
    `N.m` whose head N is a binding that owns `m`. A head named like a namespace reads the
    namespace unless MemberRule says `m` is the binding's own member: `session.ismarket`
    beside an input named `session` reads the session namespace. The Pine v6 refusal
    (pine_semantics.py) and the v5 lowering (grouped_history_lowering.py) both ask it.

    The head's binding is the one in scope where the index is read, with the type its
    declaration gives it (`declared_type`), as the scope renamer resolves `N.m`;
    each caller says which binding that is, from its own scope model."""

    def __init__(self, script: Script):
        self.member_rule = MemberRule(script)

    def is_field_read(self, value, binding_in_scope: Callable[[str], Optional[HeadBinding]]) -> bool:
        """`binding_in_scope(head)`: the binding the head names where `value` is read, None
        when it names no binding there."""
        if isinstance(value, MemberAccess) and not isinstance(value.object, str):
            return True
        if isinstance(value, MemberAccess):
            dotted = f'{value.object}.{value.member}'
        else:
            dotted = value.name if isinstance(value, Identifier) else ''
        head, dot, path = dotted.partition('.')
        binding = binding_in_scope(head) if dot else None
        if binding is None:
            return False
        return head not in PYNECORE_LIB_MODULES or self.member_rule.binding_owns(
            binding.type_name, head, path.partition('.')[0])
