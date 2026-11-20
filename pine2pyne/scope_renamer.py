"""
Scope renamer: give every Pine local that shadows an outer name its own name,
and every binding named like a pynecore.lib name a name of its own.

In Pine, `x = ...` inside a function or a block declares a NEW local, even when
an outer `x` exists (the compiler only warns about the shadowing); `x := ...`
assigns the nearest visible `x`. Generated Python cannot express that directly:
an `if`/`for` block has no scope of its own, so a block local would overwrite
the global, and the transformer's symbol lookup would give a function local the
global's `var` persistence. This pre-pass renames each shadowing
declaration, and the references that follow it in the same scope, to a fresh
name, so every later stage sees distinct names. References before the
declaration keep the outer name, as in Pine.

A binding named like a pynecore.lib name that the script also uses as that
name (`f(color = color.red) => color.r(color)`) is renamed too, together with
its uses and the keyword arguments that name it; a declared function shadows
every bare call of its name. A function named like a builtin series
(`hour(x) => x * 3`, `close(x) => ...`) is renamed too whether or not the script reads
the series: its calls go to the new name and a bare read of the name stays the series. `N.m` keeps the namespace when `m` is
not the binding's own member (see library_collision.MemberRule). Two kinds of
binding keep their name: an input (its name is the key of the script's input
settings) and an `export` declaration (its name is the library's API); a
namespace use under such a binding reads an alias of the namespace instead.
An `export` with a parameter named like a function or method of the script is split into a private `f_body<N>`
(the same body, its parameters renamed like any function's) and the public export, which keeps the parameter names
and calls it.

A local named like a builtin series variable (`close`, `time`, `hour`) gets its own name
wherever it is declared below the top level: in a function, as a loop variable or
in a block of a top-level `if`/`for`/`switch`. The script imports the builtin by
that name wherever the name is read, and pynecore then reads the library series for the
local's reads too; and a block local that kept the name would make every read of it in
main() a read of that local. A top-level declaration keeps its name (the import resolver
leaves that builtin out of the imports), and so does an input. Pine declares a global where
its declaration stands, so a read of its name before it (in a statement, a function declared
earlier, the declaration's own value `close = close * 2`, or an input's own arguments:
`close = input.source(close, ...)`) is the builtin: it reads the builtin by the namespace
alias (`_close_module`), inputs and every other global alike. A bare
call of a builtin function named like such a global or a binding that keeps the name
(`time("D")`, `hour(t)`) calls the builtin by the alias too.

A Python reserved word used as a Pine name (`lambda`, `from`) gets its spelling here
too, once per name, against every identifier of the script (emitted_names.py):
`lambda` becomes `lambda_` when that is free, else `lambda_1`, so it never merges
with a user `lambda_`. Variables, parameters, functions, methods and type names share one
spelling domain; fields and enum members another (attributes), applied to every dotted
read, constructor keyword and call keyword of a user callable. An input and an `export`
declaration keep their raw name: the one existing sanitizer spells it, so a settings id
and a library API never move, and another binding holding that spelling moves instead.

A parameter is renamed for a library collision, and when it is named like a
script global: pynecore passes a global that a nested function reads
as an argument at each call site, so inside `f(g)` a callee `reader()` that
reads the global `g` would receive f's `g`. A keyword argument that names a
renamed parameter follows it; a builtin method or namespace function of the
same name keeps its own keyword.
"""
import dataclasses
import re
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from .ast_nodes import (
    ASTNode, Assignment, EnumDecl, ForInLoop, ForLoop, FuncDecl, FunctionCall, Identifier,
    IfExpression, IfStatement, InputDecl, Literal, MemberAccess, MethodCall, Reassignment,
    Script, Statement, SwitchExpression, SwitchStatement, TupleDestructure, TypeDecl,
    VarDecl, VaripDecl, WhileLoop,
)
from .ast_walk import walk
from .emitted_names import (
    EMITTED_HELPER_NAMES, KeywordSpellings, LIBRARY_ALIAS_RENAMED_NAMES, free_spelling, identifier_names,
)
from .errors import UnsupportedFeatureError
from .node_ids import BlockNumbers, hex_suffixed
from .library_collision import (COMPUTED_SERIES, 
    MemberRule, binding_type, colliding_names, global_declarations, input_type, script_type_evidence,
)
from .pine_builtins import (
    CALLABLE_SERIES_VARIABLES, GLOBAL_SUFFIX, NUMBERED_RESERVED_NAMES, NUMBERED_RESERVED_NAMES_V4,
    PYNECORE_RESERVED_NAMES, PYNECORE_LIB_MODULES,
    PYTHON_RESERVED_WORDS,
    RENAMED_SUFFIX,
    SERIES_FUNCTIONS,
    SERIES_VARIABLES,
    rename_reserved_word,
)

FRESH_NAME_SUFFIX = '_local'
NAMESPACE_ALIAS = '_{}_module'
# The heads of a namespace function call: `array.get(...)`, Pine's `str.format(...)`
NAMESPACE_HEADS = PYNECORE_LIB_MODULES | {'str'}
SCOPED_STATEMENT_TYPES = (Assignment, VarDecl, VaripDecl, Reassignment, IfStatement,
                          ForLoop, ForInLoop, WhileLoop, SwitchStatement, FuncDecl)


class RenamedScript(NamedTuple):
    script: Script
    # pynecore.lib name -> the alias its namespace uses under a binding that keeps the name
    namespace_aliases: Dict[str, str]
    # function or method name -> {Pine parameter name: its emitted name}
    parameter_names: Dict[str, Dict[str, str]]
    # Pine line of a statement -> the id of the block it stands in (node_ids.BlockNumbers)
    line_blocks: Dict[int, int] = {}


@dataclasses.dataclass(frozen=True)
class Binding:
    """What a Pine name in a scope stands for."""
    emitted: str
    collides: bool = False  # named like a pynecore.lib name the script also uses
    type_name: Optional[str] = None  # the Pine type, when proven; a container keeps its element type


def rename_shadowing_locals(script: Script) -> RenamedScript:
    """Return a copy of `script` with shadowing locals and library-named
    bindings renamed. The input script is not modified."""
    renamer = _ScopeRenamer(script)
    return RenamedScript(renamer.run(script), renamer.namespace_aliases,
                         {name: renames for name, renames in renamer.parameter_names.items() if renames},
                         dict(renamer.block_numbers.line_blocks))


def collect_names(tree) -> Set[str]:
    """Every identifier-like string in the tree: the names a fresh name must avoid."""
    names: Set[str] = set()
    _names_in(tree, names)
    return names


def script_global_names(script: Script) -> Set[str]:
    """The names a script declares at its top level: inputs, `var`/`varip`
    declarations and assignments (tuple targets included)."""
    names = {declaration.name for declaration in script.declarations
             if isinstance(declaration, (VarDecl, VaripDecl, InputDecl))}
    for statement in script.body:
        if isinstance(statement, Assignment):
            target = statement.target
            names.update(target.names if isinstance(target, TupleDestructure) else [target])
    return names


def closure_shadows(script: Script) -> Dict[str, Set[str]]:
    """Function name -> its parameters and locals named like a script global that a function it calls reads.

    pynecore hands a global a nested function reads to it as an argument at each call site, so a call
    inside `f(g)` passes f's `g` where the callee reads the global `g` (Pine reads the
    global); pine2pyne gives such a name one of its own."""
    global_names = script_global_names(script)
    functions = [node for node in script.declarations if isinstance(node, FuncDecl)]
    if not functions or not global_names:
        return {}

    def declared(function: FuncDecl) -> Set[str]:
        names = {parameter.name for parameter in function.params}
        for node in walk(function.body):
            if isinstance(node, Assignment):
                target = node.target
                names.update(target.names if isinstance(target, TupleDestructure) else
                             [target] if isinstance(target, str) else [])
            elif isinstance(node, (VarDecl, VaripDecl)):
                names.add(node.name)
            elif isinstance(node, ForLoop):
                names.add(node.var)
            elif isinstance(node, ForInLoop):
                names.update(node.vars)
        return names

    reads_of: Dict[str, Set[str]] = {}
    for function in functions:
        reads = {node.name.split('.')[0] for node in walk(function.body) if isinstance(node, Identifier)}
        reads_of.setdefault(function.name, set()).update((reads - declared(function)) & global_names)
    calls = [node for node in walk(script) if isinstance(node, FunctionCall) and isinstance(node.func, str)]
    shadows: Dict[str, Set[str]] = {}
    for function in functions:
        called = {node.func.rpartition('.')[2] for node in walk(function.body)
                  if isinstance(node, FunctionCall) and isinstance(node.func, str)}
        read_by_callees = set().union(*(reads_of.get(name, set()) for name in called))
        risky = read_by_callees & declared(function)
        parameters = [parameter.name for parameter in function.params]
        for position, parameter in enumerate(parameters):
            if parameter in risky and _always_given_itself(function, position, parameter, calls):
                risky.discard(parameter)  # `f(src)` of the global `src` everywhere: the callee reads the same value
        if risky:
            shadows.setdefault(function.name, set()).update(risky)
    return shadows


def _always_given_itself(function: FuncDecl, position: int, parameter: str, calls: List[FunctionCall]) -> bool:
    """Whether every call of a plain function passes the global of the parameter's own name to it; a function
    the script never calls has no call to pass another value (the name is kept too)."""
    if function.is_method:
        return False
    own = [call for call in calls if call.func == function.name]
    if not own:
        return True
    for call in own:
        given = call.kwargs.get(parameter) if parameter in call.kwargs else (
            call.args[position] if position < len(call.args) else None)
        if not (isinstance(given, Identifier) and given.name == parameter):
            return False
    return True


def _declared_in(statements) -> Set[str]:
    """The names a block's own statements declare (not its nested blocks')."""
    names: Set[str] = set()
    if not isinstance(statements, list):
        return names
    for statement in statements:
        if isinstance(statement, (VarDecl, VaripDecl)):
            names.add(statement.name)
        elif isinstance(statement, Assignment):
            target = statement.target
            names.update(target.names if isinstance(target, TupleDestructure) else [target])
    return names


def _top_level_names(script: Script) -> Set[str]:
    """Every name the script's top level declares, wherever its statement was moved."""
    names: Set[str] = set()
    for statement in list(script.declarations) + list(script.body):
        if isinstance(statement, (VarDecl, VaripDecl, InputDecl)):
            names.add(statement.name)
        elif isinstance(statement, Assignment):
            target = statement.target
            names.update(target.names if isinstance(target, TupleDestructure) else [target])
    return names


def builtin_series_globals(script: Script) -> Dict[str, Any]:
    """Each global named like a builtin series variable -> its first top-level declaration (an
    input, a `var`/`varip`, an assignment or a tuple target). The import resolver leaves such a
    builtin out of the imports, so a read of it above the declaration needs the alias."""
    declarations = [node for node in script.declarations if isinstance(node, (VarDecl, VaripDecl, InputDecl))]
    declarations += [node for node in script.body if isinstance(node, Assignment)]
    first: Dict[str, Any] = {}
    for node in sorted(declarations, key=lambda node: node.line):
        target = node.target if isinstance(node, Assignment) else node.name
        for name in target.names if isinstance(target, TupleDestructure) else [target]:
            if name in SERIES_VARIABLES:
                first.setdefault(name, node)
    return first


def inner_path_segments(script: Script, heads: Optional[Set[str]] = None) -> Set[str]:
    """The segments between the head and the last name of every dotted name of `script`
    (of those whose head is in `heads`, when given)."""
    segments: Set[str] = set()
    for node in walk(script):
        dotted = None
        if isinstance(node, FunctionCall) and isinstance(node.func, str):
            dotted = node.func
        elif isinstance(node, Identifier):
            dotted = node.name
        elif isinstance(node, MemberAccess) and isinstance(node.object, str):
            dotted = f'{node.object}.{node.member}'
        if dotted and dotted.count('.') >= 2 and (heads is None or dotted.split('.', 1)[0] in heads):
            segments.update(dotted.split('.')[1:-1])
    return segments


HEX_RENAME = re.compile(r'__[0-9a-f]{8}__')
#: the built-in types a user type may not be named like, compared without case
BUILTIN_TYPE_NAMES = frozenset({'box', 'label', 'line', 'linefill', 'table'})


def builtin_call_segments(script: Script) -> Set[str]:
    """The receiver names of the dotted calls of a builtin method (`this.msi.put(k, v)`: `msi`): a local or
    parameter named like one is renamed, not for a field read (`this.sess.timeRanges`) nor for a
    method no builtin namespace has (`this.channelDesc.validate()`)."""
    from .pine_builtins import method_namespaces
    # a field holding an array, a map or a matrix: any method called on it goes through that namespace's
    # type (Trendoscope/Zigzag `candle.indicators.init(...)`)
    collection_fields = {field[0] for node in walk(script) if isinstance(node, TypeDecl) for field in node.fields
                         if str(field[1]).startswith(('array<', 'map<', 'matrix<')) or str(field[1]).endswith('[]')}
    segments: Set[str] = set()
    for node in walk(script):
        if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func.count('.') >= 2:
            *_, receiver, member = node.func.split('.')
            if method_namespaces(member) or receiver in collection_fields:  # a builtin method, or a collection's
                segments.add(receiver)  # the receiver's own name only: `this.sess.timeRanges.size()` is `timeRanges`
    return segments


def _reads_itself(declaration, name) -> bool:
    """Whether the value a declaration of `name` gives reads `name` (its own value, or the if/switch value it
    had before the value blocks were lowered: `reads_own_name`)."""
    if getattr(declaration, 'reads_own_name', False):
        return True
    return isinstance(name, str) and any(isinstance(node, Identifier) and node.name.split('.')[0] == name
                                         for node in walk(declaration.value))


def _dotted_head(node) -> Optional[str]:
    """`box` of a dotted call or name (`box.new(...)`, `box.style_solid`), else None."""
    if isinstance(node, FunctionCall) and isinstance(node.func, str) and '.' in node.func:
        return node.func.split('.')[0]
    if isinstance(node, Identifier) and '.' in node.name:
        return node.name.split('.')[0]
    if isinstance(node, MemberAccess) and isinstance(node.object, str):
        return node.object.split('.')[0]
    return None


def fresh_name(stem: str, used_names: Set[str]) -> str:
    """`stem` plus the lowest index not in `used_names`; the result is recorded
    as used. Ends in digits, so never a Python keyword nor a
    sanitize_identifier result."""
    index = 1
    while f'{stem}{index}' in used_names:
        index += 1
    name = f'{stem}{index}'
    used_names.add(name)
    return name


def _names_in(value, names: Set[str]) -> None:
    """Collect every identifier-like string in the tree (first dotted segment)."""
    if isinstance(value, str):
        names.add(value.split('.')[0])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _names_in(item, names)
    elif isinstance(value, dict):
        for item in value.values():
            _names_in(item, names)
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            _names_in(getattr(value, field.name), names)


class _ScopeRenamer:
    """Walks a Script with a stack of scopes, each mapping a Pine name to its
    Binding."""

    def __init__(self, script: Script):
        self.used_names = collect_names(script)
        self.scopes: List[Dict[str, Binding]] = []
        self.scope_declarations: List[Set[str]] = []  # per scope: every name its block declares
        # the id of the block of each scope in `scopes` (0: the top level, or not known)
        self.block_numbers = BlockNumbers(script)
        self.scope_blocks: List[int] = []
        # the index in `scopes` of each function scope being visited, innermost last
        self.function_scopes: List[int] = []
        self.shadowed_names: Set[str] = set()
        self.library_names = colliding_names(script)
        # the heads of the dotted names and calls the script reads (`box` of `box.new(...)`)
        self.read_namespaces = {head for node in walk(script) for head in [_dotted_head(node)] if head}
        # the names the script imports libraries under (an alias, else the library's name)
        self.library_alias_names = {imported.alias or imported.library for imported in script.imports}
        self.top_level_names = _top_level_names(script)
        self.renamed_exports: Set[str] = set()  # exported functions spelled `name__ren__`
        self.member_rule = MemberRule(script)
        # the user types, user function returns and user methods that prove a binding's type: the
        # type of `x = mk()`, `x = array.get(a, 0)` or `x = c ? T.new() : na`
        self.type_evidence = script_type_evidence(script)
        self.namespace_aliases: Dict[str, str] = {}
        # a library alias the compiled module binds itself -> its emitted name
        self.library_alias_spellings: Dict[str, str] = {}
        # user function named like a library name or a builtin series -> its emitted name (shared by its overloads)
        self.function_names: Dict[str, str] = {}
        # function or method name -> {renamed parameter: its emitted name}
        self.parameter_names: Dict[str, Dict[str, str]] = {}
        self.method_names = {node.name for node in script.declarations
                             if isinstance(node, FuncDecl) and node.is_method}
        self.user_function_names = {node.name for node in script.declarations
                                    if isinstance(node, FuncDecl) and not node.is_method}
        # Python reserved words used as Pine names get one spelling each, chosen against the
        # identifiers of the script (never its string literals or titles) and the names the
        # generated module defines itself
        self.spellings = KeywordSpellings(script, identifier_names(script) | EMITTED_HELPER_NAMES)
        # the declared UDT and enum names: the heads of `T.new(...)`, `Color.red`
        self.type_names = {node.name for node in script.declarations if isinstance(node, (TypeDecl, EnumDecl))}
        # the inner segments of the dotted names headed by a namespace, a library alias or a type (`risk` of
        # `strategy.risk.allow_entry_in`) and the receivers of a builtin method's dotted call (`days` of
        # `this.days.put(...)`); not a field path of a variable (`b` of `o.b.c`, `o.b.src(...)` of a user method)
        self.namespace_segments = inner_path_segments(
            script, set(PYNECORE_LIB_MODULES) | {imported.alias or imported.library for imported in script.imports}
            | self.type_names) | builtin_call_segments(script)
        # `commission` of `strategy.commission.percent`: a local named so is renamed in its block
        self.builtin_namespace_segments = inner_path_segments(script, PYNECORE_LIB_MODULES)
        self.builtin_call_segments = builtin_call_segments(script)
        # keyword-named UDT/enum -> its spelling, for type-hint tokens
        self.type_tokens = {name: self.spellings.variable(name) for name in sorted(self.type_names)
                            if name in PYTHON_RESERVED_WORDS}
        # a user type named like a built-in drawing type, whatever its case, is `Box__ren__`
        # (Trendoscope/Drawing `Line__ren__`, `Linefill__ren__`; `PolyLine` keeps its name)
        self.type_tokens.update({name: f'{name}{RENAMED_SUFFIX}' for name in self.type_names
                                 if name.lower() in BUILTIN_TYPE_NAMES and name not in self.type_tokens})
        # and one named like a pynecore namespace (`type session`): the class would hide `session.ismarket`
        self.type_tokens.update({name: f'{name}{RENAMED_SUFFIX}' for name in self.type_names
                                 if name in PYNECORE_LIB_MODULES and name not in self.type_tokens})
        # and, in a library, one an exported function of its name would hide (`export type necessaryData` beside
        # `export necessaryData(...)`): the class is spelled `necessaryData__ren__` and the exported
        # function gets a `new` attribute holding it, so `necessaryData.new(...)` still constructs
        exported_functions = {node.name for node in script.declarations
                              if isinstance(node, FuncDecl) and node.is_export and not node.is_method}
        self.type_tokens.update({name: f'{name}{RENAMED_SUFFIX}' for name in self.type_names
                                 if name in exported_functions and name not in self.type_tokens})
        # sanitized spellings of keyword-named inputs and exports: the settings id and the
        # library API keep them, so another binding of that spelling moves
        self.external_spellings: Set[str] = set()
        # keyword-named input whose spelling another input holds, or input named like a function or method of
        # the script -> its new name
        self.moved_inputs: Dict[str, str] = {}
        # the inputs named like a function or method of the script: they keep their Pine label
        self.inputs_named_like_a_function: Set[str] = set()
        self.moved_bindings = False
        # global named like a builtin series -> its first declaration: an input, a
        # var, an assignment or a tuple target
        self.builtin_globals: Dict[str, Any] = builtin_series_globals(script)
        # a binding named like a Python builtin the generated code may call -> its numbered spelling,
        # by (name, line, column) of the declaration or of the function a parameter belongs to
        version_text = getattr(script.version, 'version', '') if script.version else ''
        self.numbered_names = (NUMBERED_RESERVED_NAMES_V4 if str(version_text).isdigit() and int(version_text) <= 4
                               else NUMBERED_RESERVED_NAMES)
        self.numbered: Dict[tuple, str] = self.plan_numbered_names(script)
        self.plan_external_names(script)
        # function name -> its names hiding a global its callees read (pynecore passes it at the call)
        self.closure_shadows = closure_shadows(script)
        self.current_functions: List[str] = []
        # per function being visited: the outer bindings it has read so far
        self.outer_reads: List[Set[str]] = []
        self.plan_functions(script)

    # ---------------------------------------------------------------- names

    def fresh_name(self, name: str) -> str:
        return fresh_name(f'{name}{FRESH_NAME_SUFFIX}', self.used_names)

    def local_name(self, name: str) -> str:
        """`name_local`, else `name_local_1`, `name_local_2`, ... : the first the script does not hold."""
        candidate, index = f'{name}{FRESH_NAME_SUFFIX}', 0
        while candidate in self.used_names:
            index += 1
            candidate = f'{name}{FRESH_NAME_SUFFIX}_{index}'
        self.used_names.add(candidate)
        return candidate

    def numbered_name(self, name: str, at: Optional[tuple] = None) -> str:
        """The spelling of a binding named like a Python builtin the generated code may call: `range_1`, `range_2`,
        numbered in source order (`at`: the binding's (line, column), or its function's for a parameter)."""
        planned = self.numbered.get((name, *at)) if at is not None else None
        return planned or fresh_name(f'{name}_', self.used_names)

    def plan_numbered_names(self, script: Script) -> Dict[tuple, str]:
        """The numbered spelling of every binding named like NUMBERED_RESERVED_NAMES: the script's top-level
        bindings first, then the parameters and locals of its functions, each in source order."""
        found = []
        inside = {id(inner) for node in walk(script) if isinstance(node, FuncDecl) for inner in walk(node.body)}
        for node in walk(script):
            local = id(node) in inside
            if isinstance(node, FuncDecl):
                found += [(True, node.line, node.column, parameter.name) for parameter in node.params
                          if parameter.name in self.numbered_names]
                if node.name in self.numbered_names and not node.is_method:
                    found.append((local, node.line, node.column, node.name))
            elif isinstance(node, InputDecl) and node.name in self.numbered_names:
                found.append((local, node.line, node.column, node.name))
            elif isinstance(node, Assignment) and isinstance(node.target, str) \
                    and node.target in self.numbered_names:
                found.append((local, node.line, node.column, node.target))
            elif isinstance(node, (VarDecl, VaripDecl)) and node.name in self.numbered_names:
                found.append((local, node.line, node.column, node.name))
        # the script's own bindings first, then those of its functions, each in source order
        return {(name, line, column): fresh_name(f'{name}_', self.used_names)
                for _, line, column, name in sorted(set(found))}

    def global_name(self, name: str) -> str:
        """The spelling of a renamed global: `name__global__`."""
        spelling = f'{name}{GLOBAL_SUFFIX}'
        if spelling in self.used_names:
            return self.fresh_name(name)
        self.used_names.add(spelling)
        return spelling

    def block_local_name(self, name: str, block_id: int) -> str:
        """The spelling of a renamed local: `name__<its block's id>__`."""
        if not block_id:
            return self.fresh_name(name)
        spelling = hex_suffixed(name, block_id)
        if spelling in self.used_names:
            return self.fresh_name(name)
        self.used_names.add(spelling)
        return spelling

    def reserved_name(self, name: str) -> str:
        """The spelling of a binding named like something pynecore defines: one per name."""
        spelling = f'{name}{RENAMED_SUFFIX}'
        self.used_names.add(spelling)
        return spelling

    def lookup(self, name: str) -> Optional[Binding]:
        for depth in range(len(self.scopes) - 1, -1, -1):
            if name in self.scopes[depth]:
                if self.function_scopes and depth < self.function_scopes[-1] and self.outer_reads:
                    self.outer_reads[-1].add(name)  # the function reads an outer binding of this name
                return self.scopes[depth][name]
        return None

    def resolve(self, name: str) -> str:
        binding = self.lookup(name)
        return binding.emitted if binding else name

    def resolve_dotted(self, dotted: str, is_call: bool = False) -> str:
        """`t.bx` -> the emitted name of `t` plus `.bx`; `color.new` on a
        library-named binding `color` stays the namespace unless `new` is the
        binding's own member. The segments after a user binding or a declared
        type are its fields and members (a keyword spelling follows the field map);
        the last one of a call is a method."""
        first, dot, rest = dotted.partition('.')
        if dot and '.' not in rest and first in self.library_alias_names \
                and (rest in PYNECORE_RESERVED_NAMES or rest in SERIES_VARIABLES):
            # a library member named like a pynecore name is exported as `name__ren__` (plan_functions);
            # it is called so through the alias, `m.fill__ren__(x)`
            self.moved_bindings = True
            alias = self.lookup(first)
            return (alias.emitted if alias else first) + dot + rest + RENAMED_SUFFIX
        binding = self.lookup(first)
        library_spelling = self.library_alias_spellings.get(first, first)
        if is_call and dot and '.' not in rest and first in self.library_alias_names and binding is not None \
                and binding.emitted != library_spelling:
            # `Alert.AlertSender(...)` beside a variable named like the library alias `Alert`: the library's function
            return library_spelling + dot + rest
        if binding is not None and dot and first in self.type_names and rest.partition('.')[0] == 'new' \
                and binding.emitted != first:
            # `osc.new(...)` beside a variable named like the type `osc`: the constructor of the type
            return self.type_spelling(first) + dot + self.member_path(rest, is_call)
        if binding is None:
            if dot and first in self.type_names and not (
                    first in PYNECORE_LIB_MODULES and rest.partition('.')[0] not in ('new', 'copy')):
                # `session.new(...)` of `type session` is the user type; `session.ismarket` the namespace
                return self.type_spelling(first) + dot + self.member_path(rest, is_call)
            return dotted
        if dot and binding.collides and not self.member_rule.binding_owns(
                binding.type_name, first, rest.partition('.')[0]):
            return self.namespace(first) + dot + rest
        return binding.emitted + dot + self.member_path(rest, is_call)

    def member_path(self, path: str, is_call: bool = False) -> str:
        """The emitted spelling of `a.b.c` after the head: fields, and a user method last in a call."""
        if not path:
            return path
        *fields, last = path.split('.')
        last = self.method_spelling(last) if is_call else self.spellings.attribute(last)
        return '.'.join([self.spellings.attribute(name) for name in fields] + [last])

    def method_spelling(self, name: str) -> str:
        """A user method's emitted name; a builtin method keeps its own."""
        return self.function_names.get(name, name) if name in self.method_names else name

    def value_type(self, type_hint: Optional[str], value) -> Optional[str]:
        """The Pine type a declaration gives its binding (`binding_type`, with the script's evidence
        and the type of each variable in scope where the declaration stands)."""
        return binding_type(type_hint, value, self.type_evidence._replace(variable_type=self.variable_type))

    def variable_type(self, name: str) -> Optional[str]:
        binding = self.lookup(name)
        return binding.type_name if binding else None

    def type_spelling(self, name: str) -> str:
        return self.type_tokens.get(name, name)

    def type_hint(self, hint: Optional[str]) -> Optional[str]:
        """A type-hint string with the keyword-named user types it mentions respelled."""
        if hint is None or not (self.type_tokens or self.library_alias_spellings):
            return hint
        # a type a library exports (`lib.T`) is spelled through the alias's emitted name
        hint = re.sub(r'\b([A-Za-z_]\w*)(?=\.)',
                      lambda match: self.library_alias_spellings.get(match.group(1), match.group(1)), hint)
        return re.sub(r'\b[A-Za-z_]\w*\b', lambda match: self.type_tokens.get(match.group(0), match.group(0)), hint)

    def namespace(self, name: str) -> str:
        """How the namespace `name` is spelled here: as itself, or by an alias
        when a visible binding keeps the name (an input, an export)."""
        if not any(name in scope and scope[name].emitted == name for scope in self.scopes):
            return name
        return self.alias(name)

    def alias(self, name: str) -> str:
        """The alias the module imports the pynecore.lib name `name` by."""
        self.moved_bindings = True
        alias = self.namespace_aliases.get(name)
        if alias is None:
            alias = NAMESPACE_ALIAS.format(name)
            if alias in self.used_names:
                alias = fresh_name(alias, self.used_names)
            self.used_names.add(alias)
            self.namespace_aliases[name] = alias
        return alias

    def callee(self, name: str) -> str:
        """The emitted name of a bare call `name(...)`: the declared function of
        that name, which shadows every bare call of it; else the library
        function, by its alias when a visible input or export keeps the name
        (Pine keeps functions and variables apart, Python does not). A builtin
        function named like a builtin series (`time("D")`) is called by the alias
        beside a binding that keeps the name, or a global of that name."""
        if name in self.function_names:
            return self.function_names[name]
        builtin_function = name in CALLABLE_SERIES_VARIABLES and name not in self.user_function_names
        if builtin_function and name in self.builtin_globals:
            return self.alias(name)  # the module does not import it
        if name in self.library_names or builtin_function:
            return self.namespace(name)
        return name

    def declare(self, name: str, type_name: Optional[str] = None, at: Optional[tuple] = None,
                reads_itself: bool = False, tuple_element: bool = False) -> str:
        """Bind `name` in the innermost scope; a fresh name if it shadows an
        outer binding, collides with a library name, or hides a builtin series.
        `at`: the declaration's (line, column), for a numbered spelling; `reads_itself`: the declared value
        reads the name (`n = n * 2`), which is the outer binding there; `tuple_element`: an element of a tuple
        declaration (`[a, range] = f()`), which is not numbered (`range` stays `range`, `text` takes
        the spelling of a library name, `text__ren__`)."""
        if name == '_':
            return name  # Pine's discarded tuple element is never renamed
        if name in self.scopes[-1]:
            return self.scopes[-1][name].emitted
        # Only a binding of the same Python function is shadowed: a function's own local
        # keeps the name of a global it hides
        first = self.function_scopes[-1] if self.function_scopes else 0
        shadows = any(name in scope for scope in self.scopes[first:-1])
        # a declaration the enclosing blocks make later counts too, in a function and at the top level
        shadows = shadows or any(name in names for names in self.scope_declarations[first:-1])
        # in a block nested in a function, a local named like a global of the script takes its block's id, while
        # the function's own body keeps the name
        shadows = shadows or (bool(self.function_scopes) and len(self.scopes) - 1 > first
                              and name in self.top_level_names)
        # a local named like a global a callee of its function reads: pynecore would hand the callee this local
        shadows = shadows or (bool(self.current_functions)
                              and name in self.closure_shadows.get(self.current_functions[-1], ()))
        # a local declared after its function read the outer binding of its name: Python makes the name local to
        # the whole function, so the earlier read would fail (UnboundLocalError); Pine reads the outer one there
        shadows = shadows or (bool(self.outer_reads) and name in self.outer_reads[-1])
        if shadows:
            self.shadowed_names.add(name)
        # scopes[0] is the top level; every deeper scope is a function, a loop or a block
        # a variable named like a function or method of the script: Python reads the name as the variable all through
        # its function, so a call of the function in that scope reads the variable. At the top level too:
        # the functions are defined inside main(), where the variable is a local that hides them from every function
        names_a_function = (name in self.user_function_names or name in self.method_names) \
            and not self.function_names.get(name, '').endswith('__global__')  # a function spelled apart hides nothing
        if names_a_function:
            self.moved_bindings = True
        collides = name in self.library_names
        hides_builtin_series = name in SERIES_VARIABLES and len(self.scopes) > 1 and not (
            name in COMPUTED_SERIES and name not in self.library_names)
        if hides_builtin_series:
            self.moved_bindings = True
        # a global named like a type of the script, a namespace it uses or the alias of a library it imports
        # (`snd__global__`, `risk__global__`, `ma__global__` beside `import .../_matrix/5 as ma`): in main() the
        # variable would hide the module from every read of the alias in the function
        names_a_global_entity = len(self.scopes) == 1 and (name in self.type_names or name in self.namespace_segments
                                                           or name in self.library_alias_names)
        if name in self.numbered_names and not tuple_element:
            # `range_1`, `text_2`: numbered in source order, whatever else the name collides with
            self.moved_bindings = True
            emitted = self.numbered_name(name, at)
        elif collides and len(self.scopes) > 1 and (shadows or names_a_function):
            # a pynecore name that also hides a function or an outer binding: both spellings
            # (`size__ren____00000038__`, forex_factory_utility)
            emitted = hex_suffixed(self.reserved_name(name), self.scope_blocks[-1] if self.scope_blocks else 0)
        elif collides:
            emitted = self.reserved_name(name)
        elif len(self.scopes) == 1 and (names_a_function or names_a_global_entity):
            self.moved_bindings = True
            emitted = self.global_name(name)
        elif reads_itself and shadows and self.function_scopes and not hides_builtin_series \
                and not names_a_function:
            # a function's local whose value reads the outer binding of its name (`n = n * 2`, `m = switch m`):
            # `n_local`, then `n_local_1`, ... past every name the script holds
            self.moved_bindings = True
            emitted = self.local_name(name)
        elif shadows or hides_builtin_series or names_a_function or name in self.builtin_call_segments \
                or (len(self.scopes) > 1 and name in self.builtin_namespace_segments):
            # a local named like an inner segment of a dotted name (`commission` of
            # `strategy.commission.percent`) takes its block's spelling
            emitted = self.block_local_name(name, self.scope_blocks[-1] if self.scope_blocks else 0)
            self.moved_bindings = self.moved_bindings or emitted != name
        elif self.holds_external_spelling(name):
            self.moved_bindings = True
            emitted = self.fresh_name(name)
        else:
            emitted = self.spellings.variable(name)
        self.scopes[-1][name] = Binding(emitted, collides, type_name)
        return emitted

    def holds_external_spelling(self, name: str) -> bool:
        """Whether `name` is the spelling an input, an export or an export parameter keeps
        (`lambda_` for `lambda`): the binding that merely shares it moves."""
        if not name.endswith('_') or name[:-1] not in PYTHON_RESERVED_WORDS:
            return False
        return name in self.external_spellings or any(
            binding.emitted == name and key != name
            for scope in self.scopes for key, binding in scope.items())

    def keep(self, name: str, type_name: Optional[str]) -> Binding:
        """The binding of a name that must not change (an input, an export)."""
        return Binding(name, name in self.library_names, type_name)

    # ---------------------------------------------------------------- scopes

    def scoped_block(self, statements, declared: Iterable[str] = (),
                     parameters: Dict[str, Binding] = None, block_id: int = 0) -> Tuple[object, List[str]]:
        """Visit a nested scope. `parameters` bind as given; `declared` (loop
        variables) bind like declarations. Returns the new body and the
        emitted names of `declared`."""
        self.scopes.append(dict(parameters or {}))
        self.scope_blocks.append(block_id or self.block_numbers.of(statements))
        # what this block declares anywhere in it: a nested block's local shadows it even when the
        # declaration comes later (analytics_tables `float ret` after an inner loop's `float ret`)
        self.scope_declarations.append(_declared_in(statements))
        try:
            emitted_declared = [self.declare(name) for name in declared]
            if isinstance(statements, list):
                body = [self.statement(item) for item in statements]
            else:
                body = self.expression(statements)
        finally:
            self.scopes.pop()
            self.scope_blocks.pop()
            self.scope_declarations.pop()
        return body, emitted_declared

    def block(self, statements):
        if statements is None:
            return None
        return self.scoped_block(statements)[0]

    # ---------------------------------------------------------------- statements

    def statement(self, node):
        replace = dataclasses.replace
        if isinstance(node, Assignment):
            value = self.expression(node.value)
            if isinstance(node.target, TupleDestructure):
                names = [self.declare(name, tuple_element=True) for name in node.target.names]
                return replace(node, value=value, target=replace(node.target, names=names, value=value))
            return replace(node, value=value, type_hint=self.type_hint(node.type_hint),
                           target=self.declare(node.target, self.value_type(node.type_hint, node.value),
                                               (node.line, node.column), _reads_itself(node, node.target)))
        if isinstance(node, (VarDecl, VaripDecl)):
            value = self.expression(node.value)
            return replace(node, value=value, type_hint=self.type_hint(node.type_hint),
                           name=self.declare(node.name, self.value_type(node.type_hint, node.value),
                                             (node.line, node.column), _reads_itself(node, node.name)))
        if isinstance(node, Reassignment):
            return replace(node, value=self.expression(node.value),
                           target=self.resolve_dotted(node.target))
        if isinstance(node, IfStatement):
            return replace(node, condition=self.expression(node.condition),
                           body=self.block(node.body),
                           elseifs=[(self.expression(condition), self.block(body))
                                    for condition, body in node.elseifs],
                           else_body=self.block(node.else_body))
        if isinstance(node, (ForLoop, ForInLoop)):
            return self.loop(node)
        if isinstance(node, WhileLoop):
            return replace(node, condition=self.expression(node.condition),
                           body=self.block(node.body))
        if isinstance(node, SwitchStatement):
            return replace(node, expr=self.expression(node.expr),
                           cases=[(self.expression(case), self.block(body))
                                  for case, body in node.cases],
                           default=self.block(node.default))
        if isinstance(node, FuncDecl):
            return self.function(node)
        return self.expression(node)

    def loop(self, node):
        replace = dataclasses.replace
        if isinstance(node, ForLoop):
            from_value, to_value, step = (self.expression(node.from_val),
                                          self.expression(node.to_val),
                                          self.expression(node.step))
            body, (variable,) = self.scoped_block(node.body, declared=[node.var])
            return replace(node, var=variable, from_val=from_value, to_val=to_value,
                           step=step, body=body)
        iterable = self.expression(node.iterable)
        body, variables = self.scoped_block(node.body, declared=node.vars)
        return replace(node, vars=variables, iterable=iterable, body=body)

    def function(self, node: FuncDecl) -> FuncDecl:
        """A function's parameters and body share one scope. A default is
        evaluated where the function is declared, so it is visited first."""
        renames = self.parameter_names.get(node.name, {})
        parameters = []
        bindings: Dict[str, Binding] = {}
        for parameter in node.params:
            type_name = binding_type(parameter.type_hint, parameter.default)
            if parameter.name not in renames:
                binding = self.keep(parameter.name, type_name)
            else:
                emitted = renames[parameter.name]
                block_id = self.block_numbers.of(node)
                if HEX_RENAME.fullmatch(emitted[len(parameter.name):]) and block_id:
                    emitted = hex_suffixed(parameter.name, block_id)  # each overload its own block (chrono_utils)
                binding = Binding(emitted, parameter.name in self.library_names, type_name)
            bindings[parameter.name] = binding
            parameters.append(dataclasses.replace(parameter, name=binding.emitted,
                                                  type_hint=self.type_hint(parameter.type_hint),
                                                  default=self.expression(parameter.default)))
        self.function_scopes.append(len(self.scopes))
        self.current_functions.append(node.name)
        self.outer_reads.append(set())
        try:
            body, _ = self.scoped_block(node.body, parameters=bindings, block_id=self.block_numbers.of(node))
        finally:
            self.function_scopes.pop()
            self.current_functions.pop()
            self.outer_reads.pop()
        name = (self.function_names.get(node.name, node.name) if not node.is_export or node.name in self.renamed_exports
                else node.name)
        return dataclasses.replace(node, name=name, params=parameters, body=body,
                                   return_type=self.type_hint(node.return_type))

    def plan_functions(self, script: Script) -> None:
        """Fresh names for user functions named like a library name or a builtin
        series (every overload shares one) and their parameters named like a library name or a script global
        (one per function name, so a keyword argument maps the same way to every
        overload). An `export` declaration keeps its names."""
        global_names = script_global_names(script)
        # an `export` overload keeps its parameters, so its name's keywords must too
        exported = {declaration.name for declaration in script.declarations
                    if isinstance(declaration, FuncDecl) and declaration.is_export}
        # the functions spelled `name__global__` below: a name they had hides nothing any more
        spelled_apart = {declaration.name for declaration in script.declarations
                         if isinstance(declaration, FuncDecl) and not declaration.is_export
                         and self.spelled_as_a_global(declaration)}
        for declaration in script.declarations:
            if not isinstance(declaration, FuncDecl):
                continue
            name = declaration.name
            if declaration.is_export:
                if (name in PYNECORE_RESERVED_NAMES or name in SERIES_VARIABLES) and name not in self.function_names:
                    # an exported function or method named like a pynecore name is `fill__ren__`
                    self.function_names[name] = self.reserved_name(name)
                    self.renamed_exports.add(name)
                self.plan_export_parameters(declaration)
                # an exported parameter named like a pynecore name is renamed too (`order__ren__`)
                renames = self.parameter_names.setdefault(declaration.name, {})
                for parameter in declaration.params:
                    if parameter.name in self.library_names - SERIES_VARIABLES and parameter.name not in renames:
                        renames[parameter.name] = self.reserved_name(parameter.name)
                        if parameter.name in (self.user_function_names | self.method_names) - spelled_apart:
                            # and named like a function too (`method__ren____00000046__`, FunctionZigZagMultipleMethods)
                            renames[parameter.name] = hex_suffixed(renames[parameter.name],
                                                                   self.block_numbers.of(declaration))
                    elif (parameter.name in self.user_function_names | self.method_names | self.builtin_call_segments
                          and parameter.name not in renames):
                        # and one named like a function or method of the script, as any local (`smoothrng__0000001e__`)
                        renames[parameter.name] = self.block_local_name(parameter.name,
                                                                        self.block_numbers.of(declaration))
                continue
            name = declaration.name
            # a function named like a pynecore.lib name the script uses, or like a builtin series: the
            # module imports the builtin wherever its name is read, and pynecore reads the
            # library object for every use of an imported name, the function's calls included
            if self.spelled_as_a_global(declaration) and name not in self.function_names:
                # such a function is spelled `name__global__` (`def position__global__(...)`), and so is a method
                # named like a pynecore namespace (`method session(...)`: `session__global__`)
                self.function_names[name] = self.global_name(name)
            elif name not in self.function_names and name in self.numbered_names:
                self.function_names[name] = self.numbered_name(name, (declaration.line, declaration.column))
            elif name not in self.function_names and (
                    name in PYTHON_RESERVED_WORDS or name in self.external_spellings):
                # one spelling for every overload; a function holding an external spelling moves
                self.function_names[name] = (self.spellings.variable(name) if name in PYTHON_RESERVED_WORDS
                                             else self.fresh_name(name))
            renames = self.parameter_names.setdefault(name, {})
            # a parameter named like a script global keeps its name; one named like a
            # pynecore name is renamed, except a builtin series, which the function body reads as the parameter
            # (a parameter named `Series` keeps its name too, beside its own annotation: `def f(Series: Series)`)
            # a builtin series Pine also calls as a function (`time`, `hour`) is renamed even so (`time__ren__`)
            reserved_parameters = (self.library_names - SERIES_VARIABLES - {'Series'}) | SERIES_FUNCTIONS
            renamed_names = set(self.external_spellings)
            if name not in exported:
                # a parameter named like a function or method of the script hides it from the body
                renamed_names = renamed_names | self.user_function_names | self.method_names
            # and one named like an inner segment of a dotted call of a builtin (`this.msi.put(...)`: `msi__00000006__`)
            # or of a builtin's dotted name (`strategy.direction.long`: `direction__00000049__`)
            renamed_names = renamed_names | self.builtin_call_segments | self.builtin_namespace_segments
            for parameter in declaration.params:
                if parameter.name in self.numbered_names and parameter.name not in renames:
                    renames[parameter.name] = self.numbered_name(parameter.name, (declaration.line, declaration.column))
                elif parameter.name in reserved_parameters and parameter.name not in renames:
                    renames[parameter.name] = self.reserved_name(parameter.name)
                elif parameter.name in renamed_names and parameter.name not in renames:
                    renames[parameter.name] = self.block_local_name(parameter.name,
                                                                    self.block_numbers.of(declaration))
                elif parameter.name in PYTHON_RESERVED_WORDS and parameter.name not in renames:
                    renames[parameter.name] = self.spellings.variable(parameter.name)
                elif parameter.name in self.closure_shadows.get(name, ()) and parameter.name not in renames:
                    # named like a global a callee reads: pynecore would hand the callee this parameter (a method
                    # named like a builtin one is then chosen by the receiver's class where a call names it by
                    # keyword: each arm takes its own keyword, Transformer._route_method_call)
                    renames[parameter.name] = self.block_local_name(parameter.name, self.block_numbers.of(declaration))

    def spelled_as_a_global(self, declaration: FuncDecl) -> bool:
        """A function named like a pynecore.lib name the script binds or like a builtin series, and a method
        named like a pynecore namespace that the script binds or reads (`method box(...)` beside `box.new(...)`,
        which a function of its name would hide), and a function named like a type of the script: it is
        spelled `name__global__`."""
        name = declaration.name
        if declaration.is_method:
            return name in PYNECORE_LIB_MODULES and (name in self.library_names or name in self.read_namespaces)
        # a function named like a type of the script would hide its class: `foo__global__` beside `type foo`
        return name in self.library_names or name in SERIES_VARIABLES or name in self.type_names

    def plan_export_parameters(self, declaration: FuncDecl) -> None:
        """A keyword-named parameter of an `export` function keeps its API spelling
        (`lambda_`): the definition and the call keywords of this script use it."""
        raw_names = {parameter.name for parameter in declaration.params}
        renames = self.parameter_names.setdefault(declaration.name, {})
        for parameter in declaration.params:
            if parameter.name in PYTHON_RESERVED_WORDS and parameter.name not in renames:
                spelling = rename_reserved_word(parameter.name)
                if spelling in raw_names:
                    raise UnsupportedFeatureError(
                        f'export function {declaration.name!r}: parameters {parameter.name!r} and '
                        f'{spelling!r} would both be emitted as {spelling!r}', declaration.line)
                renames[parameter.name] = spelling

    def plan_external_names(self, script: Script) -> None:
        """The spellings an input or an export holds. An input keeps its raw name in the
        tree (codegen spells it, as before, and its settings id is that spelling); a
        keyword-named input whose spelling another input holds was a SyntaxError, so it
        takes a fresh name."""
        input_names = {node.name for node in script.declarations if isinstance(node, InputDecl)}
        for node in script.declarations:
            if isinstance(node, InputDecl) and node.name in self.numbered_names:
                # `range_1=input.int(...)`, `return_1=input(...)`
                self.moved_inputs[node.name] = self.numbered_name(node.name, (node.line, node.column))
        for name in sorted((input_names & PYTHON_RESERVED_WORDS) - self.moved_inputs.keys()):
            spelling = rename_reserved_word(name)
            if spelling in input_names:
                self.moved_inputs[name] = free_spelling(spelling, self.spellings.occupied)
            else:
                self.external_spellings.add(spelling)
        # an input named like a function or method of the script: the input is a parameter of main() and the
        # script's functions are defined inside main(), so the function overwrites the parameter
        self.inputs_named_like_a_function = {name for name in input_names
                                             if name in self.user_function_names or name in self.method_names}
        # an input named like a pynecore name is renamed like any binding (`extend__ren__=input.bool(...)`)
        for name in sorted((input_names & self.library_names) - self.moved_inputs.keys()):
            self.moved_inputs[name] = self.reserved_name(name)
        for name in sorted(self.inputs_named_like_a_function - self.moved_inputs.keys()):
            self.moved_inputs[name] = self.global_name(name)  # `correction__global__`
        # an input named like a library alias (`import .../AlertSenderLibrary/1 as Alert`) is a global too
        aliases = {imported.alias or imported.library for imported in script.imports}
        for name in sorted((input_names & aliases) - self.moved_inputs.keys()):
            self.moved_inputs[name] = self.global_name(name)
        # an input named like an inner segment of a dotted name is a global like any other
        # (`risk__global__=input.float(...)` beside `strategy.risk.max_intraday_loss`)
        for name in sorted((input_names & self.namespace_segments) - self.moved_inputs.keys()):
            self.moved_inputs[name] = self.global_name(name)
        exported = {node.name for node in script.declarations
                    if isinstance(node, FuncDecl) and node.is_export}
        for name in sorted(exported & PYTHON_RESERVED_WORDS):
            spelling = rename_reserved_word(name)
            if spelling in exported or spelling in input_names:
                raise UnsupportedFeatureError(
                    f'export function {name!r} would be emitted as {spelling!r}, a name the script '
                    f'declares as an input or another export', 0)
            self.external_spellings.add(spelling)
        self.spellings.occupied.update(self.external_spellings)

    # ---------------------------------------------------------------- expressions

    def expression(self, node):
        replace = dataclasses.replace
        if node is None or isinstance(node, (str, int, float, bool)):
            return node
        if isinstance(node, list):
            if any(isinstance(item, (Statement, VarDecl, VaripDecl)) for item in node):
                return self.block(node)
            return [self.expression(item) for item in node]
        if isinstance(node, tuple):
            return tuple(self.expression(item) for item in node)
        if isinstance(node, dict):
            return {key: self.expression(value) for key, value in node.items()}
        if isinstance(node, Identifier):
            return replace(node, name=self.resolve_dotted(node.name))
        if isinstance(node, MemberAccess):
            return self.member_access(node)
        if isinstance(node, FunctionCall) and isinstance(node.func, str):
            return self.function_call(node)
        if isinstance(node, MethodCall):
            # `x.t.m(...)` on a call result reaches here with method 't.m'
            head, dot, member = node.method.rpartition('.')
            callee = member if member in self.method_names else None
            method = self.member_path(head) + dot + self.method_spelling(member) if dot else self.method_spelling(member)
            return replace(node, object=self.expression(node.object), method=method,
                           args=self.expression(node.args), kwargs=self.keywords(callee, node.kwargs))
        if isinstance(node, IfExpression):
            return replace(node, condition=self.expression(node.condition),
                           true_expr=self.block(node.true_expr),
                           false_expr=self.block(node.false_expr))
        if isinstance(node, SwitchExpression):
            return replace(node, expr=self.expression(node.expr),
                           cases=[(self.expression(case), self.block(body))
                                  for case, body in node.cases],
                           default=self.block(node.default))
        if isinstance(node, SCOPED_STATEMENT_TYPES):
            return self.statement(node)
        if isinstance(node, ASTNode):
            return replace(node, **{field.name: self.expression(getattr(node, field.name))
                                    for field in dataclasses.fields(node)
                                    if field.name not in ('line', 'column')})
        return node

    def function_call(self, node: FunctionCall) -> FunctionCall:
        """A bare name is a function; a dotted one may be a method call on a
        renamed variable, or a namespace call."""
        constructs = False
        if '.' in node.func:
            # `array.new<lib.T>`: the call is one token; its type argument is a type hint
            callee_path, bracket, type_argument = node.func.partition('<')
            head = callee_path.partition('.')[0]
            if head in PYNECORE_LIB_MODULES and self.lookup(head) is self.scopes[0].get(head) \
                    and getattr(node, 'line', 0) and self.global_lines.get(head, 0) > node.line:
                # `label.delete(x)` above `var table label = ...`: a global is read only below its declaration, so
                # the call is the namespace's
                func = callee_path + bracket + self.type_hint(type_argument)
            else:
                func = self.resolve_dotted(callee_path, is_call=True) + bracket + self.type_hint(type_argument)
            # `t.m(...)`: only a user method's keywords can name a renamed parameter;
            # `array.get(a, index = 0)` calls the namespace, whatever user method shares the name
            member = node.func.rpartition('.')[2]
            callee = member if member in self.method_names and not self.names_a_namespace(node.func, func) else None
            head, _, constructor = node.func.partition('.')
            constructs = constructor == 'new' and head in self.type_names and self.lookup(head) is None
        else:
            func, callee = self.callee(node.func), node.func
        return dataclasses.replace(node, func=func, args=self.expression(node.args),
                                   kwargs=self.keywords(callee, node.kwargs, constructs))

    def member_access(self, node: MemberAccess) -> MemberAccess:
        """`a.b` as a node: the member is a field of a user value (a name that resolves to a user
        binding or declared type, or any computed value), never of a builtin namespace."""
        if not isinstance(node.object, str):
            return dataclasses.replace(node, object=self.expression(node.object),
                                       member=self.spellings.attribute(node.member))
        head = node.object.partition('.')[0]
        binding = self.lookup(head)
        member = node.member if binding is None and head not in self.type_names else self.spellings.attribute(node.member)
        return dataclasses.replace(node, object=self.resolve_dotted(node.object), member=member)

    def names_a_namespace(self, pine_func: str, emitted_func: str) -> bool:
        """Whether a dotted call is emitted as a namespace call (`array.get`,
        `chart.point.from_index`, an aliased namespace), not a call on a binding."""
        emitted_head = emitted_func.partition('.')[0]
        if emitted_head in self.namespace_aliases.values():
            return True
        binding = self.lookup(pine_func.partition('.')[0])
        return emitted_head in NAMESPACE_HEADS and (binding is None or binding.emitted != emitted_head)

    def keywords(self, callee: Optional[str], kwargs: dict, constructs: bool = False) -> dict:
        """Keyword arguments, visited; one that names a renamed parameter of
        the called user function or method `callee` takes the parameter's new name,
        and one of a UDT constructor `T.new(...)` follows the field map."""
        kwargs = self.expression(kwargs)
        if constructs:
            return {self.spellings.attribute(keyword): value for keyword, value in kwargs.items()}
        renames = self.parameter_names.get(callee)
        if not renames:
            return kwargs
        return {renames.get(keyword, keyword): value for keyword, value in kwargs.items()}

    # ---------------------------------------------------------------- script

    def global_names(self, script: Script) -> None:
        """Bind every global up front: a function body may read one. Globals bind in source
        order (`global_declarations`), so a value is typed from the globals declared above it."""
        for declaration in global_declarations(script):
            if isinstance(declaration, (VarDecl, VaripDecl)):
                self.declare(declaration.name, self.value_type(declaration.type_hint, declaration.value),
                             (declaration.line, declaration.column))
            elif isinstance(declaration, InputDecl):
                binding = self.keep(declaration.name, input_type(declaration))
                if declaration.name in self.moved_inputs:
                    binding = dataclasses.replace(binding, emitted=self.moved_inputs[declaration.name])
                self.scopes[0][declaration.name] = binding
            elif isinstance(declaration.target, TupleDestructure):
                for name in declaration.target.names:
                    self.declare(name, tuple_element=True)
            else:
                self.declare(declaration.target, self.value_type(declaration.type_hint, declaration.value),
                             (declaration.line, declaration.column))

    def global_declaration(self, declaration):
        replace = dataclasses.replace
        if isinstance(declaration, FuncDecl):
            return self.function(declaration)
        if isinstance(declaration, (VarDecl, VaripDecl)):
            return replace(declaration, name=self.resolve(declaration.name),
                           type_hint=self.type_hint(declaration.type_hint),
                           value=self.own_value(declaration, [declaration.name]))
        if isinstance(declaration, InputDecl):
            return self.input_declaration(declaration)
        if isinstance(declaration, TypeDecl):
            return self.type_declaration(declaration)
        if isinstance(declaration, EnumDecl):
            return self.enum_declaration(declaration)
        return declaration

    def input_declaration(self, declaration: InputDecl) -> InputDecl:
        """An input keeps its name. Its arguments are read when the module loads, before the
        input exists, so its own name there is the builtin of that name: a builtin
        series, which the import resolver does not import for the input's reads, is read by
        the namespace alias."""
        name = declaration.name
        saved = self.read_builtin_by_alias({name} & self.builtin_globals.keys(),
                                           [declaration.args, declaration.kwargs])
        arguments = self.expression(declaration.args), self.expression(declaration.kwargs)
        self.scopes[0].update(saved)
        args, kwargs = arguments
        if name in self.inputs_named_like_a_function and len(args) < 2 and 'title' not in kwargs:
            # Pine shows the variable name when the call gives no title: the rename must not change the label
            kwargs = {**kwargs, 'title': Literal(value=name, literal_type='string', is_double_quoted=True)}
        return dataclasses.replace(declaration, name=self.moved_inputs.get(name, name), args=args, kwargs=kwargs)

    def read_builtin_by_alias(self, names: Set[str], tree) -> Dict[str, Binding]:
        """Until the caller restores the returned bindings, each input of `names` that `tree`
        reads stands for the builtin of its name, spelled by the namespace alias."""
        saved = {}
        for name in sorted(names):
            if any(isinstance(node, Identifier) and node.name.partition('.')[0] == name for node in walk(tree)):
                saved[name] = self.scopes[0][name]
                self.scopes[0][name] = Binding(self.namespace(name))
                self.moved_bindings = True
        return saved

    def own_value(self, declaration, names) -> Any:
        """The value of a global's declaration. The global is declared once its declaration ends,
        so where this is the first declaration of a name like a builtin series, the value's read
        of that name is the builtin: `close = close * 2`."""
        first = {name for name in names if self.builtin_globals.get(name) is declaration}
        saved = self.read_builtin_by_alias(first, declaration.value)
        value = self.expression(declaration.value)
        self.scopes[0].update(saved)
        return value

    def in_source_order(self, visit, item):
        """Visit a top-level item; a global named like a builtin series and declared after it is
        not visible there yet, so the item's reads of that name are the builtin."""
        declared_later = {name for name, declaration in self.builtin_globals.items()
                          if 0 < item.line < declaration.line}
        saved = self.read_builtin_by_alias(declared_later, item)
        result = visit(item)
        self.scopes[0].update(saved)
        return result

    def type_declaration(self, declaration: TypeDecl) -> TypeDecl:
        """A UDT's name and fields in their emitted spelling; a field's type may name a keyword-named UDT."""
        fields = [(self.spellings.attribute(field[0]), self.type_hint(field[1]), *self.field_default(field[2:]))
                  for field in declaration.fields]
        return dataclasses.replace(declaration, name=self.type_spelling(declaration.name), fields=fields)

    def field_default(self, default_part: tuple) -> tuple:
        """A field's default, read as it stands in the module: a dotted constant headed by a renamed library alias
        (`lib.Mode.a`) starts with the alias's emitted name. Every other default is left as written: a
        built-in series (`close`) stays the built-in, whatever a later global or input of its name does."""
        if default_part and isinstance(default_part[0], Identifier):
            default = default_part[0]
            head, dot, _ = default.name.partition('.')
            if dot and head in self.library_alias_spellings:
                return (dataclasses.replace(default, name=self.resolve_dotted(default.name)), *default_part[1:])
        return default_part

    def enum_declaration(self, declaration: EnumDecl) -> EnumDecl:
        """An enum's name and members in their emitted spelling. A member without a value
        stands for its own name as a string; it keeps that value when its attribute moves."""
        members = []
        for member in declaration.members:
            if isinstance(member, tuple):
                members.append((self.spellings.attribute(member[0]), *member[1:]))
            elif self.spellings.attribute(member) != member:
                members.append((self.spellings.attribute(member),
                                Literal(value=member, literal_type='string', is_double_quoted=True)))
            else:
                members.append(member)
        return dataclasses.replace(declaration, name=self.type_spelling(declaration.name), members=members)

    def global_statement(self, statement):
        """A top-level assignment declares a global, already bound by global_names."""
        if not isinstance(statement, Assignment):
            return self.statement(statement)
        target = statement.target
        value = self.own_value(statement, target.names if isinstance(target, TupleDestructure) else [target])
        if isinstance(statement.target, TupleDestructure):
            names = [self.resolve(name) for name in statement.target.names]
            return dataclasses.replace(statement, value=value, target=dataclasses.replace(
                statement.target, names=names, value=value))
        return dataclasses.replace(statement, value=value, target=self.resolve(statement.target),
                                   type_hint=self.type_hint(statement.type_hint))

    def library_aliases(self, script: Script) -> Dict[str, str]:
        """Bind every `import user/lib/version as alias` whose alias (the library name without one) the compiled
        module binds itself to a fresh name: alias -> its emitted name. Every use of the alias then
        resolves like any renamed global."""
        for imported in script.imports:
            name = imported.alias or imported.library
            if name in self.library_alias_spellings:
                continue
            if name in LIBRARY_ALIAS_RENAMED_NAMES:
                self.library_alias_spellings[name] = self.fresh_name(name)
            elif name in PYTHON_RESERVED_WORDS:
                self.library_alias_spellings[name] = self.spellings.variable(name)  # `as class`: `class__ren__`
            else:
                continue
            self.scopes[0][name] = Binding(emitted=self.library_alias_spellings[name])
        return self.library_alias_spellings

    def run(self, script: Script) -> Script:
        self.scopes = [{}]
        self.scope_declarations = [set()]
        # the line of each global's first declaration
        self.global_lines: Dict[str, int] = {}
        for item in [*script.declarations, *script.body]:
            name = getattr(item, 'target', None) if isinstance(item, Assignment) else getattr(item, 'name', None)
            if isinstance(name, str) and getattr(item, 'line', 0):
                self.global_lines[name] = min(self.global_lines.get(name, item.line), item.line)
        renamed_aliases = self.library_aliases(script)
        self.global_names(script)
        declarations = [self.in_source_order(self.global_declaration, declaration)
                        for declaration in script.declarations]
        body = [self.in_source_order(self.global_statement, statement) for statement in script.body]
        if not (self.shadowed_names or self.library_names or any(self.parameter_names.values())
                or self.spellings.changed or self.moved_bindings or self.moved_inputs or self.function_names
                or renamed_aliases or any(name != spelling for name, spelling in self.type_tokens.items())):
            return script
        imports = [dataclasses.replace(imported, alias=renamed_aliases[imported.alias or imported.library])
                   if (imported.alias or imported.library) in renamed_aliases else imported
                   for imported in script.imports]
        return dataclasses.replace(script, declarations=declarations, body=body, imports=imports)
