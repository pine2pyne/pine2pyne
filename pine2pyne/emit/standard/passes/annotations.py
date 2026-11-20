"""Type annotations the way pine2pyne writes them.

* A name is a series (``Series``, ``PersistentSeries``) only when its scope
  reads it with history (``x[1]``); a ``var`` otherwise is ``Persistent``.
* A declaration (the first assignment of a name in a scope; later ones are
  Pine ``:=``) is annotated when its type is known without inference: the Pine
  source types it (``float x = ...``) or its value says it — a literal, a hex
  colour, ``array.new_float()``, ``array.from_items('a', 'b')``, ``line.all``.
* A series or persistent annotation keeps its element type (``Series[float]``)
  only then; otherwise it is bare (``Series``).
"""
from __future__ import annotations

import ast
import re

from .base import PassContext, function_scopes, main_function, own_nodes
from ....pine_builtins import PYNECORE_LIB_MODULES
from ....import_resolver import library_module_path

_LITERAL_TYPES = {bool: 'bool', int: 'int', float: 'float', str: 'str'}
_ARRAY_ELEMENTS = {
    'new_float': 'float', 'new_int': 'int', 'new_bool': 'bool', 'new_string': 'str', 'new_color': 'Color',
    'new_line': 'Line', 'new_label': 'Label', 'new_box': 'Box', 'new_table': 'Table',
    'new_linefill': 'LineFill', 'new_polyline': 'Polyline',
}
_ALL_COLLECTIONS = {'line': 'Line', 'label': 'Label', 'box': 'Box', 'table': 'Table',
                    'linefill': 'LineFill', 'polyline': 'Polyline'}
_SERIES_KINDS = {'Series': None, 'PersistentSeries': 'Persistent', 'IBPersistentSeries': 'IBPersistent'}
_PERSISTENT_KINDS = ('Series', 'PersistentSeries', 'IBPersistentSeries', 'Persistent', 'IBPersistent')


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return f'{_dotted(node.value)}.{node.attr}'
    return node.id if isinstance(node, ast.Name) else ''


def literal_type(value: ast.expr | None) -> str | None:
    if isinstance(value, ast.Constant):
        return _LITERAL_TYPES.get(type(value.value))
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'na'
            and len(value.args) == 1 and isinstance(value.args[0], ast.Name)
            and value.args[0].id in ('int', 'float', 'str', 'bool')):
        return value.args[0].id  # a typed na, `na(float)`
    if (isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub)
            and isinstance(value.operand, ast.Constant) and type(value.operand.value) is int):
        return 'int'
    return None


#: the type of a main() parameter by its input function
_INPUT_TYPES = {'input.string': 'str', 'input.timeframe': 'str', 'input.symbol': 'str', 'input.session': 'str',
                'input.text_area': 'str', 'input.color': 'Color', 'input.bool': 'bool', 'input.float': 'float',
                'input.int': 'int'}
#: the scalar types an `array.from` of them or an `array.get` from them is annotated with
_COLLECTED_TYPES = frozenset({'str', 'Color'})


def _scalar_type(value: ast.expr, types: dict) -> str | None:
    """`str`, `Color`, `bool` or a number type for a value typed so in an `array.from`: a string literal, a
    `string.*` call that gives a string, a hex colour (`color.new('#0980FF')`), or a name this scope typed (an
    input, an annotated declaration)."""
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return 'str'
    if isinstance(value, ast.Constant) and type(value.value) in (int, float, bool):
        return literal_type(value)  # `array.from(i1, 2)` of an int input and a literal
    if isinstance(value, ast.Call) and _dotted(value.func) == 'color.new' and len(value.args) == 1 \
            and not value.keywords and isinstance(value.args[0], ast.Constant) and isinstance(value.args[0].value, str):
        return 'Color'
    if isinstance(value, ast.Name) and types.get(value.id) in ('bool', 'float', 'int'):
        return types[value.id]
    if isinstance(value, ast.Name) and re.fullmatch(r'Persistent\[(bool|float|int|str)\]', types.get(value.id, '')):
        return types[value.id][len('Persistent['):-1]  # `var float a = 0.0`

    if isinstance(value, ast.Call) and _dotted(value.func) in ('string.format', 'string.tostring', 'string.upper',
                                                               'string.lower', 'string.replace_all'):
        return 'str'
    if isinstance(value, ast.Name) and types.get(value.id) in _COLLECTED_TYPES:
        return types[value.id]
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add) \
            and 'str' in (_scalar_type(value.left, types), _scalar_type(value.right, types)):
        return 'str'  # a concatenation (`hi + "\n" + "A"`)
    return None


_DRAWING_CONSTRUCTORS = {'table.new': 'Table', 'line.new': 'Line', 'label.new': 'Label', 'box.new': 'Box'}


def _collection_type(value: ast.expr, types: dict) -> str | None:
    """`list[str]` for `array.from(...)` of strings (or of colours), `str` for `array.get(a, i)` of a
    `list[str]` this scope declared; a number, bool or colour array's element is not annotated."""
    if not isinstance(value, ast.Call):
        return None
    function = _dotted(value.func)
    if function == 'array.from_items' and value.args and not value.keywords and any(
            (isinstance(argument, ast.Constant) and argument.value is not None)
            or (isinstance(argument, ast.Name) and _scalar_type(argument, types) is not None
                and not re.fullmatch(r'__input_\d+__', argument.id))  # an inline input does not anchor it
            or (isinstance(argument, ast.Call) and _scalar_type(argument, types) == 'Color')  # `color.new('#0980FF')`
            for argument in value.args):
        # annotated where a literal, a hex colour or a typed name is among its elements, typed by the elements
        # that carry a type (a string call too); a bare na, a ternary or a price series (`close`) types nothing:
        # `array.from(close, a)` with `var float a` is a list[float], an array of string calls alone is not annotated
        kinds = {_scalar_type(argument, types) for argument in value.args} - {None}
        if kinds == {'int', 'float'}:
            kinds = {'float'}  # `array.from(0, 10e6)`: list[float]
        if len(kinds) == 1:
            return f'list[{kinds.pop()}]'
    if function == 'array.get' and value.args and isinstance(value.args[0], ast.Name):
        # `list[str]`, or a var's `Persistent[list[str]]`
        match = re.fullmatch(r'(?:\w*Persistent\w*\[)?list\[(\w+)\]\]?', types.get(value.args[0].id, ''))
        if match and match.group(1) == 'str':  # a colour element is not annotated (ichimoku_kinko_hyo)
            return match.group(1)
    return None


def _library_constructed_type(value: ast.expr, aliases: frozenset = frozenset()) -> str | None:
    """`lib.T` of `lib.T.new(...)`, a library type's constructor (`IQ.m2022.new()`). A library's own function named
    `new` (`tl.new(...)`) is read as the constructor of a type named like the alias: `tl.tl`."""
    if isinstance(value, ast.Call):
        function = _dotted(value.func)
        if function.endswith('.new') and function.count('.') == 2 and function.split('.')[0] not in PYNECORE_LIB_MODULES:
            return function[:-len('.new')]
        if function.count('.') == 1 and function.endswith('.new') and function.split('.')[0] in aliases:
            alias = function.split('.')[0]
            return f'{alias}.{alias}'
    return None


def _typed_na_argument(call: ast.Call) -> str | None:
    """T of the `NA(T)` value of `array.new(n, NA(T))` / `matrix.new(..., initial_value=NA(T))`."""
    position = 2 if _dotted(call.func) == 'matrix.new' else 1  # `matrix.new(r, c, NA(T))`
    values = list(call.args[position:position + 1]) + [keyword.value for keyword in call.keywords
                                                        if keyword.arg == 'initial_value']
    for value in values:
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'NA'
                and len(value.args) == 1 and isinstance(value.args[0], ast.Name)):
            return value.args[0].id
    return None


def inferable_type(value: ast.expr | None, classes: frozenset = frozenset(),
                   methods: frozenset = frozenset()) -> str | None:
    """The type read off a value without inference, or None. ``classes``:
    the module's UDT classes, whose constructor call names its type; ``methods``: the user
    methods, whose call on a constructor (`T.new().m()`) is typed `T`."""
    # a negative number is a sign applied to a literal, which is not typed (`a = -1` stays bare)
    found = literal_type(value) if isinstance(value, ast.Constant) else None
    if found:
        return found
    if isinstance(value, ast.Attribute) and _dotted(value) in ('chart.fg_color', 'chart.bg_color'):
        return 'Color'  # the chart's colours
    if isinstance(value, ast.Attribute) and value.attr == 'all' and isinstance(value.value, ast.Name):
        element = _ALL_COLLECTIONS.get(value.value.id)
        return f'list[{element}]' if element else None
    if not isinstance(value, ast.Call):
        return None
    function = _dotted(value.func)
    if function in classes:
        return function
    if function.endswith('.new') and function.count('.') == 2 and function.split('.')[1][:1].isupper():
        return function[:-len('.new')]  # a library's type, `dr.AuxData.new(...)` (biased_price_target)
    receiver = None
    if function == 'method_call' and len(value.args) >= 2:
        receiver = value.args[1]
    elif function in methods and value.args:
        receiver = value.args[0]
    if isinstance(receiver, ast.Attribute) and isinstance(receiver.value, ast.Call):
        receiver = receiver.value  # a method on a field of a constructor (`T.new().a.m()`) is typed as `T`
    if isinstance(receiver, ast.Call):
        # a method called on `array.from(...)` of literals is typed as that array, whatever it returns
        # (`array.from('a', 'b').includes(x)` is `list[str]`)
        return inferable_type(receiver, classes, methods) if _dotted(receiver.func) in classes | methods \
            or _dotted(receiver.func) in ('method_call', 'array.from_items') else None
    if function == 'color.new' and len(value.args) == 1 and not value.keywords \
            and isinstance(value.args[0], ast.Constant) and isinstance(value.args[0].value, str):
        return 'Color'  # a hex colour literal
    if function == 'matrix.new' and len(value.args) >= 3 and literal_type(value.args[2]):
        return f'Matrix[{literal_type(value.args[2])}]'  # the element type of the initial value
    if function == 'matrix.new' and len(value.args) >= 3 and isinstance(value.args[2], ast.Call) \
            and isinstance(value.args[2].func, ast.Name) and value.args[2].func.id == 'cast_float':
        return 'Matrix[float]'  # `matrix.new<float>(r, c, source)`
    if function == 'matrix.new' and len(value.args) >= 3 and isinstance(value.args[2], ast.Call) \
            and _dotted(value.args[2].func) in _DRAWING_CONSTRUCTORS:
        return f'Matrix[{_DRAWING_CONSTRUCTORS[_dotted(value.args[2].func)]}]'  # `matrix.new(1, 1, table.new(...))`
    typed_na = _typed_na_argument(value)
    if function in ('matrix.new', 'array.new') and typed_na:
        return f"{'Matrix' if function == 'matrix.new' else 'list'}[{typed_na}]"  # `NA(T)`: of T
    if function == 'array.new' and len(value.args) >= 2 and isinstance(value.args[1], ast.Call) \
            and _dotted(value.args[1].func) in classes:
        return f'list[{_dotted(value.args[1].func)}]'  # `array.new(n, T(...))`: of the type T
    if function == 'array.new' and len(value.args) >= 2 and isinstance(value.args[1], ast.Call):
        filler = _dotted(value.args[1].func)
        if filler.startswith('chart.point.'):
            return 'list[ChartPoint]'  # `array.new(n, chart.point.now(...))`
        if filler == 'na' and len(value.args[1].args) == 1 and isinstance(value.args[1].args[0], ast.Subscript):
            return ast.unparse(value.args[1].args[0])  # `array.new(n, na(list[T]))`: `list[T]`
    if function.startswith('array.'):
        element = _ARRAY_ELEMENTS.get(function[len('array.'):])
        if element:
            return f'list[{element}]'
        if function == 'array.from_items' and value.args:
            types = {literal_type(argument) for argument in value.args}
            if len(types) == 1 and None not in types:
                return f'list[{types.pop()}]'
    return None


class DeclaredTypes:
    """The Python types of the Pine source's typed declarations (``float x = ...``),
    looked up by name and, when the statement's Pine line is known, by that line."""

    def __init__(self, script, library_aliases: dict | None = None) -> None:
        from ....ast_nodes import Assignment, VarDecl, VaripDecl
        from ....ast_walk import walk
        from ....transformer import convert_pine_generic_to_python
        self.by_line: dict = {}
        self.by_name: dict = {}
        self.declared_lines: dict = {}
        for node in walk([getattr(script, 'declarations', []), getattr(script, 'body', [])]):
            if isinstance(node, (Assignment, VarDecl, VaripDecl)):
                name = node.target if isinstance(node, Assignment) else node.name
                if not isinstance(name, str):
                    continue
                line = getattr(node, 'line', 0)
                self.declared_lines.setdefault(name, set()).add(line)
                type_hint = getattr(node, 'type_hint', None) or self._generic_constructor_type(node.value)
                if type_hint:
                    if library_aliases:  # a type a library exports (`lib.T`), through the alias's emitted name
                        type_hint = re.sub(r'\b([A-Za-z_]\w*)(?=\.)',
                                           lambda match: library_aliases.get(match.group(1), match.group(1)),
                                           type_hint)
                    python_type = convert_pine_generic_to_python(type_hint)
                    self.by_line[(name, line)] = python_type
                    self.by_name.setdefault(name, python_type)

    @staticmethod
    def _generic_constructor_type(value) -> str | None:
        """`map.new<string, bool>()` names its type, `map<string, bool>`; `T.copy(x)` names `T`
        (Trendoscope/Zigzag `lPivot: Pivot = udt_copy(...)`)."""
        from ....ast_nodes import FunctionCall
        if isinstance(value, FunctionCall) and isinstance(value.func, str) and value.func.startswith('map.new<'):
            return 'map' + value.func[len('map.new'):]
        if isinstance(value, FunctionCall) and isinstance(value.func, str) and value.func.endswith('.copy') \
                and value.func.count('.') == 1 and value.func[:1].isupper() and len(value.args) == 1:
            return value.func[:-len('.copy')]
        return None

    #: the suffixes renaming gives a Pine name: `x__000000c0__`, `x__ren__`, `x__global__`
    _RENAMED = re.compile(r'__[0-9a-f]{8}__$|__ren__$|__global__$')

    def get(self, name: str, line: int = 0):
        name = self._RENAMED.sub('', name)
        if line and line in self.declared_lines.get(name, ()):
            return self.by_line.get((name, line))
        return self.by_name.get(name)

    def __contains__(self, name: str) -> bool:
        return self._RENAMED.sub('', name) in self.by_name


def pine_declared_types(script, tree: ast.Module | None = None) -> DeclaredTypes:
    return DeclaredTypes(script, library_aliases(script, tree) if tree is not None else None)


def library_aliases(script, tree: ast.Module) -> dict[str, str]:
    """The Pine name of each imported library (its alias, else its name) -> the name the module reads it by,
    where the two differ: `import a/pick/1 as lib` is `import lib.a.pick.v1 as lib_local1`, and a library
    under a builtin namespace's name reads through `<name>__ren__` (`shadowed_namespace`)."""
    emitted = {alias.name: alias.asname for node in tree.body if isinstance(node, ast.Import)
               for alias in node.names if alias.asname}
    aliases = {}
    for declaration in getattr(script, 'imports', None) or ():
        module = library_module_path(declaration.user, declaration.library, declaration.version)
        name = emitted.get(module)
        if name is None:
            continue
        name = name[:-len('__lib')] if name.endswith('____lib') else name
        pine_name = declaration.alias or declaration.library
        if name != pine_name:
            aliases[pine_name] = name
    return aliases


class SeriesAnnotationPass:
    """A name is a series only where its scope reads it with history: otherwise
    ``PersistentSeries[T]`` is ``Persistent[T]`` and a ``Series`` annotation goes."""

    @staticmethod
    def _history_reads(scope: ast.AST) -> set[str]:
        """Names `scope` reads with history, its nested functions included where they do
        not bind the name themselves (a parameter `min` is not main's `min`)."""
        reads = {node.value.id for node in own_nodes(scope)
                 if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)}
        for node in own_nodes(scope):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args
                bound = {arg.arg for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
                bound |= {child.id for child in own_nodes(node)
                          if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)}
                reads |= SeriesAnnotationPass._history_reads(node) - bound
        return reads

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        # the receiver of a user method whose result is read with history (`x.src(s)[1]`, not `src(x, s)[1]`)
        receivers = getattr(context.unit.transformed, 'method_history_receivers', set())
        for scope in function_scopes(tree):
            read_with_history = self._history_reads(scope) | receivers
            blocks = [node for node in own_nodes(scope) if hasattr(node, 'body')
                      and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            for owner in [scope] + blocks:
                for field in ('body', 'orelse', 'finalbody'):
                    block = getattr(owner, field, None)
                    if isinstance(block, list):
                        # a name declared in a block is the block's own: a series only where that block reads it
                        # with history (`x = ...` in an else branch beside `x[1]` of the if branch's `x` is not)
                        reads = read_with_history if owner is scope else self._block_history_reads(block) | receivers
                        block[:] = [self._demoted(statement, reads) for statement in block]

    @staticmethod
    def _block_history_reads(block: list) -> set[str]:
        return {node.value.id for statement in block for node in ast.walk(statement)
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)}

    @staticmethod
    def _demoted(statement: ast.stmt, read_with_history: set) -> ast.stmt:
        if (isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
                and statement.target.id in read_with_history):
            # a varip read with history: spelled IBPersistentSeries, pynecore would keep it a plain
            # persistent (its PersistentSeries transformer splits PersistentSeries only), so `x[1]` indexes the
            # value. PersistentSeries runs, and a historical bar has no intrabar update for varip to keep
            annotation = statement.annotation
            head = annotation.value if isinstance(annotation, ast.Subscript) else annotation
            if isinstance(head, ast.Name) and head.id == 'IBPersistentSeries':
                head.id = 'PersistentSeries'
            return statement
        if not (isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)):
            return statement
        annotation = statement.annotation
        head = annotation.value if isinstance(annotation, ast.Subscript) else annotation
        if not isinstance(head, ast.Name) or head.id not in _SERIES_KINDS:
            return statement
        replacement = _SERIES_KINDS[head.id]
        if replacement is None:  # a plain series: no annotation
            if statement.value is None:
                return statement
            return ast.copy_location(ast.Assign(targets=[statement.target], value=statement.value), statement)
        if isinstance(annotation, ast.Subscript):
            annotation.value = ast.Name(id=replacement, ctx=ast.Load())
        else:
            statement.annotation = ast.Name(id=replacement, ctx=ast.Load())
        return statement


def _unquoted(annotation) -> str:
    """An annotation's text, a string annotation (`'list[Pivot]'`) unquoted."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    return ast.unparse(annotation)


def _element_of_kind(annotation: str | None) -> str | None:
    """`Persistent[Pivot]`, `Series[Pivot]` -> `Pivot`."""
    if annotation is None:
        return None
    match = re.fullmatch(r'(?:Persistent|PersistentSeries|Series|IBPersistent|IBPersistentSeries)\[(.*)\]', annotation)
    return match.group(1) if match else annotation


class DeclarationAnnotationPass:
    """Declarations carry a type known without inference; series and persistent annotations
    keep their element type only then (see the module docstring)."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        typed = pine_declared_types(context.unit.pine_script, tree)
        # a type the module renames (`type order` is `class order__ren__`) is spelled so in the declared types too
        renamed = {node.name[:-len('__ren__')]: node.name for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name.endswith('__ren__')}
        if renamed:
            pattern = re.compile(r'\b(' + '|'.join(map(re.escape, renamed)) + r')\b(?!__)')
            for table in (typed.by_line, typed.by_name):
                for key, hint in list(table.items()):
                    if isinstance(hint, str):
                        table[key] = pattern.sub(lambda match: renamed[match.group(1)], hint)
        self._origins = context.origins
        self._classes = frozenset(node.name for node in tree.body if isinstance(node, ast.ClassDef))
        self._methods = frozenset(node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                                  and any(isinstance(d, ast.Name) and d.id == 'method' for d in node.decorator_list))
        # the aliases of the imported libraries (`import lib.X.Y.v3 as tl`)
        self._library_aliases = frozenset(alias.asname or alias.name for node in tree.body
                                          if isinstance(node, ast.Import)
                                          for alias in node.names if alias.name.startswith('lib.'))
        # each UDT's fields and their annotations: a field read is typed by its field
        self._fields = {node.name: {field.target.id: _unquoted(field.annotation) for field in node.body
                                    if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)}
                        for node in tree.body if isinstance(node, ast.ClassDef)}
        main = main_function(tree)
        main_inputs: dict[str, str] = {}
        if main is not None:
            for arg, default in zip(reversed(main.args.args), reversed(main.args.defaults)):
                kind = _INPUT_TYPES.get(_dotted(default.func)) if isinstance(default, ast.Call) else None
                if kind:
                    main_inputs[arg.arg] = kind
        nested = {id(node) for node in ast.walk(main) if isinstance(node, ast.FunctionDef) and node is not main} \
            if main is not None else set()
        scopes = list(function_scopes(tree))
        if main is not None and main in scopes:  # main first: its names' types reach the functions in it
            scopes.remove(main)
            scopes.insert(0, main)
        main_types: dict[str, str] = dict(main_inputs)
        for scope in scopes:
            declared: set[str] = set()
            types: dict[str, str] = {}
            if id(scope) in nested:
                # a function in main() reads main's inputs and typed declarations so, where it binds no name of its
                # own so (`levels = array.from(fib0, fib1)` with `fib0=input.float(...)`, `array.get(names, i)`)
                bound = {arg.arg for arg in scope.args.posonlyargs + scope.args.args + scope.args.kwonlyargs}
                bound |= {node.id for node in own_nodes(scope) if isinstance(node, ast.Name)
                          and isinstance(node.ctx, ast.Store)}
                types.update({name: kind for name, kind in main_types.items() if name not in bound})
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = scope.args
                for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs:
                    declared.add(arg.arg)
                    if arg.annotation is not None:
                        types[arg.arg] = _unquoted(arg.annotation)
                # main()'s inputs are typed by their input function (`tf1=input.timeframe(...)` is a str)
                for arg, default in zip(reversed(arguments.args), reversed(arguments.defaults)):
                    kind = _INPUT_TYPES.get(_dotted(default.func)) if isinstance(default, ast.Call) else None
                    if kind and arg.arg not in types:
                        types[arg.arg] = kind
            self._annotate(scope.body, declared, typed, types)
            if scope is main:
                main_types.update(types)

    _classes: frozenset = frozenset()
    _methods: frozenset = frozenset()
    _fields: dict = {}
    _origins: dict = {}

    def _known(self, name: str, value, typed, statement=None) -> str | None:
        line = self._origins.get(getattr(statement, 'lineno', 0), 0) if statement is not None else 0
        return typed.get(name, line) or inferable_type(value, self._classes, self._methods)

    def _bare_unless_known(self, statement: ast.AnnAssign, typed: dict[str, str],
                           types: dict[str, str] | None = None) -> None:
        annotation = statement.annotation
        if isinstance(annotation, ast.Name) and annotation.id in _PERSISTENT_KINDS:
            # a bare kind takes the type of a constructor's method chain (`var w = T.new().init()`), of a
            # library type's constructor (`var s = lib.T.new()`) (chrono_utils, biased_price_target) or of the
            # field it reads (`cl: Series[float] = c.c`, Trendoscope/oota)
            element = self._constructed_type(statement.value) or self._field_type(statement.value, typed, types or {})
            if element is None and isinstance(statement.value, ast.Call) \
                    and _dotted(statement.value.func) == 'array.from_items':
                element = _collection_type(statement.value, types or {})  # `var a = array.from(b1, b2)` of inputs
            if element is None and isinstance(statement.value, ast.Call) and _dotted(statement.value.func) == 'array.get':
                element = _collection_type(statement.value, types or {})  # `Series[str]` of a string array's element
            if element:
                statement.annotation = ast.Subscript(value=annotation, slice=ast.parse(element, mode='eval').body,
                                                     ctx=ast.Load())
            return
        if not (isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name)
                and annotation.value.id in _PERSISTENT_KINDS):
            return
        if isinstance(statement.value, ast.Call) and _dotted(statement.value.func) == 'array.from_items' \
                and _collection_type(statement.value, types or {}):
            # `var a = array.from(b1, b2)` of bool inputs: `Persistent[list[bool]]`
            annotation.slice = ast.parse(_collection_type(statement.value, types or {}), mode='eval').body
            return
        if not (self._known(statement.target.id, statement.value, typed, statement)
                or self._field_type(statement.value, typed, types or {})
                or self._constructed_type(statement.value)):
            statement.annotation = ast.Name(id=annotation.value.id, ctx=ast.Load())
        elif literal_type(statement.value) and statement.target.id not in typed:
            # a literal names the element type itself: `x: Series[float] = 0.0`
            annotation.slice = ast.Name(id=literal_type(statement.value), ctx=ast.Load())

    def _constructed_type(self, value) -> str | None:
        if isinstance(value, ast.Call):
            function = _dotted(value.func)
            if function.endswith('.new') and function.count('.') == 2 \
                    and function.split('.')[0] not in PYNECORE_LIB_MODULES:
                return function[:-len('.new')]  # `dr.DistSettings.new(...)`, `IQ.sweeps.new(...)`
            if function in self._methods or function == 'method_call':
                found = inferable_type(value, self._classes, self._methods)
                if found is None and function == 'method_call' and len(value.args) > 1:
                    # a library method on a library type's constructor (`chr.DateTimeWindow.new().init(...)`): the
                    # library type (the library's method is read as returning its receiver)
                    found = self._constructed_type(value.args[1])
                return found
            if function == 'array.new' and len(value.args) > 1 and isinstance(value.args[1], ast.Call) \
                    and _dotted(value.args[1].func) == 'NA' and len(value.args[1].args) == 1:
                element = _dotted(value.args[1].args[0])
                if element.count('.') == 1 and element.split('.')[0] not in PYNECORE_LIB_MODULES:
                    return f'list[{element}]'  # `array.new<lib.T>()`: list[lib.T]
        return None

    def _field_type(self, value, typed, types: dict[str, str]) -> str | None:
        """The declared type of the UDT field `value` reads (`p.x`, `a.b.c`), when its receiver's type is known."""
        if not isinstance(value, ast.Attribute):
            return None
        receiver = value.value
        if isinstance(receiver, ast.Name):
            owner = types.get(receiver.id)  # only a type this scope declares (Zigzag `llastPivot` is untyped)
        else:
            owner = self._field_type(receiver, typed, types)
        owner = _element_of_kind(owner)
        return self._fields.get(owner, {}).get(value.attr) if owner else None

    def _annotate(self, body: list[ast.stmt], declared: set[str], typed: dict[str, str],
                  types: dict[str, str] | None = None) -> None:
        types = {} if types is None else types
        for index, statement in enumerate(body):
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
                    and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                value = statement.value
                if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) \
                        and re.fullmatch(r'__loop_\d+__', value.value.id):
                    declared.add(name)  # a `pine_loop` variable's value is never annotated (analytics_tables)
                    continue
                annotation = self._known(name, statement.value, typed, statement)
                if annotation is None and name not in declared:
                    annotation = (self._field_type(statement.value, typed, types)
                                  or _collection_type(statement.value, types)
                                  or _library_constructed_type(statement.value, self._library_aliases))
                if annotation is None and name not in declared and _scalar_type(value, types) == 'str':
                    types[name] = 'str'  # unannotated, but a string to an `array.from` of it (`s = a + "b"`)
                if name not in declared and annotation and not name.startswith('__'):
                    body[index] = ast.copy_location(ast.AnnAssign(
                        target=statement.targets[0], annotation=ast.parse(annotation, mode='eval').body,
                        value=statement.value, simple=1), statement)
                    types[name] = annotation
                declared.add(name)
            elif isinstance(statement, ast.Assign):  # `a, b = f()` declares its names
                declared.update(node.id for target in statement.targets for node in ast.walk(target)
                                if isinstance(node, ast.Name))
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                self._bare_unless_known(statement, typed, types)
                declared.add(statement.target.id)
                types[statement.target.id] = _unquoted(statement.annotation)
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                # a loop variable is the loop's own: a declaration of its name after the loop is annotated
                loop_names = {node.id for node in ast.walk(statement.target) if isinstance(node, ast.Name)}
                if isinstance(statement.target, ast.Name) and isinstance(statement.iter, ast.Name):
                    # `for sr in levels` over a `list[level]`: `sr` is a level, so `start = sr.start` takes the
                    # field's type
                    element = re.fullmatch(r'list\[(\w+(?:\.\w+)?)\]', _element_of_kind(types.get(statement.iter.id)) or '')
                    if element:
                        types[statement.target.id] = element.group(1)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                declared.add(statement.name)
                continue
            for name in ('body', 'orelse', 'finalbody'):
                nested = getattr(statement, name, None)
                if isinstance(nested, list):
                    # a Pine block is a scope: what it declares is unknown to its siblings
                    inner = set(declared)
                    if isinstance(statement, (ast.For, ast.AsyncFor)) and name == 'body':
                        inner |= loop_names
                    self._annotate(nested, inner, typed, dict(types))


_CLASS_SPELLINGS = {'footprint': 'Footprint', 'volume_row': 'VolumeRow'}


class BuiltinClassNamePass:
    """`footprint` / `volume_row` named as a type, in an annotation or a typed na (`na(T)`, `NA(T)`), are the
    runtime's classes `Footprint` / `VolumeRow`: `bar_print: Footprint = na(Footprint)`. The namespace
    of their functions keeps its name (`footprint.buy_volume(bar_print)`)."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        def respell(node) -> None:
            for name in ast.walk(node):
                if isinstance(name, ast.Name) and name.id in _CLASS_SPELLINGS:
                    name.id = _CLASS_SPELLINGS[name.id]
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                respell(node.annotation)
            elif isinstance(node, ast.arg) and node.annotation is not None:
                respell(node.annotation)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('na', 'NA') \
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                respell(node.args[0])


class ColorCastPass:
    """A colour value is not wrapped in ``Color(...)``: ``GREEN: Color = color.new('#00FF00')``."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        for node in ast.walk(tree):
            for field, value in ast.iter_fields(node):
                if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'Color'
                        and len(value.args) == 1 and not value.keywords and isinstance(value.args[0], ast.Call)):
                    setattr(node, field, value.args[0])


def _in_pine_order(node: ast.AST, statements: frozenset = frozenset()):
    """`node` and every node below it, in the order pine2pyne reads them: a ternary's test before its arms
    (`c ? x : y` is `x if c else y`); an if or while written as one in Pine (`statements`, by identity) its body
    before its condition, then its else branch. The cases of a switch are read condition first."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, ast.IfExp):
            children = [current.test, current.body, current.orelse]
        elif id(current) in statements:
            children = [*current.body, current.test, *current.orelse]
        else:
            children = list(ast.iter_child_nodes(current))
        stack.extend(reversed(children))


_PINE_CONDITIONAL = re.compile(r'^(?:else\s+)?if\b|^while\b|=\s*if\b')


def _pine_conditionals(function: ast.FunctionDef, context: PassContext) -> frozenset:
    """The ifs and whiles of `function` written as an if or a while in Pine (not a switch lowered to one), by
    identity; an `elif` follows the `if` it continues."""
    lines = (context.unit.pine_source or '').splitlines()
    found = set()
    for node in ast.walk(function):
        if not isinstance(node, (ast.If, ast.While)) or id(node) in found:
            continue
        line = context.origins.get(node.lineno, 0)
        if not (0 < line <= len(lines) and _PINE_CONDITIONAL.search(lines[line - 1].strip())):
            continue
        found.add(id(node))
        branch = node
        while isinstance(branch, ast.If) and len(branch.orelse) == 1 and isinstance(branch.orelse[0], ast.If) \
                and not context.origins.get(branch.orelse[0].lineno):
            branch = branch.orelse[0]
            found.add(id(branch))
    return frozenset(found)


def _input_default_is_a_series(call: ast.Call) -> bool:
    """`input(close, ...)` / `input(defval=open, ...)`: a source input."""
    default = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == 'defval'), None)
    return isinstance(default, ast.Name) and default.id not in ('True', 'False', 'na')


class ClosureHistoryPass:
    """A function that reads the history of a variable of its enclosing function
    (``avg[1]`` with ``avg`` a main() variable) reads it from its own series,
    ``__hist_avg__: Series = avg[0]``, bound first in its body. The snapshot of a
    tuple's element (`a, b = f()`) is `Series[void]`, one of a Pine v4 source input (`src=input(close)`) is a bare
    `Series`."""

    PREFIX, SUFFIX = '__hist_', '__'

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        main = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main'), None)
        if main is not None:
            # the snapshot keeps the element type the variable is declared with in main():
            # `x: PersistentSeries[float]` or `x: float` gives `__hist_x__: Series[float]`
            self._series_types = {}
            declared = [(arg.arg, arg.annotation) for arg in main.args.args]
            declared += [(node.target.id, node.annotation) for node in own_nodes(main)
                         if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
            # a Pine v4 source input (`src=input(close)`) is snapshotted untyped, a tuple's element `Series[void]`
            sources = {arg.arg for arg, default in zip(reversed(main.args.args), reversed(main.args.defaults))
                       if isinstance(default, ast.Call) and isinstance(default.func, ast.Name)
                       and default.func.id == 'input' and _input_default_is_a_series(default)}
            for name, annotation in declared:
                element = self._element_type(annotation)
                if element is not None and name not in self._series_types and name not in sources:
                    self._series_types[name] = ast.Subscript(value=ast.Name(id='Series', ctx=ast.Load()),
                                                             slice=element, ctx=ast.Load())
            for node in own_nodes(main):
                # Pine v6: the elements of a call's tuple (`[m, s, h] = ta.macd(...)`, `[a, b] = f()`) are snapshotted
                # `Series`; those of an if/switch value (`[a, b] = switch c`) keep `Series[void]`, as v5 does for any
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple) \
                        and (context.pine_version < 6 or not isinstance(node.value, ast.Call)):
                    for element in node.targets[0].elts:
                        if isinstance(element, ast.Name):
                            self._series_types.setdefault(element.id, ast.Subscript(
                                value=ast.Name(id='Series', ctx=ast.Load()), slice=ast.Name(id='void', ctx=ast.Load()),
                                ctx=ast.Load()))
            self._context = context
            self._visit(main, set())

    _series_types: dict = {}
    _context = None

    @staticmethod
    def _element_type(annotation):
        if annotation is None:
            return None
        if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name) \
                and annotation.value.id in _PERSISTENT_KINDS:
            return annotation.slice
        if isinstance(annotation, ast.Name) and annotation.id not in _PERSISTENT_KINDS:
            return annotation
        return None

    @classmethod
    def _snapshot(cls, name: str) -> str:
        """`__hist_x__` for the variable `x`, spelled by its Pine name (`osc__global__` is `__hist_osc__`)."""
        if name.endswith('__global__'):
            name = name[:-len('__global__')]
        return f'{cls.PREFIX}{name}{cls.SUFFIX}'

    @staticmethod
    def _own_bindings(function: ast.FunctionDef) -> set[str]:
        arguments = function.args
        bound = {arg.arg for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
        for node in own_nodes(function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                bound.add(node.name)
        return bound

    def _visit(self, function: ast.FunctionDef, enclosing: set[str]) -> None:
        own = self._own_bindings(function)
        if enclosing:
            # each history read moves its name's snapshot first in the body, read in Pine order (an if's
            # body before its condition): the name of the last read comes first
            snapshots: list[str] = []
            statements = _pine_conditionals(function, self._context) if self._context is not None else frozenset()
            for node in _in_pine_order(function, statements):
                if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                        and node.value.id in enclosing and node.value.id not in own):
                    if node.value.id in snapshots:
                        snapshots.remove(node.value.id)
                    snapshots.insert(0, node.value.id)
            if snapshots:
                for node in ast.walk(function):
                    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                            and node.value.id in snapshots):
                        node.value = ast.Name(id=self._snapshot(node.value.id), ctx=ast.Load())
                first = function.body[0] if function.body else function
                function.body[:0] = [ast.copy_location(ast.AnnAssign(
                    target=ast.Name(id=self._snapshot(name), ctx=ast.Store()),
                    annotation=self._series_types.get(name) or ast.Name(id='Series', ctx=ast.Load()),
                    value=ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()), slice=ast.Constant(0),
                                        ctx=ast.Load()), simple=1), first) for name in snapshots]
        for node in own_nodes(function):
            if isinstance(node, ast.FunctionDef):
                self._visit(node, (enclosing | own) - self._own_bindings(node))
