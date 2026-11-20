"""Emitted spellings that must not collide with the names a script already uses.

Pine names that are Python reserved words (`lambda`, `from`, `pass`) cannot be
emitted as they are. They used to be spelled by `sanitize_identifier` (`lambda_`)
independently at each place, so `lambda` and a user's `lambda_` became one name,
and a reference that the escaping missed (a call keyword, a field read) stayed
invalid Python. Here a spelling is chosen ONCE per name against the complete set
of names the script holds: the preferred spelling is taken only when it is free,
else the lowest free numeric suffix (`lambda_1`).

Two spelling domains, because Python keeps them apart:
- variables, parameters, functions, methods and type names share one scope, so
  `variable()` avoids every identifier of the script;
- fields and enum members are attributes, so `attribute()` avoids only attribute
  names (declared fields, enum members, and the tail of dotted names). A field
  `lambda` therefore still becomes `lambda_` next to a variable `lambda_`.

Both maps are uniform: one Pine name has one spelling in a domain whatever the
receiver's type, which is what lets an untyped `p.lambda` read the right field.

What never occupies a spelling: a string literal or a script title (`"lambda_"`).
"""
import dataclasses
import re
from typing import Dict, Set

from .pine_builtins import PYNECORE_LIB_MODULES, PYTHON_RESERVED_WORDS, rename_reserved_word

# Names the generated module imports or defines itself; no generated spelling may equal one
MODULE_HELPER_NAMES = frozenset({
    'Series', 'Persistent', 'udt', 'method', 'pine_range', 'pine_loop', 'method_call', 'na', 'main', 'script',
})
EMITTED_HELPER_NAMES = MODULE_HELPER_NAMES | frozenset(PYNECORE_LIB_MODULES)

# A library alias spelled like one of these is renamed (scope_renamer): `lib` is the pynecore library object pynecore's
# own transformers read in a compiled module, the rest are the helpers the module imports. A namespace or series name
# is not here: `import x as ta` is pynecore's shadowed_namespace, which a rename would drop (import_resolver refuses it)
LIBRARY_ALIAS_RENAMED_NAMES = MODULE_HELPER_NAMES | {'lib'}

# Fields that hold text, not names: a title or a literal is not an identifier
_TEXT_FIELDS = {('IndicatorDecl', 'title'), ('StrategyDecl', 'title'), ('LibraryDecl', 'title'),
                ('RawCode', 'code')}
_TEXT_NODES = {'Literal', 'ImportDecl'}


def identifier_names(tree) -> Set[str]:
    """Every identifier-bearing string of the tree (first and later segments of a
    dotted name, targets, fields, parameters, type-hint tokens, keyword names). A
    string literal, a title or an import is not one."""
    names: Set[str] = set()
    _identifiers_in(tree, names)
    return names


def _identifiers_in(value, names: Set[str]) -> None:
    if isinstance(value, str):
        names.update(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', value))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _identifiers_in(item, names)
    elif isinstance(value, dict):
        for key, item in value.items():
            _identifiers_in(key, names)
            _identifiers_in(item, names)
    elif dataclasses.is_dataclass(value) and type(value).__name__ not in _TEXT_NODES:
        kind = type(value).__name__
        for field in dataclasses.fields(value):
            if field.name in ('line', 'column', 'literal_type') or (kind, field.name) in _TEXT_FIELDS:
                continue
            _identifiers_in(getattr(value, field.name), names)


def attribute_names(tree) -> Set[str]:
    """Names that are attributes: the declared fields and enum members, and every
    segment after the head of a dotted name."""
    names: Set[str] = set()
    _attributes_in(tree, names)
    return names


def _attributes_in(value, names: Set[str]) -> None:
    kind = type(value).__name__
    if isinstance(value, str):
        names.update(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', value.partition('.')[2]))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _attributes_in(item, names)
    elif isinstance(value, dict):
        for item in value.values():
            _attributes_in(item, names)
    elif dataclasses.is_dataclass(value) and kind not in _TEXT_NODES:
        if kind == 'TypeDecl':
            names.update(field[0] for field in value.fields)
        elif kind == 'EnumDecl':
            names.update(member[0] if isinstance(member, tuple) else member for member in value.members)
        elif kind == 'MemberAccess':
            names.add(value.member)
        for field in dataclasses.fields(value):
            if field.name in ('line', 'column', 'literal_type') or (kind, field.name) in _TEXT_FIELDS:
                continue
            _attributes_in(getattr(value, field.name), names)


def free_spelling(preferred: str, occupied: Set[str]) -> str:
    """`preferred` when no name in `occupied` holds it, else the lowest free
    numeric suffix of it; the result is recorded as occupied."""
    spelling, index = preferred, 0
    while spelling in occupied:
        index += 1
        spelling = f'{preferred}{index}'
    occupied.add(spelling)
    return spelling


class KeywordSpellings:
    """The spelling of every Pine name that is a Python reserved word, in the two domains."""

    def __init__(self, tree, occupied: Set[str]):
        # shared with the renamer: identifiers of the script, helper names, names planned so far
        self.occupied = occupied
        self.attribute_occupied = attribute_names(tree)
        self.variables: Dict[str, str] = {}
        self.attributes: Dict[str, str] = {}
        # a UDT field named like a pynecore namespace or function is `name__ren__` wherever it is read
        # (`this.map__ren__`, `table__ren__`, `bar.time__ren__`); a series that is no function keeps its name
        # (`bar.close`, `bar.timenow`, `bar.ask`)
        from .ast_nodes import TypeDecl
        from .pine_builtins import PYNECORE_RESERVED_NAMES, SERIES_FUNCTIONS, SERIES_VARIABLES
        renamed = (PYNECORE_RESERVED_NAMES - SERIES_VARIABLES - {'ask', 'bid', 'updatetime', 'Series'}) | SERIES_FUNCTIONS
        fields = {field[0] for node in getattr(tree, 'declarations', ()) if isinstance(node, TypeDecl)
                  for field in node.fields}
        self.reserved_fields = fields & renamed
        # a field whose name starts with two underscores is `name__ren___`: Python mangles such a name in
        # the class body (`_T__p`), so the plain spelling read outside the class would name no field
        self.private_fields = {name for name in fields if name.startswith('__') and not name.endswith('__')}

    def variable(self, name: str) -> str:
        """The spelling of a variable, parameter, function, method or type named `name`."""
        if name not in PYTHON_RESERVED_WORDS:
            return name
        if name not in self.variables:
            self.variables[name] = free_spelling(rename_reserved_word(name), self.occupied)
        return self.variables[name]

    def attribute(self, name: str) -> str:
        """The spelling of a field or enum member named `name`, whatever its type."""
        if name in self.reserved_fields:
            return f'{name}__ren__'
        if name in self.private_fields:
            return f'{name}__ren___'
        if name not in PYTHON_RESERVED_WORDS:
            return name
        if name not in self.attributes:
            self.attributes[name] = free_spelling(rename_reserved_word(name), self.attribute_occupied)
        return self.attributes[name]

    @property
    def changed(self) -> bool:
        return bool(self.variables or self.attributes or self.reserved_fields or self.private_fields)
