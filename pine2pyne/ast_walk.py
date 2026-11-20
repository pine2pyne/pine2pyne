"""
AST walk: the child traversal shared by the Pine-AST pre-passes and scans.

A Pine AST node is a dataclass; its children sit in its fields, directly or
inside lists, tuples (the arms of an else-if, the cases of a switch) and dicts
(the keyword arguments of a call). A walk that forgets one of these containers
silently misses what is inside it: a history index in a keyword argument or in
an else-if arm was not seen. Every generic walk goes through
`walk` (to find nodes) or `map_children` (to rebuild them), so the containers
are decided in one place.

Every transpile walks every file, so `walk` is iterative (a deep expression
does not use up the recursion limit) and caches the child fields per node type.
"""
import dataclasses
from typing import Callable, Dict, Iterator, Tuple, Union

# The source position is not a child
POSITION_FIELDS = frozenset({'line', 'column'})
# Field values that are never a node nor hold one
LEAF_TYPES = frozenset({str, int, float, bool, type(None)})
CONTAINER_TYPES = frozenset({list, tuple})

# node type -> its child field names, last first; False for a type that is not a node
_CHILD_FIELDS_LAST_FIRST: Dict[type, Union[Tuple[str, ...], bool]] = {}
# node type -> its child field names in declaration order
_CHILD_FIELDS_IN_ORDER: Dict[type, Tuple[str, ...]] = {}


def child_field_names(node_type: type) -> Tuple[str, ...]:
    """The fields of a node type that can hold children, in declaration order."""
    fields = _CHILD_FIELDS_IN_ORDER.get(node_type)
    if fields is None:
        fields = _CHILD_FIELDS_IN_ORDER[node_type] = tuple(reversed(_child_fields_last_first(node_type) or ()))
    return fields


def _child_fields_last_first(value_type: type):
    fields = _CHILD_FIELDS_LAST_FIRST.get(value_type)
    if fields is None:
        is_node_type = hasattr(value_type, '__dataclass_fields__')
        fields = tuple(reversed([field.name for field in dataclasses.fields(value_type)
                                 if field.name not in POSITION_FIELDS])) if is_node_type else False
        _CHILD_FIELDS_LAST_FIRST[value_type] = fields
    return fields


def is_node(value) -> bool:
    """True for an AST node instance (a dataclass instance, not a class)."""
    return _child_fields_last_first(type(value)) is not False and not isinstance(value, type)


def walk(tree) -> Iterator:
    """Every node in `tree` (a node, or lists, tuples and dicts of them),
    each parent before its children, in source order."""
    pending = [tree]
    while pending:
        value = pending.pop()
        value_type = type(value)
        if value_type in CONTAINER_TYPES:
            pending.extend(item for item in reversed(value) if type(item) not in LEAF_TYPES)
        elif value_type is dict:
            pending.extend(item for item in reversed(list(value.values())) if type(item) not in LEAF_TYPES)
        elif value_type not in LEAF_TYPES:
            fields = _child_fields_last_first(value_type)
            if fields is False:
                continue
            yield value
            for name in fields:
                child = getattr(value, name)
                if type(child) not in LEAF_TYPES:
                    pending.append(child)


def contains(tree, predicate: Callable) -> bool:
    """True when a node in `tree` satisfies `predicate`."""
    for node in walk(tree):
        if predicate(node):
            return True
    return False


def map_children(value, rebuild: Callable):
    """`value` with `rebuild` applied to each node it holds directly: a list,
    tuple or dict is copied around the rebuilt items, anything else that is
    not a node (a name, a number) is kept."""
    if isinstance(value, list):
        return [map_children(item, rebuild) for item in value]
    if isinstance(value, tuple):
        return tuple(map_children(item, rebuild) for item in value)
    if isinstance(value, dict):
        return {key: map_children(item, rebuild) for key, item in value.items()}
    return rebuild(value) if is_node(value) else value
