"""
`once` lowering: Pine v6's `once` structure as a persistent flag plus an `if`.

    once close > mySMA            var bool once_fired1 = false
        printLabel(...)     ->    if not once_fired1 and close > mySMA
                                      once_fired1 := true
                                      printLabel(...)

This is the equivalent code the Pine v6 User Manual gives for `once`
("Conditional structures", section "`once` structure"): the block runs the
first time its condition is true and never again. The condition is optional
and `true` by default, so a bare `once` checks only its flag. `and`
short-circuits, so the condition is not evaluated once the structure fired
("evaluated only once ... and then never again after that"). The flag is set
before the block, so a `break` or `continue` in it cannot skip that.

The flag is a `var`, so it has a `var`'s scope: one per function call site,
one for all iterations of a loop, initialised the first time its block runs.
The lowering runs at the end of parsing, so no later stage sees a `once`.
"""
import dataclasses

from .ast_nodes import (
    BinaryOp, Identifier, IfStatement, Literal, OnceStatement, Reassignment,
    Script, UnaryOp, VarDecl,
)
from .ast_walk import contains, is_node
from .scope_renamer import collect_names, fresh_name

FLAG_STEM = 'once_fired'


def lower_once_structures(script: Script) -> Script:
    """Return a copy of `script` with every `once` structure lowered to a
    `var` flag and an `if`. A script without one is returned unchanged."""
    if not contains(script, lambda node: isinstance(node, OnceStatement)):
        return script
    return _OnceLowering(collect_names(script)).rebuild(script)


class _OnceLowering:
    def __init__(self, used_names):
        self.used_names = used_names

    def rebuild(self, value):
        """`value` with each statement list's `once` structures expanded in place."""
        if isinstance(value, list):
            lowered = []
            for item in value:
                rebuilt = self.rebuild(item)
                lowered.extend(self.flag_and_if(rebuilt) if isinstance(rebuilt, OnceStatement) else [rebuilt])
            return lowered
        if isinstance(value, tuple):
            return tuple(self.rebuild(item) for item in value)
        if isinstance(value, dict):
            return {key: self.rebuild(item) for key, item in value.items()}
        if not is_node(value):
            return value
        children = {field.name: self.rebuild(getattr(value, field.name))
                    for field in dataclasses.fields(value) if field.name not in ('line', 'column')}
        return dataclasses.replace(value, **children)

    def flag_and_if(self, once: OnceStatement) -> list:
        position = {'line': once.line, 'column': once.column}
        flag = fresh_name(FLAG_STEM, self.used_names)
        not_fired = UnaryOp(op='not', operand=Identifier(name=flag, **position), **position)
        condition = not_fired if once.condition is None else BinaryOp(
            left=not_fired, op='and', right=once.condition, **position)
        fired = Reassignment(target=flag, value=Literal(value=True, literal_type='bool', **position), **position)
        return [
            VarDecl(name=flag, value=Literal(value=False, literal_type='bool', **position), type_hint='bool',
                    **position),
            IfStatement(condition=condition, body=[fired] + once.body, **position),
        ]
