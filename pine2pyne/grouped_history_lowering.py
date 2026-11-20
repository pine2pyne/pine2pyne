"""
Grouped-history lowering: a history index on an expression that is not a name.

Pine's `(a + b)[1]` is the grouped value one bar back, `a[1] + b[1]`, and
`f()[1]` is the value the call returned one bar back. Python has no series of
an expression: the emitter bound `[1]` to the last operand, and a
call's value is a plain number. This pre-pass names the object of every such
history index as a Pine local, in the idiom a Pine author writes by hand:

    y = (close + open)[1]      ->    grouped_history1 = close + open
                                     y = grouped_history1[1]

The local sits immediately before the statement that uses it, in the same
statement list, so every later stage (symbol table, type inference, Series
annotation, function-instance series) treats it as an ordinary user local.
A group in a ternary arm or an and/or operand is therefore evaluated on every
bar its statement runs; a group in an if/switch-expression branch stays in
that branch (the branch becomes a block, which block-value lowering handles);
a group in an else-if condition, or in a later switch case, runs in the else
block, where Pine puts it. A top-level `var` whose initialiser holds a group
moves into the body, after its local. A group whose position has no statement
to precede it (a while condition, the header of a loop used as a value) is
refused with its line, never emitted wrong.

The history of a field of a user-defined type, `t.x[1]`, is such a site too:
Pine v5 reads the field's own per-bar series, as `value = t.x; value[1]` does, and
Python's `t.x` is a plain number. Pine v6 refuses the form (pine_semantics.py, which
runs before this pass and asks the same rule about the head's binding in scope), so only
a v5 script gets here. A dotted name is a field read when its head is a binding that owns
the member (`library_collision.FieldReadRule`); a namespace member (`ta.tr[1]`,
`session.ismarket[1]` beside an input named `session`) is not.

This pass runs after the scope renamer, which has resolved every `N.m` against the binding
in scope: a binding named like a namespace that owns the member was given a fresh name, and
a namespace read under a binding that keeps its name (an input, an export) reads an alias.
So the binding a head names here is the one the script binds under that emitted name, and
its type is the one its declarations give it (`library_collision.binding_types`).
One approximation is left: an `export` function's parameter keeps a namespace's name in its
own body only, so a namespace read with history elsewhere, or a second such parameter of
another type, is judged by the script-wide type of that name.
"""
import dataclasses
from typing import List, Optional

from .ast_nodes import (
    Assignment, BinaryOp, Expression, ExpressionStatement, ForInLoop, ForLoop,
    FuncDecl, FunctionCall, Identifier, IfExpression, IfStatement, IndexAccess,
    Literal, MethodCall, Reassignment, Script, SwitchExpression, SwitchStatement,
    TernaryOp, UnaryOp, VarDecl, VaripDecl, WhileLoop,
)
from .ast_walk import child_field_names, contains, map_children
from .errors import UnsupportedFeatureError
from .library_collision import FieldReadRule, HeadBinding, binding_types
from .scope_renamer import collect_names, fresh_name

# The objects of a history index that are named as a local: every expression
# whose value Python cannot index, i.e. all but a plain name (a field read is
# decided by `FieldReadRule`)
HISTORY_SOURCES = (BinaryOp, UnaryOp, TernaryOp, IndexAccess, FunctionCall, MethodCall)
LOCAL_STEM = 'grouped_history'
SUBJECT_STEM = 'switch_subject'
ASSIGNING_STATEMENTS = (Assignment, Reassignment, VarDecl, VaripDecl)
LOOP_HEADER_FIELDS = ('from_val', 'to_val', 'step', 'iterable')
LOOPS = (WhileLoop, ForLoop, ForInLoop)


def lower_grouped_history(script: Script) -> Script:
    """Return a copy of `script` with the object of every history index that
    is not a name bound to a local. A script without one is returned unchanged."""
    field_read_rule = _FieldReads(script)
    if not has_history_site(script, field_read_rule):
        return script
    return _GroupedHistoryLowering(collect_names(script), field_read_rule).script(script)


class _FieldReads:
    """`FieldReadRule` over the renamed script: a head names the binding of its emitted name."""

    def __init__(self, script: Script):
        self.rule = FieldReadRule(script)
        self.types = binding_types(script)

    def binding(self, head: str) -> Optional[HeadBinding]:
        return HeadBinding(self.types[head]) if head in self.types else None

    def is_field_read(self, value) -> bool:
        return self.rule.is_field_read(value, self.binding)


def is_history_site(node, field_read_rule: Optional[_FieldReads] = None) -> bool:
    """A history index whose object is an expression or a field read, not a name."""
    return isinstance(node, IndexAccess) and (
        isinstance(node.object, HISTORY_SOURCES)
        or (field_read_rule is not None and field_read_rule.is_field_read(node.object)))


def has_history_site(tree, field_read_rule: Optional[_FieldReads] = None) -> bool:
    return contains(tree, lambda node: is_history_site(node, field_read_rule))


def _switch_case_has_site(switch, field_read_rule: _FieldReads) -> bool:
    return any(has_history_site(case, field_read_rule) for case, _ in switch.cases)


class _GroupedHistoryLowering:
    def __init__(self, used_names, field_read_rule: _FieldReads):
        self.used_names = used_names
        self.field_read_rule = field_read_rule
        self.line = 0  # nearest enclosing source line, for refusals and new nodes

    def refuse(self, position: str):
        raise UnsupportedFeatureError(
            f'a history index on an expression in {position} (assign the expression to a variable first)',
            self.line)

    # ------------------------------------------------------------ expressions

    def expression(self, value, hoisted: List):
        """`value` with the object of each history site named as a local; the
        locals' assignments are appended to `hoisted`, innermost first."""
        if isinstance(value, (IfExpression, SwitchExpression)):
            return self.value_branches(value, hoisted)
        if isinstance(value, (list, tuple, dict)):
            return map_children(value, lambda item: self.expression(item, hoisted))
        if isinstance(value, LOOPS):
            return self.loop_value(value)
        if not isinstance(value, Expression):
            return value
        children = {name: self.expression(getattr(value, name), hoisted) for name in child_field_names(type(value))}
        rebuilt = dataclasses.replace(value, **children)
        return self.named_source(rebuilt, hoisted) if is_history_site(rebuilt, self.field_read_rule) else rebuilt

    def named_source(self, site: IndexAccess, hoisted: List) -> IndexAccess:
        """The site reads a new local that holds its object."""
        name = fresh_name(LOCAL_STEM, self.used_names)
        hoisted.append(Assignment(target=name, value=site.object, line=self.line))
        return dataclasses.replace(site, object=Identifier(name=name, line=self.line))

    def value_branches(self, value, hoisted: List):
        """An if/switch expression: the condition and subject run with the
        statement; each branch keeps its own locals (a Pine local block)."""
        if isinstance(value, SwitchExpression) and _switch_case_has_site(value, self.field_read_rule):
            # A later case runs only when the earlier ones fail: an else-if chain
            value = self.switch_as_if_expression(value, hoisted)
        if isinstance(value, IfExpression):
            return dataclasses.replace(value, condition=self.expression(value.condition, hoisted),
                                       true_expr=self.branch(value.true_expr),
                                       false_expr=self.branch(value.false_expr))
        return dataclasses.replace(value, expr=self.expression(value.expr, hoisted),
                                   cases=[(case, self.branch(body)) for case, body in value.cases],
                                   default=self.branch(value.default))

    def branch(self, branch):
        """A branch of an if/switch expression: a block keeps its statements'
        locals; a single expression with a site becomes a block ending in it."""
        if branch is None or isinstance(branch, list):
            return self.block(branch)
        hoisted: List = []
        value = self.expression(branch, hoisted)
        return hoisted + [ExpressionStatement(expr=value, line=self.line)] if hoisted else value

    # ------------------------------------------------------------ switch cases

    def switch_as_if_expression(self, switch: SwitchExpression, hoisted: List) -> IfExpression:
        """`switch` as nested if expressions over its case conditions."""
        case_condition = self.case_condition_builder(switch.expr, hoisted)
        chain = switch.default
        for case, body in reversed(switch.cases):
            chain = IfExpression(condition=case_condition(case), true_expr=body, false_expr=chain, line=self.line)
        return chain

    def switch_as_if_statement(self, switch: SwitchStatement, hoisted: List) -> IfStatement:
        """`switch` as an if / else-if chain over its case conditions."""
        case_condition = self.case_condition_builder(switch.expr, hoisted)
        (first_case, first_body), *other_cases = switch.cases
        return IfStatement(condition=case_condition(first_case), body=first_body,
                           elseifs=[(case_condition(case), body) for case, body in other_cases],
                           else_body=switch.default, line=self.line,
                           switch_arms='statement' if switch.expr is None else '')

    def case_condition_builder(self, subject, hoisted: List):
        """The condition of a case: the case itself without a subject, else
        `subject == case`. A subject that is not a plain name or a literal is
        evaluated once, into a local, as Pine evaluates it once."""
        if subject is None:
            return lambda case: case
        subject = self.expression(subject, hoisted)
        if not isinstance(subject, Literal) and not (isinstance(subject, Identifier) and '.' not in subject.name):
            name = fresh_name(SUBJECT_STEM, self.used_names)
            hoisted.append(Assignment(target=name, value=subject, line=self.line))
            subject = Identifier(name=name, line=self.line)
        return lambda case: BinaryOp(left=dataclasses.replace(subject), op='==', right=case, line=self.line)

    # ------------------------------------------------------------ statements

    def block(self, body):
        if body is None:
            return None
        lowered = []
        for statement in body:
            lowered.extend(self.statement(statement))
        return lowered

    def statement(self, node) -> list:
        """The statement lowered, preceded by the locals its sites read."""
        enclosing_line, self.line = self.line, node.line or self.line
        try:
            hoisted: List = []
            lowered = self.lowered_statement(node, hoisted)
            return hoisted + [lowered]
        finally:
            self.line = enclosing_line

    def lowered_statement(self, node, hoisted: List):
        if isinstance(node, IfStatement):
            return self.if_statement(node, hoisted)
        if isinstance(node, SwitchStatement):
            return self.switch_statement(node, hoisted)
        if isinstance(node, LOOPS):
            return self.loop(node, hoisted)
        if isinstance(node, FuncDecl):
            return self.function(node)
        if isinstance(node, ASSIGNING_STATEMENTS):
            return dataclasses.replace(node, value=self.expression(node.value, hoisted))
        if isinstance(node, ExpressionStatement):
            return dataclasses.replace(node, expr=self.expression(node.expr, hoisted))
        if isinstance(node, Expression):
            # A block branch may end in a bare expression
            return self.expression(node, hoisted)
        return node

    def loop(self, node, hoisted: List):
        """A for header runs once, before the loop; a while condition runs per
        iteration, where a statement cannot precede it, and so does the end
        bound of a v6 for loop that re-reads it."""
        if isinstance(node, WhileLoop):
            if has_history_site(node.condition, self.field_read_rule):
                self.refuse('a while condition')
            return dataclasses.replace(node, body=self.block(node.body))
        if getattr(node, 'counter_name', '') and has_history_site(node.to_val, self.field_read_rule):
            self.refuse('the end bound of a for loop whose body changes it')
        header = {name: self.expression(getattr(node, name), hoisted)
                  for name in LOOP_HEADER_FIELDS if hasattr(node, name)}
        return dataclasses.replace(node, body=self.block(node.body), **header)

    def loop_value(self, loop):
        """A loop used as a value (a bound value, an operand, a while
        too): its header has no statement of its own to precede."""
        if has_history_site([getattr(loop, name, None) for name in LOOP_HEADER_FIELDS + ('condition',)],
                            self.field_read_rule):
            self.refuse('the header of a loop used as a value')
        return dataclasses.replace(loop, body=self.block(loop.body))

    def if_statement(self, node: IfStatement, hoisted: List) -> IfStatement:
        """An else-if is an if inside the else block: a group in its condition
        is named there, so it runs only on bars that reach that condition."""
        condition = self.expression(node.condition, hoisted)
        body = self.block(node.body)
        for index, (elseif_condition, elseif_body) in enumerate(node.elseifs):
            if has_history_site(elseif_condition, self.field_read_rule):
                rest = IfStatement(condition=elseif_condition, body=elseif_body, elseifs=node.elseifs[index + 1:],
                                   else_body=node.else_body, line=node.line)
                return dataclasses.replace(
                    node, condition=condition, body=body,
                    elseifs=[(earlier_condition, self.block(earlier_body))
                             for earlier_condition, earlier_body in node.elseifs[:index]],
                    else_body=self.statement(rest))
        return dataclasses.replace(node, condition=condition, body=body,
                                   elseifs=[(elseif_condition, self.block(elseif_body))
                                            for elseif_condition, elseif_body in node.elseifs],
                                   else_body=self.block(node.else_body))

    def switch_statement(self, node: SwitchStatement, hoisted: List):
        """A site in a case: the switch becomes the if / else-if chain Pine runs."""
        if _switch_case_has_site(node, self.field_read_rule):
            return self.if_statement(self.switch_as_if_statement(node, hoisted), hoisted)
        return dataclasses.replace(node, expr=self.expression(node.expr, hoisted),
                                   cases=[(case, self.block(body)) for case, body in node.cases],
                                   default=self.block(node.default))

    def function(self, node: FuncDecl) -> FuncDecl:
        if isinstance(node.body, list):
            return dataclasses.replace(node, body=self.block(node.body))
        return dataclasses.replace(node, body=self.branch(node.body))

    # ------------------------------------------------------------ script

    def script(self, script: Script) -> Script:
        declarations: List = []
        moved_to_body: List = []
        for declaration in script.declarations:
            if isinstance(declaration, FuncDecl):
                declarations.extend(self.statement(declaration))
            elif isinstance(declaration, (VarDecl, VaripDecl)) and has_history_site(declaration.value, self.field_read_rule):
                # The local must precede the declaration, so both run as body statements
                moved_to_body.extend(self.statement(declaration))
            else:
                declarations.append(declaration)
        body = _merged_by_line(self.block(script.body), moved_to_body)
        return dataclasses.replace(script, declarations=declarations, body=body)


def _merged_by_line(body: list, moved: list) -> list:
    """`moved` placed among `body` by source line: each goes before the first
    body statement on a later line. The order within each list is kept."""
    merged = list(body)
    for statement in moved:
        position = next((index for index, item in enumerate(merged) if (item.line or 0) > statement.line),
                        len(merged))
        merged = merged[:position] + [statement] + merged[position:]
    return merged
