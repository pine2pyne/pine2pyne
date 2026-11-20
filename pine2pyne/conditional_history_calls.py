"""A `ta.*` call that runs on only some bars.

Pine keeps a call site's series history by running the call on every bar. A `ta.*` call
reached only when a condition holds keeps a history that skips the other bars, so its value
differs from the same call made on every bar (measured: the transpiled output
matches Pine's call-site semantics in every construct, and a hoisted call gives another value).
The Pine User Manual (Execution model) says to run such functions in the global scope for
consistent results. The transpiler never hoists: it keeps the call where the script has it and
reports the site, as TradingView does, so the author decides.

A call is conditional when it sits in:
- the body of an `if`, `else if` or `else`, or an `else if` condition (it is read only when the
  earlier ones failed), as a statement or as a value;
- a branch of `?:`;
- a `switch` arm or its default, or a case after the first (the first case is always read);
- the right operand of `and` / `or` in Pine v6, which evaluates them lazily. Pine v5 and earlier
  evaluate both operands, so the right operand there is not conditional.

A function body starts unconditional again: whether the function is called conditionally is not
decided here. A loop body is not conditional. `ta.vwap` and the other argument-free `ta.`
variables are not calls and are hoisted by the transpiler already.
"""
from typing import List

from .ast_nodes import (
    BinaryOp, FuncDecl, FunctionCall, IfExpression, IfStatement, Script, SwitchExpression, SwitchStatement,
    TernaryOp,
)
from .ast_walk import child_field_names, is_node
from .strict_boolean import BOOLEAN_OPERATORS, evaluates_and_or_strictly

HISTORY_CALL_PREFIX = 'ta.'


def conditional_history_calls(script: Script) -> List[FunctionCall]:
    """Every `ta.*` call of `script` that runs on only some bars, in source order."""
    finder = _ConditionalCallFinder(evaluates_and_or_strictly(script))
    finder.visit(script, False)
    return sorted(finder.calls, key=lambda call: (call.line, call.column))


class _ConditionalCallFinder:
    def __init__(self, right_operand_always_runs: bool):
        self.right_operand_always_runs = right_operand_always_runs
        self.calls: List[FunctionCall] = []

    def visit(self, value, conditional: bool) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                self.visit(item, conditional)
        elif isinstance(value, dict):
            for item in value.values():
                self.visit(item, conditional)
        elif is_node(value):
            self.visit_node(value, conditional)

    def visit_node(self, node, conditional: bool) -> None:
        if isinstance(node, (IfStatement, IfExpression, TernaryOp)):
            self.visit_branches(node, conditional)
        elif isinstance(node, (SwitchStatement, SwitchExpression)):
            self.visit_switch(node, conditional)
        elif isinstance(node, BinaryOp) and node.op in BOOLEAN_OPERATORS:
            self.visit(node.left, conditional)
            self.visit(node.right, conditional or not self.right_operand_always_runs)
        elif isinstance(node, FuncDecl):
            self.visit(node.params, False)
            self.visit(node.body, False)
        else:
            if conditional and isinstance(node, FunctionCall) and _is_history_call(node):
                self.calls.append(node)
            for name in child_field_names(type(node)):
                self.visit(getattr(node, name), conditional)

    def visit_branches(self, node, conditional: bool) -> None:
        """The condition always runs; each branch and every `else if` condition only after it."""
        self.visit(node.condition, conditional)
        if isinstance(node, IfStatement):
            self.visit(node.body, True)
            for condition, body in node.elseifs:
                self.visit(condition, True)
                self.visit(body, True)
            self.visit(node.else_body, True)
        else:
            self.visit(node.true_expr, True)
            self.visit(node.false_expr, True)

    def visit_switch(self, node, conditional: bool) -> None:
        self.visit(node.expr, conditional)
        for index, (case, body) in enumerate(node.cases):
            self.visit(case, conditional if index == 0 else True)
            self.visit(body, True)
        self.visit(node.default, True)


def _is_history_call(call: FunctionCall) -> bool:
    return isinstance(call.func, str) and call.func.startswith(HISTORY_CALL_PREFIX)
