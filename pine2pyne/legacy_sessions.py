"""Pine v4 session strings, read the way Pine v5 reads them.

A Pine v4 session without days (``"0930-1600"``) means weekdays; Pine v5 reads it as
every day. The session argument of ``time()`` and ``time_close()`` of a Pine v4 script
is converted::

    time(tf, "0930-1600")   time(tf, '0930-1600:23456')
    time(tf, sess)          time(tf, sess if sess == '' else sess + ('23456' if string.endswith(sess, ':')
                                     else '' if string.contains(sess, ':') else ':23456'))
    time(tf, sess + "x")    time(tf, __pc_session_days(sess + 'x'))   (a helper defined first in main)

A literal that names its days (``"0930-1600:1234567"``) is kept, and so is the empty session ``""``,
which is no session at all (``time(tf, "")`` is the bar's time).
"""
from __future__ import annotations

import dataclasses

from .ast_nodes import BinaryOp, FuncDecl, FunctionCall, Identifier, Literal, Parameter, Script, TernaryOp
from .ast_walk import walk

SESSION_FUNCTIONS = frozenset({'time', 'time_close'})
SESSION_HELPER = '__pc_session_days'
WEEKDAYS = '23456'


def _string(value: str) -> Literal:
    return Literal(value=value, literal_type='string')


def with_weekdays(session) -> TernaryOp:
    """``session`` read with Pine v4's default days, as an expression of it."""
    def copy():
        return dataclasses.replace(session)
    days = TernaryOp(
        condition=FunctionCall(func='str.endswith', args=[copy(), _string(':')]),
        true_expr=_string(WEEKDAYS),
        false_expr=TernaryOp(condition=FunctionCall(func='str.contains', args=[copy(), _string(':')]),
                             true_expr=_string(''), false_expr=_string(f':{WEEKDAYS}')))
    return TernaryOp(condition=BinaryOp(left=copy(), op='==', right=_string('')),
                     true_expr=copy(), false_expr=BinaryOp(left=copy(), op='+', right=days))


def qualify_legacy_sessions(script: Script) -> Script:
    """A Pine v4 script with the sessions of its ``time()``/``time_close()`` calls given days."""
    version = script.version.version if script.version else ''
    if not (str(version).isdigit() and int(version) == 4):
        return script
    needs_helper = False
    for node in walk(script):
        if not (isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in SESSION_FUNCTIONS
                and len(node.args) >= 2):
            continue
        session = node.args[1]
        if isinstance(session, Literal):
            # An empty session is no session, in Pine v4 as in v5: it keeps no days
            if session.literal_type == 'string' and str(session.value) and ':' not in str(session.value):
                node.args[1] = dataclasses.replace(session, value=f'{session.value}:{WEEKDAYS}')
        elif isinstance(session, Identifier):
            node.args[1] = with_weekdays(session)
        else:
            node.args[1] = FunctionCall(func=SESSION_HELPER, args=[session], line=getattr(session, 'line', 0),
                                        column=getattr(session, 'column', 0))
            needs_helper = True
    if not needs_helper:
        return script
    line = script.script_decl.line if script.script_decl is not None else 1
    helper = FuncDecl(name=SESSION_HELPER, params=[Parameter(name='s')], body=with_weekdays(Identifier(name='s')),
                      line=line)
    return dataclasses.replace(script, declarations=[helper] + list(script.declarations))
