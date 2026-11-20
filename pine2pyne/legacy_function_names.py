"""Pine v1-v4 call names, spelled the way Pine v5 spells them.

Before Pine v5 the technical-analysis, math, request and ticker functions were global: `sma(close, 5)`,
`mom(close, 2)`, `security(...)`, `heikenashi(...)`. v5 moved them into the `ta.`, `math.`, `str.`, `request.`
and `ticker.` namespaces. The transformer and the runtime know only the namespaced names, so a bare `sma(...)`
reached the generated code as a bare `sma(...)` that nothing imports: `NameError` at run.

`qualify_legacy_function_names` renames those calls in a script that declares //@version 1-4. Only a call of a
name from the table is renamed, and only when the script declares no FUNCTION of that name itself: a user's own
`sma(x, n) => ...` stays. A variable, a parameter or a loop counter of that name does not stop the rename: a call
`sma(...)` looks up a function, a variable lives apart from it (`rsi = rsi(close, 14)` is a common idiom) or is out of
scope at the call, so the builtin is the callee. A source with no //@version line, or one at v5 or later, is not
touched.

Left out on purpose: names a Python builtin of the same spelling already answers
(`abs`, `max`, `min`, `round`, `pow`, `range`), which run today with Python semantics; and the legacy names
that are series, not calls (a bare `tr`, `vwap`, `obv`, `tickerid`, ...).
"""
from __future__ import annotations

from typing import Dict, Set

import dataclasses

from .ast_nodes import FuncDecl, FunctionCall, IndexAccess, Script
from .ast_walk import is_node, map_children, walk

TECHNICAL_ANALYSIS_NAMES = (
    'alma atr barssince bb bbw cci change cmo cog correlation cross crossover crossunder cum dev dmi ema falling '
    'hma highest highestbars kc kcw linreg lowest lowestbars macd mfi mom percentile_linear_interpolation '
    'percentile_nearest_rank percentrank pivothigh pivotlow rising rma roc rsi sar sma stdev stoch supertrend '
    'swma tr tsi valuewhen variance vwma wma wpr'
).split()
MATH_NAMES = ('acos asin atan avg ceil cos exp floor log log10 random round_to_mintick sign sin sqrt sum tan '
              'todegrees toradians').split()

# call name before Pine v5 -> the name Pine v5 gives it
LEGACY_FUNCTION_NAMES: Dict[str, str] = {
    **{name: f'ta.{name}' for name in TECHNICAL_ANALYSIS_NAMES},
    **{name: f'math.{name}' for name in MATH_NAMES},
    'tostring': 'str.tostring',
    'tonumber': 'str.tonumber',
    'security': 'request.security',
    'heikenashi': 'ticker.heikinashi',
    'heikinashi': 'ticker.heikinashi',
    'tickerid': 'ticker.new',
    'renko': 'ticker.renko',
    'linebreak': 'ticker.linebreak',
    'kagi': 'ticker.kagi',
    'pointfigure': 'ticker.pointfigure',
    'financial': 'request.financial',
    'quandl': 'request.quandl',
    # `earnings(t, earnings.actual, ...)`: the call is v5's request.earnings; `earnings.actual` stays a constant
    'earnings': 'request.earnings',
    'dividends': 'request.dividends',
    'splits': 'request.splits',
}


def _declared_function_names(script: Script) -> Set[str]:
    """The names of the functions the script itself declares, anywhere in it."""
    return {node.name for node in walk(script) if isinstance(node, FuncDecl) and isinstance(node.name, str)}


def _declared_version(script: Script) -> int:
    """The //@version number, or 0 when absent or not a number."""
    text = script.version.version if script.version else ''
    return int(text) if str(text).isdigit() else 0


def qualify_legacy_function_names(script: Script) -> Script:
    """The script with the calls of a Pine v1-v4 script spelled as Pine v5 spells them."""
    if not 1 <= _declared_version(script) <= 4:
        return script
    own_names = _declared_function_names(script)
    renames = {legacy: current for legacy, current in LEGACY_FUNCTION_NAMES.items() if legacy not in own_names}
    for node in walk(script):
        if isinstance(node, FunctionCall) and isinstance(node.func, str) and node.func in renames:
            if node.func == 'security' and 'resolution' in node.kwargs:
                # Pine v4 `security(resolution = r, ...)` is v5's `request.security(timeframe = r, ...)`
                node.kwargs = {('timeframe' if key == 'resolution' else key): value
                               for key, value in node.kwargs.items()}
            node.func = renames[node.func]
    if 'offset' not in own_names:
        script = _offsets_as_history(script)
    return script


def _is_offset_call(node) -> bool:
    return isinstance(node, FunctionCall) and node.func == 'offset' and len(node.args) == 2 and not node.kwargs


def _offsets_as_history(node):
    """Pine v1-v4 `offset(source, n)` is `source[n]`, a history read, and is written so; a variable named
    `offset` is no callee, so the call is the builtin."""
    if _is_offset_call(node):
        return IndexAccess(object=_offsets_as_history(node.args[0]), index=_offsets_as_history(node.args[1]),
                           line=node.line, column=node.column)
    if not is_node(node):
        return node
    changes = {field.name: map_children(getattr(node, field.name), _offsets_as_history)
               for field in dataclasses.fields(node) if field.name not in ('line', 'column')}
    return dataclasses.replace(node, **changes)
