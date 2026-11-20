"""Imports that match what the rewritten module actually uses."""
from __future__ import annotations

import ast
import builtins

from .base import PassContext

#: Where a name is imported from, for the names that do not live in
#: ``pynecore.lib``
KNOWN_ORIGINS: dict[str, str] = {
    'pine_range': 'pynecore', 'pine_loop': 'pynecore',
    'overload': 'pynecore.core.overload',
    'cast_bool': 'pynecore.core.pine_cast', 'cast_int': 'pynecore.core.pine_cast',
    'cast_float': 'pynecore.core.pine_cast', 'cast_string': 'pynecore.core.pine_cast',
    'Exported': 'pynecore.core.pine_export', 'export': 'pynecore.core.pine_export',
    'shadowed_namespace': 'pynecore.core.pine_import',
    'method_call': 'pynecore.core.pine_method', 'method': 'pynecore.core.pine_method',
    'udt': 'pynecore.core.pine_udt', 'udt_copy': 'pynecore.core.pine_udt',
    'inline_series': 'pynecore.core.series',
    'Color': 'pynecore.types', 'Line': 'pynecore.types', 'NA': 'pynecore.types',
    'Persistent': 'pynecore.types', 'PersistentSeries': 'pynecore.types', 'Series': 'pynecore.types',
    'Box': 'pynecore.types', 'Label': 'pynecore.types', 'Table': 'pynecore.types',
    'IBPersistent': 'pynecore.types', 'IBPersistentSeries': 'pynecore.types', 'LineFill': 'pynecore.types',
    'Matrix': 'pynecore.types', 'Polyline': 'pynecore.types', 'Footprint': 'pynecore.types', 'VolumeRow': 'pynecore.types',
    'ChartPoint': 'pynecore.types.chart',
    'field': 'dataclasses', 'StrEnum': 'enum',
}

#: Names imported from ``pynecore.lib``
LIB_NAMES = frozenset((
    'adjustment alert alertcondition array ask bar_index barcolor barmerge barstate bgcolor bid box chart '
    'close color currency dayofmonth dayofweek display dividends earnings extend fill fixnan font footprint '
    'format high hl2 hlc3 hlcc4 hline hour input label last_bar_index last_bar_time line linefill location '
    'log low map math matrix max_bars_back minute month na nz ohlc4 open order plot plotarrow plotbar '
    'plotcandle plotchar plotshape polyline position request runtime scale script second session shape '
    'size splits strategy string syminfo ta table text ticker time time_close time_tradingday timeframe '
    'timenow timestamp volume weekofyear xloc year yloc').split())

_MANAGED_PREFIXES = ('pynecore', 'dataclasses', 'enum')
#: the computed builtin series a script may bind as a local of the same name
_COMPUTED_SERIES = frozenset({'hl2', 'hlc3', 'ohlc4', 'hlcc4'})
#: Names imported under an alias: name -> (module, imported name)
KNOWN_ALIASES = {'_dataclass_field': ('dataclasses', 'field'), 'Enum': ('enum', 'StrEnum')}


def _bound_names(tree: ast.Module) -> set[str]:
    """Names the module binds itself (anywhere), which it must not import."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return bound


def _loads_outside_parameters(tree: ast.AST) -> set:
    """The names the module reads, but where a function reads its own parameter (`def m(bar_index): ...`):
    no builtin a parameter hides is imported."""
    used: set = set()

    def visit(node, hidden: frozenset) -> None:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in hidden:
            used.add(node.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            for default in arguments.defaults + [d for d in arguments.kw_defaults if d is not None]:
                visit(default, hidden)
            for decorator in getattr(node, 'decorator_list', []):
                visit(decorator, hidden)
            parameters = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
            for parameter in parameters:
                if parameter.annotation is not None:
                    visit(parameter.annotation, hidden)
            inner = hidden | {parameter.arg for parameter in parameters}
            body = node.body if isinstance(node.body, list) else [node.body]
            for statement in body:
                visit(statement, inner)
            if getattr(node, 'returns', None) is not None:
                visit(node.returns, hidden)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, hidden)

    visit(tree, frozenset())
    return used


class ImportReconciliationPass:
    """Drops imported names nothing uses, adds the ones a rewrite introduced,
    and imports each from the module it lives in (``NA`` from
    ``pynecore.types``, not ``pynecore.lib``). Run it last."""

    def apply(self, tree: ast.Module, context: PassContext) -> None:
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
        managed = [node for node in imports if (node.module or '').startswith(_MANAGED_PREFIXES)]
        used = _loads_outside_parameters(tree)
        used |= {decorator.id for node in ast.walk(tree)
                 for decorator in getattr(node, 'decorator_list', []) if isinstance(decorator, ast.Name)}
        local = _bound_names(tree) - {alias.asname or alias.name for node in imports for alias in node.names}

        wanted: dict[str, list[str]] = {}
        aliases: dict[str, list[ast.alias]] = {}
        for node in managed:
            for alias in node.names:
                visible = alias.asname or alias.name
                if visible not in used or (visible in _COMPUTED_SERIES and visible in _bound_names(tree)):
                    continue  # unused, or a computed series the module binds itself (`hl2 = (h + l) / 2`)
                if alias.asname:
                    aliases.setdefault(node.module, []).append(alias)
                    continue
                module = KNOWN_ORIGINS.get(alias.name, node.module)
                wanted.setdefault(module, [])
                if alias.name not in wanted[module]:
                    wanted[module].append(alias.name)
        already = {name for names in wanted.values() for name in names}
        already |= {alias.asname for names in aliases.values() for alias in names}
        for name in sorted(used - already - local - set(dir(builtins))):
            if name in KNOWN_ALIASES:
                module, imported = KNOWN_ALIASES[name]
                aliases.setdefault(module, []).append(ast.alias(name=imported, asname=name))
                continue
            module = KNOWN_ORIGINS.get(name) or ('pynecore.lib' if name in LIB_NAMES else None)
            if module:
                wanted.setdefault(module, []).append(name)

        rebuilt = [ast.ImportFrom(module=module, names=[ast.alias(name=name) for name in names]
                                  + aliases.pop(module, []), level=0)
                   for module, names in wanted.items() if names]
        rebuilt += [ast.ImportFrom(module=module, names=names, level=0) for module, names in aliases.items()]
        first = next((index for index, node in enumerate(tree.body) if node in managed), None)
        # a `typing` import none of whose names the module uses goes too (`Any` of an annotation a rewrite
        # replaced); a library's `Protocol, Any` stays as written
        unused_typing = [node for node in imports if node.module == 'typing'
                         and not any((alias.asname or alias.name) in used for alias in node.names)]
        tree.body[:] = [node for node in tree.body if node not in managed and node not in unused_typing]
        if rebuilt:
            position = first if first is not None else _after_docstring(tree)
            tree.body[position:position] = rebuilt


def _after_docstring(tree: ast.Module) -> int:
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        return 1
    return 0
