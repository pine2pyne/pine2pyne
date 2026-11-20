"""
Import resolver for PyneCore transpiled code.

Collects all used symbols and generates appropriate import statements.
"""
import keyword
from typing import Set, List
from .ast_nodes import *
from .errors import UnsupportedFeatureError
from .ast_walk import walk
from .block_value_lowering import bool_is_never_na
from .udt_field_defaults import FIELD_FACTORY_IMPORT, emitted_default_requires_na, is_dotted_default, is_series_default
from .pine_builtins import (
    PYNECORE_LIB_MODULES, DRAWING_TYPES, CHART_TYPES,
    SERIES_VARIABLES, is_series_variable, needs_plot_style_remap
)


def library_module_path(user, library: str, version) -> str:
    """The module a library import loads, `lib.<user>.<library>.v<version>` (`-` of a user
    read as `_`); a segment that is a Python keyword (`class`) is spelled `class__ren__`, as a name so spelled
    is, since Python cannot import a module path that holds a keyword."""
    def segment(name: str) -> str:
        return f'{name}__ren__' if keyword.iskeyword(name) else name
    path = f"lib.{segment(str(user).replace('-', '_'))}.{segment(library)}"
    if version not in (None, ''):
        path += f'.v{version}'
    return path


class ImportResolver:
    """Collects used symbols and generates import statements."""

    def __init__(self):
        self.uses_series = False
        self.uses_persistent = False
        self.uses_persistent_series = False
        self.uses_na = False
        self.bool_is_never_na = True  # Pine v6 and later; set per script by analyze()
        self.uses_any = False
        self.uses_udt = False  # Custom type definitions
        self.uses_udt_copy = False  # udt_copy for UDT field copying
        self.uses_field_factory = False  # a UDT field default read per instance
        self.uses_method = False  # User-defined methods
        self.uses_method_call = False  # method_call for explicit method calling
        self.uses_overload = False  # pynecore @overload for same-name declarations
        self.uses_export = False  # a library's `export` functions: the Exported proxy and the @export decorator
        self.library_imports: List[ImportDecl] = []  # `import user/lib/version as alias`, one Python import each
        self.uses_pine_range = False  # pine_range for loops
        self.pine_range_name = 'pine_range'  # the name pine_range is imported under; the transformer sets an alias when the script uses the name
        self.pine_loop_name = ''  # the name pine_loop is imported under: a v6 for loop re-reads its end
        self.uses_cast_int = False  # cast_int for type casting
        self.uses_cast_float = False  # cast_float for type casting
        self.uses_cast_bool = False  # cast_bool for type casting (Pine v6)
        self.uses_cast_string = False  # cast_string for `string(x)`
        # Note: SeriesImpl for function params is handled by runtime SeriesTransformer
        self.lib_modules: Set[str] = set()
        self.builtin_variables: Set[str] = set()
        # Global-scope names the script declares itself (var/varip/assignment). Pine
        # lets a user variable shadow a built-in variable of the same name (they are
        # not reserved keywords), so such a name must NOT be treated/imported as a
        # built-in series — e.g. `var array<bool> hl2 = ...` shadows the `hl2`
        # ((high+low)/2) price source, and importing the built-in would let the
        # runtime overwrite the user's array each bar.
        self.declared_names: Set[str] = set()
        self.drawing_types: Set[str] = set()
        self.chart_types: Set[str] = set()
        self.uses_plot_style = False
        # A namespace spelled by an alias because an input or export keeps its name
        # (scope_renamer): module name -> alias ('size' -> '_size_module')
        self.aliased_modules: dict[str, str] = {}

    def analyze(self, script: Script) -> None:
        """Analyze script to determine what needs to be imported."""
        self.bool_is_never_na = bool_is_never_na(script)
        self.library_imports = list(script.imports)
        # the UDTs and enums the script declares: a UDT-typed field is made per instance; an enum is a StrEnum
        self._udt_names = {decl.name for decl in script.declarations if isinstance(decl, TypeDecl)}
        self.uses_enum = any(isinstance(decl, EnumDecl) for decl in script.declarations)
        # Pre-pass: collect every name the script declares, so a user variable that
        # shadows a built-in (e.g. hl2) is not mistaken for the built-in below.
        self._collect_declared_names(script)
        # a computed series a function binds as its own local, where nothing reads the builtin (`hl2 = ...`)
        from .library_collision import COMPUTED_SERIES, colliding_names
        self.declared_names |= {name for name in COMPUTED_SERIES if name not in colliding_names(script)
                                and any(isinstance(node, (Assignment, VarDecl, VaripDecl))
                                        and name in (getattr(node, 'target', None), getattr(node, 'name', None))
                                        for node in walk(script))}

        # Check for Series usage (var declarations, globals)
        for decl in script.declarations:
            if isinstance(decl, (VarDecl, VaripDecl)):
                self.uses_persistent = True
            elif isinstance(decl, Assignment):
                # Global assignments are Series
                self.uses_series = True

        # Check body for Series usage
        if script.body:
            self.uses_series = True

        # Scan every node once; the shared walk also enters keyword arguments
        # and else-if arms (a nested call was rescanned per level)
        for node in walk(script):
            self._scan_node(node)

    def _collect_declared_names(self, node: Any) -> None:
        """Recursively collect GLOBAL-scope names the script declares (var/varip/
        assignment/input) so they can shadow a same-named built-in variable.

        Only global-scope declarations shadow a built-in everywhere the built-in
        would be imported. Function parameters and function-local declarations are
        scoped to that function, so they must NOT suppress the global import — a
        script can have a function parameter named ``time`` while still using the
        built-in ``time`` at global scope. We therefore skip ``FuncDecl`` subtrees
        entirely and never collect ``Parameter`` names.

        Pine declarations use a plain ``str`` name/target; a tuple target's names
        are declarations too (`[close, unit] = pair()` at the top level).

        An input is a global too: pynecore reads the library series for
        every read of an imported name except a function parameter, and a source
        input is emitted as a local of main() (``close: Series[float] = _close_src``).
        A block local named like a built-in never reaches here: the scope renamer
        gives it its own name.
        """
        if isinstance(node, FuncDecl):
            return  # function-scoped names never shadow a global built-in import
        if isinstance(node, (VarDecl, VaripDecl, InputDecl)):
            if isinstance(node.name, str):
                self.declared_names.add(node.name)
        elif isinstance(node, Assignment):
            if isinstance(node.target, str):
                self.declared_names.add(node.target)
            elif isinstance(node.target, TupleDestructure):  # `[close, x] = f()`
                self.declared_names.update(node.target.names)

        if isinstance(node, (list, tuple)):
            for item in node:
                self._collect_declared_names(item)
        elif hasattr(node, '__dict__'):
            for attr_value in node.__dict__.values():
                if isinstance(attr_value, (ASTNode, list, tuple, dict)):
                    self._collect_declared_names(attr_value)

    def _scan_node(self, node: Any) -> None:
        """Record the symbols of one AST node (not of its children)."""
        # Handle NaLiteral
        if isinstance(node, NaLiteral):
            self.uses_na = True

        # Handle function calls
        if isinstance(node, FunctionCall):
            self._process_function_call(node)

        # Handle member access
        if isinstance(node, MemberAccess):
            self._process_member_access(node)

        # Handle identifiers
        if isinstance(node, Identifier):
            self._process_identifier(node.name)

        # Handle script declaration
        if isinstance(node, (IndicatorDecl, StrategyDecl, LibraryDecl)):
            self.lib_modules.add('script')

        # A library's exported function
        if isinstance(node, FuncDecl) and node.is_export:
            self.uses_export = True

        # Handle input declarations
        if isinstance(node, InputDecl):
            self.lib_modules.add('input')

        # Handle var/varip declarations (including inside blocks)
        if isinstance(node, (VarDecl, VaripDecl)):
            self.uses_persistent = True

        # Handle type declarations (custom types)
        if isinstance(node, TypeDecl):
            self.uses_udt = True
            # Scan field types for drawing type imports (e.g., lbl: Label)
            from .pine_builtins import get_type_name
            from .transformer import convert_pine_generic_to_python
            for field_info in node.fields:
                if len(field_info) >= 2:
                    pine_type = field_info[1]  # type_hint string (lowercase from Pine, e.g., "label")
                    default_val = field_info[2] if len(field_info) >= 3 else None
                    if isinstance(pine_type, str):
                        # Convert Pine type to Python type (e.g., "label" → "Label")
                        py_type = get_type_name(pine_type)
                        # Check if it's a drawing or chart type
                        if py_type in DRAWING_TYPES:
                            self.drawing_types.add(py_type)
                        elif py_type in CHART_TYPES:
                            self.chart_types.add(py_type)
                        # ... or holds one: the field annotation `matrix<float>` -> `Matrix[float]`
                        self._add_types_named_in(convert_pine_generic_to_python(pine_type))
                        # Codegen emits na or NA(Type) where emitted_default_requires_na says so
                        holds_a_udt = pine_type in self._udt_names and (default_val is None
                                                                        or isinstance(default_val, NaLiteral))
                        if holds_a_udt or is_series_default(default_val) or is_dotted_default(default_val):
                            self.uses_field_factory = True
                        if emitted_default_requires_na(py_type, default_val, self.bool_is_never_na):
                            self.uses_na = True

        # Handle method declarations
        if isinstance(node, FuncDecl):
            if hasattr(node, 'is_method') and node.is_method:
                self.uses_method = True

        # Handle for loops (need pine_range)
        if isinstance(node, ForLoop):
            if node.counter_name:
                self.pine_loop_name = node.counter_helper
            else:
                self.uses_pine_range = True

        # Check for Any in type hints
        if isinstance(node, (Assignment, Reassignment, VarDecl, VaripDecl, Parameter)):
            if hasattr(node, 'type_hint') and node.type_hint:
                type_hint_str = str(node.type_hint)
                if 'Any' in type_hint_str:
                    self.uses_any = True

                # Extract drawing/structural types from type hints like Persistent[Line], Matrix[float], etc.
                self._add_types_named_in(type_hint_str)

    def _add_types_named_in(self, type_hint: str) -> None:
        """Import the drawing and chart types a Python type hint names:
        Persistent[Line], Matrix[float], dict[str, Line], ..."""
        import re
        for match in re.finditer(r'\b([A-Z][a-zA-Z]*)\b', type_hint):
            self.add_drawing_type(match.group(1))

    def _process_function_call(self, call: FunctionCall) -> None:
        """Process a function call to extract module dependencies."""
        func_name = None

        if isinstance(call.func, str):
            func_name = call.func
            # Handle dotted function names like "box.new" -> extract "box"
            if '.' in func_name:
                self._add_namespace(func_name.split('.')[0])
            # Handle standalone function names like "plot", "bgcolor", "barcolor"
            elif func_name in PYNECORE_LIB_MODULES:
                self.lib_modules.add(func_name)
            # Handle series variables used as functions (e.g., time(), time_close())
            elif is_series_variable(func_name):
                self.builtin_variables.add(func_name)
                self.uses_series = True
        elif isinstance(call.func, NaLiteral):
            # Special case: na() used as a function call, not a literal
            # Import the na function from pynecore.lib
            self.lib_modules.add('na')
        elif isinstance(call.func, MemberAccess):
            # Extract module name from member access
            if isinstance(call.func.object, str):
                self._add_namespace(call.func.object)
            elif isinstance(call.func.object, Identifier):
                self._add_namespace(call.func.object.name)

        # Plot style references in kwargs (the walk visits the values themselves)
        for value in call.kwargs.values():
            if isinstance(value, MemberAccess):
                # Handle both Identifier and str object types
                module_name = value.object.name if isinstance(value.object, Identifier) else str(value.object)
                full_name = f"{module_name}.{value.member}"
                if needs_plot_style_remap(full_name):
                    self.uses_plot_style = True

    def _process_member_access(self, member: MemberAccess) -> None:
        """Process member access to extract module dependencies."""
        if isinstance(member.object, str):
            module = member.object
            self._add_namespace(module)

            # Check for plot style
            full_name = f"{module}.{member.member}"
            if needs_plot_style_remap(full_name):
                self.uses_plot_style = True

        elif isinstance(member.object, Identifier):
            module = member.object.name
            self._add_namespace(module)

            # Check for plot style
            full_name = f"{module}.{member.member}"
            if needs_plot_style_remap(full_name):
                self.uses_plot_style = True

    def _process_identifier(self, name: str) -> None:
        """Process identifier to check for built-in variables and module constants."""
        # Check for dotted identifiers like "xloc.bar_time" or "line.style_dashed"
        if '.' in name:
            self._add_namespace(name.split('.')[0])
        # Check for built-in Series variables — but a name the script declares
        # itself shadows the built-in, so it must not be imported as one.
        elif is_series_variable(name) and name not in self.declared_names:
            self.builtin_variables.add(name)
            self.uses_series = True

    def _add_namespace(self, module: str) -> None:
        """Import what a dotted name starts with: a lib namespace (Pine's `str`
        is pynecore's `string`), or a built-in series used as a method receiver
        (`close.isUp()`). An alias head is imported by generate_imports."""
        if module in PYNECORE_LIB_MODULES:
            self.lib_modules.add(module)
        elif module == 'str':
            self.lib_modules.add('string')
        elif is_series_variable(module) and module not in self.declared_names:
            self.builtin_variables.add(module)
            self.uses_series = True

    def add_drawing_type(self, type_name: str) -> None:
        """Manually add a drawing type that was detected."""
        if type_name in DRAWING_TYPES:
            self.drawing_types.add(type_name)
        elif type_name in CHART_TYPES:
            self.chart_types.add(type_name)

    def add_module(self, module_name: str) -> None:
        """Manually add a module dependency."""
        if module_name in PYNECORE_LIB_MODULES:
            self.lib_modules.add(module_name)

    #: built-in namespace alias of a library -> the name the library module is imported under
    shadowed_namespaces: dict = None

    def library_import_lines(self) -> List[str]:
        """`import user/lib/version as alias` is the Python import of the module `lib.<user>.<lib>.v<version>` under
        `alias`: the user and the version are part of the module path, so two libraries of one
        name from two users or in two versions are two modules. A segment Python cannot spell (a keyword, `class`)
        takes the keyword spelling of a name, `class__ren__`, which is the folder the library is compiled to. An alias
        that is a keyword is renamed the same way (by the scope renamer, as is one spelled like `lib` or a helper)."""
        lines: List[str] = []
        self.shadowed_namespaces = {}
        for imported in self.library_imports:
            alias = imported.alias or imported.library
            path = library_module_path(imported.user, imported.library, imported.version)
            if alias in self.lib_modules | self.builtin_variables | PYNECORE_LIB_MODULES:
                # an alias that is a built-in namespace (`import TradingView/ta/12` as `ta`): the module is
                # imported aside and `ta__ren__ = shadowed_namespace(ta__ren____lib, ta)` reads the library
                # first and the built-in after; every `ta.` of the script goes through it
                self.shadowed_namespaces[alias] = f'{alias}__ren____lib'
                line = f'import {path} as {alias}__ren____lib'
            else:
                line = f'import {path} as {alias}'
            if line not in lines:
                lines.append(line)
        return lines

    def generate_imports(self) -> List[str]:
        """Generate import statements in the correct order."""
        imports = self.library_import_lines()

        # 0. Typing imports (if used)
        if self.uses_any:
            imports.append("from typing import Any")

        # 0b. A UDT field default read per instance: field(default_factory=...)
        if self.uses_field_factory:
            imports.append(FIELD_FACTORY_IMPORT)
        if getattr(self, 'uses_enum', False):
            imports.append('from enum import StrEnum as Enum')

        # 1. Pine range and the re-reading loop counter (if used)
        if self.pine_loop_name == 'pine_loop':
            imports.append("from pynecore import pine_loop")
        elif self.pine_loop_name:
            imports.append(f"from pynecore import pine_loop as {self.pine_loop_name}")
        if self.uses_pine_range and self.pine_range_name == 'pine_range':
            imports.append("from pynecore import pine_range")
        elif self.uses_pine_range:
            imports.append(f"from pynecore import pine_range as {self.pine_range_name}")

        # 2. Pine cast (if used)
        if self.uses_cast_int or self.uses_cast_float or self.uses_cast_bool or self.uses_cast_string:
            cast_funcs = []
            if self.uses_cast_bool:
                cast_funcs.append('cast_bool')
            if self.uses_cast_string:
                cast_funcs.append('cast_string')
            if self.uses_cast_float:
                cast_funcs.append('cast_float')
            if self.uses_cast_int:
                cast_funcs.append('cast_int')
            imports.append(f"from pynecore.core.pine_cast import {', '.join(sorted(cast_funcs))}")

        # 3. Pine method (only import what's used)
        if self.uses_method or self.uses_method_call:
            method_imports = []
            if self.uses_method:
                method_imports.append('method')
            if self.uses_method_call:
                method_imports.append('method_call')
            imports.append(f"from pynecore.core.pine_method import {', '.join(method_imports)}")

        # 3a. A library's export surface
        if self.uses_export:
            imports.append("from pynecore.core.pine_export import Exported, export")

        # 3b. Overload dispatcher for same-name declarations
        if self.uses_overload:
            imports.append("from pynecore.core.overload import overload")

        # 4. Pine UDT (only import what's used)
        if self.uses_udt or self.uses_udt_copy:
            udt_imports = []
            if self.uses_udt:
                udt_imports.append('udt')
            if self.uses_udt_copy:
                udt_imports.append('udt_copy')
            imports.append(f"from pynecore.core.pine_udt import {', '.join(udt_imports)}")

        # 5. PyneCore core types - Series only (Persistent moved to pynecore.types)
        # Note: Series import is now handled in types import section with Persistent

        # 6. Main library imports + built-in variables (combined into multi-line statement)
        # Ensure na is imported when used as bare identifier (e.g., f = na)
        if self.uses_na:
            self.lib_modules.add('na')
            self.lib_modules.add('NA')

        all_lib_imports = []
        if self.lib_modules:
            all_lib_imports.extend(sorted(self.lib_modules))
        if self.builtin_variables:
            all_lib_imports.extend(sorted(self.builtin_variables))

        # An aliased namespace follows the plain names (`size as _size_module`)
        aliased_imports = [f"{module} as {alias}" for module, alias in sorted(self.aliased_modules.items())]

        if all_lib_imports or aliased_imports:
            combined_imports = sorted(set(all_lib_imports)) + aliased_imports

            # Generate multi-line import statement like ground truth
            # Format: from pynecore.lib import (\n    item1, item2, item3,\n    item4, item5\n)
            import_str = "from pynecore.lib import (\n    "
            import_str += ', '.join(combined_imports)
            import_str += "\n)"
            imports.append(import_str)

        # 8. Plot styles (if used)
        if self.uses_plot_style:
            imports.append("from pynecore.lib import plot_style")

        # 9. Types from pynecore.types (drawing types + Persistent + Series + Matrix if used)
        types_import = []
        if self.drawing_types:
            types_import.extend(sorted(self.drawing_types))
        if self.uses_persistent:
            types_import.append('Persistent')
        if self.uses_persistent_series:
            types_import.append('PersistentSeries')
        if self.uses_series:
            types_import.append('Series')
        # Matrix is always imported if we have drawing types or persistent (it's a common type)
        # Actually, only import if explicitly used

        if types_import:
            imports.append(f"from pynecore.types import {', '.join(sorted(set(types_import)))}")

        # 10. Chart types (if used)
        if self.chart_types:
            types = sorted(self.chart_types)
            imports.append(f"from pynecore.types.chart import {', '.join(types)}")

        return imports
