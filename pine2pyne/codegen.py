"""
Code generator: Emit Python source code from transformed AST.

Generates PEP 8 compliant Python code compatible with PyneCore.
"""
import re
from typing import Dict, List, Union
from .ast_nodes import *
from .transformer import PyneTransformedScript
from .errors import CodeGenError
from .udt_field_defaults import field_value
from .ast_walk import walk
from .strict_boolean import strict_operands

# Pine binary operator precedence, lowest to highest, mirroring the parser's
# chain parse_or -> parse_multiplication. Every level is left-associative.
BINARY_OPERATOR_PRECEDENCE = {
    'or': 1,
    'and': 2,
    '==': 3, '!=': 3,
    '<': 4, '>': 4, '<=': 4, '>=': 4,
    '+': 5, '-': 5,
    '*': 6, '/': 6, '%': 6,
}
COMPARISON_OPERATORS = frozenset({'==', '!=', '<', '>', '<=', '>='})


class CodeGenerator:
    """Generates Python source code from transformed AST."""

    def __init__(self, indent_size: int = 4, keep_keyword_string_quotes: bool = False):
        self.indent_size = indent_size
        # False: a keyword argument's string is double-quoted (pine2pyne's layout);
        # True: it keeps the quotes of the Pine source
        self.keep_keyword_string_quotes = keep_keyword_string_quotes
        self.indent_level = 0
        self.output: List[str] = []
        # The Pine line of the top-level statement being emitted. The parser
        # stamps only top-level statements, so a refusal of a nested node
        # reports this line
        self.top_level_line = 0
        # Pine v5 and earlier evaluate both operands of and/or
        self.strict_boolean_operators = False
        # Pine v6 gives a UDT bool field without a default false; before v6 it is na
        self.bool_is_never_na = True
        self.user_function_names: frozenset = frozenset()
        # Output line index -> the Pine line its statement came from. A layout
        # printer (pine2pyne.emit) reads it to place blank lines the way the source does
        self.line_origins: Dict[int, int] = {}

    @staticmethod
    def _esc(name: str) -> str:
        """Escape Python reserved words used as identifiers."""
        from .pine_builtins import rename_reserved_word
        return rename_reserved_word(name)

    def _get_all_type_hints(self, script) -> List[str]:
        """Extract all type hints from script for Any detection."""
        hints = []

        # Check main parameters
        for param in script.main_params:
            if param.type_hint:
                hints.append(param.type_hint)

        # Check main body statements
        for stmt in script.main_body:
            if hasattr(stmt, 'type_hint') and stmt.type_hint:
                hints.append(stmt.type_hint)

        # Check global vars
        for var in script.global_vars:
            if hasattr(var, 'type_hint') and var.type_hint:
                hints.append(var.type_hint)

        # Check helper functions
        for func in script.helper_functions:
            if func.return_type:
                hints.append(func.return_type)
            for param in func.params:
                if param.type_hint:
                    hints.append(param.type_hint)
            # Check function body (handle both list and single statement)
            body = func.body if isinstance(func.body, list) else [func.body]
            for stmt in body:
                if hasattr(stmt, 'type_hint') and stmt.type_hint:
                    hints.append(stmt.type_hint)

        return hints

    def generate(self, script: PyneTransformedScript) -> str:
        """Generate complete Python source code."""
        self.output = []
        self.line_origins = {}
        self.top_level_line = 0
        self.strict_boolean_operators = script.strict_boolean_operators
        self.bool_is_never_na = script.bool_is_never_na
        self.user_function_names = frozenset(
            node.name for node in walk([script.helper_functions, script.main_body, script.global_vars])
            if isinstance(node, FuncDecl))

        # 1. @pyne magic docstring with attribution
        self.emit('"""')
        self.emit('@pyne')
        self.emit('')
        self.emit('This code was compiled by Pine2Pyne — the Pine Script to PyneCore\'s Python compiler.')        
        self.emit('"""')
        self.emit('')

        # 2. Import statements
        # Check if Any is used in type hints
        uses_any = any('[Any]' in stmt or ': Any' in stmt for stmt in self._get_all_type_hints(script))

        if uses_any:
            self.emit('from typing import Any')

        for import_stmt in script.imports:
            self.emit(import_stmt)
        if script.imports or uses_any:
            self.emit('')

        # 3-5. Module prelude in three phases: enum classes, then the module
        # constants (they may read an enum member), then the UDT classes (a field default may
        # read an enum member, and cannot read a constant)
        # Enum declarations
        for enum_decl in script.enum_declarations:
            self.top_level_line = enum_decl.line
            self.generate_enum_declaration(enum_decl)
            self.emit('')

        # Module-level constants (e.g., const string group labels, pre-main literals)
        # These must be at module level because they are referenced by main()
        # parameter defaults which Python evaluates at function definition time.
        if script.module_constants:
            for const_stmt in script.module_constants:
                self.generate_top_level_statement(const_stmt)
            self.emit('')

        # Type declarations (custom types with @udt)
        self._udt_names = {type_decl.name for type_decl in script.type_declarations}
        # Pine v6: a gap in the source between two fields is a blank line between them
        self._keeps_field_gaps = getattr(script, 'pine_version', 6) >= 6
        for type_decl in script.type_declarations:
            self.top_level_line = type_decl.line
            self.generate_type_declaration(type_decl)
            self.emit('')

        # A library's export surface: one Exported proxy per exported name, published in __all__.
        # pynecore's `@export` resolves the proxy by name in this module's globals
        exported_names = list(dict.fromkeys(
            statement.name for statement in script.main_body if isinstance(statement, FuncDecl) and statement.is_export))
        if exported_names:
            for exported_name in exported_names:
                self.emit(f'{self._esc(exported_name)} = Exported()')
            self.emit('')
            self.emit('__all__ = [' + ', '.join(f"'{self._esc(name)}'" for name in exported_names) + ']')
            self.emit('')

        # 4. Main function with decorator (helper functions now inside main)
        if script.main_decorator:
            self.top_level_line = script.main_decorator.line
            decorator = self.generate_decorator(script.main_decorator)
            self.emit(decorator)

        # Generate main function signature
        self.emit('def main(')
        self.indent_level += 1

        # Generate parameters
        if script.main_params:
            for i, param in enumerate(script.main_params):
                self.top_level_line = param.line  # the Pine input it came from
                param_str = self.generate_parameter(param)
                if i < len(script.main_params) - 1:
                    param_str += ','
                self.emit(param_str)
        else:
            # No parameters - but Python requires at least empty ()
            pass

        self.indent_level -= 1

        # Close parameter list and add colon
        if script.main_params:
            self.emit('):')
        else:
            self.output[-1] += '):'  # Add closing paren and colon to def line

        self.indent_level += 1

        main_body_start = len(self.output)

        # Generate global var declarations (Persistent)
        for var_decl in script.global_vars:
            self.generate_top_level_statement(var_decl)

        # Generate main body with blank lines between statements
        # Rule: add blank line after function defs and between different statement kinds
        prev_kind = None  # 'persistent', 'func', 'body'
        prev_source_line = 0
        for i, stmt in enumerate(script.main_body):
            source_line = getattr(stmt, 'line', 0)

            # Determine statement kind
            if isinstance(stmt, Assignment) and isinstance(stmt.type_hint, str) and 'Persistent' in stmt.type_hint:
                kind = 'persistent'
            elif isinstance(stmt, FuncDecl):
                kind = 'func'
            else:
                kind = 'body'

            # Add blank lines based on context
            if i > 0:
                gap = source_line - prev_source_line if source_line > 0 and prev_source_line > 0 else 0

                if prev_kind == 'func':
                    # Always blank line after function definition
                    self.emit('')
                elif prev_kind != kind:
                    # Transition between statement types
                    if gap > 5:
                        # Large gap (section change) - add extra blank lines
                        blanks = min((gap - 1) // 10 + 1, 3)
                        for _ in range(blanks):
                            self.emit('')
                    else:
                        self.emit('')
                elif kind == 'persistent' and gap >= 3:
                    # Gap between Persistent groups (e.g., maps → arrays)
                    self.emit('')
                elif kind == 'body' and gap >= 3:
                    # Gap between body statements (comment/blank line in source)
                    self.emit('')

            self.generate_top_level_statement(stmt)
            prev_kind = kind
            prev_source_line = source_line

        # A script with no executable statement (a type-only or comment-only
        # Pine script) still needs a statement under `def main():`
        if len(self.output) == main_body_start:
            self.emit('pass')

        self.indent_level -= 1

        return '\n'.join(self.output)

    # ========================================================================
    # Emission helpers
    # ========================================================================

    def emit(self, line: str = '') -> None:
        """Emit a line of code with current indentation."""
        if line:
            indent = ' ' * (self.indent_level * self.indent_size)
            self.output.append(indent + line)
        else:
            self.output.append('')

    def emit_inline(self, text: str) -> None:
        """Emit text without newline (append to last line)."""
        if self.output:
            self.output[-1] += text
        else:
            self.output.append(text)

    # ========================================================================
    # Top-level code generation
    # ========================================================================

    def generate_decorator(self, decl: Union[IndicatorDecl, StrategyDecl, LibraryDecl]) -> str:
        """Generate @script.indicator, @script.strategy or @script.library decorator."""
        name = {IndicatorDecl: 'indicator', LibraryDecl: 'library'}.get(type(decl), 'strategy')
        # A None title is an expression, emitted as the 'title' keyword
        arguments = [] if decl.title is None else [f"'{self._escape_string(decl.title, chr(39))}'"]
        arguments += [f'{key}={self.generate_decorator_value(value)}' for key, value in decl.kwargs.items()]
        return f"@script.{name}({', '.join(arguments)})"

    def generate_decorator_value(self, value: Any) -> str:
        """Generate decorator kwarg value."""
        if isinstance(value, bool):
            return 'True' if value else 'False'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, Literal):
            return self.generate_literal(value)
        elif isinstance(value, MemberAccess):
            return self.generate_member_access(value)
        elif isinstance(value, Identifier):
            return value.name
        elif isinstance(value, Expression):
            # Handle any other expression type
            return self.generate_expression(value)
        else:
            return str(value)

    def generate_parameter(self, param: Parameter) -> str:
        """Generate function parameter."""
        param_str = self._esc(param.name)

        # Add type hint
        if param.type_hint:
            param_str += f': {param.type_hint}'

        # Add default value
        if param.default:
            default_str = self.generate_expression(param.default)
            # Fix NA() to na for function default parameters
            if default_str == 'NA()':
                default_str = 'na'
            # PEP 8: spaces around = in defaults when type annotation present
            if param.type_hint:
                param_str += f' = {default_str}'
            else:
                param_str += f'={default_str}'

        return param_str

    # ========================================================================
    # Type declaration generation
    # ========================================================================

    def generate_type_declaration(self, type_decl: TypeDecl) -> None:
        """Generate custom type definition with @udt decorator."""
        from .transformer import convert_pine_generic_to_python
        from .pine_builtins import get_type_name

        # Emit @udt decorator
        self.emit('@udt')
        self.emit(f'class {self._esc(type_decl.name)}:')
        self.indent_level += 1

        # Generate fields: a series built-in default is read per instance, a
        # field without a default is na (udt_field_defaults)
        lines = list(getattr(type_decl, 'field_lines', ()) or ())
        for index, (field_name, field_type, default_value) in enumerate(type_decl.fields):
            if getattr(self, '_keeps_field_gaps', False) and 0 < index < len(lines) and lines[index] - lines[index - 1] > 1:
                self.emit('')
            # Convert Pine type to Python type (handles generics like array<T> → list[T])
            py_type = convert_pine_generic_to_python(field_type)
            fname = self._esc(field_name)

            # For NA() calls, use base type only (NA(list) not NA(list[X]))
            na_type = py_type.split('[')[0] if '[' in py_type else py_type

            # A field that holds a UDT (its own class, or one declared later) is annotated with a
            # string, and its na is made per instance (`p: 'P' = field(default_factory=...)`)
            self_referential = re.search(rf'\b{re.escape(type_decl.name)}\b', py_type) is not None
            holds_a_udt = na_type in getattr(self, '_udt_names', ()) or na_type == type_decl.name
            # and so is a collection of one (`tp: 'list[PosAction]' = na(list)`); an enum is not
            mentions_a_udt = any(re.search(rf'\b{re.escape(name)}\b', py_type) for name in getattr(self, '_udt_names', ()))
            annotation = f"'{py_type}'" if self_referential or holds_a_udt or mentions_a_udt else py_type
            generated_default = (self.generate_expression(default_value)
                                 if default_value is not None and not isinstance(default_value, NaLiteral) else None)
            value = field_value(py_type, na_type, default_value, generated_default, holds_a_udt,
                                self.bool_is_never_na)
            self.emit(f'{fname}: {annotation} = {value}')

        self.indent_level -= 1

    def generate_enum_declaration(self, enum_decl: EnumDecl) -> None:
        """Generate an enum: a `StrEnum` (imported as `Enum`) whose members
        are their titles, a member without one its own name (`a = 'a'`)."""
        self.uses_enum = True
        self.emit(f'class {self._esc(enum_decl.name)}(Enum):')
        self.indent_level += 1

        for member in enum_decl.members:
            if isinstance(member, tuple):
                name, value = member
                value_str = (repr(value.value) if isinstance(value, Literal) and isinstance(value.value, str)
                             else self.generate_expression(value))
                self.emit(f'{self._esc(name)} = {value_str}')
            else:
                # No explicit value — use member name as string value
                self.emit(f'{self._esc(member)} = {member!r}')

        self.indent_level -= 1

    # ========================================================================
    # Function generation
    # ========================================================================

    def generate_function(self, func: FuncDecl) -> None:
        """Generate function definition."""
        # Emit @method decorator if this is a method
        if hasattr(func, 'is_method') and func.is_method:
            self.emit('@method')
        # A library's exported function; pynecore's @export goes above @overload
        if func.is_export:
            self.emit('@export')
        # Same-name declarations share one pynecore @overload dispatcher
        if func.is_overload:
            self.emit('@overload')

        # Function signature
        params = ', '.join(self.generate_parameter(p) for p in func.params)
        self.emit(f'def {self._esc(func.name)}({params}):')

        self.indent_level += 1

        # Track if we emitted any body content
        body_was_empty = False

        # Function body
        if isinstance(func.body, (list, tuple)):
            for i, stmt in enumerate(func.body):
                is_last = (i == len(func.body) - 1)
                # Last expression in body is the return value
                if is_last and isinstance(stmt, Expression):
                    self.emit(f'return {self.generate_expression(stmt)}')
                else:
                    self.generate_statement(stmt)
            # If body list was empty or all statements were None, add pass
            if not func.body or all(s is None for s in func.body):
                self.emit('pass')
                body_was_empty = True
        elif isinstance(func.body, Expression):
            # Single expression - add return
            self.emit(f'return {self.generate_expression(func.body)}')
        elif isinstance(func.body, ReturnStatement):
            self.generate_statement(func.body)
        else:
            # Empty or None body - add pass
            self.emit('pass')
            body_was_empty = True

        self.indent_level -= 1

    # ========================================================================
    # Statement generation
    # ========================================================================

    def generate_top_level_statement(self, stmt: Statement) -> None:
        """Generate a statement of the module or of main(), recording its Pine line."""
        self.top_level_line = getattr(stmt, 'line', 0)
        self.generate_statement(stmt)

    def refusal_line(self, node: ASTNode) -> int:
        """The Pine line to report for a node codegen refuses: its own, or its
        top-level statement's when the parser did not stamp it."""
        return getattr(node, 'line', 0) or self.top_level_line

    def generate_statement(self, stmt: Statement) -> None:
        """Generate a statement."""
        # Flatten lists of statements (from transforms that expand one stmt into multiple)
        if isinstance(stmt, list):
            for s in stmt:
                self.generate_statement(s)
            return
        pine_line = getattr(stmt, 'line', 0) or getattr(stmt, 'source_line', 0)
        if pine_line:
            self.line_origins.setdefault(len(self.output), pine_line)
        if isinstance(stmt, FuncDecl):
            # Functions can now appear inside main()
            self.generate_function(stmt)
            self.emit('')  # Add blank line after function
        elif isinstance(stmt, Assignment):
            self.generate_assignment(stmt)
        elif isinstance(stmt, IfStatement):
            self.generate_if_statement(stmt)
        elif isinstance(stmt, ForLoop):
            self.generate_for_loop(stmt)
        elif isinstance(stmt, ForInLoop):
            self.generate_for_in_loop(stmt)
        elif isinstance(stmt, WhileLoop):
            self.generate_while_loop(stmt)
        elif isinstance(stmt, BreakStatement):
            self.emit('break')
        elif isinstance(stmt, ContinueStatement):
            self.emit('continue')
        elif isinstance(stmt, ReturnStatement):
            self.generate_return_statement(stmt)
        elif isinstance(stmt, ExpressionStatement):
            self.emit(self.generate_expression(stmt.expr))
        elif isinstance(stmt, RawCode):
            for line in stmt.code.split('\n'):
                self.emit(line)
        else:
            # Refuse rather than drop the statement: a comment in its place
            # changes what the program computes, or leaves a block empty
            raise CodeGenError(f'Unsupported statement type: {type(stmt).__name__}',
                               line=self.refusal_line(stmt))

    def generate_assignment(self, stmt: Assignment) -> None:
        """Generate assignment statement."""
        from .ast_nodes import BinaryOp, Identifier, MemberAccess, TupleDestructure

        # Handle tuple destructuring: [a, b, c] = expr
        if isinstance(stmt.target, TupleDestructure):
            names = ', '.join(self._esc(n) for n in stmt.target.names)
            if len(stmt.target.names) == 1:
                names += ','  # `[sig] = f()`: the one element of the function's tuple, `sig, = f()`
            value = self.generate_expression(stmt.value)
            type_hint = f': {stmt.type_hint}' if stmt.type_hint else ''
            self.emit(f'{names}{type_hint} = {value}')
            return

        target = self._esc(stmt.target if isinstance(stmt.target, str) else str(stmt.target))

        # A compound assignment the source wrote as one (x += y), not x := x + y
        if isinstance(stmt.value, BinaryOp) and stmt.value.compound and not stmt.type_hint:
            left_matches = False

            # Check if the left side of the BinaryOp matches the target
            if isinstance(stmt.value.left, Identifier):
                left_matches = (stmt.value.left.name == target)
            elif isinstance(stmt.value.left, MemberAccess):
                # For member access like this.wins, reconstruct the full name
                left_name = f"{stmt.value.left.object}.{stmt.value.left.member}"
                left_matches = (left_name == target)

            # If it's a compound assignment pattern, generate +=, -=, etc.
            if left_matches and stmt.value.op in ['+', '-', '*', '/', '%']:
                compound_op = f'{stmt.value.op}='
                right_value = self.generate_expression(stmt.value.right)
                self.emit(f'{target} {compound_op} {right_value}')
                return

        # Regular assignment
        value = self.generate_expression(stmt.value)

        # Wrap string literals assigned to Color type as Color('...')
        if stmt.type_hint == 'Color' and value.startswith("'") and value.endswith("'"):
            value = f'Color({value})'

        # Add type hint if present
        if stmt.type_hint:
            self.emit(f'{target}: {stmt.type_hint} = {value}')
        else:
            self.emit(f'{target} = {value}')

    def generate_if_statement(self, stmt: IfStatement) -> None:
        """Generate if statement."""
        condition = self.generate_expression(stmt.condition)
        self.emit(f'if {condition}:')

        self.indent_level += 1
        for s in stmt.body:
            self.generate_statement(s)
        self.indent_level -= 1

        # Generate elif clauses
        for elif_cond, elif_body in stmt.elseifs:
            elif_condition = self.generate_expression(elif_cond)
            self.emit(f'elif {elif_condition}:')
            self.indent_level += 1
            for s in elif_body:
                self.generate_statement(s)
            self.indent_level -= 1

        # Generate else clause
        if stmt.else_body:
            self.emit('else:')
            self.indent_level += 1
            for s in stmt.else_body:
                self.generate_statement(s)
            self.indent_level -= 1

    def generate_for_loop(self, stmt: ForLoop) -> None:
        """Generate for loop with pine_range(), or with pine_loop() when the end
        bound is re-read on every pass (Pine v6)."""
        # Convert Pine 'for i = from to to_val [by step]' to Python pine_range()
        from_val = self.generate_expression(stmt.from_val)
        to_val = self.generate_expression(stmt.to_val)

        # Pine's 'to' is inclusive, pine_range handles this correctly
        # No need to add 1 like we did with range()
        var = self._esc(stmt.var)
        if stmt.counter_name and stmt.counter_entry:
            # from, then the end read once on entry, then by, then the end before every pass
            step = self.generate_expression(stmt.step) if stmt.step else ''
            self.emit(f'{stmt.counter_name} = {stmt.counter_helper}({from_val}, entry=True)')
            self.emit(f'while {stmt.counter_name}.step({to_val}):')
            self.indent_level += 1
            self.emit(f'if {stmt.counter_name}.entering:')
            self.indent_level += 1
            self.emit(f'{stmt.counter_name}.enter({step})')
            self.emit('continue')
            self.indent_level -= 1
            self.emit(f'{var} = {stmt.counter_name}.value')
            for s in stmt.body:
                self.generate_statement(s)
            self.indent_level -= 1
            return
        if stmt.counter_name:
            # from and by are evaluated once, the end before every pass
            step = f', {self.generate_expression(stmt.step)}' if stmt.step else ''
            self.emit(f'{stmt.counter_name} = {stmt.counter_helper}({from_val}{step})')
            self.emit(f'while {stmt.counter_name}.step({to_val}):')
            self.indent_level += 1
            self.emit(f'{var} = {stmt.counter_name}.value')
            for s in stmt.body:
                self.generate_statement(s)
            self.indent_level -= 1
            return
        range_helper = stmt.range_helper or 'pine_range'
        if stmt.step:
            step = self.generate_expression(stmt.step)
            self.emit(f'for {var} in {range_helper}({from_val}, {to_val}, {step}):')
        else:
            self.emit(f'for {var} in {range_helper}({from_val}, {to_val}):')

        self.indent_level += 1
        for s in stmt.body:
            self.generate_statement(s)
        self.indent_level -= 1

    def generate_for_in_loop(self, stmt: ForInLoop) -> None:
        """Generate for-in loop."""
        vars_str = ', '.join(self._esc(v) for v in stmt.vars)
        iterable = self.generate_expression(stmt.iterable)

        # `for [k, v] in x` reaches here with its pairs already chosen by the
        # transformer: enumerate(), a map's keys, or a run-time choice
        self.emit(f'for {vars_str} in {iterable}:')

        self.indent_level += 1
        for s in stmt.body:
            self.generate_statement(s)
        self.indent_level -= 1

    def generate_while_loop(self, stmt: WhileLoop) -> None:
        """Generate while loop."""
        condition = self.generate_expression(stmt.condition)
        self.emit(f'while {condition}:')

        self.indent_level += 1
        for s in stmt.body:
            self.generate_statement(s)
        self.indent_level -= 1

    def generate_return_statement(self, stmt: ReturnStatement) -> None:
        """Generate return statement."""
        if stmt.expr:
            self.emit(f'return {self.generate_expression(stmt.expr)}')
        else:
            self.emit('return')

    # ========================================================================
    # Expression generation
    # ========================================================================

    def generate_expression(self, expr: Expression) -> str:
        """Generate an expression."""
        if isinstance(expr, BinaryOp):
            return self.generate_binary_op(expr)
        elif isinstance(expr, UnaryOp):
            return self.generate_unary_op(expr)
        elif isinstance(expr, TernaryOp):
            return self.generate_ternary_op(expr)
        elif isinstance(expr, FunctionCall):
            return self.generate_function_call(expr)
        elif isinstance(expr, MethodCall):
            return self.generate_method_call(expr)
        elif isinstance(expr, MemberAccess):
            return self.generate_member_access(expr)
        elif isinstance(expr, IndexAccess):
            return self.generate_index_access(expr)
        elif isinstance(expr, ArrayLiteral):
            return self.generate_array_literal(expr)
        elif isinstance(expr, Identifier):
            return self.generate_identifier(expr)
        elif isinstance(expr, NamedExpression):
            return self.generate_named_expression(expr)
        elif isinstance(expr, Literal):
            return self.generate_literal(expr)
        elif isinstance(expr, NaLiteral):
            return 'na'
        else:
            # A comment in expression position is not Python; refuse
            raise CodeGenError(f'Unsupported expression type: {type(expr).__name__}',
                               line=self.refusal_line(expr))

    def generate_identifier(self, expr: Identifier) -> str:
        """Generate identifier."""
        from .pine_builtins import TA_VARIABLE_FUNCTIONS

        name = self._esc(expr.name)

        # Pine Script "variable-functions": used without () in Pine Script
        # but implemented as functions in PyneCore (e.g., ta.vwap -> ta.vwap())
        if name in TA_VARIABLE_FUNCTIONS:
            # The table's value is the argument text; every entry is '' today
            defaults = TA_VARIABLE_FUNCTIONS.get(name, '')
            return f'{name}({defaults})'

        return name

    def generate_named_expression(self, expr: NamedExpression) -> str:
        """Generate `(name := value)`. Always parenthesised: a bare walrus is
        illegal in most positions, and the parentheses keep a comparison in
        `value` from chaining with an enclosing `==`."""
        return f'({self._esc(expr.name)} := {self.generate_expression(expr.value)})'

    def generate_binary_op(self, expr: BinaryOp) -> str:
        """Generate binary operation."""
        if self.strict_boolean_operators:
            strict = strict_operands(expr, self.user_function_names)
            if strict:
                return self._generate_strict_boolean(expr.op, strict)
        left = self.generate_expression(expr.left)
        right = self.generate_expression(expr.right)

        if self._operand_needs_parentheses(expr.op, expr.left, is_right_operand=False):
            left = f'({left})'
        if self._operand_needs_parentheses(expr.op, expr.right, is_right_operand=True):
            right = f'({right})'

        return f'{left} {expr.op} {right}'

    def _generate_strict_boolean(self, operator: str, operands: List[Expression]) -> str:
        """`all((a, b))` / `any((a, b))`: the tuple evaluates every operand, in order."""
        function = 'all' if operator == 'and' else 'any'
        return f'{function}(({", ".join(self.generate_expression(operand) for operand in operands)}))'

    @staticmethod
    def _operand_needs_parentheses(parent_operator: str, operand: Expression, is_right_operand: bool) -> bool:
        """Whether an operand must be parenthesised to keep its Pine grouping.

        The parser keeps no parenthesis node, so grouping exists only in the
        tree shape and every operator-precedence parenthesis between binary
        and unary operators is re-derived here (index and member access on a
        grouped object are not).
        """
        if isinstance(operand, UnaryOp):
            # Pine `not` binds tightest; Python `not` binds looser than any
            # comparison or arithmetic, so `q == not r` would not even parse.
            return operand.op == 'not'
        if not isinstance(operand, BinaryOp):
            return False
        if parent_operator in COMPARISON_OPERATORS and operand.op in COMPARISON_OPERATORS:
            # Python chains `a < b == c` into `a < b and b == c`.
            return True
        parent_precedence = BINARY_OPERATOR_PRECEDENCE[parent_operator]
        operand_precedence = BINARY_OPERATOR_PRECEDENCE[operand.op]
        if operand_precedence != parent_precedence:
            return operand_precedence < parent_precedence
        # Same level: Pine is left-associative, so a right operand at this
        # level was grouped explicitly (`a - (b - c)`, `a * (b % c)`).
        return is_right_operand

    def generate_unary_op(self, expr: UnaryOp) -> str:
        """Generate unary operation."""
        operand = self.generate_expression(expr.operand)

        # Convert Pine operators to Python
        op_map = {
            'not': 'not ',
        }
        op = op_map.get(expr.op, expr.op)

        # Pine unary operators bind tighter than every binary operator, so a
        # binary operand was grouped explicitly: -(a + b), not (a == b).
        if isinstance(expr.operand, BinaryOp):
            return f'{op}({operand})'

        return f'{op}{operand}'

    def generate_ternary_op(self, expr: TernaryOp) -> str:
        """Generate ternary operation (Python if/else expression)."""
        condition = self.generate_expression(expr.condition)
        true_expr = self.generate_expression(expr.true_expr)
        false_expr = self.generate_expression(expr.false_expr) if expr.false_expr else 'None'

        return f'({true_expr} if {condition} else {false_expr})'

    def generate_function_call(self, call: FunctionCall) -> str:
        """Generate function call."""
        # Generate function name
        if isinstance(call.func, str):
            func = call.func
        elif isinstance(call.func, MemberAccess):
            func = self.generate_member_access(call.func)
        elif isinstance(call.func, NaLiteral):
            # Special case: na(x) is a function call, not NA() literal
            func = 'na'
            # If NA() with no args, just return 'na' (for default params)
            if not call.args and not call.kwargs:
                return 'na'
        else:
            func = self.generate_expression(call.func)

        # Generate arguments (positional - use single quotes for strings)
        args = [self.generate_expression(arg) for arg in call.args]

        # Generate keyword arguments (use double quotes for string literals)
        kwargs = []
        for key, value in call.kwargs.items():
            # Special case: options parameter should use tuple () instead of list []
            if key == 'options' and isinstance(value, ArrayLiteral):
                elements = [self._generate_kwarg_value(e) for e in value.elements]
                kwargs.append(f'{key}=({", ".join(elements)})')
            else:
                kwargs.append(f'{key}={self._generate_kwarg_value(value)}')

        all_args = args + kwargs
        return f'{func}({", ".join(all_args)})'

    def _generate_kwarg_value(self, value: Expression) -> str:
        """Generate kwarg value with double quotes for strings (or the source's quotes)."""
        if isinstance(value, Literal) and value.literal_type == 'string' and not self.keep_keyword_string_quotes:
            return f'"{self._escape_string(value.value, chr(34))}"'
        elif isinstance(value, Literal) and value.literal_type in ('int', 'float', 'bool'):
            # For non-string literals, use normal generation
            return self.generate_literal(value)
        else:
            return self.generate_expression(value)

    def generate_method_call(self, call: MethodCall) -> str:
        """Generate method call. The transformer routes every builtin method to its
        namespace, so what reaches here is a Python method call."""
        from .ast_nodes import BinaryOp, UnaryOp, TernaryOp, IfExpression

        obj = self.generate_expression(call.object)
        args = [self.generate_expression(arg) for arg in call.args]
        kwargs = [f'{key}={self.generate_expression(value)}' for key, value in call.kwargs.items()]

        # FIX: Wrap complex expressions in parentheses to preserve precedence
        # This ensures (expression).method() keeps correct semantics
        if isinstance(call.object, (BinaryOp, UnaryOp, TernaryOp, IfExpression)):
            obj = f'({obj})'

        # Otherwise, keep as method call
        all_args = args + kwargs
        return f'{obj}.{call.method}({", ".join(all_args)})'

    def generate_member_access(self, expr: MemberAccess) -> str:
        """Generate member access."""
        from .pine_builtins import TA_VARIABLE_FUNCTIONS

        if isinstance(expr.object, str):
            obj = expr.object
        elif isinstance(expr.object, Expression):
            obj = self.generate_expression(expr.object)
        else:
            obj = str(expr.object)

        result = f'{obj}.{expr.member}'

        # Pine Script "variable-functions": used without () in Pine Script
        # but implemented as functions in PyneCore (e.g., ta.vwap -> ta.vwap())
        if result in TA_VARIABLE_FUNCTIONS:
            defaults = TA_VARIABLE_FUNCTIONS.get(result, '')
            result = f'{result}({defaults})'

        return result

    def generate_index_access(self, expr: IndexAccess) -> str:
        """Generate index access."""
        obj = self.generate_expression(expr.object)
        index = self.generate_expression(expr.index)
        return f'{obj}[{index}]'

    def generate_array_literal(self, expr: ArrayLiteral) -> str:
        """Generate a Pine tuple expression as a Python tuple.

        Pine writes tuple literals with square brackets and has no bracket
        syntax for arrays -- those come from ``array.new``/``array.from`` -- so
        every bracket that reaches here is a tuple, never a collection literal.
        Emitting a list instead makes the runtime take its array branch: it
        answers ``request.seed``/``request.security`` with a single ``na``
        rather than one ``na`` per element, and the caller's destructuring then
        dies with "na is not iterable".
        """
        elements = [self.generate_expression(e) for e in expr.elements]
        if len(elements) == 1:
            # A lone parenthesised value is not a tuple without the comma.
            return f'({elements[0]},)'
        return f'({", ".join(elements)})'

    def _escape_string(self, value: str, quote: str) -> str:
        """Escape a string value for Python output."""
        escaped = value.replace('\\', '\\\\')
        escaped = escaped.replace('\n', '\\n')
        escaped = escaped.replace('\t', '\\t')
        escaped = escaped.replace('\r', '\\r')
        escaped = escaped.replace(quote, f'\\{quote}')
        return escaped

    def generate_literal(self, expr: Literal) -> str:
        """Generate literal value."""
        if expr.literal_type == 'string':
            if expr.is_double_quoted:
                return f'"{self._escape_string(expr.value, chr(34))}"'
            else:
                return f"'{self._escape_string(expr.value, chr(39))}'"
        elif expr.literal_type == 'bool':
            return 'True' if expr.value else 'False'
        elif expr.literal_type == 'color':
            if expr.is_double_quoted:
                return f'"{self._escape_string(expr.value, chr(34))}"'
            else:
                return f"'{self._escape_string(expr.value, chr(39))}'"
        else:
            return str(expr.value)


def generate_code(script: PyneTransformedScript) -> str:
    """Convenience function to generate Python code from transformed script."""
    codegen = CodeGenerator()
    return codegen.generate(script)
