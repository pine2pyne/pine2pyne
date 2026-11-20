"""
AST (Abstract Syntax Tree) node definitions for Pine Script v4, v5 and v6.

All node types used by the parser to represent Pine Script syntax.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


# Base class for all AST nodes
@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    line: int = 0
    column: int = 0
    # the Pine line a statement built by a lowering is laid out at, where `line` stays unset
    source_line: int = field(default=0, compare=False, repr=False)
    # the last source line of a declaration the parser read (its value may continue on later lines)
    end_line: int = field(default=0, compare=False, repr=False)
    # the line a declaration's value starts on (`up =` with the value below: a later line)
    value_line: int = field(default=0, compare=False, repr=False)


# ============================================================================
# Top-level nodes
# ============================================================================

@dataclass
class Script(ASTNode):
    """Root node representing the entire Pine Script."""
    version: Optional['VersionAnnotation'] = None
    script_decl: Optional[Union['IndicatorDecl', 'StrategyDecl', 'LibraryDecl']] = None
    imports: List['ImportDecl'] = field(default_factory=list)
    declarations: List[Union['VarDecl', 'VaripDecl', 'TypeDecl', 'EnumDecl', 'FuncDecl', 'InputDecl']] = field(default_factory=list)
    body: List['Statement'] = field(default_factory=list)


@dataclass
class VersionAnnotation(ASTNode):
    """//@version=6 annotation."""
    version: str = ""  # "6" or "5"


@dataclass
class IndicatorDecl(ASTNode):
    """indicator("Title", ...) declaration."""
    title: Optional[str] = ""  # None: the title is an expression, kept in kwargs['title']
    kwargs: dict = field(default_factory=dict)
    # How the source wrote the call: how many arguments were positional, and the
    # title's quotes. Codegen ignores it; a layout printer may reproduce it
    positional_count: int = 0
    title_double_quoted: bool = False
    keyword_order: tuple = ()  # the keyword arguments' names as the source orders them


@dataclass
class StrategyDecl(ASTNode):
    """strategy("Title", ...) declaration."""
    title: Optional[str] = ""  # None: the title is an expression, kept in kwargs['title']
    kwargs: dict = field(default_factory=dict)
    # How the source wrote the call: how many arguments were positional, and the
    # title's quotes. Codegen ignores it; a layout printer may reproduce it
    positional_count: int = 0
    title_double_quoted: bool = False
    keyword_order: tuple = ()  # the keyword arguments' names as the source orders them


@dataclass
class LibraryDecl(ASTNode):
    """library("Title", ...) declaration."""
    title: Optional[str] = ""  # None: the title is an expression, kept in kwargs['title']
    kwargs: dict = field(default_factory=dict)
    # How the source wrote the call: how many arguments were positional, and the
    # title's quotes. Codegen ignores it; a layout printer may reproduce it
    positional_count: int = 0
    title_double_quoted: bool = False
    keyword_order: tuple = ()  # the keyword arguments' names as the source orders them


@dataclass
class ImportDecl(ASTNode):
    """import user/library/version as alias"""
    user: str = ""
    library: str = ""
    version: Optional[str] = None
    alias: Optional[str] = None


# ============================================================================
# Declarations
# ============================================================================

@dataclass
class VarDecl(ASTNode):
    """var [type] name = value"""
    name: str = ""
    value: 'Expression' = None
    type_hint: Optional[str] = None
    is_var: bool = True  # True for 'var', False for 'varip'


@dataclass
class VaripDecl(ASTNode):
    """varip [type] name = value"""
    name: str = ""
    value: 'Expression' = None
    type_hint: Optional[str] = None


@dataclass
class TypeDecl(ASTNode):
    """type MyType ... (User-defined type)"""
    name: str = ""
    fields: List[tuple] = field(default_factory=list)  # [(field_name, type_hint, default_value), ...] or [(field_name, type_hint), ...]
    field_lines: List[int] = field(default_factory=list)  # each field's Pine line
    is_export: bool = False  # `export type T`: part of a library's API


@dataclass
class EnumDecl(ASTNode):
    """enum MyEnum ... (Enum definition)"""
    name: str = ""
    members: List = field(default_factory=list)  # List[str] or List[tuple[str, Expression]]
    is_export: bool = False  # `export enum E`: part of a library's API


@dataclass
class FuncDecl(ASTNode):
    """Function declaration: name(params) => body"""
    name: str = ""
    params: List['Parameter'] = field(default_factory=list)
    body: Union['Expression', List['Statement']] = None
    return_type: Optional[str] = None
    is_method: bool = False
    is_export: bool = False
    is_overload: bool = False  # One of several same-name declarations
    indexed_params: set = field(default_factory=set)  # Params that use [n] history access
    value_is_bool: bool = False  # Pine v6: its value is of type bool


@dataclass
class Parameter(ASTNode):
    """Function parameter: [type] name [= default]"""
    name: str = ""
    type_hint: Optional[str] = None
    default: Optional['Expression'] = None
    is_series: bool = False  # declared `series T`: annotated Series[T]
    qualifier: Optional[str] = None  # the form keyword it was declared with: simple, series, const, input


@dataclass
class InputDecl(ASTNode):
    """Input declaration: name = input.*(...)"""
    name: str = ""
    func: str = ""  # e.g., "input.int"
    args: List['Expression'] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


# ============================================================================
# Expressions
# ============================================================================

@dataclass
class Expression(ASTNode):
    """Base class for all expressions."""
    pass


@dataclass
class BinaryOp(Expression):
    """Binary operation: left op right"""
    left: Expression = None
    op: str = ""
    right: Expression = None
    compound: bool = False  # written `x op= y`: printed that way, `x := x op y` is not
    # written in parentheses, `(a and b)`; not part of the node's value
    parenthesized: bool = field(default=False, compare=False, repr=False)
    # `x op= <if/switch value>` read into a block result: Pine v4/v5 keep the operator there too
    of_a_block_value: bool = field(default=False, compare=False, repr=False)


@dataclass
class UnaryOp(Expression):
    """Unary operation: op operand"""
    op: str = ""
    operand: Expression = None


@dataclass
class TernaryOp(Expression):
    """Ternary operation: condition ? true_expr : false_expr"""
    condition: Expression = None
    true_expr: Expression = None
    false_expr: Expression = None
    # written in parentheses, `(c ? a : b)`; not part of the node's value
    parenthesized: bool = field(default=False, compare=False, repr=False)
    # the Pine line of the `?` (the condition may end a line above the arms)
    question_line: int = field(default=0, compare=False, repr=False)
    # the Pine line of the `:` (the first arm may end a line above the second)
    colon_line: int = field(default=0, compare=False, repr=False)


@dataclass
class FunctionCall(Expression):
    """Function call: func(args, kwargs)"""
    func: Union[str, 'MemberAccess'] = ""  # Can be "ta.sma" or MemberAccess node
    args: List[Expression] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    # written in parentheses, `(f(x))`; not part of the node's value
    parenthesized: bool = field(default=False, compare=False, repr=False)
    # the keywords written more than once (`title = "a", title = "b"`): kwargs keeps the first value
    repeated_kwargs: list = field(default_factory=list, compare=False, repr=False)


@dataclass
class MethodCall(Expression):
    """Method call: object.method(args)"""
    object: Expression = None
    method: str = ""
    args: List[Expression] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


@dataclass
class MemberAccess(Expression):
    """Member access: object.member"""
    object: Union[str, Expression] = ""  # Can be identifier string or expression
    member: str = ""


@dataclass
class IndexAccess(Expression):
    """Index access: object[index]"""
    object: Expression = None
    index: Expression = None
    # written in parentheses, `(t[1])`; not part of the node's value
    parenthesized: bool = field(default=False, compare=False, repr=False)


@dataclass
class ArrayLiteral(Expression):
    """Array literal: [elem1, elem2, ...]"""
    elements: List[Expression] = field(default_factory=list)


@dataclass
class TupleDestructure(Expression):
    """Tuple destructuring: [a, b, c] = expr"""
    names: List[str] = field(default_factory=list)
    value: Expression = None


@dataclass
class Identifier(Expression):
    """Identifier (variable name)"""
    name: str = ""
    # written in parentheses, `(s)`; not part of the node's value
    parenthesized: bool = field(default=False, compare=False, repr=False)
    # written `x[0]`, the current bar's value: still a history read for Pine v5's na_bool
    read_at_zero: bool = field(default=False, compare=False, repr=False)


@dataclass
class NamedExpression(Expression):
    """Transformer output only: binds `name` to `value` and is that value
    (Python `(name := value)`). A switch subject is bound this way in the
    first case test, so it is evaluated once, where Pine evaluates it."""
    name: str = ""
    value: Expression = None


@dataclass
class Literal(Expression):
    """Literal value (int, float, string, bool, color)"""
    value: Any = None
    literal_type: str = ""  # 'int', 'float', 'string', 'bool', 'color'
    is_double_quoted: bool = False  # Track if original source used double quotes


@dataclass
class NaLiteral(Expression):
    """The 'na' literal value"""
    pass


@dataclass
class IfExpression(Expression):
    """If expression (returns a value): if cond\n    val1\nelse\n    val2"""
    condition: Expression = None
    true_expr: Union[Expression, List['Statement']] = None
    false_expr: Optional[Union[Expression, List['Statement']]] = None
    value_is_bool: bool = False  # Pine v6: its value is of type bool
    chained: bool = False  # written `else if` after the if before it (not an `if` inside an `else` block)


@dataclass
class SwitchExpression(Expression):
    """Switch expression (returns a value)"""
    expr: Optional[Expression] = None
    cases: List[tuple[Expression, Union[Expression, List['Statement']]]] = field(default_factory=list)
    default: Optional[Union[Expression, List['Statement']]] = None
    value_is_bool: bool = False  # Pine v6: its value is of type bool


# ============================================================================
# Statements
# ============================================================================

@dataclass
class Statement(ASTNode):
    """Base class for all statements."""
    pass


@dataclass
class Assignment(Statement):
    """Assignment statement: target = value"""
    target: Union[str, 'TupleDestructure'] = ""
    value: Expression = None
    type_hint: Optional[str] = None
    is_const: bool = False  # declared `const T name = ...`: a top-level one moves to the module
    # the form keyword written before the type (`simple float x = ...`): simple, series, const or input
    qualifier: Optional[str] = field(default=None, compare=False, repr=False)


@dataclass
class Reassignment(Statement):
    """Reassignment statement: target := value"""
    target: str = ""
    value: Expression = None


@dataclass
class IfStatement(Statement):
    """If statement: if cond\n    body\nelse if ...\nelse ..."""
    condition: Expression = None
    body: List[Statement] = field(default_factory=list)
    elseifs: List[tuple[Expression, List[Statement]]] = field(default_factory=list)
    else_body: Optional[List[Statement]] = None
    #: the if chain a `switch` without a subject runs: 'statement' for a switch statement, 'value' for one whose
    #: value is used; '' for an if statement
    switch_arms: str = ''
    #: the if of an if expression whose value is assigned (`x = if c`)
    of_an_assigned_value: bool = field(default=False, compare=False, repr=False)


@dataclass
class ForLoop(Statement):
    """For loop: for var = from to to_val [by step]"""
    var: str = ""
    from_val: Expression = None
    to_val: Expression = None
    step: Optional[Expression] = None
    body: List[Statement] = field(default_factory=list)
    value_is_bool: bool = False  # Pine v6: its value is of type bool
    # Pine v6, an end bound that can change between passes: the name of the runtime
    # counter (`pine_loop`) that re-reads it; empty for a fixed `pine_range`
    counter_name: str = ""
    counter_helper: str = ""  # the name `pine_loop` is imported under (an alias when the script uses the name)
    # Pine v6, an end bound that calls a function (`nz(n)`, `f()`): the counter reads it once on entry before `by`
    # and the first check (`pine_loop(from, entry=True)`, `enter(by)`), as TradingView reads the header
    counter_entry: bool = field(default=False, compare=False, repr=False)
    range_helper: str = ""  # the name `pine_range` is imported under for a fixed loop; empty is `pine_range`
    # the value of an assignment (`x = for ...`): no default step is written for it, even in Pine v4/v5
    is_assigned_value: bool = False


@dataclass
class ForInLoop(Statement):
    """For-in loop: for [vars] in iterable"""
    vars: List[str] = field(default_factory=list)
    iterable: Expression = None
    body: List[Statement] = field(default_factory=list)
    value_is_bool: bool = False  # Pine v6: its value is of type bool


@dataclass
class WhileLoop(Statement):
    """While loop: while condition\n    body"""
    condition: Expression = None
    body: List[Statement] = field(default_factory=list)
    value_is_bool: bool = False  # Pine v6: its value is of type bool


@dataclass
class SwitchStatement(Statement):
    """Switch statement (imperative form)"""
    expr: Optional[Expression] = None
    cases: List[tuple[Expression, List[Statement]]] = field(default_factory=list)
    default: Optional[List[Statement]] = None


@dataclass
class OnceStatement(Statement):
    """Pine v6 once structure: once [cond]\n    body. The parser lowers it to
    a `var` flag and an `if` (once_lowering.py); no later stage sees it."""
    condition: Optional[Expression] = None
    body: List[Statement] = field(default_factory=list)


@dataclass
class BreakStatement(Statement):
    """Break statement"""
    pass


@dataclass
class ContinueStatement(Statement):
    """Continue statement"""
    pass


@dataclass
class ExpressionStatement(Statement):
    """Expression as statement (e.g., function call on its own line)"""
    expr: Expression = None


@dataclass
class ReturnStatement(Statement):
    """Explicit return statement (for codegen)"""
    expr: Optional[Expression] = None


@dataclass
class RawCode(Statement):
    """Raw Python code to be emitted as-is by codegen."""
    code: str = ""


# ============================================================================
# Type annotations
# ============================================================================

# Forward references for type checking
Statement = Union[
    Assignment, Reassignment, IfStatement, ForLoop, ForInLoop,
    WhileLoop, SwitchStatement, BreakStatement, ContinueStatement,
    ExpressionStatement, ReturnStatement, VarDecl, VaripDecl,
    FuncDecl, InputDecl
]
