"""The `transpile()` function: what it returns, what it refuses, what it warns about."""
import ast
import warnings

import pytest

import pine2pyne
from pine2pyne import Pine2PyneError, SemanticError, SemanticWarning, transpile
from pine2pyne.core import OutputProfile

import pine_samples as samples


def main_parameters(code: str) -> list[str]:
    """The parameter names of the generated `main()`."""
    module = ast.parse(code)
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    return [argument.arg for argument in main.args.args]


def test_an_indicator_becomes_a_pynecore_module():
    code = transpile(samples.INDICATOR_V6)
    module = ast.parse(code)  # valid Python
    assert module.body[0].value.value.strip().startswith('@pyne')  # the magic docstring the runtime looks for
    assert '@script.indicator("My MA", overlay=True)' in code
    assert main_parameters(code) == ['length', 'src']  # every input is a parameter of main, in source order
    assert 'ta.sma(src, length)' in code
    assert code.rstrip().endswith('run(__file__)')  # the standalone footer


def test_a_strategy_keeps_its_orders():
    code = transpile(samples.STRATEGY_V6)
    ast.parse(code)
    assert '@script.strategy(' in code
    assert "strategy.entry('L', strategy.long)" in code
    assert "strategy.close('L')" in code


def test_every_import_comes_from_pynecore_or_the_standard_library():
    allowed = ('pynecore', 'lib', 'typing', 'enum')
    for source in (samples.INDICATOR_V6, samples.STRATEGY_V6, samples.TYPES_V6, samples.INDICATOR_V5,
                   samples.STUDY_V4, samples.LIBRARY_V6):
        code = transpile(source)
        for node in ast.parse(code).body:
            if isinstance(node, ast.ImportFrom):
                assert node.module.split('.')[0] in allowed, ast.unparse(node)
            elif isinstance(node, ast.Import):
                assert all(alias.name.split('.')[0] in allowed for alias in node.names), ast.unparse(node)


def test_the_banner_names_the_version():
    code = transpile(samples.INDICATOR_V6)
    assert f'compiled by pine2pyne {pine2pyne.__version__}' in code


def test_a_comment_only_source_gives_an_empty_string():
    assert transpile(samples.COMMENT_ONLY) == ''
    assert transpile('') == ''


def test_a_utf8_bom_is_not_source():
    assert transpile('﻿' + samples.INDICATOR_V6) == transpile(samples.INDICATOR_V6)


def test_what_tradingview_refuses_is_refused_with_its_position():
    with pytest.raises(SemanticError) as refused:
        transpile(samples.REFUSED_HISTORY_TARGET)
    error = refused.value
    assert error.code == 'history-target'
    assert (error.line, error.column) == (4, 6)
    assert str(error).startswith('Error at 4:6: Pine refuses:')
    assert '[history-target]' in str(error)


def test_a_syntax_error_names_its_line():
    with pytest.raises(Pine2PyneError) as failed:
        transpile(samples.SYNTAX_ERROR)
    assert failed.value.line >= 3


def test_a_conditional_ta_call_is_converted_with_a_warning():
    with pytest.warns(SemanticWarning) as warned:
        code = transpile(samples.CONDITIONAL_TA_CALL)
    ast.parse(code)
    warning = warned[0].message
    assert warning.code == 'conditional-history-call'
    assert (warning.line, warning.column) == (5, 10)
    assert "'ta.sma'" in str(warning)


def test_warnings_can_be_silenced_with_the_standard_filters():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SemanticWarning)
        ast.parse(transpile(samples.CONDITIONAL_TA_CALL))


def test_the_output_profile_is_standard_by_default():
    assert OutputProfile.parse(None) is OutputProfile.STANDARD
    assert OutputProfile.parse('legacy') is OutputProfile.LEGACY
    with pytest.raises(ValueError):
        OutputProfile.parse('nope')
    standard, legacy = transpile(samples.INDICATOR_V6), transpile(samples.INDICATOR_V6, profile='legacy')
    assert standard != legacy
    ast.parse(legacy)
    assert "ta.sma(src, length)" in legacy
