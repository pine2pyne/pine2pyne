"""Pine v4, v5 and v6 constructs convert to the PyneCore constructs that model them."""
import ast
import re

from pine2pyne import transpile

import pine_samples as samples


def test_v4_study_with_unprefixed_functions():
    code = transpile(samples.STUDY_V4)
    ast.parse(code)
    assert '@script.indicator(' in code  # `study()` is an indicator
    assert 'ta.sma(close, len)' in code and 'ta.stdev(close, len)' in code  # v4 names get their v5 namespace
    assert "len=input(20, 'Length')" in code  # a v4 `input()` stays generic; `type=` is dropped
    assert 'na_bool=True' in code  # a v4 bool has three states
    assert 'color.new(color.blue, 50)' in code  # `transp=` becomes the colour's transparency
    assert 'transp=' not in code


def test_v5_var_switch_value_and_history():
    code = transpile(samples.INDICATOR_V5)
    ast.parse(code)
    assert re.search(r'peak: Persistent(Series)?\[float\] = ', code)  # `var` lives across bars
    assert 'src[length]' in code  # a history read stays a subscript
    assert 'state[1]' in code
    assert 'plotshape(' in code
    assert '__switch__' not in code  # a switch without a subject is an if/elif chain


def test_v6_types_enum_method_and_for_in():
    code = transpile(samples.TYPES_V6)
    module = ast.parse(code)
    classes = {node.name for node in ast.walk(module) if isinstance(node, ast.ClassDef)}
    assert {'Side', 'Level'} <= classes
    assert re.search(r'def age\(this', code)  # a method is a function with the receiver first
    assert 'for lvl in levels' in code
    assert 'levels.append(Level(' in code or 'Level.new(' in code or 'Level(' in code


def test_a_library_exports_its_functions():
    code = transpile(samples.LIBRARY_V6)
    module = ast.parse(code)
    assert '@script.library(' in code
    assert '@export' in code
    exported = next(node for node in module.body if isinstance(node, ast.Assign)
                    and getattr(node.targets[0], 'id', None) == '__all__')
    assert {constant.value for constant in exported.value.elts} == {'clamp', 'half'}


def test_a_library_import_is_a_python_import_from_lib():
    code = transpile(samples.USES_LIBRARY_V6)
    ast.parse(code)
    assert 'import lib.someone.MathLib.v1 as ml' in code
    assert 'ml.clamp(close, low, high)' in code
