"""The `pine2pyne` command: inputs, outputs, exit codes, messages."""
import subprocess
import sys
import warnings

import pytest

import pine2pyne
from pine2pyne.cli import main
from pine2pyne.errors import SemanticWarning

import pine_samples as samples


@pytest.fixture
def pine(tmp_path):
    def write(name: str, source: str):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
        return path
    return write


def test_one_file_to_stdout(pine, capsys):
    path = pine('a.pine', samples.INDICATOR_V6)
    assert main([str(path)]) == 0
    out, err = capsys.readouterr()
    assert out.startswith('"""\n@pyne') and 'def main(' in out  # the code and nothing else
    assert err == ''


def test_one_file_to_a_path_creates_the_directories(pine, tmp_path, capsys):
    path = pine('a.pine', samples.INDICATOR_V6)
    target = tmp_path / 'out' / 'deep' / 'a.py'
    assert main([str(path), '-o', str(target)]) == 0
    assert target.read_text(encoding='utf-8') == pine2pyne.transpile(samples.INDICATOR_V6)
    assert capsys.readouterr().out.strip() == f'✓ {path} → {target}'


def test_several_files_to_a_directory_with_a_summary(pine, tmp_path, capsys):
    a, b = pine('a.pine', samples.INDICATOR_V6), pine('b.pine', samples.STRATEGY_V6)
    out = tmp_path / 'out'
    assert main([str(a), str(b), '-o', str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ['a.py', 'b.py']
    assert capsys.readouterr().out.rstrip().endswith('Summary: 2 succeeded, 0 failed')


def test_a_pattern_is_expanded(pine, tmp_path, capsys):
    pine('src/a.pine', samples.INDICATOR_V6)
    pine('src/b.pine', samples.INDICATOR_V5)
    assert main([str(tmp_path / 'src' / '*.pine'), '-o', str(tmp_path / 'out')]) == 0
    assert sorted(p.name for p in (tmp_path / 'out').iterdir()) == ['a.py', 'b.py']


def test_inputs_sharing_a_name_do_not_overwrite_each_other(pine, tmp_path, capsys):
    a, b = pine('one/x.pine', samples.INDICATOR_V6), pine('two/x.pine', samples.INDICATOR_V5)
    assert main([str(a), str(b), '-o', str(tmp_path / 'out')]) == 0
    assert sorted(p.name for p in (tmp_path / 'out').iterdir()) == ['x.py', 'x_2.py']
    assert 'written as x_2.py' in capsys.readouterr().err


def test_a_failed_file_does_not_stop_the_batch(pine, tmp_path, capsys):
    good, bad = pine('good.pine', samples.INDICATOR_V6), pine('bad.pine', samples.REFUSED_HISTORY_TARGET)
    assert main([str(good), str(bad), '-o', str(tmp_path / 'out')]) == 1
    out, err = capsys.readouterr()
    assert [p.name for p in (tmp_path / 'out').iterdir()] == ['good.py']
    assert f'✗ {bad}: Error at 4:6: Pine refuses:' in err and '[history-target]' in err
    assert 'Summary: 1 succeeded, 1 failed' in out


def test_validate_checks_everything_and_writes_nothing(pine, tmp_path, capsys):
    good, bad = pine('good.pine', samples.INDICATOR_V6), pine('bad.pine', samples.REFUSED_HISTORY_TARGET)
    assert main([str(good), '--validate']) == 0
    assert capsys.readouterr().out.strip() == f'✓ {good} - Valid syntax'
    assert main([str(bad), '--validate']) == 1
    assert '[history-target]' in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ['bad.pine', 'good.pine']  # nothing written


def test_a_warning_is_reported_and_the_file_is_still_written(pine, tmp_path, capsys):
    path = pine('warn.pine', samples.CONDITIONAL_TA_CALL)
    target = tmp_path / 'warn.py'
    with warnings.catch_warnings():
        warnings.simplefilter('default', SemanticWarning)  # the test suite makes warnings errors; the CLI user does not
        assert main([str(path), '-o', str(target)]) == 0
    out, err = capsys.readouterr()
    assert err.startswith(f'⚠ {path}:5:10:') and '[conditional-history-call]' in err
    assert target.is_file() and out.strip() == f'✓ {path} → {target}'


def test_a_warning_made_an_error_by_the_filters_fails_the_file(pine, tmp_path, capsys):
    path = pine('warn.pine', samples.CONDITIONAL_TA_CALL)
    with warnings.catch_warnings():
        warnings.simplefilter('error', SemanticWarning)  # what `PYTHONWARNINGS=error` does
        assert main([str(path), '-o', str(tmp_path / 'warn.py')]) == 1
    assert 'made an error by the warning filters' in capsys.readouterr().err
    assert not (tmp_path / 'warn.py').exists()


def test_a_comment_only_file_is_skipped(pine, tmp_path, capsys):
    path = pine('empty.pine', samples.COMMENT_ONLY)
    assert main([str(path), '-o', str(tmp_path / 'empty.py')]) == 0
    assert not (tmp_path / 'empty.py').exists()


def test_a_bom_is_accepted(pine, tmp_path):
    plain, bom = pine('plain.pine', samples.INDICATOR_V6), pine('bom.pine', '﻿' + samples.INDICATOR_V6)
    assert main([str(plain), str(bom), '-o', str(tmp_path / 'out')]) == 0
    assert (tmp_path / 'out' / 'plain.py').read_bytes() == (tmp_path / 'out' / 'bom.py').read_bytes()


def test_a_missing_input_is_an_error(tmp_path, capsys):
    assert main([str(tmp_path / 'nope.pine')]) == 1
    assert 'is not a file or valid pattern' in capsys.readouterr().err
    assert main([str(tmp_path / '*.pine')]) == 1
    assert 'No input files found' in capsys.readouterr().err


def test_the_module_entry_point_and_the_version():
    result = subprocess.run([sys.executable, '-m', 'pine2pyne', '--version'], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == f'pine2pyne {pine2pyne.__version__}'
