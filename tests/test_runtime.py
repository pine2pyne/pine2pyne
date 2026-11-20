"""A converted strategy runs on the PyneCore runtime (`pip install pynesys-pynecore`; skipped without it)."""
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pine2pyne import transpile

import pine_samples as samples

pytest.importorskip('pynecore', reason='the PyneCore runtime is not installed')


def write_candles(path: Path, bars: int = 300) -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows, price = ['timestamp,open,high,low,close,volume'], 100.0
    for index in range(bars):
        close = price + math.sin(index / 7) * 2
        rows.append(f'{int((start + timedelta(hours=index)).timestamp())},{price:.2f},'
                    f'{max(price, close) + 1:.2f},{min(price, close) - 1:.2f},{close:.2f},{1000 + index}')
        price = close
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')


def run_standalone(tmp_path: Path, name: str, source: str) -> Path:
    """`python <name>.py candles.csv`, as the generated footer runs it; returns the plot CSV."""
    script = tmp_path / f'{name}.py'
    script.write_text(transpile(source), encoding='utf-8')
    write_candles(tmp_path / 'candles.csv')
    result = subprocess.run([sys.executable, str(script), 'candles.csv'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    return tmp_path / f'{name}.csv'


def test_a_strategy_trades(tmp_path):
    plotted = run_standalone(tmp_path, 'crossing', samples.STRATEGY_V6).read_text(encoding='utf-8').splitlines()
    assert len(plotted) == 301  # a header and one row per bar
    trades = (tmp_path / 'crossing_trades.csv').read_text(encoding='utf-8').splitlines()
    assert len(trades) > 1


def test_a_v4_study_plots(tmp_path):
    plotted = run_standalone(tmp_path, 'bands', samples.STUDY_V4).read_text(encoding='utf-8').splitlines()
    assert len(plotted) == 301


def test_user_types_and_methods_run(tmp_path):
    plotted = run_standalone(tmp_path, 'levels', samples.TYPES_V6).read_text(encoding='utf-8').splitlines()
    assert len(plotted) == 301
