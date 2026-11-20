<p align="center">
  <img src="https://raw.githubusercontent.com/pine2pyne/.github/main/images/logo.png" height="115px" alt="pine2pyne" />
</p>

<p align="center">
  <strong>TradingView Pine Script, transpiled to Python on your own machine</strong><br>
  Indicators, strategies and libraries in Pine Script v4, v5 and v6, compiled to PyneCore.<br>
  No API key, no quota, no network.
</p>

> pine2pyne converts TradingView Pine Script indicators, strategies and libraries into Python scripts
for [PyneCore](https://pynecore.org), the open-source runtime that runs Pine-style code bar by bar.
A converted script can backtest offline on any candles you have, without a TradingView account or
chart.

## Install

Requires Python 3.11+.

```bash
pip install pine2pyne
```

This installs the `pine2pyne` command. `python -m pine2pyne` does the same.

## Usage

```bash
pine2pyne strategy.pine -o strategy.py
```

```
pine2pyne [-h] [-o OUTPUT] [--validate] [-v] [--version] input [input ...]
```

| Option | Meaning |
|---|---|
| `input` | One or more `.pine` files. A quoted pattern such as `"src/*.pine"` is expanded by pine2pyne itself |
| `-o`, `--output` | Output file, or a directory for several inputs. Without it, the code goes to stdout |
| `--validate` | Check the script, write nothing |
| `-v`, `--verbose` | Print each step |
| `--version` | Print the version and exit |

## Examples

| Command | Result |
|---|---|
| `pine2pyne a.pine` | The code on stdout, so `pine2pyne a.pine > a.py` gives a clean file |
| `pine2pyne a.pine -o out/a.py` | Written to `out/a.py`. Missing directories are created |
| `pine2pyne a.pine b.pine -o out/` | `out/a.py` and `out/b.py` |
| `pine2pyne "src/*.pine" -o out/` | One `<name>.py` in `out/` for every match of the pattern |
| `pine2pyne a.pine --validate` | `✓ a.pine - Valid syntax`, or the error |

With several inputs, `-o` is a directory. A failed file does not stop the batch: every input is
processed, then a summary is printed:

```
Summary: 41 succeeded, 2 failed
```

## Errors

Errors and warnings go to stderr with the Pine line and column:

```
✗ bad.pine: Error at 4:6: Pine refuses: cannot assign with ':=' to a history reference (x[n]) [history-target]
```

A warning (`⚠`) does not stop the file from being written.

| Exit code | When |
|---|---|
| `0` | Every input was converted or validated |
| `1` | At least one input failed, or no input file was found |

## License

Apache-2.0. PyneCore is Powered by PyneSys (https://pynesys.io).

