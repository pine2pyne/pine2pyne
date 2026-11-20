"""The fixed text around every module pine2pyne writes.

The module opens with the ``@pyne`` magic docstring the PyneCore runtime looks for and
names the program that wrote it; it ends with the ``__main__`` footer that runs it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

REPOSITORY_URL = 'https://github.com/pine2pyne/pine2pyne'


def _version() -> str:
    from ... import __version__
    return __version__


@dataclass(frozen=True)
class ModuleBanner:
    """Module docstring and ``__main__`` footer of a generated module."""

    compiler: str = field(default_factory=lambda: f'pine2pyne {_version()}')
    magic: str = '@pyne edge'

    def header(self, magic: str | None = None) -> list[str]:
        return [
            '"""',
            magic or self.magic,
            '',
            f'This code was compiled by {self.compiler} — the Pine Script to PyneCore Python transpiler.',
            'Run with open-source PyneCore: https://pynecore.org',
            f'Source: {REPOSITORY_URL}',
            '"""',
        ]

    def footer(self) -> list[str]:
        return [
            'if __name__ == "__main__":',
            '    from pynecore.standalone import run',
            '    run(__file__)',
        ]
