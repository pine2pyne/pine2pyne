"""Room for the recursive walks of a deep tree, for the length of one transpile.

The parser, the passes and the code generator walk the syntax tree recursively, and a long `if` / `else if` chain, a
`switch` of hundreds of arms or a long concatenation nests as deep as it is long: past Python's default limit of 1000
frames a walk raises RecursionError. The limit is raised while pine2pyne works and put back afterwards, so that the
process that called it, and the modules it goes on to run, see the limit they had.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator

#: frames the walks may use; a frame of a pure Python call does not use the C stack
RECURSION_LIMIT = 20000


@contextmanager
def recursion_headroom(limit: int = RECURSION_LIMIT) -> Iterator[None]:
    previous = sys.getrecursionlimit()
    if previous < limit:
        sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        if sys.getrecursionlimit() != previous:
            sys.setrecursionlimit(previous)
