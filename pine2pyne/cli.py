"""
Command-line interface for pine2pyne transpiler.
"""
import argparse
import sys
import warnings
from pathlib import Path
from typing import Callable, List, Optional
from . import transpile, __version__, _is_comment_only
from .core.contracts import OutputProfile
from .errors import Pine2PyneError, SemanticError, SemanticWarning


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='pine2pyne',
        description='Transpile Pine Script v4, v5 and v6 to PyneCore Python',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Output to stdout
  python -m pine2pyne strategy.pine

  # Output to file
  python -m pine2pyne strategy.pine -o strategy.py

  # Validate syntax only
  python -m pine2pyne strategy.pine --validate

  # Batch convert multiple files
  python -m pine2pyne strategy/*.pine -o output/
        '''
    )

    parser.add_argument(
        'input',
        nargs='+',
        type=str,
        help='Input Pine Script file(s) (.pine)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file or directory. If directory, outputs will have .py extension'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate syntax only (parse without generating code)'
    )

    parser.add_argument(
        '--profile',
        choices=[profile.value for profile in OutputProfile],
        default=OutputProfile.STANDARD.value,
        help='Output layout: standard (default) or legacy (the earlier pine2pyne layout)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    return parser.parse_args(args)


def reporting_semantic_warnings(input_path: Path, action: Callable[[], object]):
    """Run `action`, printing each SemanticWarning it raises as
    `⚠ <file>:<line>:<column>: <message> [<code>]`. The warning
    filters stay the user's: a fresh catch_warnings resets Python's
    once-per-location record, so every file reports its own warnings, and
    `-W ignore::pine2pyne.errors.SemanticWarning` silences them. A filter that
    makes one an error turns it into a refusal with its code. Other warnings
    are re-issued unchanged once recording has stopped."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            return action()
    except SemanticWarning as escalated:
        raise SemanticError(escalated.code, f'{escalated.text} (a warning, made an error by the warning filters)',
                            escalated.line, escalated.column) from escalated
    finally:
        for record in caught:
            if isinstance(record.message, SemanticWarning):
                print(f'⚠ {input_path}:{record.message}', file=sys.stderr)
            else:
                warnings.warn_explicit(record.message, record.category, record.filename, record.lineno)


def transpile_file(input_path: Path, output_path: Optional[Path], validate_only: bool, verbose: bool,
                   profile: Optional[str] = None) -> bool:
    """
    Transpile a single Pine Script file in output ``profile`` (default: standard).

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read input file
        if verbose:
            print(f'Reading {input_path}...')

        source = input_path.read_text(encoding='utf-8-sig')

        if validate_only:
            # the whole pipeline, its text discarded: valid exactly when the full run accepts it
            from .di import get_transpiler

            if verbose:
                print('Tokenizing...')
                print('Parsing...')
                print('Transforming and emitting (the text is discarded)...')
            reporting_semantic_warnings(input_path, lambda: get_transpiler(profile).validate(source))

            print(f'✓ {input_path} - Valid syntax')
            return True

        # Full transpilation
        if verbose:
            print('Transpiling...')

        if profile is None or OutputProfile.parse(profile) is OutputProfile.STANDARD:
            output_code = reporting_semantic_warnings(input_path, lambda: transpile(source))
        else:
            output_code = reporting_semantic_warnings(input_path, lambda: transpile(source, profile=profile))

        # Empty output is only legitimate for a comment-only / empty source.
        # If the source carries real Pine code but the transpiler produced
        # nothing, an unsupported construct was silently dropped — fail loudly
        # instead of reporting a bogus success (exit 0 with no file written).
        if not output_code:
            if _is_comment_only(source):
                if verbose:
                    print(f'Skipping {input_path} (comment-only file)')
                return True
            print(f'✗ {input_path}: transpiler produced no output for non-comment '
                  f'source (unsupported construct silently dropped)', file=sys.stderr)
            return False

        # Write output
        if output_path:
            if verbose:
                print(f'Writing to {output_path}...')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_code, encoding='utf-8')
            print(f'✓ {input_path} → {output_path}')
        else:
            # Output to stdout
            print(output_code)

        return True

    except Pine2PyneError as e:
        print(f'✗ {input_path}: {e}', file=sys.stderr)
        # A repaired source that still fails: the first error above, then where it stopped
        cause = e.__cause__
        while isinstance(cause, Pine2PyneError):
            print(f'  after repairing it as valid Pine, the source stops at: {cause}', file=sys.stderr)
            cause = cause.__cause__
        return False
    except FileNotFoundError:
        print(f'✗ {input_path}: File not found', file=sys.stderr)
        return False
    except Exception as e:
        print(f'✗ {input_path}: Unexpected error: {e}', file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def _output_names(input_paths: List[Path]) -> dict:
    """The output file name of each input converted into a directory: its stem with `.py`, and a numeric suffix
    (`foo_2.py`, `foo_3.py`, ...) where another input of the batch, from another directory, has the same stem, so
    that no output overwrites another. Each such rename is reported on stderr."""
    names: dict = {}
    taken: dict = {}  # output name -> the resolved input that has it
    for input_path in input_paths:
        source = input_path.resolve()
        name = input_path.stem + '.py'
        suffix = 1
        while name in taken and taken[name] != source:
            suffix += 1
            name = f'{input_path.stem}_{suffix}.py'
        if suffix > 1:
            print(f'Note: {input_path} shares its name with another input; written as {name}', file=sys.stderr)
        taken[name] = source
        names[input_path] = name
    return names


def use_utf8_streams() -> None:
    """
    Write stdout and stderr as UTF-8 on every platform, without PYTHONUTF8.

    On Windows a piped or redirected stream takes the ANSI code page (cp1251, cp1252...), and printing a
    status mark or a non-Latin character of a Pine source raised UnicodeEncodeError. Linux and macOS
    streams are UTF-8 already, so there this changes nothing.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, 'encoding', None) or '').lower().replace('-', '')
        if encoding != 'utf8' and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    use_utf8_streams()
    args = parse_args(argv)

    # Resolve input files
    input_paths: List[Path] = []
    for input_pattern in args.input:
        path = Path(input_pattern)
        if path.is_file():
            input_paths.append(path)
        elif '*' in input_pattern or '?' in input_pattern:
            # Glob pattern
            from glob import glob
            for match in glob(input_pattern):
                input_paths.append(Path(match))
        else:
            print(f'Error: {input_pattern} is not a file or valid pattern', file=sys.stderr)
            return 1

    if not input_paths:
        print('Error: No input files found', file=sys.stderr)
        return 1

    # Determine output strategy
    output_is_dir = False
    output_base = None

    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir() or (not output_path.exists() and len(input_paths) > 1):
            output_is_dir = True
            output_base = output_path
        elif len(input_paths) > 1:
            print('Error: Multiple input files require output to be a directory', file=sys.stderr)
            return 1
        else:
            output_base = output_path

    # Process files
    success_count = 0
    fail_count = 0
    output_names = _output_names(input_paths) if output_is_dir else {}

    for input_path in input_paths:
        # Determine output path
        if args.output is None:
            output_path = None  # stdout
        elif output_is_dir:
            output_path = output_base / output_names[input_path]
        else:
            output_path = output_base

        # Transpile
        if transpile_file(input_path, output_path, args.validate, args.verbose, profile=args.profile):
            success_count += 1
        else:
            fail_count += 1

    # Print summary for multiple files
    if len(input_paths) > 1:
        print(f'\nSummary: {success_count} succeeded, {fail_count} failed')

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
