#!/usr/bin/env python3
"""Enforce inline comments in Python source files.

The hook intentionally uses Python's tokenizer so ``#`` characters inside
strings and docstrings are not mistaken for comments.  Standalone comments
are moved to the next code line when ``--fix`` is supplied; the hook exits
non-zero after modifying a file so the change can be reviewed and staged.
"""

from __future__ import annotations

import argparse
import io
import sys
import tokenize
from pathlib import Path


_NON_CODE_TOKENS = {
    tokenize.COMMENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.NEWLINE,
    tokenize.NL,
}


def _read_source(path: Path) -> tuple[str, str]:
    """Read a Python file while preserving its declared encoding."""
    raw = path.read_bytes()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    return raw.decode(encoding), encoding


def _token_lines(source: str) -> tuple[list[tokenize.TokenInfo], set[int], set[int]]:
    """Return tokens, lines containing code, and lines containing comments."""
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    code_lines = {token.start[0] for token in tokens if token.type not in _NON_CODE_TOKENS}
    comment_lines = {token.start[0] for token in tokens if token.type == tokenize.COMMENT}
    return tokens, code_lines, comment_lines


def _is_required_header_comment(line_number: int, comment: str) -> bool:
    """Allow Python's interpreter directive and encoding declaration."""
    if line_number == 1 and comment.startswith("#!"):
        return True
    return line_number <= 2 and "coding" in comment


def _standalone_comments(source: str) -> list[tokenize.TokenInfo]:
    """Find comment tokens whose physical line contains no code before them."""
    tokens, _, _ = _token_lines(source)
    lines = source.splitlines()
    standalone = []
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        line = lines[token.start[0] - 1] if token.start[0] <= len(lines) else ""
        if line[: token.start[1]].strip():
            continue
        if not _is_required_header_comment(token.start[0], token.string):
            standalone.append(token)
    return standalone


def fix_inline_comments(source: str) -> tuple[str, list[str]]:
    """Move standalone comments inline and return the fixed source and errors."""
    tokens, code_lines, comment_lines = _token_lines(source)
    lines = source.splitlines(keepends=True)
    standalone = []
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        line = lines[token.start[0] - 1] if token.start[0] <= len(lines) else ""
        if line[: token.start[1]].strip():
            continue
        if not _is_required_header_comment(token.start[0], token.string):
            standalone.append(token)

    errors: list[str] = []
    for token in reversed(standalone):
        source_line = token.start[0] - 1
        target_line = None

        candidates = sorted(code_lines - comment_lines)
        for candidate in candidates:
            if candidate <= token.start[0]:
                continue
            content = lines[candidate - 1].rstrip("\r\n").rstrip()
            if content and not content.endswith("\\"):
                target_line = candidate - 1
                break
        if target_line is None:
            for candidate in reversed(candidates):
                if candidate >= token.start[0]:
                    continue
                content = lines[candidate - 1].rstrip("\r\n").rstrip()
                if content and not content.endswith("\\"):
                    target_line = candidate - 1
                    break

        if target_line is None:
            errors.append(f"line {token.start[0]}: no safe code line for {token.string}")
            continue

        target = lines[target_line]
        newline = ""
        if target.endswith("\r\n"):
            newline = "\r\n"
            target = target[:-2]
        elif target.endswith(("\n", "\r")):
            newline = target[-1]
            target = target[:-1]
        lines[target_line] = f"{target.rstrip()}  {token.string}{newline}"

        original = lines[source_line]
        if original.endswith("\r\n"):
            lines[source_line] = "\r\n"
        elif original.endswith(("\n", "\r")):
            lines[source_line] = original[-1]
        else:
            lines[source_line] = ""

    return "".join(lines), errors


def _files_from_args(filenames: list[str]) -> list[Path]:
    return [Path(filename) for filename in filenames if Path(filename).suffix == ".py"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="move standalone comments inline")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    violations = []
    changed = []
    for path in _files_from_args(args.filenames):
        try:
            source, encoding = _read_source(path)
            fixed, errors = fix_inline_comments(source)
        except (OSError, SyntaxError, tokenize.TokenError, UnicodeError) as exc:
            violations.append(f"{path}: unable to inspect file: {exc}")
            continue

        if errors:
            violations.extend(f"{path}: {error}" for error in errors)
        if fixed != source:
            if args.fix:
                path.write_bytes(fixed.encode(encoding))
                changed.append(str(path))
            else:
                violations.extend(
                    f"{path}: line {token.start[0]}: standalone comment {token.string}"
                    for token in _standalone_comments(source)
                )

    if changed:
        print("Moved standalone comments inline in:")
        print("\n".join(f"  {path}" for path in changed))
        return 1
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
