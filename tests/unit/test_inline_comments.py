"""Tests for the inline-comment pre-commit fixer."""

from scripts.check_inline_comments import fix_inline_comments


def test_standalone_comments_are_moved_to_the_next_code_line():
    source = "# explain the assignment\nvalue = 1\n"

    fixed, errors = fix_inline_comments(source)

    assert errors == []
    assert fixed == "\nvalue = 1  # explain the assignment\n"


def test_comments_inside_strings_and_inline_comments_are_unchanged():
    source = 'text = "# not a comment"  # keep this\n'

    fixed, errors = fix_inline_comments(source)

    assert errors == []
    assert fixed == source


def test_shebang_and_encoding_comments_are_allowed():
    source = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nvalue = 1\n"

    fixed, errors = fix_inline_comments(source)

    assert errors == []
    assert fixed == source
