"""Tests for the individual tools."""

import os

import pytest

from agent.tools import (
    ToolError,
    make_read_file,
    make_search,
    safe_calculate,
)

SANDBOX = os.path.join(os.path.dirname(__file__), "..", "sandbox")


def test_calculator_basic():
    assert safe_calculate("12 * 7") == "84"
    assert safe_calculate("2 ** 10") == "1024"
    assert safe_calculate("(3 + 4) * 2") == "14"
    assert safe_calculate("10 / 4") == "2.5"


def test_calculator_rejects_non_arithmetic():
    with pytest.raises(ToolError):
        safe_calculate("__import__('os').system('echo hi')")
    with pytest.raises(ToolError):
        safe_calculate("abs(-5)")
    with pytest.raises(ToolError):
        safe_calculate("")


def test_search_finds_known_fact():
    search = make_search()
    assert "Shakespeare" in search("who wrote hamlet")
    assert "da Vinci" in search("mona lisa")


def test_search_missing_topic():
    search = make_search()
    assert "No knowledge-base entry" in search("flux capacitor schematics")


def test_read_file_reads_inside_sandbox():
    read_file = make_read_file(SANDBOX)
    content = read_file("notes.txt")
    assert "release is scheduled" in content


def test_read_file_blocks_traversal():
    read_file = make_read_file(SANDBOX)
    with pytest.raises(ToolError):
        read_file("../agent/core.py")
    with pytest.raises(ToolError):
        read_file("/etc/passwd")
    with pytest.raises(ToolError):
        read_file("does_not_exist.txt")
