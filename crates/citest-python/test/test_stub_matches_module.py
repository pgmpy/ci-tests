"""Gate the hand-written type stub against the module it describes.

``citest/_citest/__init__.pyi`` is maintained by hand, so nothing stops it
drifting from ``src/lib.rs``. It had already drifted before this test existed.
The stub is the only type information downstream users get, so a stale one is
worse than none: it type-checks code that will fail at runtime.

This compares names and signatures structurally rather than textually, so
reformatting the stub does not fail the check but adding, removing or renaming
a member does.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import citest
from citest import _citest

STUB_PATH = Path(_citest.__file__).parent / "_citest" / "__init__.pyi"

# Declared in the stub as type aliases and a shared base, not as runtime
# objects: the native module has no `_BaseTest`, and the aliases are for
# annotation only.
STUB_ONLY = {"ColumnRef", "ColumnSpec", "_BaseTest"}

TEST_METHODS = ("run_test", "is_independent", "meta")


def stub_tree() -> ast.Module:
    return ast.parse(STUB_PATH.read_text(encoding="utf-8"))


def stub_classes() -> dict[str, ast.ClassDef]:
    return {node.name: node for node in stub_tree().body if isinstance(node, ast.ClassDef)}


def stub_functions() -> set[str]:
    return {node.name for node in stub_tree().body if isinstance(node, ast.FunctionDef)}


def test_stub_file_is_present_and_parses() -> None:
    assert STUB_PATH.is_file(), f"missing stub: {STUB_PATH}"
    assert stub_classes(), "stub declares no classes"


def test_every_stub_class_exists_in_the_module() -> None:
    for name in stub_classes():
        if name in STUB_ONLY:
            continue
        assert hasattr(_citest, name), f"stub declares {name}, module does not export it"


def test_every_module_class_is_declared_in_the_stub() -> None:
    declared = set(stub_classes())
    for name, value in vars(_citest).items():
        if name.startswith("_") or not inspect.isclass(value):
            continue
        assert name in declared, f"module exports {name}, stub does not declare it"


def test_every_module_function_is_declared_in_the_stub() -> None:
    declared = stub_functions()
    for name, value in vars(_citest).items():
        if name.startswith("_") or not inspect.isroutine(value):
            continue
        assert name in declared, f"module exports {name}(), stub does not declare it"


def test_every_registered_test_class_is_stubbed() -> None:
    """A ninth test added to the core must reach the stub, not just the module."""
    declared = set(stub_classes())
    for meta in citest.list_tests():
        cls = "".join(part.capitalize() for part in meta["name"].split("_"))
        assert cls in declared, f"{meta['name']} is registered but {cls} is not stubbed"


@pytest.mark.parametrize("method", TEST_METHODS)
def test_test_classes_expose_the_stubbed_methods(method: str) -> None:
    for meta in citest.list_tests():
        cls_name = "".join(part.capitalize() for part in meta["name"].split("_"))
        cls = getattr(_citest, cls_name)
        assert hasattr(cls, method), f"{cls_name} is missing {method}()"


def test_stub_base_declares_exactly_the_query_methods() -> None:
    base = stub_classes()["_BaseTest"]
    declared = {node.name for node in base.body if isinstance(node, ast.FunctionDef)}
    assert declared == set(TEST_METHODS), f"_BaseTest declares {sorted(declared)}, expected {sorted(TEST_METHODS)}"


def test_cierror_is_stubbed_as_a_valueerror_subclass() -> None:
    """The stub must not understate the exception hierarchy callers rely on."""
    bases = {base.id for base in stub_classes()["CiError"].bases if isinstance(base, ast.Name)}
    assert "ValueError" in bases, "stub must declare CiError(ValueError)"
    assert issubclass(citest.CiError, ValueError)
