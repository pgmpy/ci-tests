#!/usr/bin/env python3
"""Synchronize the R source package's copies of canonical repository assets."""

from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path


CANONICAL_CORE = Path("crates/ci-core")
PACKAGED_CORE = Path("crates/ci-r/src/rust/ci-core")
CANONICAL_FIXTURE = Path("tests/fixtures/golden.json")
PACKAGED_FIXTURE = Path("crates/ci-r/tests/testthat/fixtures/golden.json")
R_RUST_MANIFEST = Path("crates/ci-r/src/rust/Cargo.toml")
ROOT_MANIFEST = Path("Cargo.toml")


def _core_inputs(root: Path) -> list[Path]:
    core = root / CANONICAL_CORE
    return [core / "Cargo.toml", core / "README.md", *sorted((core / "src").rglob("*"))]


def _expected_assets(root: Path) -> dict[Path, bytes]:
    expected: dict[Path, bytes] = {}
    for source in _core_inputs(root):
        if source.is_file():
            relative = source.relative_to(root / CANONICAL_CORE)
            expected[PACKAGED_CORE / relative] = source.read_bytes()
    expected[PACKAGED_FIXTURE] = (root / CANONICAL_FIXTURE).read_bytes()
    return expected


def _workspace_drift(root: Path) -> list[str]:
    root_manifest = tomllib.loads((root / ROOT_MANIFEST).read_text(encoding="utf-8"))
    r_manifest = tomllib.loads((root / R_RUST_MANIFEST).read_text(encoding="utf-8"))
    errors: list[str] = []
    for key in ("dependencies", "lints"):
        expected = root_manifest["workspace"].get(key)
        actual = r_manifest["workspace"].get(key)
        if actual != expected:
            errors.append(f"R Rust workspace {key} differ from the canonical workspace")
    return errors


def find_drift(root: Path) -> list[str]:
    """Return deterministic diagnostics for every missing, modified, or stale copy."""
    expected = _expected_assets(root)
    errors: list[str] = []
    for relative, wanted in sorted(expected.items(), key=lambda item: str(item[0])):
        destination = root / relative
        if not destination.is_file():
            errors.append(f"missing packaged asset: {relative.as_posix()}")
        elif destination.read_bytes() != wanted:
            errors.append(f"modified packaged asset: {relative.as_posix()}")

    packaged_files = {
        path.relative_to(root)
        for path in (root / PACKAGED_CORE).rglob("*")
        if path.is_file()
    }
    expected_core_files = {
        path for path in expected if path.is_relative_to(PACKAGED_CORE)
    }
    for relative in sorted(packaged_files - expected_core_files, key=str):
        errors.append(f"unexpected packaged asset: {relative.as_posix()}")
    errors.extend(_workspace_drift(root))
    return errors


def sync_assets(root: Path) -> None:
    """Replace packaged project copies with their canonical repository bytes."""
    packaged_core = root / PACKAGED_CORE
    if packaged_core.exists():
        shutil.rmtree(packaged_core)
    for relative, content in _expected_assets(root).items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def copy_canonical_inputs(source_root: Path, destination_root: Path) -> None:
    """Copy only inputs needed to exercise synchronization in isolation."""
    inputs = [
        source_root / ROOT_MANIFEST,
        source_root / R_RUST_MANIFEST,
        source_root / CANONICAL_FIXTURE,
        *_core_inputs(source_root),
    ]
    for source in inputs:
        if source.is_file():
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check", action="store_true", help="report drift without writing"
    )
    action.add_argument("--sync", action="store_true", help="replace packaged copies")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    if args.sync:
        sync_assets(root)
    errors = find_drift(root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("R package assets match canonical sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
