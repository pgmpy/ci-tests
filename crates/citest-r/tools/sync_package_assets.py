#!/usr/bin/env python3
"""Synchronize the R source package's copies of canonical repository assets."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tomllib
from collections.abc import Sequence
from pathlib import Path


CANONICAL_CORE = Path("crates/citest")
PACKAGED_CORE = Path("crates/citest-r/src/rust/citest")
CANONICAL_FIXTURE = Path("tests/fixtures/golden.json")
PACKAGED_FIXTURE = Path("crates/citest-r/tests/testthat/fixtures/golden.json")
R_RUST_MANIFEST = Path("crates/citest-r/src/rust/Cargo.toml")
ROOT_MANIFEST = Path("Cargo.toml")

# Distribution contracts for the redistributed Rust dependency bundle. The R
# source package vendors its whole dependency tree so the built archive
# compiles outside this monorepo without network access; that makes the
# repository a redistributor, with the licensing obligations that implies.
R_RUST_LOCKFILE = Path("crates/citest-r/src/rust/Cargo.lock")
VENDOR_ARCHIVE = Path("crates/citest-r/src/rust/vendor.tar.xz")
THIRD_PARTY_NOTICES = Path("crates/citest-r/THIRD-PARTY-NOTICES")

# Crates whose source is redistributed under extendr's own MIT terms.
EXTENDR_CRATES = frozenset({"extendr-api", "extendr-ffi", "extendr-macros"})

# Upstream's copyright line points at a CONTRIBUTORS.md that is not part of the
# vendored tarball, so the notice reproduces the list. Losing an entry would
# make the redistributed copyright reference incomplete.
EXTENDR_COPYRIGHT = "Copyright (c) 2021 The Extendr contributors (See CONTRIBUTORS.md)"
EXTENDR_CONTRIBUTORS = (
    "Andy Thomason",
    "Thomas Down",
    "Mossa Merhi Reimert",
    "Claus O. Wilke",
    "Hiroaki Yutani",
    "Ilia Kosenkov",
    "Daniel Falbel",
    "Genomics PLC",
)
MIT_CLAUSES = (
    "Permission is hereby granted, free of charge, to any person obtaining a copy",
    "The above copyright notice and this permission notice shall be included",
)


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
    for key in ("package", "dependencies", "lints"):
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


def _registry_crates(root: Path) -> set[tuple[str, str]]:
    """Every `(name, version)` the R Rust lockfile resolves from a registry."""
    lockfile = tomllib.loads((root / R_RUST_LOCKFILE).read_text(encoding="utf-8"))
    return {
        (package["name"], package["version"])
        for package in lockfile["package"]
        if package.get("source", "").startswith("registry+")
    }


def _vendored_manifests(
    root: Path,
) -> tuple[dict[tuple[str, str], dict[str, object]], set[str]]:
    """Parse `vendor/<crate>/Cargo.toml` out of the archive, with its members."""
    manifests: dict[tuple[str, str], dict[str, object]] = {}
    with tarfile.open(root / VENDOR_ARCHIVE, "r:xz") as archive:
        members = {member.name for member in archive.getmembers()}
        for name in sorted(members):
            parts = Path(name).parts
            if len(parts) == 3 and parts[0] == "vendor" and parts[-1] == "Cargo.toml":
                handle = archive.extractfile(name)
                if handle is None:
                    continue
                package = tomllib.loads(handle.read().decode("utf-8"))["package"]
                manifests[(package["name"], package["version"])] = package
    return manifests, members


def _vendor_drift(root: Path) -> list[str]:
    """The vendored bundle matches the lockfile and declares its licences."""
    if not (root / VENDOR_ARCHIVE).is_file():
        return [f"missing vendor archive: {VENDOR_ARCHIVE.as_posix()}"]

    errors: list[str] = []
    manifests, members = _vendored_manifests(root)
    locked = _registry_crates(root)

    for name, version in sorted(locked - set(manifests)):
        errors.append(f"locked crate not vendored: {name} {version}")
    for name, version in sorted(set(manifests) - locked):
        errors.append(f"vendored crate not in lockfile: {name} {version}")
    for name in sorted(EXTENDR_CRATES - {name for name, _ in manifests}):
        errors.append(f"extendr crate missing from vendor archive: {name}")

    for (name, version), package in sorted(manifests.items()):
        expression = str(package.get("license", "")).strip()
        license_file = str(package.get("license-file", "")).strip()
        if not expression and not license_file:
            errors.append(f"vendored crate declares no licence: {name} {version}")
        if license_file and f"vendor/{name}/{license_file}" not in members:
            errors.append(
                f"vendored crate declares a licence file that is absent: "
                f"{name} {version} -> {license_file}"
            )
    return errors


def _notice_drift(root: Path) -> list[str]:
    """The third-party notice reproduces extendr's upstream MIT attribution.

    NOTE: this covers only the extendr crates. The archive vendors the full
    dependency tree, and the notice does not yet name every redistributed
    crate; widening it is tracked separately as CRAN preparation work.
    """
    notice_path = root / THIRD_PARTY_NOTICES
    if not notice_path.is_file():
        return [f"missing third-party notice: {THIRD_PARTY_NOTICES.as_posix()}"]

    notice = notice_path.read_text(encoding="utf-8")
    errors: list[str] = []
    if "https://github.com/extendr/extendr" not in notice:
        errors.append("third-party notice omits the extendr upstream source URL")
    for crate in sorted(EXTENDR_CRATES):
        if crate not in notice:
            errors.append(f"third-party notice omits redistributed crate: {crate}")
    if EXTENDR_COPYRIGHT not in notice:
        errors.append("third-party notice omits the extendr copyright line")
    for contributor in EXTENDR_CONTRIBUTORS:
        if contributor not in notice:
            errors.append(f"third-party notice omits contributor: {contributor}")
    for clause in MIT_CLAUSES:
        if clause not in notice:
            errors.append(f"third-party notice omits an MIT clause: {clause[:40]}...")
    return errors


def find_distribution_drift(root: Path) -> list[str]:
    """Licensing contracts for the redistributed Rust dependency bundle.

    Kept separate from [`find_drift`], which reports asset-copy drift and must
    stay runnable against a minimal tree holding only the canonical inputs.
    These checks need the full R package (vendor archive, notice file).
    """
    return _vendor_drift(root) + _notice_drift(root)


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
    errors = find_drift(root) + find_distribution_drift(root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("R package assets and distribution licensing match canonical sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
