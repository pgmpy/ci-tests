#!/usr/bin/env python3
"""Assert every package manifest agrees on the version being released.

Run before publishing anything. A mismatch caught here costs nothing; the same
mismatch caught after one registry has accepted an upload is permanent, because
crates.io, PyPI and npm all refuse to reuse a version number.

The R package is allowed to trail: CRAN review latency is outside the
project's control, so `crates/citest-r/DESCRIPTION` is reported but does not
fail the check unless it is *ahead* of the tag.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def workspace_version() -> str:
    manifest = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    return manifest["workspace"]["package"]["version"]


def pyproject_version() -> str:
    manifest = tomllib.loads(
        (REPO_ROOT / "crates/citest-python/pyproject.toml").read_text(encoding="utf-8")
    )
    return manifest["project"]["version"]


def description_version() -> str:
    text = (REPO_ROOT / "crates/citest-r/DESCRIPTION").read_text(encoding="utf-8")
    match = re.search(r"^Version:\s*(\S+)$", text, re.MULTILINE)
    if match is None:
        raise SystemExit("DESCRIPTION has no Version field")
    return match.group(1)


def as_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def main(argv: list[str]) -> int:
    tag = argv[1] if len(argv) > 1 else ""
    expected = tag[1:] if tag.startswith("v") else tag

    workspace = workspace_version()
    python = pyproject_version()
    r_version = description_version()

    print(f"  workspace (crates.io, npm) : {workspace}")
    print(f"  pyproject (PyPI)           : {python}")
    print(f"  DESCRIPTION (CRAN)         : {r_version}")
    if expected:
        print(f"  git tag                    : {expected}")

    errors: list[str] = []
    if python != workspace:
        errors.append(f"pyproject {python} != workspace {workspace}")
    if expected and workspace != expected:
        errors.append(f"workspace {workspace} != tag {expected}")

    # The R package may trail, but must never claim a version that has not been
    # released elsewhere.
    if as_tuple(r_version) > as_tuple(workspace):
        errors.append(f"DESCRIPTION {r_version} is ahead of workspace {workspace}")
    elif r_version != workspace:
        print(f"  note: CRAN package trails at {r_version}; this is allowed")

    if errors:
        for error in errors:
            print(f"::error::{error}")
        return 1

    print("versions agree")
    if (output := os.environ.get("GITHUB_OUTPUT")) is not None:
        Path(output).open("a", encoding="utf-8").write(f"version={workspace}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
