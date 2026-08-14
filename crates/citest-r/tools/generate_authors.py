#!/usr/bin/env python3
"""Generate inst/AUTHORS from the vendored Rust crate bundle.

CRAN's Rust policy requires the authorship and copyright information for
redistributed Rust sources to be documented. Deriving the file from the vendor
archive keeps it complete by construction: a crate cannot enter the bundle
without appearing here.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tomllib
from collections.abc import Sequence
from pathlib import Path

VENDOR_ARCHIVE = Path("crates/citest-r/src/rust/vendor.tar.xz")
AUTHORS_FILE = Path("crates/citest-r/inst/AUTHORS")
EXTENDR_NOTICE = Path("crates/citest-r/tools/extendr-notice.txt")

HEADER = """\
Authors and copyright holders of the bundled Rust crates
========================================================

The R source package bundles its Rust dependencies so that it can be built
without network access, as recommended by CRAN's "Using Rust in CRAN packages".
This file records the authorship, copyright and licence of every crate in
`src/rust/vendor.tar.xz`, and is generated from that archive by
`tools/generate_authors.py`.

The citest R package itself, and the Rust core it wraps, are licensed MIT --
see the LICENSE file. The bundled crates are licensed as listed below. Note
that `approx` is Apache-2.0 only, and `unicode-ident` additionally carries
Unicode-3.0 terms.

"""


def crates(root: Path) -> list[tuple[str, str, str, list[str]]]:
    rows = []
    with tarfile.open(root / VENDOR_ARCHIVE, "r:xz") as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            parts = Path(member.name).parts
            if len(parts) == 3 and parts[0] == "vendor" and parts[-1] == "Cargo.toml":
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                pkg = tomllib.loads(handle.read().decode("utf-8"))["package"]
                licence = pkg.get("license") or f"see {pkg.get('license-file', '?')}"
                rows.append(
                    (pkg["name"], pkg["version"], licence, pkg.get("authors") or [])
                )
    return sorted(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]

    body = [HEADER]
    for name, version, licence, authors in crates(root):
        body.append(f"{name} {version}\n")
        body.append(f"  Licence: {licence}\n")
        if authors:
            for author in authors:
                body.append(f"  Copyright: {author}\n")
        else:
            body.append("  Copyright: see the crate's own source distribution\n")
        body.append("\n")

    # extendr's crates declare no `authors` key, and upstream's copyright line
    # points at a CONTRIBUTORS.md that is not part of the vendored tarball, so
    # the full upstream notice is reproduced verbatim.
    body.append((root / EXTENDR_NOTICE).read_text(encoding="utf-8"))
    text = "".join(body)

    destination = root / AUTHORS_FILE
    if args.check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != text:
            print(f"{AUTHORS_FILE.as_posix()} is stale; run without --check", file=sys.stderr)
            return 1
        print("inst/AUTHORS matches the vendored crate bundle")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    print(f"wrote {AUTHORS_FILE.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
