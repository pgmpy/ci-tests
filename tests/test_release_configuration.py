from __future__ import annotations

from pathlib import Path
import tarfile
import tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
R_PACKAGE = REPO_ROOT / "crates" / "ci-r"
R_RUST = R_PACKAGE / "src" / "rust"
VENDOR_ARCHIVE = R_RUST / "vendor.tar.xz"
THIRD_PARTY_NOTICE = R_PACKAGE / "THIRD-PARTY-NOTICES"
EXTENDR_CRATES = {"extendr-api", "extendr-ffi", "extendr-macros"}


def registry_crates_from_lockfile() -> set[tuple[str, str]]:
    lockfile = tomllib.loads((R_RUST / "Cargo.lock").read_text())
    return {
        (package["name"], package["version"])
        for package in lockfile["package"]
        if package.get("source", "").startswith("registry+")
    }


def vendored_crate_manifests() -> dict[tuple[str, str], dict[str, object]]:
    manifests: dict[tuple[str, str], dict[str, object]] = {}
    with tarfile.open(VENDOR_ARCHIVE, "r:xz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if len(parts) == 3 and parts[0] == "vendor" and parts[-1] == "Cargo.toml":
                handle = archive.extractfile(member)
                assert handle is not None
                manifest = tomllib.loads(handle.read().decode("utf-8"))
                package = manifest["package"]
                manifests[(package["name"], package["version"])] = package
    return manifests


def load_workflow(name: str) -> dict[str, object]:
    return yaml.safe_load((REPO_ROOT / ".github" / "workflows" / name).read_text())


def named_step(workflow: dict[str, object], job: str, name: str) -> dict[str, object]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    selected_job = jobs[job]
    assert isinstance(selected_job, dict)
    steps = selected_job["steps"]
    assert isinstance(steps, list)
    return next(
        step for step in steps if isinstance(step, dict) and step.get("name") == name
    )


def test_r_msrv_is_advertised_preflighted_and_exercised() -> None:
    manifest = tomllib.loads(
        (REPO_ROOT / "crates" / "ci-r" / "src" / "rust" / "Cargo.toml").read_text()
    )
    assert manifest["package"]["rust-version"] == "1.81"

    description = (REPO_ROOT / "crates" / "ci-r" / "DESCRIPTION").read_text()
    assert "rustc >= 1.81.0" in description

    workflow = load_workflow("r.yml")
    rust_step = named_step(workflow, "r-test", "Install Rust MSRV")
    assert rust_step["with"] == {"toolchain": "1.81.0"}


def test_js_distribution_license_and_wasm_pack_version_are_pinned() -> None:
    manifest = tomllib.loads(
        (REPO_ROOT / "crates" / "ci-js" / "Cargo.toml").read_text()
    )
    assert manifest["package"]["license"] == "MIT"
    assert (REPO_ROOT / "crates" / "ci-js" / "LICENSE").read_text() == (
        REPO_ROOT / "LICENSE"
    ).read_text()

    workflow = load_workflow("js.yml")
    wasm_step = named_step(workflow, "js-test", "Install wasm-pack")
    assert wasm_step["with"] == {"version": "0.15.0"}


def test_r_workflow_checks_the_built_source_archive_with_warnings_as_errors() -> None:
    workflow = load_workflow("r.yml")
    check_step = named_step(workflow, "r-test", "Build and check source package")
    command = check_step["run"]
    assert isinstance(command, str)
    assert "devtools::build" in command
    assert "devtools::check_built" in command
    assert 'error_on = "warning"' in command


def test_r_package_completes_the_mit_license_template() -> None:
    description = (REPO_ROOT / "crates" / "ci-r" / "DESCRIPTION").read_text()
    assert "License: MIT + file LICENSE" in description
    assert (REPO_ROOT / "crates" / "ci-r" / "LICENSE").read_text() == (
        "YEAR: 2026\nCOPYRIGHT HOLDER: GIP House\n"
    )


def test_r_builds_use_the_packaged_lockfile_and_keep_linting_enabled() -> None:
    for makevars in ("Makevars.in", "Makevars.win.in"):
        contents = (REPO_ROOT / "crates" / "ci-r" / "src" / makevars).read_text()
        assert "cargo build --locked" in contents
        assert "cargo run --locked" in contents

    lintr_config = (REPO_ROOT / "crates" / "ci-r" / ".lintr").read_text()
    assert "object_usage_linter = NULL" not in lintr_config


def test_r_vendor_archive_matches_the_locked_crate_set_and_declares_licenses() -> None:
    """Every locked registry crate is vendored with a Cargo license declaration."""
    manifests = vendored_crate_manifests()
    assert set(manifests) == registry_crates_from_lockfile()
    assert EXTENDR_CRATES <= {name for name, _ in manifests}
    assert all(
        "license" in manifest or "license-file" in manifest
        for manifest in manifests.values()
    )


def test_r_package_includes_the_upstream_extendr_mit_notice() -> None:
    """Redistributed extendr crates retain their upstream copyright and MIT terms."""
    notice = THIRD_PARTY_NOTICE.read_text(encoding="utf-8")
    assert "https://github.com/extendr/extendr" in notice
    assert all(crate in notice for crate in EXTENDR_CRATES)
    assert "Copyright (c) 2020 Andy Thomason, Claus O. Wilke" in notice
    assert (
        "Permission is hereby granted, free of charge, to any person obtaining a copy"
        in notice
    )
    assert (
        "The above copyright notice and this permission notice shall be included"
        in notice
    )


def test_r_source_archive_excludes_python_cache_artifacts() -> None:
    """R CMD build must not distribute local Python bytecode or cache directories."""
    buildignore = (R_PACKAGE / ".Rbuildignore").read_text(encoding="utf-8")
    assert r"^tools/__pycache__$" in buildignore
    assert r"^[^/]+/.*[.]pyc$" in buildignore
