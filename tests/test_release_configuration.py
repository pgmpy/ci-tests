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


def test_js_uses_one_npm_project() -> None:
    js = REPO_ROOT / "crates" / "ci-js"
    assert not (js / "tests" / "package.json").exists()
    assert not (js / "tests" / "package-lock.json").exists()

    workflow = load_workflow("js.yml")
    for job_name in ("js-lint", "js-test"):
        job = workflow["jobs"][job_name]
        setup = next(
            step
            for step in job["steps"]
            if step.get("uses") == "actions/setup-node@v4"
        )
        assert setup["with"]["cache-dependency-path"] == (
            "crates/ci-js/package-lock.json"
        )
        npm_steps = [
            step for step in job["steps"] if step.get("run") in {"npm ci", "npm test"}
        ]
        assert all(step["working-directory"] == "crates/ci-js" for step in npm_steps)


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
    with tarfile.open(VENDOR_ARCHIVE, "r:xz") as archive:
        members = {member.name for member in archive.getmembers()}
    for (name, _), manifest in manifests.items():
        license_expression = manifest.get("license", "")
        license_file = manifest.get("license-file", "")
        assert isinstance(license_expression, str)
        assert isinstance(license_file, str)
        assert license_expression.strip() or license_file.strip()
        if license_file:
            assert f"vendor/{name}/{license_file}" in members


def test_r_package_includes_the_upstream_extendr_mit_notice() -> None:
    """Redistributed extendr crates retain their upstream copyright and MIT terms."""
    notice = THIRD_PARTY_NOTICE.read_text(encoding="utf-8")
    assert "https://github.com/extendr/extendr" in notice
    assert all(crate in notice for crate in EXTENDR_CRATES)
    assert "Copyright (c) 2021 The Extendr contributors (See CONTRIBUTORS.md)" in notice
    for contributor in (
        "Andy Thomason",
        "Thomas Down",
        "Mossa Merhi Reimert",
        "Claus O. Wilke",
        "Hiroaki Yutani",
        "Ilia Kosenkov",
        "Daniel Falbel",
        "Genomics PLC",
    ):
        assert contributor in notice
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


def test_ci_omits_unused_openblas_and_redundant_core_build() -> None:
    docs = (REPO_ROOT / ".github" / "workflows" / "docs.yml").read_text()
    assert "libopenblas-dev" not in docs

    rust = load_workflow("rust.yml")
    steps = rust["jobs"]["rust_test"]["steps"]
    commands = [step.get("run", "") for step in steps if isinstance(step, dict)]
    assert not any("cargo build -p ci_core" in command for command in commands)
    assert any("cargo test -p ci_core" in command for command in commands)


def test_rust_distribution_dependency_tree_is_minimal() -> None:
    root = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text())
    r_rust = tomllib.loads((R_RUST / "Cargo.toml").read_text())
    expected = {"version": "0.18.0", "default-features": False}
    assert root["workspace"]["dependencies"]["statrs"] == expected
    assert r_rust["workspace"]["dependencies"]["statrs"] == expected

    js = tomllib.loads((REPO_ROOT / "crates" / "ci-js" / "Cargo.toml").read_text())
    assert "getrandom" not in js["dependencies"]
    assert "getrandom" not in js.get("target", {}).get(
        'cfg(target_arch = "wasm32")', {}
    ).get("dependencies", {})


def test_leaf_docs_examples_and_package_metadata_are_current() -> None:
    crates_readme = (REPO_ROOT / "crates" / "README.md").read_text()
    core_readme = (REPO_ROOT / "crates" / "ci-core" / "README.md").read_text()
    packaged_core_readme = (
        REPO_ROOT / "crates" / "ci-r" / "src" / "rust" / "ci-core" / "README.md"
    ).read_text()
    example = (REPO_ROOT / "crates" / "ci-js" / "examples" / "test.html").read_text()
    assert "Asynchronous API" not in crates_readme
    assert "TestRegistry" not in core_readme
    assert "src/tests/" not in core_readme
    assert "pearson_correlation_test" not in example
    assert "new Dataset" in example
    assert "new PearsonCorrelation" in example

    for readme in (core_readme, packaged_core_readme):
        assert "https://github.com/pgmpy/ci-tests/blob/main/README.md" in readme
        assert "https://github.com/pgmpy/ci-tests/blob/main/CONTRIBUTING.md" in readme
        assert "../../README.md" not in readme
        assert "../../CONTRIBUTING.md" not in readme

    for relative in (
        "crates/ci-core/Cargo.toml",
        "crates/ci-python/Cargo.toml",
        "crates/ci-js/Cargo.toml",
    ):
        package = tomllib.loads((REPO_ROOT / relative).read_text())["package"]
        assert package["license"] == "MIT"
        assert package["repository"] == "https://github.com/pgmpy/ci-tests"
        assert "Your Team" not in " ".join(package.get("authors", []))


def test_rust_lockfiles_preserve_unaffected_binding_versions() -> None:
    def versions(lockfile: Path, names: set[str]) -> dict[str, str]:
        packages = tomllib.loads(lockfile.read_text())["package"]
        return {
            package["name"]: package["version"]
            for package in packages
            if package["name"] in names
        }

    assert versions(
        REPO_ROOT / "Cargo.lock",
        {
            "js-sys",
            "thiserror",
            "thiserror-impl",
            "wasm-bindgen",
            "wasm-bindgen-macro",
            "wasm-bindgen-macro-support",
            "wasm-bindgen-shared",
        },
    ) == {
        "js-sys": "0.3.99",
        "thiserror": "2.0.18",
        "thiserror-impl": "2.0.18",
        "wasm-bindgen": "0.2.122",
        "wasm-bindgen-macro": "0.2.122",
        "wasm-bindgen-macro-support": "0.2.122",
        "wasm-bindgen-shared": "0.2.122",
    }
    assert versions(
        R_RUST / "Cargo.lock", {"readonly", "thiserror", "thiserror-impl"}
    ) == {
        "readonly": "0.2.13",
        "thiserror": "2.0.20",
        "thiserror-impl": "2.0.20",
    }
