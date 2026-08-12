from __future__ import annotations

from pathlib import Path
import tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


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
