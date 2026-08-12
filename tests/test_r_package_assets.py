from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_TOOL = REPO_ROOT / "crates" / "ci-r" / "tools" / "sync_package_assets.py"


def load_sync_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_package_assets", SYNC_TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_r_package_assets_match_canonical_sources() -> None:
    sync_tool = load_sync_tool()
    assert sync_tool.find_drift(REPO_ROOT) == []


def test_drift_check_reports_modified_and_unexpected_assets(tmp_path: Path) -> None:
    sync_tool = load_sync_tool()
    repo = tmp_path / "repo"
    sync_tool.copy_canonical_inputs(REPO_ROOT, repo)
    sync_tool.sync_assets(repo)

    fixture = (
        repo / "crates" / "ci-r" / "tests" / "testthat" / "fixtures" / "golden.json"
    )
    fixture.write_text("modified\n", encoding="utf-8")
    stale = repo / "crates" / "ci-r" / "src" / "rust" / "ci-core" / "src" / "stale.rs"
    stale.write_text("// stale\n", encoding="utf-8")

    errors = sync_tool.find_drift(repo)
    assert errors == [
        "modified packaged asset: crates/ci-r/tests/testthat/fixtures/golden.json",
        "unexpected packaged asset: crates/ci-r/src/rust/ci-core/src/stale.rs",
    ]
