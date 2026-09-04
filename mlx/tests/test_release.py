"""Release metadata must identify one reviewed source and parity record."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

from release import ReleaseError, project_version, validate_release  # noqa: E402


def _release_tree(tmp_path: Path, version: str = "1.2.3") -> Path:
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "example"\nversion = "{version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(f"# Changelog\n\n## [{version}] - 2026-09-03\n\n- Shipped.\n")
    (tmp_path / "PARITY.md").write_text(f"# Numerical parity\n\n## [{version}] - 2026-09-03\n\n- Passed.\n")
    return tmp_path


def test_release_metadata_agrees(tmp_path: Path) -> None:
    root = _release_tree(tmp_path)
    version, notes = validate_release("v1.2.3", root)
    assert version == project_version(root) == "1.2.3"
    assert notes == "- Shipped.\n"


@pytest.mark.parametrize(
    ("tag", "missing", "message"),
    [
        ("v1.2.4", None, "release tag must be v1.2.3"),
        ("v1.2.3", "CHANGELOG.md", "CHANGELOG.md has no"),
        ("v1.2.3", "PARITY.md", "PARITY.md has no"),
    ],
)
def test_release_metadata_rejects_mismatch(tmp_path: Path, tag: str, missing: str | None, message: str) -> None:
    root = _release_tree(tmp_path)
    if missing is not None:
        (root / missing).write_text(f"# {missing}\n")
    with pytest.raises(ReleaseError, match=message):
        validate_release(tag, root)


def test_monorepo_release_is_namespaced_and_tag_bound() -> None:
    # In the public repository this test lives under mlx/tests/. In navi, the
    # canonical source lives under apps/t0-mlx/ and the workflow is in the
    # adjacent T0 public template.
    candidates = (
        Path(__file__).parents[2] / ".github/workflows/release-mlx.yml",
        Path(__file__).parents[2] / "t0/tools/public_template/.github/workflows/release-mlx.yml",
    )
    workflow_path = next(path for path in candidates if path.is_file())
    workflow = workflow_path.read_text()
    assert 'tags: ["tfc-t0-mlx-v*"]' in workflow
    assert "refs/tags/tfc-t0-mlx-v*" in workflow
    assert "environment: pypi" in workflow
    assert "working-directory: mlx" in workflow
