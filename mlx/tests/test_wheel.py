"""Build and inspect the public wheel boundary."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("uv") is None:
        pytest.skip("uv is required to inspect the wheel")
    output = tmp_path_factory.mktemp("wheel-build")
    subprocess.check_call(
        ["uv", "build", "--wheel", "--out-dir", str(output)],
        cwd=str(PACKAGE_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=sys.stderr,
    )
    wheels = list(output.glob("*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def test_wheel_contains_only_package_and_metadata(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()
    unexpected = [name for name in names if not (name.startswith("t0_mlx/") or ".dist-info/" in name)]
    assert not unexpected, unexpected


def test_wheel_includes_legal_files(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()
    assert any(name.endswith("/LICENSE") for name in names)
    assert any(name.endswith("/NOTICE") for name in names)


def test_wheel_excludes_public_repository_support_files(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()
    forbidden = ("tests/", "tools/", "README.INTERNAL.md", "BENCHMARKS.md", "PARITY.md", "CHANGELOG.md")
    assert not [name for name in names if name.startswith(forbidden) or Path(name).name in forbidden]
