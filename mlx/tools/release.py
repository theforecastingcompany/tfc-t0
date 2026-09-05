"""Validate release metadata and extract notes for a tagged release."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')


class ReleaseError(RuntimeError):
    """Raised when a release tag and the checked-out source disagree."""


def project_version(root: Path = PACKAGE_ROOT) -> str:
    """Read ``project.version`` from the package's intentionally simple TOML."""
    in_project = False
    for line in (root / "pyproject.toml").read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
            continue
        if in_project and (match := VERSION_PATTERN.fullmatch(stripped)):
            return match.group(1)
    raise ReleaseError("pyproject.toml has no project.version")


def _section(path: Path, version: str) -> str:
    """Return the non-empty Markdown section headed by ``version``."""
    prefix = f"## [{version}]"
    lines = path.read_text().splitlines()
    start = next((index + 1 for index, line in enumerate(lines) if line.startswith(prefix)), None)
    if start is None:
        raise ReleaseError(f"{path.name} has no {prefix} section")
    stop = next((index for index in range(start, len(lines)) if lines[index].startswith("## [")), len(lines))
    section = "\n".join(lines[start:stop]).strip()
    if not section:
        raise ReleaseError(f"{path.name} has an empty {prefix} section")
    return section + "\n"


def validate_release(tag: str, root: Path = PACKAGE_ROOT) -> tuple[str, str]:
    """Require tag, package version, changelog, and parity record to agree."""
    version = project_version(root)
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ReleaseError(f"release tag must be {expected_tag}, got {tag}")
    notes = _section(root / "CHANGELOG.md", version)
    _section(root / "PARITY.md", version)
    return version, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="annotated release tag, for example v0.1.0a0")
    parser.add_argument("--root", type=Path, default=PACKAGE_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--notes-output", type=Path, help="write the matching changelog section here")
    args = parser.parse_args()

    try:
        version, notes = validate_release(args.tag, args.root)
    except (OSError, ReleaseError) as error:
        raise SystemExit(f"release validation failed: {error}") from error
    if args.notes_output is not None:
        args.notes_output.write_text(notes)
    print(f"validated release {version}")


if __name__ == "__main__":
    main()
