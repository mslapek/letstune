import re
from pathlib import Path

from letstune import __version__

EXPECTED_VERSION = "0.3.0"
IS_DEV_VERSION = EXPECTED_VERSION.endswith("-dev")


def find_project_root() -> Path:
    p = Path().resolve()
    while not (p / "pyproject.toml").exists():
        p = p.parent

    return p


def test_version_is_major_minor_patch() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(-dev)?", EXPECTED_VERSION)


def test_version_in_letstune_py() -> None:
    assert __version__ == EXPECTED_VERSION


def test_version_in_pyproject() -> None:
    assert (
        f'version = "{EXPECTED_VERSION}"'
        in (find_project_root() / "pyproject.toml").read_text().splitlines()
    )


def test_version_in_changelog() -> None:
    if IS_DEV_VERSION:
        return  # this test is only for release commits

    changelog = (find_project_root() / "CHANGELOG.md").read_text().splitlines()

    assert any(line.startswith(f"## [{EXPECTED_VERSION}] - ") for line in changelog)

    assert (
        f"[{EXPECTED_VERSION}]: "
        f"https://github.com/mslapek/letstune/releases/tag/v{EXPECTED_VERSION}"
        in changelog
    )
    assert (
        f"[Unreleased]: "
        f"https://github.com/mslapek/letstune/compare/v{EXPECTED_VERSION}...HEAD"
        in changelog
    )
