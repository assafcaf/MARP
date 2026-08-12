"""Every module in the commons_game_marp package must import cleanly."""

import importlib
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "commons_game_marp"


def _iter_module_names():
    """Discover modules from the filesystem.

    Deliberately does not use pkgutil.walk_packages, which would require
    importing src first -- that is exactly what this test is checking.
    """
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        relative = path.relative_to(SRC.parent).with_suffix("")
        yield ".".join(relative.parts)


@pytest.mark.parametrize("module_name", list(_iter_module_names()))
def test_module_imports(module_name):
    importlib.import_module(module_name)
