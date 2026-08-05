"""The harness must not know which system it is looking at.

Enforced by reading the source rather than by convention, because a convention is exactly
what an import added in a hurry breaks. Every module of the layer is parsed and every
import it names is checked, which catches an import inside a function or inside a
`TYPE_CHECKING` block as readily as one at the top of a file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import nnphysics.evals

FORBIDDEN = ("nnphysics.systems", "nnphysics.models", "nnphysics.training")
"""`systems` and `models` are what the plan forbids. `training` is added because it sits
outside this layer, and an evaluation harness that imported the training loop could not
score anything the loop had not produced."""

ROOT = Path(nnphysics.evals.__file__).parent


def _modules() -> list[Path]:
    return sorted(ROOT.rglob("*.py"))


def _imported(path: Path) -> set[str]:
    """Every module name a file imports, however it imports it."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


class TestImports:
    def test_the_layer_has_modules_to_check(self) -> None:
        """A test that silently checks nothing would pass forever."""
        assert len(_modules()) > 5

    @pytest.mark.parametrize("path", _modules(), ids=lambda path: path.name)
    def test_no_module_imports_an_outer_or_sibling_layer(self, path: Path) -> None:
        offending = sorted(
            name
            for name in _imported(path)
            for forbidden in FORBIDDEN
            if name == forbidden or name.startswith(f"{forbidden}.")
        )
        assert not offending, f"{path.name} imports {offending}"

    def test_the_check_would_notice_a_violation(self, tmp_path: Path) -> None:
        """The guard is only worth having if it can fail."""
        module = tmp_path / "offender.py"
        module.write_text("from nnphysics.systems.nbody import state\n", encoding="utf-8")

        assert "nnphysics.systems.nbody" in _imported(module)

    def test_the_layer_does_depend_on_the_ones_it_may(self) -> None:
        """Depending inwards is the other half of the rule, and it is not vacuous here."""
        imported = {name for path in _modules() for name in _imported(path)}

        assert any(name.startswith("nnphysics.core") for name in imported)
        assert any(name.startswith("nnphysics.data") for name in imported)


class TestRuntime:
    def test_no_system_is_reachable_through_the_layer(self) -> None:
        """A module object that held a system would have been imported by some path."""
        for name, value in vars(nnphysics.evals).items():
            module = getattr(value, "__module__", "")
            assert not module.startswith("nnphysics.systems"), name
