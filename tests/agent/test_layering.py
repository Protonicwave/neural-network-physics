"""The agent must not reach out into the command line layer.

The layer diagram puts `agent` and `reporting` at the same indentation and so leaves the
direction between them unsaid. It is resolved in `nnphysics.agent`: an agent whose job is
to read reports has to be outside the layer that writes them, so `agent` imports
`reporting` and never the other way round. Both halves of that are checked here, along
with the part the diagram does say, that nothing below `cli` imports `cli`.

Read from the source rather than trusted to convention, the same way the evaluation
harness checks its own rule, because a convention is exactly what an import added in a
hurry breaks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import nnphysics.agent
import nnphysics.reporting

FORBIDDEN = ("nnphysics.cli",)
"""The command line is where the side effects live and it is outside this layer. An agent
that imported it could run training, which the brief puts out of scope."""

ROOT = Path(nnphysics.agent.__file__).parent
REPORTING_ROOT = Path(nnphysics.reporting.__file__).parent


def _modules(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


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
        assert len(_modules(ROOT)) > 5

    @pytest.mark.parametrize("path", _modules(ROOT), ids=lambda path: path.name)
    def test_no_module_imports_an_outer_layer(self, path: Path) -> None:
        offending = sorted(
            name
            for name in _imported(path)
            for forbidden in FORBIDDEN
            if name == forbidden or name.startswith(f"{forbidden}.")
        )
        assert not offending, f"{path.name} imports {offending}"

    def test_reporting_does_not_import_the_agent(self) -> None:
        """The other half of the rule, and the half the diagram does not state."""
        imported = {name for path in _modules(REPORTING_ROOT) for name in _imported(path)}

        assert not [name for name in imported if name.startswith("nnphysics.agent")]

    def test_the_layer_does_depend_on_the_ones_it_may(self) -> None:
        """Depending inwards is not vacuous here: the whole phase reads run records."""
        imported = {name for path in _modules(ROOT) for name in _imported(path)}

        assert any(name.startswith("nnphysics.reporting") for name in imported)
        assert any(name.startswith("nnphysics.core") for name in imported)

    def test_the_check_would_notice_a_violation(self, tmp_path: Path) -> None:
        """The guard is only worth having if it can fail."""
        module = tmp_path / "offender.py"
        module.write_text("from nnphysics.cli import train\n", encoding="utf-8")

        assert "nnphysics.cli" in _imported(module)
