"""The theme has to be the only place a colour is written down.

Two of these tests read the source of the package rather than calling into it. A shared
stylesheet is only shared while nobody writes a second one, and that is a property of the
files, not of any function.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

import nnphysics.reporting
from nnphysics.core.errors import ValidationError
from nnphysics.reporting.theme import (
    DARK,
    LIGHT,
    TYPEFACES,
    Palette,
    Typefaces,
    css_var,
    report_stylesheet,
    token_names,
    tokens_css,
)

REPORTING_ROOT = Path(nnphysics.reporting.__file__).parent

ALLOWED_COLOUR_LITERALS = ("theme.py", "style.py")
"""`theme` holds document colour and `style` holds the figure palette matplotlib draws
with. Anywhere else, a colour literal is a second stylesheet starting."""

COLOUR_LITERAL = re.compile(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(")

DECLARATION = re.compile(r"(--[a-z0-9-]+):")
REFERENCE = re.compile(r"var\((--[a-z0-9-]+)\)")


class TestTokens:
    def test_every_token_has_a_name_in_both_themes(self) -> None:
        """A token defined for one theme only is invisible in the other."""
        assert [field.name for field in fields(LIGHT)] == [field.name for field in fields(DARK)]

    def test_no_token_is_empty(self) -> None:
        values = [
            getattr(palette, name) for palette in (LIGHT, DARK) for name in token_names()[:-3]
        ]

        assert all(value.strip() for value in values)

    def test_the_names_are_the_colours_then_the_type_families(self) -> None:
        assert token_names()[:2] == ("plane", "surface")
        assert token_names()[-3:] == ("serif", "sans", "mono")

    def test_the_plate_stays_light_in_the_dark_theme(self) -> None:
        """The plots are drawn on white. Inverting the plate hides them."""
        assert DARK.plate == LIGHT.plate

    def test_the_type_families_name_no_hosted_font(self) -> None:
        for family in (TYPEFACES.serif, TYPEFACES.sans, TYPEFACES.mono):
            assert "url(" not in family
            assert "http" not in family


class TestCssVar:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [("ink", "var(--ink)"), ("ink_2", "var(--ink-2)"), ("nbody_soft", "var(--nbody-soft)")],
    )
    def test_an_underscore_becomes_a_hyphen(self, name: str, expected: str) -> None:
        assert css_var(name) == expected

    def test_a_type_family_is_a_token_too(self) -> None:
        assert css_var("mono") == "var(--mono)"

    def test_an_unknown_token_raises(self) -> None:
        """A misspelled token resolves to nothing and draws an invisible mark."""
        with pytest.raises(ValidationError, match="no theme token named 'accent'"):
            css_var("accent")

    def test_the_property_form_is_not_accepted(self) -> None:
        """One spelling of a token name, the one the dataclasses use."""
        with pytest.raises(ValidationError):
            css_var("--ink")


class TestTokensCss:
    def test_the_light_theme_declares_every_token(self) -> None:
        declared = set(DECLARATION.findall(tokens_css()))

        assert declared == {css_var(name)[4:-1] for name in token_names()}

    def test_the_system_preference_carries_the_dark_theme(self) -> None:
        css = tokens_css()

        assert "@media (prefers-color-scheme: dark)" in css
        assert DARK.plane in css

    def test_a_report_offers_no_override(self) -> None:
        """A report has no theme button, so it has no attribute to honour."""
        assert "data-theme" not in tokens_css()

    def test_the_attribute_switch_overrides_in_both_directions(self) -> None:
        css = tokens_css("attribute")

        assert "@media (prefers-color-scheme: dark)" in css
        assert ':root[data-theme="dark"]' in css
        assert ':root[data-theme="light"]' in css

    def test_the_attribute_switch_tells_the_browser_which_theme_it_is_in(self) -> None:
        """Without this the scrollbars and the form controls stay on the other theme."""
        css = tokens_css("attribute")

        assert "color-scheme: dark;" in css
        assert "color-scheme: light;" in css

    def test_it_is_the_same_text_every_time(self) -> None:
        assert tokens_css() == tokens_css()


class TestReportStylesheet:
    def test_every_reference_is_to_a_declared_token(self) -> None:
        """A typo in the rules would otherwise silently drop a colour."""
        css = report_stylesheet()

        assert set(REFERENCE.findall(css)) <= set(DECLARATION.findall(css))

    def test_it_references_no_host(self) -> None:
        css = report_stylesheet()

        for reference in ("http", "@import", "url("):
            assert reference not in css

    def test_the_rules_name_no_colour_of_their_own(self) -> None:
        """Colour enters the stylesheet once, in the token block, and is referenced after."""
        rules = report_stylesheet().split("* { box-sizing: border-box; }")[1]

        assert not COLOUR_LITERAL.findall(rules)

    def test_it_is_the_same_text_every_time(self) -> None:
        assert report_stylesheet() == report_stylesheet()


class TestOneStylesheet:
    def test_the_package_has_modules_to_check(self) -> None:
        assert len(sorted(REPORTING_ROOT.glob("*.py"))) > 5

    @pytest.mark.parametrize(
        "path",
        [
            path
            for path in sorted(REPORTING_ROOT.glob("*.py"))
            if path.name not in ALLOWED_COLOUR_LITERALS
        ],
        ids=lambda path: path.name,
    )
    def test_no_other_module_writes_a_colour(self, path: Path) -> None:
        found = COLOUR_LITERAL.findall(path.read_text(encoding="utf-8"))

        assert not found, f"{path.name} names {found}; use nnphysics.reporting.theme"

    def test_the_check_would_notice_a_violation(self, tmp_path: Path) -> None:
        offender = tmp_path / "offender.py"
        offender.write_text('CSS = "body { color: #ff0000; }"\n', encoding="utf-8")

        assert COLOUR_LITERAL.findall(offender.read_text(encoding="utf-8"))


class TestPublicSurface:
    def test_the_package_exports_the_theme(self) -> None:
        """The chart phases reach the tokens through the package, not through the module."""
        for name in ("LIGHT", "DARK", "TYPEFACES", "css_var", "tokens_css", "report_stylesheet"):
            assert name in nnphysics.reporting.__all__

    def test_the_palettes_are_frozen(self) -> None:
        """Global mutable state dressed as a constant would be worse than a literal."""
        with pytest.raises(AttributeError):
            LIGHT.ink = "#000000"  # type: ignore[misc]

    def test_the_dataclasses_are_the_token_list(self) -> None:
        assert len(token_names()) == len(fields(Palette)) + len(fields(Typefaces))
