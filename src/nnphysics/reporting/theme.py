"""Design tokens and the rules the report and the landing page share.

Both artefacts are read one after the other, a landing page linking to the report of a
particular run, so they have to look like one thing. Holding the colours, the three type
families and the treatment of body text, tables and figures here is what stops them
drifting: there is one definition, not two that agree today.

The tokens are available as data as well as as text. Generated markup needs to name a
colour, an SVG bar in a chart for instance, and `css_var` is how it does that without a
second copy of the value appearing in a builder.

Colour appears twice in this package for two different media. This module holds document
colour, meaning the CSS that dresses a report or a page. `style` holds figure colour,
meaning the Okabe and Ito series palette that matplotlib draws plots with. They are not
the same palette and should not be merged: one is a house style, the other is chosen so
that eight series stay distinguishable to a colour blind reader and in greyscale print.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

from nnphysics.core.errors import ValidationError

__all__ = [
    "DARK",
    "LIGHT",
    "TYPEFACES",
    "DarkSwitch",
    "Palette",
    "Typefaces",
    "css_var",
    "landing_stylesheet",
    "report_stylesheet",
    "token_names",
    "tokens_css",
]


@dataclass(frozen=True, slots=True)
class Palette:
    """The colour tokens of one theme.

    Field names are the token names: `ink_2` is written `--ink-2` in the stylesheet and
    reached as `css_var("ink_2")` from generated markup.

    Attributes:
        plane: The page behind everything.
        surface: A panel raised off the plane, behind a table row or a chart.
        ink: Body text at full strength, and headings.
        ink_2: Body text at reading strength, a step down from `ink`.
        muted: Labels, captions and axis text.
        grid: Chart grid lines.
        axis: Chart axis lines and tick marks.
        rule: Hairline borders. Translucent, so it works over either background.
        plate: Behind a rendered plot. The plots are drawn on white, so this stays white
            in the dark theme rather than inverting and leaving the plot unreadable.
        nbody: The N-body system.
        nbody_soft: The N-body system, held back, for a second series in the same panel.
        fluid: The fluid system.
        fluid_soft: The fluid system, held back.
        fail: A diverged rollout, a failed check, a negative result.
        warn_ink: Text stating a qualified result.
        good_ink: Text stating a result that held.
    """

    plane: str
    surface: str
    ink: str
    ink_2: str
    muted: str
    grid: str
    axis: str
    rule: str
    plate: str
    nbody: str
    nbody_soft: str
    fluid: str
    fluid_soft: str
    fail: str
    warn_ink: str
    good_ink: str


@dataclass(frozen=True, slots=True)
class Typefaces:
    """The three type families, as CSS font stacks.

    Every family names only fonts that ship with an operating system. A report is often
    read offline and must reference no host, which rules out a web font.

    Attributes:
        serif: Headings and the sentence that answers the question.
        sans: Body text and controls.
        mono: Numbers, identifiers and labels.
    """

    serif: str
    sans: str
    mono: str


LIGHT = Palette(
    plane="#f9f9f7",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    rule="rgba(11, 11, 11, 0.10)",
    plate="#ffffff",
    nbody="#4a3aa7",
    nbody_soft="#b6aee0",
    fluid="#1baf7a",
    fluid_soft="#a9e0c9",
    fail="#d03b3b",
    warn_ink="#8a6100",
    good_ink="#006300",
)

DARK = Palette(
    plane="#0d0d0d",
    surface="#1a1a19",
    ink="#ffffff",
    ink_2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    rule="rgba(255, 255, 255, 0.10)",
    plate="#ffffff",
    nbody="#9085e9",
    nbody_soft="#4b448c",
    fluid="#199e70",
    fluid_soft="#175c45",
    fail="#e66767",
    warn_ink="#d9a341",
    good_ink="#0ca30c",
)

TYPEFACES = Typefaces(
    serif='"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif',
    sans='ui-sans-serif, system-ui, "Segoe UI", Inter, sans-serif',
    mono='ui-monospace, "Cascadia Mono", Consolas, monospace',
)

type DarkSwitch = Literal["system", "attribute"]
"""How a document chooses its theme.

`system` follows the reader's system preference and offers no way to override it, which is
all a report needs. `attribute` follows the same preference by default and then lets a
`data-theme` attribute on the root element override it, which is what a theme button on
the landing page sets.
"""


def token_names() -> tuple[str, ...]:
    """Every token name, in stylesheet order.

    Returns:
        The colour names followed by the type family names.
    """
    return tuple(field.name for field in fields(Palette)) + tuple(
        field.name for field in fields(Typefaces)
    )


def css_var(name: str) -> str:
    """Reference a token from generated markup.

    Args:
        name: A token name, as spelled on `Palette` or `Typefaces`.

    Returns:
        The CSS `var()` reference, for example `var(--ink-2)`.

    Raises:
        ValidationError: If no token has that name. A misspelled token would otherwise
            resolve to nothing and draw an invisible mark.
    """
    if name not in token_names():
        raise ValidationError(f"no theme token named {name!r}; known tokens: {token_names()}")
    return f"var({_property(name)})"


def tokens_css(dark_switch: DarkSwitch = "system") -> str:
    """The token declarations.

    Args:
        dark_switch: How the reader's theme is chosen. See `DarkSwitch`.

    Returns:
        CSS declaring every token for the light theme and redeclaring the colours for the
        dark one, ending in a newline.
    """
    light = _declarations(LIGHT) + _declarations(TYPEFACES)
    dark = _declarations(DARK)
    blocks = [
        _rule(":root", ["color-scheme: light dark;", *light]),
        f"@media (prefers-color-scheme: dark) {{\n{_indent(_rule(':root', dark))}}}\n",
    ]
    if dark_switch == "attribute":
        blocks.append(_rule(':root[data-theme="dark"]', ["color-scheme: dark;", *dark]))
        blocks.append(_rule(':root[data-theme="light"]', ["color-scheme: light;", *light]))
    return "".join(blocks)


def report_stylesheet() -> str:
    """The whole stylesheet for a run report.

    Returns:
        The tokens followed by the shared rules, ending in a newline. A report adds
        nothing of its own: everything it renders is a heading, a paragraph, a list, a
        table or a figure, and those are exactly what the shared rules cover.
    """
    return tokens_css() + _SHARED_RULES


def landing_stylesheet() -> str:
    """The whole stylesheet for the landing page.

    Returns:
        The tokens with the theme button's override, the shared rules, and the rules for
        the sections only the landing page has, ending in a newline. The shared rules come
        first and the landing rules override them where a section is treated differently,
        so that the two artefacts share a definition rather than agreeing by coincidence.
    """
    return tokens_css("attribute") + _SHARED_RULES + _LANDING_RULES


def _property(name: str) -> str:
    """A token name as a custom property, so `ink_2` becomes `--ink-2`."""
    return f"--{name.replace('_', '-')}"


def _declarations(tokens: Palette | Typefaces) -> list[str]:
    """One declaration per field, in declaration order."""
    return [f"{_property(field.name)}: {getattr(tokens, field.name)};" for field in fields(tokens)]


def _rule(selector: str, declarations: list[str]) -> str:
    """A selector and its declarations, one per line."""
    body = "".join(f"  {declaration}\n" for declaration in declarations)
    return f"{selector} {{\n{body}}}\n"


def _indent(css: str) -> str:
    """Indent a block by two spaces, leaving blank lines alone."""
    return "".join(f"  {line}\n" if line else "\n" for line in css.split("\n")[:-1])


# The rules both artefacts share: the reset, the measure, the masthead, and the treatment
# of headings, prose, tables and figures. The landing page layers its own sections on top
# of these and overrides where a section needs it. Nothing here is report specific, and
# nothing here names a colour or a font directly.
_SHARED_RULES = """\
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--plane); color: var(--ink);
  font-family: var(--sans); font-size: 16px; line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 62rem; margin: 0 auto; padding: 0 1.5rem; }
.masthead {
  border-bottom: 1px solid var(--rule); margin-bottom: 3.5rem;
  font-family: var(--mono); font-size: 0.78rem; color: var(--muted);
}
.masthead .wrap { padding-top: 0.9rem; padding-bottom: 0.9rem; }
main { padding-bottom: 6rem; }
h1, h2, h3, h4, h5, h6 {
  font-family: var(--serif); font-weight: 600; line-height: 1.2;
  letter-spacing: -0.01em; color: var(--ink);
}
h1 { font-size: clamp(1.9rem, 4vw, 2.6rem); margin: 0 0 1.5rem; max-width: 24ch; }
h2 {
  font-size: clamp(1.3rem, 2.6vw, 1.6rem); margin: 3.5rem 0 1rem;
  padding-top: 1.75rem; border-top: 1px solid var(--rule);
}
h3 { font-size: 1.1rem; margin: 2.25rem 0 0.6rem; }
p, li { color: var(--ink-2); }
p { max-width: 38rem; }
ul { max-width: 38rem; padding-left: 1.1rem; }
li { margin-bottom: 0.3rem; }
code { font-family: var(--mono); font-size: 0.86em; }
table {
  border-collapse: collapse; width: 100%; margin: 1.75rem 0;
  font-size: 0.88rem; display: block; overflow-x: auto;
}
caption {
  text-align: left; padding-bottom: 0.6rem;
  font-size: 0.86rem; color: var(--muted);
}
th, td { border-bottom: 1px solid var(--rule); padding: 0.55rem 0.8rem; text-align: left; }
th {
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--muted); white-space: nowrap;
}
td { color: var(--ink-2); }
td:first-child { color: var(--ink); }
td:not(:first-child), th:not(:first-child) {
  text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono);
}
th:not(:first-child) { font-family: var(--sans); }
tbody tr:hover { background: var(--surface); }
figure { margin: 2rem 0; }
figure img {
  display: block; width: 100%; height: auto;
  background: var(--plate); border: 1px solid var(--rule);
  border-radius: 8px; padding: 0.75rem;
}
figcaption { font-size: 0.86rem; color: var(--muted); padding-top: 0.9rem; max-width: 44rem; }
"""

# What the landing page has and a report does not: a navigation bar, a hero, sections with
# a kicker, a register of runs, and a footer. Everything here either introduces one of
# those or overrides a shared rule the page treats differently, and the override is
# written next to what it overrides so the reason stays visible.
_LANDING_RULES = """\
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
a { color: inherit; }
strong { color: var(--ink); font-weight: 600; }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
/* The page reads at a narrower measure than a report, and sets it on the block rather
   than on the paragraph, so that a list inside a passage lines up with the prose. */
.prose { max-width: 34rem; }
.prose p, .prose ul { max-width: none; }
main { padding-bottom: 0; }

nav {
  position: sticky; top: 0; z-index: 10;
  background: color-mix(in srgb, var(--plane) 92%, transparent);
  backdrop-filter: blur(8px); border-bottom: 1px solid var(--rule);
}
nav .wrap { display: flex; align-items: center; gap: 1.25rem; height: 3rem; }
nav .brand {
  font-family: var(--mono); font-size: 0.78rem; color: var(--ink); letter-spacing: -0.01em;
}
nav .links { display: flex; gap: 1.1rem; margin-left: auto; font-size: 0.8rem; }
nav a { color: var(--muted); text-decoration: none; }
nav a:hover { color: var(--ink); }
nav button {
  font: inherit; font-size: 0.78rem; color: var(--muted); background: none;
  border: 1px solid var(--rule); border-radius: 999px; padding: 0.15rem 0.7rem; cursor: pointer;
}
@media (max-width: 720px) { nav .links a { display: none; } }

.hero { padding: 5rem 0 3rem; }
.eyebrow {
  font-family: var(--mono); font-size: 0.74rem; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 1.25rem;
}
.hero h1 { font-size: clamp(2rem, 5vw, 3.1rem); margin: 0 0 1.25rem; max-width: 22ch; }
.hero .answer {
  font-family: var(--serif); font-size: clamp(1.1rem, 2.2vw, 1.35rem);
  line-height: 1.45; color: var(--ink-2); max-width: 40rem; margin: 0;
}
.headline-figs {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 2.5rem;
  margin: 3.5rem 0 0; padding-top: 2rem; border-top: 1px solid var(--rule);
}
.fig-big { font-family: var(--serif); font-size: 2.6rem; line-height: 1; letter-spacing: -0.02em; }
.fig-big.f-nbody { color: var(--nbody); }
.fig-big.f-fluid { color: var(--fluid); }
.fig-big.f-fail { color: var(--fail); }
.fig-big.f-muted { color: var(--muted); }
.fig-lead { font-size: 0.95rem; color: var(--ink); margin: 0.6rem 0 0.25rem; font-weight: 500; }
.fig-note { font-size: 0.84rem; color: var(--muted); margin: 0; max-width: none; }
@media (max-width: 720px) { .headline-figs { grid-template-columns: 1fr; gap: 1.75rem; } }
.aside-link {
  margin-top: 2.5rem; padding: 0.9rem 1.1rem; border: 1px solid var(--rule);
  border-radius: 8px; background: var(--surface); font-size: 0.9rem; color: var(--ink-2);
  max-width: none;
}
.aside-link a { color: var(--ink); text-underline-offset: 3px; }

section { padding: 4.5rem 0; border-top: 1px solid var(--rule); }
section > .wrap > .kicker {
  font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 0.9rem;
}
/* A section carries its own rule above it, so its heading does not carry a second one. */
section h2 {
  font-size: clamp(1.5rem, 3vw, 2rem); margin: 0 0 1rem; max-width: 24ch;
  border-top: 0; padding-top: 0;
}
figure { margin: 2.5rem 0 0; }
figcaption strong { color: var(--ink-2); font-weight: 600; }
.chart-card {
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 10px; padding: 1.5rem;
}
.chart-title { font-size: 0.95rem; font-weight: 600; margin: 0 0 0.2rem; color: var(--ink); }
.chart-sub { font-size: 0.84rem; color: var(--muted); margin: 0 0 1.25rem; max-width: none; }
.chart-note { font-size: 0.8rem; color: var(--muted); margin: 1.1rem 0 0; max-width: none; }
/* A card whose last line is prose rather than a chart should not carry the gap that
   separates a subtitle from the drawing below it. */
.chart-card > p:last-child { margin-bottom: 0; }
.legend {
  display: flex; flex-wrap: wrap; gap: 1.25rem; margin-bottom: 1rem;
  font-size: 0.8rem; color: var(--ink-2);
}
.legend span { display: inline-flex; align-items: center; gap: 0.45rem; }
.swatch { width: 11px; height: 11px; border-radius: 2px; flex: none; }
svg { display: block; max-width: 100%; height: auto; }

.controls {
  display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; margin-bottom: 1.25rem;
}
.ctrl-group { display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; }
.ctrl-label {
  font-family: var(--mono); font-size: 0.7rem; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--muted);
}
.seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 999px; }
/* The row of predictors belonging to a system the reader has not selected. Stated because
   the display above would otherwise beat the browser's own rule for a hidden element. */
.seg[hidden] { display: none; }
.seg button {
  font: inherit; font-size: 0.82rem; color: var(--ink-2); background: transparent;
  border: 0; padding: 0.28rem 0.8rem; cursor: pointer;
}
.seg button:first-child { border-radius: 999px 0 0 999px; }
.seg button:last-child { border-radius: 0 999px 999px 0; }
.seg button + button { border-left: 1px solid var(--rule); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--plane); }
/* The controls are the first thing on the page a reader drives from the keyboard, so the
   focus ring is stated rather than left to the browser, which draws it in a colour the
   dark theme can swallow. */
.seg button:focus-visible { outline: 2px solid var(--fluid); outline-offset: 2px; }

/* The plots are drawn on white, so the plate stays white in the dark theme rather than
   inverting and leaving a light figure floating on a dark page. The shared rule dresses a
   figure's image itself; here the plate carries the border and the image sits flat on it. */
.plate {
  background: var(--plate); border: 1px solid var(--rule);
  border-radius: 8px; padding: 0.75rem;
}
.plate img { border: 0; border-radius: 0; padding: 0; background: none; }
.strip-note {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem;
  margin-top: 1.25rem; font-size: 0.88rem; color: var(--ink-2);
}
.strip-note .h { font-weight: 600; color: var(--ink); display: block; margin-bottom: 0.15rem; }
@media (max-width: 720px) { .strip-note { grid-template-columns: 1fr; gap: 0.9rem; } }

/* A report right aligns every column but the first, because every one of its tables is a
   table of numbers. The page has tables of words, so alignment is asked for by the cell. */
td:not(:first-child), th:not(:first-child) { text-align: left; font-family: var(--sans); }
th.n, td.n { text-align: right; }
td.n { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.tag {
  display: inline-block; font-size: 0.72rem; padding: 0.05rem 0.45rem;
  border-radius: 3px; border: 1px solid currentColor;
}
.t-hidden { color: var(--fail); }
.t-visible { color: var(--muted); }
.rank-1 { color: var(--good-ink); font-weight: 600; }
.rank-3 { color: var(--warn-ink); font-weight: 600; }

.findings { list-style: none; padding: 0; margin: 2rem 0 0; max-width: none; }
.findings li {
  padding: 1.1rem 0; border-top: 1px solid var(--rule); margin-bottom: 0;
  display: grid; grid-template-columns: 12rem 1fr; gap: 1.5rem;
}
.findings li:last-child { border-bottom: 1px solid var(--rule); }
.findings .what { font-weight: 600; font-size: 0.95rem; color: var(--ink); }
.findings .why { color: var(--ink-2); font-size: 0.92rem; margin: 0; max-width: none; }
@media (max-width: 720px) { .findings li { grid-template-columns: 1fr; gap: 0.35rem; } }

.runs { margin-top: 2rem; }
.run {
  display: grid; grid-template-columns: 1fr 8rem 9rem 6rem; gap: 1.25rem;
  align-items: baseline; padding: 0.95rem 0.9rem; border-bottom: 1px solid var(--rule);
  text-decoration: none; border-left: 2px solid transparent; --accent: var(--muted);
}
.run:hover { background: var(--surface); border-left-color: var(--accent); }
.run[data-system="nbody"] { --accent: var(--nbody); }
.run[data-system="fluid"] { --accent: var(--fluid); }
.run .name { font-weight: 600; font-size: 0.95rem; color: var(--ink); }
.run .desc { font-size: 0.82rem; color: var(--muted); }
.run .cell { font-size: 0.84rem; color: var(--ink-2); }
.run .cell small {
  display: block; font-size: 0.72rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em;
}
.run .rid { font-family: var(--mono); font-size: 0.74rem; color: var(--muted); text-align: right; }
.status { font-weight: 600; }
.s-good { color: var(--good-ink); }
.s-warn { color: var(--warn-ink); }
.s-fail { color: var(--fail); }
.s-none { color: var(--muted); }
@media (max-width: 860px) {
  .run { grid-template-columns: 1fr 1fr; }
  .run .rid { text-align: left; }
}

footer {
  padding: 3rem 0 5rem; border-top: 1px solid var(--rule);
  font-size: 0.86rem; color: var(--muted);
}
footer p { color: var(--muted); }
footer code {
  font-family: var(--mono); font-size: 0.8rem; background: var(--surface);
  border: 1px solid var(--rule); border-radius: 4px; padding: 0.1rem 0.35rem; color: var(--ink-2);
}
"""
