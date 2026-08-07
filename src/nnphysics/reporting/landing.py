"""The landing page, written from the page model and the curated prose.

The page answers one question for a reader who knows none of the vocabulary. This module
places the sentences from `prose` beside the numbers from `page` and writes the result as
one HTML file. It derives nothing and it writes no sentence of its own: a claim on the page
comes from `prose`, a number comes from `page`, and this module decides only where each
goes and what colour it is.

Images are referenced, not embedded. A run report embeds its images as data URIs because a
report is meant to be sent to one person as one file. The landing page references the plots
of every run and would reach several megabytes if it embedded them, and it is always served
from the directory that holds them, so a relative path is enough. That is the one place the
two artefacts deliberately differ.

Everything that reaches the page is escaped. Curated text is escaped first and then given
its emphasis, so `**bold**` in `prose` becomes markup while a less than sign in a run name
stays text.
"""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

from nnphysics.reporting import charts, prose
from nnphysics.reporting.page import (
    PERSISTENCE,
    TOP_ONE,
    TOP_THREE,
    TRAINED_ON_SPLIT,
    VerdictKind,
)
from nnphysics.reporting.theme import css_var, landing_stylesheet

if TYPE_CHECKING:
    from nnphysics.reporting.page import Diagnosis, Horizon, PageModel, RunCard

__all__ = ["LANDING_NAME", "render_landing"]

LANDING_NAME = "index.html"
"""What the page is called in the runs root, so that serving the directory serves it."""

TIMES = prose.TIMES
"""The multiplication sign. Held in `prose` so that the charts and the register state a
ratio the same way, and re-exported here because it is what the register prints."""

_TONED_SYSTEMS = frozenset({"nbody", "fluid"})
"""The systems the theme has a colour for. A system added later is drawn in the neutral
colour rather than reaching for a token that does not exist."""

_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_CODE = re.compile(r"`([^`]+)`")

_TEENS = range(10, 21)
"""The positions that take `th` whatever digit they end in."""

_THEME_SCRIPT = """\
const root = document.documentElement;
const button = document.getElementById("theme");
const apply = (theme) => {
  root.dataset.theme = theme;
  button.textContent = theme === "dark" ? "light" : "dark";
};
apply(matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
button.addEventListener("click", () => {
  apply(root.dataset.theme === "dark" ? "light" : "dark");
});
"""
"""The whole of the page's behaviour so far: choose the theme the reader's system asks for,
and let the button change it. The page is readable with the script blocked, because the
light theme is the default and every section is already rendered."""


def render_landing(model: PageModel) -> str:
    """Write the landing page.

    Args:
        model: Everything the page shows, derived from a runs root.

    Returns:
        The page as one HTML file, ending in a newline. It references nothing outside the
        runs root: no font, no script and no stylesheet from a host, and only relative
        paths to the reports and the plots beside it.
    """
    sections = [
        _premise(),
        _gap_section(prose.DRIFT, prose.DRIFT_PLACEHOLDER),
        _chart_section(prose.TRUST, charts.usable_steps_chart(model)),
        _cost(model),
    ]
    if model.diagnosis is not None:
        sections.append(_diagnosis(model.diagnosis))
    sections.append(_findings())
    sections.append(_runs(model))
    body = "\n".join([_nav(model), "<main>", _hero(model), *sections, "</main>", _footer()])
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en-GB">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_escape(prose.TITLE)}</title>\n"
        f"<style>\n{landing_stylesheet()}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        f"<script>\n{_THEME_SCRIPT}</script>\n"
        "</body>\n"
        "</html>\n"
    )


def _escape(text: str) -> str:
    """Anything at all, as text rather than as markup."""
    return html.escape(text, quote=True)


def _inline(text: str) -> str:
    """Curated text as markup, escaped before it is emphasised.

    Escaping first is what makes the emphasis safe: by the time `**` is looked for, every
    angle bracket in the sentence is already an entity, so the only markup that can come
    out is the markup this function puts in.
    """
    escaped = _escape(text)
    return _CODE.sub(r"<code>\1</code>", _BOLD.sub(r"<strong>\1</strong>", escaped))


def _fill(template: str, **values: str) -> str:
    """Curated text with derived values placed into it.

    Args:
        template: A sentence from `prose`, with a named field per value.
        values: What to put in each field, as plain text.

    Returns:
        The sentence as markup, with every value escaped.
    """
    return _inline(template).format(**{name: _escape(value) for name, value in values.items()})


def _paragraphs(texts: tuple[str, ...], **fields: str) -> str:
    """A run of curated paragraphs, with any named fields filled in."""
    return "\n".join(f"<p>{_fill(text, **fields)}</p>" for text in texts)


def _steps(value: float) -> str:
    """A count of stored steps, which is a fraction of a step often enough to show one."""
    return f"{value:g}"


def _speed(value: float) -> str:
    """A speedup, to the two figures the benchmark's own spread supports."""
    return f"{value:.2g}"


def _ordinal(position: int) -> str:
    """A one based position as a reader writes it."""
    suffix = (
        "th" if position % 100 in _TEENS else {1: "st", 2: "nd", 3: "rd"}.get(position % 10, "th")
    )
    return f"{position}{suffix}"


def _nav(model: PageModel) -> str:
    """The sticky navigation, the wordmark and the theme button."""
    links = "".join(
        f'<a href="#{_escape(anchor)}">{_escape(label)}</a>'
        for anchor, label in prose.NAV_LINKS
        if anchor != prose.DIAGNOSIS.anchor or model.diagnosis is not None
    )
    light, _dark = prose.THEME_BUTTON
    return (
        "<nav>\n"
        '<div class="wrap">'
        f'<span class="brand">{_escape(prose.WORDMARK)}</span>'
        f'<span class="links">{links}</span>'
        f'<button id="theme" type="button">{_escape(light)}</button>'
        "</div>\n"
        "</nav>"
    )


def _system_colour(system: str) -> str:
    """The class that colours a figure by the system it came from."""
    return f"f-{system}" if system in _TONED_SYSTEMS else "f-muted"


def _system_of(model: PageModel, run_id: str) -> str:
    """Which system a headline figure came from, so the figure can be coloured by it."""
    return next((run.system for run in model.runs if run.run_id == run_id), "")


def _headline(value: str, colour: str, headline: prose.Headline, **fields: str) -> str:
    """One of the three figures in the hero."""
    return (
        "<div>"
        f'<div class="fig-big {colour}">{_escape(value)}</div>'
        f'<p class="fig-lead">{_inline(headline.lead)}</p>'
        f'<p class="fig-note">{_fill(headline.note, **fields)}</p>'
        "</div>"
    )


def _hero(model: PageModel) -> str:
    """The question, the answer, the three figures and the note about the agent."""
    figures = model.headlines
    usable, unseen, slowdown = prose.HEADLINES
    # The unseen figure is a failure only when nothing survived. Colouring it red whatever
    # it says would be the page deciding the result before the record does.
    unseen_colour = (
        "f-fail"
        if figures.held_out_completed == 0
        else _system_colour(_system_of(model, figures.held_out_run_id))
    )
    cards = "\n".join(
        [
            _headline(
                f"{_steps(figures.usable_steps)} of {figures.usable_of}",
                _system_colour(_system_of(model, figures.usable_run_id)),
                usable,
            ),
            _headline(
                f"{figures.held_out_completed} of {figures.held_out_of}",
                unseen_colour,
                unseen,
                predictor=prose.model_label(figures.held_out_predictor),
            ),
            _headline(
                f"{figures.slowdown:.0f}{TIMES} slower",
                _system_colour(_system_of(model, figures.slowdown_run_id)),
                slowdown,
            ),
        ]
    )
    return (
        '<header class="hero">\n'
        '<div class="wrap">\n'
        f'<p class="eyebrow">{_inline(prose.EYEBROW)}</p>\n'
        f"<h1>{_inline(prose.TITLE)}</h1>\n"
        f'<p class="answer">{_inline(prose.ANSWER)}</p>\n'
        f'<div class="headline-figs">\n{cards}\n</div>\n'
        f"{_aside(model.diagnosis)}"
        "</div>\n"
        "</header>"
    )


def _aside(diagnosis: Diagnosis | None) -> str:
    """The note pointing at the diagnosis section, or nothing when there is no section."""
    if diagnosis is None:
        return ""
    sentence = _fill(
        prose.ASIDE,
        faults=str(diagnosis.agent.faults),
        agent=str(diagnosis.agent.top1_count),
        baseline=f"{diagnosis.baseline.top1_count} of {diagnosis.baseline.faults}",
    )
    link = f'<a href="#{_escape(prose.DIAGNOSIS.anchor)}">{_escape(prose.ASIDE_LINK)}</a>.'
    return f'<p class="aside-link">{sentence} {link}</p>\n'


def _open_section(section: prose.Section, **fields: str) -> str:
    """A section, its kicker, its heading and its introduction."""
    return (
        f'<section id="{_escape(section.anchor)}">\n'
        '<div class="wrap">\n'
        f'<p class="kicker">{_inline(section.kicker)}</p>\n'
        f"<h2>{_inline(section.heading)}</h2>\n"
        f'<div class="prose">\n{_paragraphs(section.paragraphs, **fields)}\n</div>'
    )


def _close_section() -> str:
    """The end of a section."""
    return "</div>\n</section>"


def _premise() -> str:
    """Why the question is worth asking, and the schematic that shows it."""
    schematic = charts.Chart(
        title=prose.SCHEMATIC.title,
        subtitle=prose.SCHEMATIC.subtitle,
        legend=(),
        svg=prose.SCHEMATIC_SVG.format(alt=_escape(prose.SCHEMATIC.alt)),
        note="",
        caption=prose.SCHEMATIC.caption,
    )
    return "\n".join([_open_section(prose.PREMISE), _figure(schematic), _close_section()])


def _legend(entries: tuple[charts.Legend, ...]) -> str:
    """What each colour in a chart means, or nothing where colour carries nothing."""
    if not entries:
        return ""
    items = "".join(
        f'<span><i class="swatch" style="background:{css_var(entry.colour)}"></i>'
        f"{_escape(entry.label)}</span>"
        for entry in entries
    )
    return f'<div class="legend">{items}</div>\n'


def _figure(chart: charts.Chart) -> str:
    """One chart on its card, with its title, its legend, its note and its caption."""
    note = f'<p class="chart-note">{_inline(chart.note)}</p>\n' if chart.note else ""
    return (
        "<figure>\n"
        '<div class="chart-card">\n'
        f'<p class="chart-title">{_inline(chart.title)}</p>\n'
        f'<p class="chart-sub">{_inline(chart.subtitle)}</p>\n'
        f"{_legend(chart.legend)}"
        f"{chart.svg}"
        f"{note}"
        "</div>\n"
        f"<figcaption>{_inline(chart.caption)}</figcaption>\n"
        "</figure>"
    )


def _gap(placeholder: prose.Placeholder) -> str:
    """Where a later phase draws a figure.

    Marked with an identifier so that phase can find it rather than search the markup for
    a sentence.
    """
    return (
        "<figure>\n"
        f'<div class="placeholder" id="{_escape(placeholder.anchor)}">'
        f"{_inline(placeholder.text)}</div>\n"
        "</figure>"
    )


def _gap_section(section: prose.Section, placeholder: prose.Placeholder) -> str:
    """A section whose figure a later phase draws."""
    return "\n".join([_open_section(section), _gap(placeholder), _close_section()])


def _chart_section(section: prose.Section, chart: charts.Chart | None) -> str:
    """A section built around one chart, or its prose alone where there is none to draw."""
    parts = [_open_section(section)]
    if chart is not None:
        parts.append(_figure(chart))
    parts.append(_close_section())
    return "\n".join(parts)


def _cost(model: PageModel) -> str:
    """The cost comparison, and a note on any benchmark the chart cannot draw."""
    parts = [_open_section(prose.COST)]
    chart = charts.cost_chart(model)
    if chart is not None:
        parts.append(_figure(chart))
    asides = charts.cost_asides(model)
    if asides:
        sentences = "".join(f'<p class="chart-sub">{_inline(text)}</p>\n' for text in asides)
        parts.append(
            "<figure>\n"
            '<div class="chart-card">\n'
            f'<p class="chart-title">{_inline(prose.COST_ASIDE_TITLE)}</p>\n'
            f"{sentences}"
            "</div>\n"
            "</figure>"
        )
    parts.append(_close_section())
    return "\n".join(parts)


def _diagnosis(diagnosis: Diagnosis) -> str:
    """The agent, how it scored against the baseline, and every fault it was scored on."""
    header = _open_section(prose.DIAGNOSIS, faults=str(diagnosis.agent.faults))
    figure = _figure(charts.diagnosis_chart(diagnosis))
    closing = f'<div class="prose">\n{_paragraphs(prose.DIAGNOSIS_CLOSING)}\n</div>'
    return "\n".join([header, figure, _fault_table(diagnosis), closing, _close_section()])


def _fault_table(diagnosis: Diagnosis) -> str:
    """Every injected fault, and where the agent ranked the true cause."""
    last = len(prose.DIAGNOSIS_TABLE.headers) - 1
    headers = "".join(
        f'<th class="n">{_escape(header)}</th>' if index == last else f"<th>{_escape(header)}</th>"
        for index, header in enumerate(prose.DIAGNOSIS_TABLE.headers)
    )
    rows = "".join(
        _fault_row(entry.fault, entry.true_cause, entry.rank) for entry in diagnosis.agent.ranks
    )
    return (
        "<table>"
        f"<caption>{_inline(prose.DIAGNOSIS_TABLE.caption)}</caption>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


def _fault_row(fault: str, true_cause: str, rank: int | None) -> str:
    """One fault, in the reader's terms, with where the agent placed it."""
    note = prose.fault_note(fault, true_cause)
    label = prose.HIDDEN_LABEL if note.hidden else prose.VISIBLE_LABEL
    tag = f'<span class="tag t-{_escape(label)}">{_escape(label)}</span>'
    if rank is None:
        placing, css = prose.UNKNOWN_RANK, "n"
    elif rank == TOP_ONE:
        placing, css = _ordinal(rank), "n rank-1"
    elif rank <= TOP_THREE:
        placing, css = _ordinal(rank), "n rank-3"
    else:
        placing, css = _ordinal(rank), "n"
    return (
        "<tr>"
        f"<td>{_escape(note.name)}</td>"
        f"<td>{_escape(note.plain)}</td>"
        f"<td>{tag}</td>"
        f'<td class="{css}">{_escape(placing)}</td>'
        "</tr>"
    )


def _findings() -> str:
    """The negative results, kept rather than compressed."""
    items = "\n".join(
        "<li>"
        f'<span class="what">{_inline(finding.what)}</span>'
        f'<p class="why">{_inline(finding.why)}</p>'
        "</li>"
        for finding in prose.FINDING_LIST
    )
    return "\n".join(
        [
            _open_section(prose.FINDINGS),
            f'<ul class="findings">\n{items}\n</ul>',
            _close_section(),
        ]
    )


def _learned(run: RunCard) -> Horizon | None:
    """What the run's own model managed on the split it trained on."""
    return next(
        (
            entry
            for entry in run.horizons
            if entry.learned and entry.split == TRAINED_ON_SPLIT and entry.steps is not None
        ),
        None,
    )


def _beats_baseline(run: RunCard) -> bool:
    """Whether the run's model stayed usable longer than doing nothing did."""
    baseline = run.horizon(PERSISTENCE, TRAINED_ON_SPLIT)
    learned = _learned(run)
    if baseline is None or baseline.steps is None or learned is None or learned.steps is None:
        return False
    return learned.steps > baseline.steps


def _tone(run: RunCard) -> str:
    """How strongly to state a verdict, read from the record rather than from its wording.

    The phrase and the colour are decided separately on purpose. A verdict that changes its
    wording should not silently change its colour, and a colour chosen by matching a
    substring of a sentence would.
    """
    kind = run.verdict.kind
    if kind is VerdictKind.HARNESS_CHECK:
        return "s-none"
    if kind in (VerdictKind.DIVERGES, VerdictKind.CANNOT_RUN_UNSEEN):
        return "s-fail"
    if kind is VerdictKind.COST_BENCHMARK:
        if run.cost is None or not run.cost.matched:
            return "s-none"
        if all(entry.break_even_rollouts is None for entry in run.cost.matched):
            return "s-fail"
        return "s-warn"
    return "s-good" if _beats_baseline(run) else "s-warn"


def _role(run: RunCard) -> str:
    """What the run was for, in the reader's terms."""
    if run.model is not None:
        return prose.model_label(run.model)
    learned = tuple(dict.fromkeys(entry.predictor for entry in run.horizons if entry.learned))
    if learned:
        return ", ".join(prose.model_label(predictor) for predictor in learned)
    return prose.HARNESS_ROLE


def _description(run: RunCard) -> str:
    """What the run cost to produce, or what it was made of."""
    if run.parameters is not None and run.epochs is not None:
        return f"{run.parameters:,} parameters, {run.epochs} epochs"
    if run.cost is not None:
        threads = "thread" if run.cost.threads == 1 else "threads"
        return f"speed benchmark, {run.cost.threads} {threads}, {run.cost.trials} timed trials"
    return f"{len({entry.predictor for entry in run.horizons})} harness predictors, no model"


def _measure(run: RunCard) -> tuple[str, str]:
    """The one number the register shows for a run, and what it measures."""
    if run.verdict.kind is VerdictKind.COST_BENCHMARK and run.cost is not None and run.cost.matched:
        speeds = sorted(entry.speedup for entry in run.cost.matched)
        low, high = _speed(speeds[0]), _speed(speeds[-1])
        return prose.SPEED_CELL, f"{low}{TIMES}" if low == high else f"{low} to {high}{TIMES}"
    entry = _learned(run)
    if entry is None or entry.steps is None:
        return prose.USABLE_CELL, prose.NO_MODEL
    return prose.USABLE_CELL, f"{_steps(entry.steps)} of {entry.rollout_steps} steps"


def _identifier(run_id: str) -> str:
    """A run identifier over two lines, so a column of them stays narrow."""
    half = len(run_id) // 2
    return f"{_escape(run_id[:half])}<br>{_escape(run_id[half:])}" if half else _escape(run_id)


def _run_row(run: RunCard) -> str:
    """One run, linking to its report."""
    label, value = _measure(run)
    title = f"{prose.system_label(run.system)}, {_role(run)}"
    return (
        f'<a class="run" data-system="{_escape(run.system)}" href="{_escape(run.report)}">'
        f'<span><span class="name">{_escape(title)}</span><br>'
        f'<span class="desc">{_escape(_description(run))}</span></span>'
        f'<span class="cell"><small>{_escape(label)}</small>{_escape(value)}</span>'
        f'<span class="cell"><small>{_escape(prose.VERDICT_CELL)}</small>'
        f'<span class="status {_tone(run)}">{_escape(run.verdict.phrase)}</span></span>'
        f'<span class="rid">{_identifier(run.run_id)}</span>'
        "</a>"
    )


def _runs(model: PageModel) -> str:
    """Every run the page reports on, grouped by system."""
    ordered = sorted(model.reported, key=lambda run: (run.system, run.created, run.run_id))
    rows = "\n".join(_run_row(run) for run in ordered)
    return "\n".join(
        [_open_section(prose.RUNS), f'<div class="runs">\n{rows}\n</div>', _close_section()]
    )


def _footer() -> str:
    """How to rebuild everything the page links to, and where the limitations are."""
    return f'<footer>\n<div class="wrap">\n{_paragraphs(prose.FOOTER)}\n</div>\n</footer>'
