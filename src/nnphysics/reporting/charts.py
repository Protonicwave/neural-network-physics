"""The three data charts, drawn from the page model.

These charts are the argument. A reader who reads nothing else should be able to see that
a surrogate is usable for a small fraction of the rollout, that it loses to the solver on
cost, and that the diagnosis agent beats its baseline. Everything here is derived: this
module places marks, and the sentences around them come from `prose`.

Each chart returns its own text as well as its marks, because the words that make a chart
readable, its title, its legend, its note and its caption, are part of the chart rather
than decoration the page adds afterwards. The builder places them; it does not choose
them.

A chart is drawn from the runs the page reports on, never from one run. A predictor is
measured in more than one run and the chart shows the longest stretch it reached, so that
the page states the strongest case for the surrogate and still shows it falling short.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nnphysics.reporting import draw, prose
from nnphysics.reporting.page import HELD_OUT_SPLIT, PERSISTENCE, TRAINED_ON_SPLIT

if TYPE_CHECKING:
    from nnphysics.reporting.page import (
        CostLadder,
        CostPoint,
        Diagnosis,
        MatchedCost,
        PageModel,
        RunCard,
    )

__all__ = [
    "Chart",
    "Legend",
    "cost_asides",
    "cost_chart",
    "diagnosis_chart",
    "usable_steps_chart",
]

_TONES = {"nbody": ("nbody", "nbody_soft"), "fluid": ("fluid", "fluid_soft")}
"""The colour pair each system is drawn in: the full strength one for the split it
trained on, the held back one for the split it never saw. A system the theme has no pair
for is drawn in text colours, which are legible in both themes and claim nothing."""

_NEUTRAL = ("ink_2", "muted")

_BASELINE = ("muted", "muted")
"""The free baseline is grey in both panels and on both splits, because the reader is
meant to compare against it rather than to tell its two bars apart by colour."""

_WIDTH = 720.0
"""Viewport width every chart is drawn in. The page scales it to whatever width it has,
so this is a ratio rather than a size in pixels."""

_TRACK_LEFT = 150.0
_TRACK_RIGHT = 650.0
_VALUE_X = 720.0
_ROW = 32.0
_BAR = 7.0
_PAIR = 11.0
"""The bar chart's frame: labels to the left of 150, tracks to 650, values ending at the
right edge so that a long one cannot run off it. A row is 32 apart, each bar 7 thick, the
second 11 below the first."""

_TITLE_DROP = 18.0
_PANEL_GAP = 56.0
_TEXT_DROP = 12.0

_PLOT_LEFT = 60.0
_PLOT_RIGHT = 690.0
_PLOT_TOP = 30.0
_PLOT_BOTTOM = 350.0
_PLOT_HEIGHT = 400.0
"""The cost chart's frame, with room below the axis for its ticks and its title."""

_PAD_DECADES = 0.15
"""How much room to leave at each end of the logarithmic axis, in decades, so that the
cheapest and the slowest points do not sit on the frame."""

_HEADROOM = 1.05
"""How far above the largest error the linear axis reaches, for the same reason."""

_MS = 1000.0

_DIAGNOSIS_LEFT = 220.0
_DIAGNOSIS_RIGHT = 645.0
_DIAGNOSIS_HEIGHT = 200.0
_GROUP = 90.0
_THICK = 18.0
"""The diagnosis chart's frame: two groups 90 apart, two bars of 18 in each."""

_AGENT_COLOUR = "nbody"
"""What the agent's bars are drawn in. The theme has an accent per system and the agent
belongs to neither, so it borrows one rather than the theme growing a token that means
nothing anywhere else."""


@dataclass(frozen=True, slots=True)
class Legend:
    """One entry in a chart's legend.

    Attributes:
        colour: Theme token name of the swatch.
        label: What that colour means.
    """

    colour: str
    label: str


@dataclass(frozen=True, slots=True)
class Chart:
    """One chart, with everything a reader needs to read it.

    Attributes:
        title: What the chart is.
        subtitle: How to read its axes.
        legend: What each colour means, empty when colour carries nothing.
        svg: The marks.
        note: What the chart does not say for itself, empty when there is nothing to add.
        caption: What to conclude from it.
    """

    title: str
    subtitle: str
    legend: tuple[Legend, ...]
    svg: str
    note: str
    caption: str


@dataclass(frozen=True, slots=True)
class _Row:
    """One predictor in one panel of the usable steps chart.

    Attributes:
        label: The predictor as the reader knows it.
        trained: Steps it stayed usable on the split it trained on, or `None` when it
            never ran there.
        unseen: The same on the split it never saw.
        tones: The colour pair its two bars are drawn in.
    """

    label: str
    trained: float | None
    unseen: float | None
    tones: tuple[str, str]


@dataclass(frozen=True, slots=True)
class _Panel:
    """One system's rows, on one axis.

    Attributes:
        system: The system as the record names it.
        rollout_steps: Steps the rollout asked for, which is the length of the axis.
        rows: The predictors, in the order the page met them.
    """

    system: str
    rollout_steps: int
    rows: tuple[_Row, ...]


def usable_steps_chart(model: PageModel) -> Chart | None:
    """Draw how far each surrogate stays usable, one panel per system.

    Args:
        model: Everything the page shows.

    Returns:
        The chart, or `None` when no run trained a model there is a stretch to draw.
    """
    panels = _panels(model)
    if not panels:
        return None
    marks: list[str] = []
    top = 0.0
    for panel in panels:
        marks.extend(_panel_marks(panel, top))
        top = _panel_bottom(panel, top) + _PANEL_GAP
    height = top - _PANEL_GAP + _TEXT_DROP
    summary = " ".join(
        prose.TRUST_SUMMARY.format(
            system=prose.system_label(panel.system),
            steps=_steps(_best(panel)),
            total=panel.rollout_steps,
        )
        for panel in panels
    )
    trained, unseen = prose.TRUST_LEGEND
    legend = tuple(
        Legend(colour=tone, label=text.format(system=prose.system_label(panel.system)))
        for panel in panels
        for tone, text in zip(_tones(panel.system), (trained, unseen), strict=True)
    )
    return Chart(
        title=prose.TRUST_CHART.title,
        subtitle=prose.TRUST_CHART.subtitle,
        legend=(*legend, Legend(colour=_BASELINE[0], label=prose.TRUST_BASELINE)),
        svg=draw.canvas(
            _WIDTH, height, tuple(marks), prose.TRUST_CHART.alt.format(summary=summary)
        ),
        note=prose.TRUST_NOTE,
        caption=prose.TRUST_CHART.caption,
    )


def cost_chart(model: PageModel) -> Chart | None:
    """Draw accuracy against wall clock for the benchmark the page argues from.

    One benchmark is drawn rather than all of them: the one carrying the largest matched
    slowdown, which is the comparison the hero quotes. A second benchmark on the same
    axes would need an error axis wide enough for a diverged surrogate, and every solver
    setting would collapse onto the origin.

    Args:
        model: Everything the page shows.

    Returns:
        The chart, or `None` when no benchmark measured a surrogate against the solver.
    """
    chosen = _chosen(model)
    if chosen is None:
        return None
    cost, matched = chosen
    points = (*cost.solver, *cost.surrogates)
    time = draw.Log(
        lo=min(point.seconds_per_step for point in points) * _MS * 10.0**-_PAD_DECADES,
        hi=max(point.seconds_per_step for point in points) * _MS * 10.0**_PAD_DECADES,
        start=_PLOT_LEFT,
        end=_PLOT_RIGHT,
    )
    worst = max(point.error for point in points)
    error = draw.Linear(
        lo=0.0,
        hi=worst * _HEADROOM if worst > 0.0 else 1.0,
        start=_PLOT_BOTTOM,
        end=_PLOT_TOP,
    )
    marks = (
        *_cost_axes(time, error),
        *_connector(cost, matched, time, error),
        *_ladder_marks(cost, time, error),
        *_surrogate_marks(cost, time, error),
    )
    summary = prose.COST_SUMMARY.format(
        predictor=prose.model_label(matched.predictor),
        surrogate=_ms(matched.seconds_per_step),
        matched=_ms(matched.matched_seconds_per_step),
    )
    solver, surrogate = prose.COST_LEGEND
    return Chart(
        title=prose.COST_CHART.title.format(system=prose.system_label(cost.system)),
        subtitle=prose.COST_CHART.subtitle.format(
            threads=_threads(cost.threads), trials=cost.trials
        ),
        legend=(
            Legend(colour="muted", label=solver),
            Legend(colour=_tones(cost.system)[0], label=surrogate),
        ),
        svg=draw.canvas(_WIDTH, _PLOT_HEIGHT, marks, prose.COST_CHART.alt.format(summary=summary)),
        note="",
        caption=prose.COST_CHART.caption.format(
            predictor=prose.model_label(matched.predictor),
            surrogate=_ms(matched.seconds_per_step),
            matched=_ms(matched.matched_seconds_per_step),
            payback=_payback(cost),
        ),
    )


def cost_asides(model: PageModel) -> tuple[str, ...]:
    """State the benchmarks the cost chart does not draw.

    A page that drew one benchmark and said nothing about the others would be reporting
    the least favourable result and hiding the rest.

    Args:
        model: Everything the page shows.

    Returns:
        One sentence per benchmark the chart leaves out, in run order.
    """
    chosen = _chosen(model)
    drawn = chosen[0] if chosen is not None else None
    return tuple(_aside(cost) for cost in model.benchmarks if cost is not drawn and cost.matched)


def diagnosis_chart(diagnosis: Diagnosis) -> Chart:
    """Draw the agent against the rule based baseline, on both measures.

    Args:
        diagnosis: Both diagnosers, scored on the same faults.

    Returns:
        The chart.
    """
    faults = diagnosis.agent.faults
    counts = draw.Linear(lo=0.0, hi=float(faults), start=_DIAGNOSIS_LEFT, end=_DIAGNOSIS_RIGHT)
    measured = (
        (prose.DIAGNOSIS_ROWS[0], diagnosis.agent.top1_count, diagnosis.baseline.top1_count),
        (prose.DIAGNOSIS_ROWS[1], diagnosis.agent.top3_count, diagnosis.baseline.top3_count),
    )
    marks: list[str] = []
    for index, (measure, agent, baseline) in enumerate(measured):
        marks.extend(_diagnosis_group(counts, index, measure, (agent, baseline), faults))
    marks.extend(_diagnosis_axis(faults))
    summary = " ".join(
        prose.DIAGNOSIS_SUMMARY.format(
            measure=measure,
            agent=_share(agent, faults),
            baseline=_share(baseline, faults),
        )
        for measure, agent, baseline in measured
    )
    agent_label, baseline_label = prose.DIAGNOSIS_LEGEND
    missed = sum(1 for entry in diagnosis.baseline.ranks if entry.rank is None)
    return Chart(
        title=prose.DIAGNOSIS_CHART.title.format(faults=faults),
        subtitle=prose.DIAGNOSIS_CHART.subtitle,
        legend=(
            Legend(colour=_AGENT_COLOUR, label=agent_label),
            Legend(colour="muted", label=baseline_label),
        ),
        svg=draw.canvas(
            _WIDTH,
            _DIAGNOSIS_HEIGHT,
            tuple(marks),
            prose.DIAGNOSIS_CHART.alt.format(summary=summary),
        ),
        note="",
        caption=prose.DIAGNOSIS_CHART.caption.format(missed=missed, faults=faults),
    )


def _tones(system: str) -> tuple[str, str]:
    """The colour pair one system is drawn in."""
    return _TONES.get(system, _NEUTRAL)


def _systems(model: PageModel) -> tuple[str, ...]:
    """Every system the page reports on, in the order it first met one."""
    return tuple(dict.fromkeys(run.system for run in model.reported))


def _trained_models(runs: tuple[RunCard, ...]) -> tuple[str, ...]:
    """The models one system's runs trained, oldest first.

    A predictor the harness supplies is not one of these, and neither is the ensemble,
    which is four of these averaged rather than a model anybody trained. Both are
    reported elsewhere: the harness baselines against the grey track here, the ensemble on
    the cost chart it was benchmarked in.
    """
    return tuple(dict.fromkeys(run.model for run in runs if run.model is not None))


def _reach(runs: tuple[RunCard, ...], predictor: str, split: str) -> float | None:
    """The longest stretch one predictor stayed usable for on one split, over the runs."""
    measured = [
        entry.steps
        for run in runs
        for entry in run.horizons
        if entry.predictor == predictor and entry.split == split and entry.steps is not None
    ]
    return max(measured) if measured else None


def _panels(model: PageModel) -> tuple[_Panel, ...]:
    """One panel per system, each carrying the models that system trained."""
    panels = []
    for system in _systems(model):
        runs = tuple(run for run in model.reported if run.system == system)
        rows = tuple(
            _Row(
                label=prose.model_label(predictor),
                trained=_reach(runs, predictor, TRAINED_ON_SPLIT),
                unseen=_reach(runs, predictor, HELD_OUT_SPLIT),
                tones=_tones(system),
            )
            for predictor in _trained_models(runs)
        )
        if not rows:
            continue
        baseline = _Row(
            label=prose.model_label(PERSISTENCE),
            trained=_reach(runs, PERSISTENCE, TRAINED_ON_SPLIT),
            unseen=_reach(runs, PERSISTENCE, HELD_OUT_SPLIT),
            tones=_BASELINE,
        )
        panels.append(
            _Panel(
                system=system,
                rollout_steps=max(run.rollout_steps for run in runs),
                rows=(*rows, baseline),
            )
        )
    return tuple(panels)


def _panel_bottom(panel: _Panel, top: float) -> float:
    """Where one panel ends, given where it starts."""
    return top + _TITLE_DROP + len(panel.rows) * _ROW


def _best(panel: _Panel) -> float | None:
    """The longest stretch any model in one panel reached on the split it trained on."""
    reached = [row.trained for row in panel.rows[:-1] if row.trained is not None]
    return max(reached) if reached else None


def _panel_marks(panel: _Panel, top: float) -> tuple[str, ...]:
    """One system's title, rows and axis note."""
    steps = draw.Linear(lo=0.0, hi=float(panel.rollout_steps), start=_TRACK_LEFT, end=_TRACK_RIGHT)
    marks = [
        draw.Label(
            x=0.0,
            y=top + _TEXT_DROP + 2.0,
            text=prose.system_label(panel.system),
            colour="ink",
            size=12.0,
            bold=True,
        ).draw(),
        draw.Label(
            x=_WIDTH,
            y=top + _TEXT_DROP + 2.0,
            text=prose.TRUST_ROLLOUT.format(steps=panel.rollout_steps),
            anchor="end",
        ).draw(),
    ]
    for index, row in enumerate(panel.rows):
        marks.extend(_row_marks(row, steps, top + _TITLE_DROP + index * _ROW))
    return tuple(marks)


def _row_marks(row: _Row, steps: draw.Linear, top: float) -> tuple[str, ...]:
    """One predictor: its name, its two tracks, its two bars and its two numbers."""
    marks = [
        draw.Label(x=0.0, y=top + _TEXT_DROP, text=row.label, colour="ink_2", size=12.0).draw(),
        draw.Bar(
            x=_TRACK_LEFT,
            y=top,
            width=_TRACK_RIGHT - _TRACK_LEFT,
            height=_BAR,
            colour="grid",
        ).draw(),
        draw.Bar(
            x=_TRACK_LEFT,
            y=top + _PAIR,
            width=_TRACK_RIGHT - _TRACK_LEFT,
            height=_BAR,
            colour="grid",
        ).draw(),
    ]
    for offset, value, colour in (
        (0.0, row.trained, row.tones[0]),
        (_PAIR, row.unseen, row.tones[1]),
    ):
        if value is not None:
            marks.append(
                draw.Bar(
                    x=_TRACK_LEFT,
                    y=top + offset,
                    width=steps.length(value),
                    height=_BAR,
                    colour=colour,
                ).draw()
            )
    marks.append(
        draw.Label(
            x=_VALUE_X,
            y=top + _TEXT_DROP,
            text=f"{_steps(row.trained)} / {_steps(row.unseen)}",
            colour="ink_2",
            anchor="end",
            mono=True,
        ).draw()
    )
    return tuple(marks)


def _chosen(model: PageModel) -> tuple[CostLadder, MatchedCost] | None:
    """The benchmark the cost chart draws, and the comparison it draws in full.

    The largest slowdown, ties broken by predictor name, which is the same rule the hero
    figure is selected by, so the chart and the figure cannot quote different runs. A
    benchmark that measured a comparison but timed no points is not drawn: it is stated in
    a sentence instead, by `cost_asides`.
    """
    candidates = [
        (cost, entry)
        for cost in model.benchmarks
        if cost.solver or cost.surrogates
        for entry in cost.matched
        if entry.slowdown is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda pair: (pair[1].slowdown or 0.0, pair[1].predictor))


def _cost_axes(time: draw.Log, error: draw.Linear) -> tuple[str, ...]:
    """The gridlines, the ticks and the two axis titles."""
    up, along = prose.COST_AXES
    marks = []
    for value in draw.ticks(error.lo, error.hi, 4):
        at = error.at(value)
        marks.append(
            draw.Stroke(points=((_PLOT_LEFT, at), (_PLOT_RIGHT, at)), colour="grid").draw()
        )
        marks.append(
            draw.Label(x=_PLOT_LEFT - 10.0, y=at + 4.0, text=_axis(value), anchor="end").draw()
        )
    marks.append(
        draw.Label(
            x=18.0,
            y=(_PLOT_TOP + _PLOT_BOTTOM) / 2.0,
            text=up,
            anchor="middle",
            rotate=-90.0,
        ).draw()
    )
    marks.append(
        draw.Stroke(
            points=((_PLOT_LEFT, _PLOT_BOTTOM), (_PLOT_RIGHT, _PLOT_BOTTOM)), colour="axis"
        ).draw()
    )
    for value in draw.decade_ticks(time.lo, time.hi):
        marks.append(
            draw.Label(
                x=time.at(value), y=_PLOT_BOTTOM + 22.0, text=_axis(value), anchor="middle"
            ).draw()
        )
    marks.append(
        draw.Label(
            x=(_PLOT_LEFT + _PLOT_RIGHT) / 2.0,
            y=_PLOT_BOTTOM + 42.0,
            text=along,
            anchor="middle",
        ).draw()
    )
    return tuple(marks)


def _connector(
    cost: CostLadder, matched: MatchedCost, time: draw.Log, error: draw.Linear
) -> tuple[str, ...]:
    """The dashed line between a surrogate and the solver setting as accurate as it."""
    surrogate = next(
        (point for point in cost.surrogates if point.predictor == matched.predictor), None
    )
    solver = next(
        (point for point in cost.solver if point.substeps == matched.matched_substeps), None
    )
    if surrogate is None or solver is None or matched.slowdown is None:
        return ()
    ends = (
        (time.at(solver.seconds_per_step * _MS), error.at(solver.error)),
        (time.at(surrogate.seconds_per_step * _MS), error.at(surrogate.error)),
    )
    # Above the higher of the two ends rather than above the middle of the line, because
    # the line is diagonal and a label on its midpoint is a label with a line through it.
    middle = ((ends[0][0] + ends[1][0]) / 2.0, min(ends[0][1], ends[1][1]))
    return (
        draw.Stroke(points=ends, colour="fail", width=1.5, dashed=True).draw(),
        draw.Label(
            x=middle[0],
            y=middle[1] - 10.0,
            text=prose.COST_MATCHED.format(slowdown=_slowdown(matched.slowdown)),
            colour="fail",
            size=11.5,
            anchor="middle",
            bold=True,
        ).draw(),
    )


def _ladder_marks(cost: CostLadder, time: draw.Log, error: draw.Linear) -> tuple[str, ...]:
    """The solver's quality dial, joined cheapest to most accurate."""
    if not cost.solver:
        return ()
    places = [_place(point, time, error) for point in cost.solver]
    # One setting is a point rather than a curve. Joining it to itself would draw a mark
    # of zero length and claim the dial was turned.
    joined = len(places) > 1
    marks = [draw.Stroke(points=tuple(places), colour="muted", width=2.0).draw()] if joined else []
    marks.extend(draw.Dot(x=x, y=y, colour="muted").draw() for x, y in places)
    cheapest, exact = prose.COST_LADDER_ENDS
    marks.append(
        draw.Label(x=places[0][0], y=places[0][1] - 12.0, text=cheapest, anchor="middle").draw()
    )
    marks.append(draw.Label(x=places[-1][0] + 8.0, y=places[-1][1] - 12.0, text=exact).draw())
    return tuple(marks)


def _surrogate_marks(cost: CostLadder, time: draw.Log, error: draw.Linear) -> tuple[str, ...]:
    """Every surrogate the benchmark timed, each named beside its point."""
    colour = _tones(cost.system)[0]
    marks = []
    for point in cost.surrogates:
        x, y = _place(point, time, error)
        marks.append(draw.Dot(x=x, y=y, colour=colour, radius=6.0).draw())
        marks.append(
            draw.Label(
                x=x,
                y=y - 14.0,
                text=prose.model_label(point.predictor),
                colour=colour,
                size=11.5,
                anchor="middle",
                bold=True,
            ).draw()
        )
    return tuple(marks)


def _place(point: CostPoint, time: draw.Log, error: draw.Linear) -> tuple[float, float]:
    """Where one measured point sits."""
    return time.at(point.seconds_per_step * _MS), error.at(point.error)


def _payback(cost: CostLadder) -> str:
    """What the benchmark says about repaying the cost of training."""
    rollouts = [
        entry.break_even_rollouts for entry in cost.matched if entry.break_even_rollouts is not None
    ]
    if not rollouts:
        return prose.COST_NEVER_PAYS
    return prose.COST_PAYBACK.format(rollouts=f"{round(min(rollouts)):,}")


def _aside(cost: CostLadder) -> str:
    """One benchmark the chart does not draw, in a sentence."""
    best = max(cost.matched, key=lambda entry: (entry.speedup, entry.predictor))
    parts = [
        prose.COST_ASIDE.format(
            system=prose.system_label(cost.system),
            predictor=prose.model_label(best.predictor),
            speed=_slowdown(best.speedup),
        )
    ]
    if not best.bracketed:
        parts.append(prose.COST_ASIDE_BOUND)
    if best.break_even_rollouts is None:
        parts.append(prose.COST_ASIDE_NEVER)
    else:
        parts.append(
            prose.COST_ASIDE_PAYBACK.format(rollouts=f"{round(best.break_even_rollouts):,}")
        )
    return " ".join(parts)


def _diagnosis_group(
    counts: draw.Linear, index: int, measure: str, scored: tuple[int, int], faults: int
) -> tuple[str, ...]:
    """One measure: its name, the agent's bar and the baseline's."""
    top = index * _GROUP + 26.0
    marks = [draw.Label(x=0.0, y=top + 14.0, text=measure, colour="ink_2", size=12.0).draw()]
    for offset, count, colour, ink in (
        (0.0, scored[0], _AGENT_COLOUR, _AGENT_COLOUR),
        (_THICK + 6.0, scored[1], "muted", "ink_2"),
    ):
        width = counts.length(float(count))
        marks.append(
            draw.Bar(
                x=_DIAGNOSIS_LEFT,
                y=top + offset,
                width=width,
                height=_THICK,
                colour=colour,
                radius=4.0,
            ).draw()
        )
        marks.append(
            draw.Label(
                x=_DIAGNOSIS_LEFT + width + 12.0,
                y=top + offset + 14.0,
                text=_percent(count, faults),
                colour=ink,
                size=12.0,
                bold=True,
            ).draw()
        )
    return tuple(marks)


def _diagnosis_axis(faults: int) -> tuple[str, ...]:
    """The line under the bars and what its two ends mean."""
    axis = _DIAGNOSIS_HEIGHT - 20.0
    return (
        draw.Stroke(
            points=((_DIAGNOSIS_LEFT, axis), (_DIAGNOSIS_RIGHT, axis)), colour="axis"
        ).draw(),
        draw.Label(x=_DIAGNOSIS_LEFT, y=axis + 16.0, text="0").draw(),
        draw.Label(
            x=_DIAGNOSIS_RIGHT,
            y=axis + 16.0,
            text=prose.DIAGNOSIS_AXIS.format(faults=faults),
            anchor="end",
        ).draw(),
    )


def _steps(value: float | None) -> str:
    """A count of stored steps, or a word where a model never ran."""
    return prose.TRUST_NONE if value is None else f"{value:.1f}"


def _axis(value: float) -> str:
    """A tick value, as short as it can be written."""
    return f"{value:g}"


def _threads(threads: int) -> str:
    """How many threads the timings were taken on, as a phrase."""
    one, many = prose.COST_THREADS
    return one if threads == 1 else many.format(threads=threads)


def _ms(seconds: float) -> str:
    """A time per step in milliseconds, to the two figures a benchmark supports."""
    return f"{seconds * _MS:.2g}"


def _slowdown(value: float) -> str:
    """A ratio against the solver, to the two figures the benchmark's own spread supports."""
    return f"{value:.2g}{prose.TIMES}"


def _share(count: int, faults: int) -> str:
    """A count out of the faults it was scored on."""
    return f"{count} of {faults}"


def _percent(count: int, faults: int) -> str:
    """A count as the percentage the chart labels its bars with."""
    return f"{round(count / faults * 100)}%"
