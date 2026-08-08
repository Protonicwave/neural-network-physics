from __future__ import annotations

import re

import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.reporting import prose
from nnphysics.reporting.charts import (
    cost_asides,
    cost_chart,
    diagnosis_chart,
    usable_steps_chart,
)
from nnphysics.reporting.page import (
    HELD_OUT_SPLIT,
    PERSISTENCE,
    TRAINED_ON_SPLIT,
    CostLadder,
    CostPoint,
    DiagnoserScore,
    Diagnosis,
    FaultRank,
    Headlines,
    Horizon,
    MatchedCost,
    PageModel,
    RunCard,
    RunVerdict,
    VerdictKind,
)

ROLLOUT_STEPS = 100
"""Chosen so that a value of ten is a tenth of the axis, which is the geometry the bar
chart is asserted against."""

TRACK = 500.0
"""The length of a bar chart track, from the module's own frame."""

BAR = re.compile(r'<rect x="150" y="([\d.]+)" width="([\d.]+)"')
DOT = re.compile(r'<circle cx="([\d.]+)" cy="([\d.]+)"')


def _horizon(
    predictor: str,
    split: str,
    steps: float | None,
    *,
    learned: bool = True,
) -> Horizon:
    """One predictor on one split, carrying only what a chart reads."""
    return Horizon(
        predictor=predictor,
        split=split,
        learned=learned,
        steps=steps,
        rollout_steps=ROLLOUT_STEPS,
        whole_rollout=False,
        rollouts=4,
        completed=4,
        diverged=False,
        failed=steps is None,
    )


def _card(  # noqa: PLR0913 - a run card has a field per column of the register
    *,
    run_id: str = "0123456789abcdef",
    system: str = "fluid",
    created: str = "2026-01-01T00:00:00+00:00",
    model: str | None = "convolution",
    horizons: tuple[Horizon, ...] | None = None,
    cost: CostLadder | None = None,
) -> RunCard:
    """One run, carrying only what a chart reads."""
    return RunCard(
        run_id=run_id,
        name="example",
        directory=f"{system}-{run_id}",
        system=system,
        created=created,
        commit="a" * 40,
        model=model,
        parameters=1234,
        epochs=7,
        fault=False,
        rollout_steps=ROLLOUT_STEPS,
        horizons=horizons
        if horizons is not None
        else (
            _horizon(PERSISTENCE, TRAINED_ON_SPLIT, 5.0, learned=False),
            _horizon(PERSISTENCE, HELD_OUT_SPLIT, 5.0, learned=False),
            _horizon("convolution", TRAINED_ON_SPLIT, 10.0),
            _horizon("convolution", HELD_OUT_SPLIT, 20.0),
        ),
        verdict=RunVerdict(VerdictKind.STABLE, "stable"),
        cost=cost,
    )


def _point(predictor: str, error: float, seconds: float, *, substeps: int = 0) -> CostPoint:
    """One timed point."""
    return CostPoint(
        label=predictor,
        predictor=predictor,
        solver=substeps > 0,
        substeps=substeps,
        error=error,
        seconds_per_step=seconds,
        relative_spread=0.1,
    )


def _matched(
    predictor: str = "convolution",
    *,
    speedup: float = 0.1,
    substeps: int = 2,
    bracketed: bool = True,
    break_even: float | None = None,
) -> MatchedCost:
    """One surrogate against the cheapest solver setting as accurate as it."""
    return MatchedCost(
        predictor=predictor,
        speedup=speedup,
        slowdown=1.0 / speedup if speedup < 1.0 else None,
        seconds_per_step=0.002,
        matched_seconds_per_step=0.002 * speedup,
        matched_substeps=substeps,
        bracketed=bracketed,
        break_even_rollouts=break_even,
    )


def _ladder(
    *,
    system: str = "fluid",
    solver: tuple[CostPoint, ...] = (),
    surrogates: tuple[CostPoint, ...] = (),
    matched: tuple[MatchedCost, ...] = (),
) -> CostLadder:
    """One benchmark."""
    return CostLadder(
        system=system,
        split=TRAINED_ON_SPLIT,
        threads=1,
        trials=15,
        solver=solver,
        surrogates=surrogates,
        matched=matched,
    )


def _benchmark() -> CostLadder:
    """A benchmark with a solver ladder, one surrogate and one comparison."""
    return _ladder(
        solver=(
            _point("reference", 1.0, 0.0001, substeps=1),
            _point("reference", 0.5, 0.0004, substeps=2),
            _point("reference", 0.0, 0.001, substeps=4),
        ),
        surrogates=(_point("convolution", 0.5, 0.002),),
        matched=(_matched(speedup=0.2, substeps=2),),
    )


def _headlines() -> Headlines:
    """The hero figures, which no chart reads but a model carries."""
    return Headlines(
        usable_steps=10.0,
        usable_of=ROLLOUT_STEPS,
        usable_predictor="convolution",
        usable_run_id="0123456789abcdef",
        held_out_completed=0,
        held_out_of=4,
        held_out_predictor="convolution",
        held_out_run_id="0123456789abcdef",
        slowdown=5.0,
        slowdown_predictor="convolution",
        slowdown_run_id="0123456789abcdef",
    )


def _model(*runs: RunCard) -> PageModel:
    """A page model carrying the runs a test needs."""
    return PageModel(
        runs=runs if runs else (_card(),), headlines=_headlines(), diagnosis=None, drift=None
    )


def _score(source: str, ranks: tuple[int | None, ...]) -> DiagnoserScore:
    """One diagnoser, scored on the ranks it returned."""
    entries = tuple(
        FaultRank(fault=f"fault_{index}", true_cause=f"cause_{index}", rank=rank)
        for index, rank in enumerate(ranks)
    )
    top1 = sum(1 for entry in entries if entry.rank == 1)
    top3 = sum(1 for entry in entries if entry.rank is not None and entry.rank <= 3)
    return DiagnoserScore(
        source=source,
        model=source,
        faults=len(entries),
        top1=top1 / len(entries),
        top3=top3 / len(entries),
        top1_count=top1,
        top3_count=top3,
        ranks=entries,
    )


def _diagnosis() -> Diagnosis:
    """Four faults, the agent first on two and the baseline first on one."""
    return Diagnosis(
        agent=_score("agent", (1, 1, 3, 2)),
        baseline=_score("rule_based", (1, None, None, 5)),
    )


def _bars(chart: str) -> list[tuple[float, float]]:
    """Every bar in a chart, as its top and its width."""
    return [(float(top), float(width)) for top, width in BAR.findall(chart)]


class TestUsableSteps:
    def test_a_value_a_tenth_of_the_rollout_is_a_tenth_of_the_track(self) -> None:
        chart = usable_steps_chart(_model())
        assert chart is not None
        widths = {width for _top, width in _bars(chart.svg)}

        assert TRACK / 10.0 in widths

    def test_each_predictor_draws_a_track_for_both_splits(self) -> None:
        chart = usable_steps_chart(_model())
        assert chart is not None
        tracks = [width for _top, width in _bars(chart.svg) if width == TRACK]

        # One model and the baseline, two splits each.
        assert len(tracks) == 4

    def test_a_model_that_could_not_run_unseen_draws_no_bar_and_says_none(self) -> None:
        card = _card(
            horizons=(
                _horizon(PERSISTENCE, TRAINED_ON_SPLIT, 5.0, learned=False),
                _horizon("convolution", TRAINED_ON_SPLIT, 10.0),
                _horizon("convolution", HELD_OUT_SPLIT, None),
            )
        )
        chart = usable_steps_chart(_model(card))
        assert chart is not None
        drawn = [width for _top, width in _bars(chart.svg) if width != TRACK]

        assert prose.TRUST_NONE in chart.svg
        # The model on the split it trained on, and the baseline on the same split. The
        # split it could not run on draws nothing at all.
        assert drawn == [TRACK / 10.0, TRACK / 20.0]

    def test_one_panel_per_system(self) -> None:
        model = _model(
            _card(system="fluid"),
            _card(run_id="f" * 16, system="nbody", model="graph", created="2026-01-02T00:00:00Z"),
        )
        chart = usable_steps_chart(model)
        assert chart is not None

        assert prose.system_label("fluid") in chart.svg
        assert prose.system_label("nbody") in chart.svg

    def test_the_longest_stretch_a_predictor_reached_is_the_one_drawn(self) -> None:
        """The same model is evaluated in more than one run. The page shows its best."""
        first = _card(horizons=(_horizon("convolution", TRAINED_ON_SPLIT, 10.0),))
        second = _card(
            run_id="f" * 16,
            created="2026-01-02T00:00:00+00:00",
            horizons=(_horizon("convolution", TRAINED_ON_SPLIT, 25.0),),
        )
        chart = usable_steps_chart(_model(first, second))
        assert chart is not None
        drawn = [width for _top, width in _bars(chart.svg) if width != TRACK]

        assert drawn == [TRACK / 4.0]

    def test_the_baseline_is_drawn_in_grey_in_every_panel(self) -> None:
        chart = usable_steps_chart(_model())
        assert chart is not None

        grey = re.findall(r'<rect [^>]*fill="var\(--muted\)"', chart.svg)

        assert len(grey) == 2

    def test_a_system_the_theme_has_no_colour_for_is_drawn_neutral(self) -> None:
        chart = usable_steps_chart(_model(_card(system="plasma", model="graph")))
        assert chart is not None

        assert "var(--nbody)" not in chart.svg
        assert "var(--ink-2)" in chart.svg

    def test_a_page_with_no_trained_model_has_no_chart(self) -> None:
        card = _card(
            model=None,
            horizons=(_horizon(PERSISTENCE, TRAINED_ON_SPLIT, 5.0, learned=False),),
        )

        assert usable_steps_chart(_model(card)) is None

    def test_the_text_alternative_states_how_far_the_best_model_got(self) -> None:
        chart = usable_steps_chart(_model())
        assert chart is not None

        assert f"10.0 of {ROLLOUT_STEPS} steps" in chart.svg

    def test_a_stretch_longer_than_the_rollout_raises(self) -> None:
        """Drawing it would put a bar outside the frame and read as a longer rollout."""
        card = _card(horizons=(_horizon("convolution", TRAINED_ON_SPLIT, ROLLOUT_STEPS + 1.0),))

        with pytest.raises(ValidationError, match="outside the axis"):
            usable_steps_chart(_model(card))


class TestCost:
    def test_the_reference_sits_on_the_error_axis(self) -> None:
        """Its error is zero, and zero is where the axis is."""
        chart = cost_chart(_model(_card(cost=_benchmark())))
        assert chart is not None
        heights = [float(y) for _x, y in DOT.findall(chart.svg)]

        assert max(heights) == 350.0

    def test_the_matched_pair_is_joined_and_carries_the_slowdown(self) -> None:
        chart = cost_chart(_model(_card(cost=_benchmark())))
        assert chart is not None

        assert "stroke-dasharray" in chart.svg
        assert f"5{prose.TIMES} the time" in chart.svg

    def test_the_benchmark_with_the_largest_slowdown_is_the_one_drawn(self) -> None:
        slower = _card(
            run_id="f" * 16,
            system="nbody",
            created="2026-01-02T00:00:00+00:00",
            model=None,
            cost=_ladder(
                system="nbody",
                solver=(_point("reference", 1.0, 0.0001, substeps=1),),
                surrogates=(_point("graph", 1.0, 0.01),),
                matched=(_matched("graph", speedup=0.01, substeps=1),),
            ),
        )
        chart = cost_chart(_model(_card(cost=_benchmark()), slower))
        assert chart is not None

        assert prose.system_label("nbody") in chart.title

    def test_a_benchmark_that_never_pays_back_says_so(self) -> None:
        chart = cost_chart(_model(_card(cost=_benchmark())))
        assert chart is not None

        assert chart.caption.endswith(prose.COST_NEVER_PAYS)

    def test_a_benchmark_that_pays_back_states_the_count(self) -> None:
        cost = _ladder(
            solver=(_point("reference", 1.0, 0.0001, substeps=2),),
            surrogates=(_point("convolution", 0.5, 0.002),),
            matched=(_matched(speedup=0.2, break_even=1400.4),),
        )
        chart = cost_chart(_model(_card(cost=cost)))
        assert chart is not None

        assert "1,400 rollouts" in chart.caption

    def test_a_page_with_no_benchmark_has_no_chart(self) -> None:
        assert cost_chart(_model()) is None

    def test_a_benchmark_the_chart_leaves_out_is_stated_in_a_sentence(self) -> None:
        undecided = _card(
            run_id="f" * 16,
            system="nbody",
            created="2026-01-02T00:00:00+00:00",
            model=None,
            cost=_ladder(
                system="nbody",
                solver=(_point("reference", 1.0, 0.0001, substeps=1),),
                surrogates=(_point("graph", 1.0, 0.01),),
                matched=(_matched("graph", speedup=1.3, bracketed=False, break_even=14079.0),),
            ),
        )
        asides = cost_asides(_model(_card(cost=_benchmark()), undecided))

        assert len(asides) == 1
        assert prose.COST_ASIDE_BOUND in asides[0]
        assert "14,079 rollouts" in asides[0]

    def test_the_benchmark_the_chart_draws_is_not_repeated_in_a_sentence(self) -> None:
        assert cost_asides(_model(_card(cost=_benchmark()))) == ()


class TestDiagnosis:
    def test_each_bar_is_the_share_of_the_faults_it_named(self) -> None:
        track = 645.0 - 220.0
        chart = diagnosis_chart(_diagnosis())

        # The agent named the true cause first in two of four, the baseline in one.
        assert f'width="{track / 2:g}"' in chart.svg
        assert f'width="{track / 4:g}"' in chart.svg

    def test_the_bars_are_labelled_as_percentages(self) -> None:
        chart = diagnosis_chart(_diagnosis())

        assert ">50%<" in chart.svg
        assert ">25%<" in chart.svg

    def test_the_text_alternative_states_both_measures(self) -> None:
        chart = diagnosis_chart(_diagnosis())

        for measure in prose.DIAGNOSIS_ROWS:
            assert measure in chart.svg

    def test_the_caption_counts_what_the_baseline_never_named(self) -> None:
        chart = diagnosis_chart(_diagnosis())

        assert "2 of 4" in chart.caption

    def test_the_axis_ends_at_every_fault(self) -> None:
        chart = diagnosis_chart(_diagnosis())

        assert "4 of 4 faults" in chart.svg


class TestPresentation:
    def test_no_chart_writes_a_colour(self) -> None:
        model = _model(_card(cost=_benchmark()))
        charts = [usable_steps_chart(model), cost_chart(model), diagnosis_chart(_diagnosis())]

        for chart in charts:
            assert chart is not None
            assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(", chart.svg)

    def test_every_chart_carries_a_text_alternative(self) -> None:
        model = _model(_card(cost=_benchmark()))
        charts = [usable_steps_chart(model), cost_chart(model), diagnosis_chart(_diagnosis())]

        for chart in charts:
            assert chart is not None
            assert 'aria-label="' in chart.svg
            assert chart.caption.strip()

    def test_every_series_is_labelled_as_well_as_coloured(self) -> None:
        """A reader who cannot see colour still has to be able to tell the series apart."""
        model = _model(_card(cost=_benchmark()))
        for chart in (usable_steps_chart(model), cost_chart(model)):
            assert chart is not None
            assert chart.legend
            assert all(entry.label.strip() for entry in chart.legend)

    def test_a_run_name_in_a_chart_is_escaped(self) -> None:
        card = _card(model="<script>", horizons=(_horizon("<script>", TRAINED_ON_SPLIT, 10.0),))
        chart = usable_steps_chart(_model(card))
        assert chart is not None

        assert "<script>" not in chart.svg
        assert "&lt;script&gt;" in chart.svg
