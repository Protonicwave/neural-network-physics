from __future__ import annotations

import html
import json
import re
from dataclasses import replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import pytest

from nnphysics.core.config import RunConfig
from nnphysics.core.errors import ConfigurationError
from nnphysics.evals.result import (
    MetricRecord,
    PredictorResult,
    RolloutRecord,
    SuiteResult,
    SuiteSettings,
)
from nnphysics.evals.speed import SpeedPoint, SpeedReport, matched_speedup
from nnphysics.reporting import prose
from nnphysics.reporting.environment import EnvironmentRecord
from nnphysics.reporting.landing import CONTENT_ID, LANDING_NAME, TIMES, render_landing
from nnphysics.reporting.layout import RECORD_NAME
from nnphysics.reporting.page import (
    HELD_OUT_SPLIT,
    PERSISTENCE,
    TRAINED_ON_SPLIT,
    USABLE_THRESHOLD,
    CostLadder,
    DiagnoserScore,
    Diagnosis,
    DriftFrame,
    DriftViewer,
    FaultRank,
    Headlines,
    Horizon,
    MatchedCost,
    PageModel,
    RunCard,
    RunVerdict,
    VerdictKind,
    build_page,
)
from nnphysics.reporting.record import RunRecord, write_record

if TYPE_CHECKING:
    from pathlib import Path

DT = 0.01
ROLLOUT_STEPS = 100

CONFIG: dict[str, Any] = {
    "name": "example",
    "seed": 5,
    "system": {"name": "fluid"},
    "data": {
        "n_trajectories": 8,
        "n_steps": 8,
        "dt": DT,
        "regimes": ["hot"],
        "held_out_regimes": ["cold"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
    },
    "model": {"name": "placeholder"},
    "evaluation": {"name": "standard", "metrics": ["rollout_error"], "rollout_steps": 4},
}

ENVIRONMENT = EnvironmentRecord(
    python="3.12.0",
    implementation="CPython",
    platform="Linux 6.1",
    machine="x86_64",
    cpu_count=8,
    packages={"numpy": "2.1.0"},
)
"""Fixed rather than read from the machine, so the same runs root gives the same page
everywhere."""


def _horizon(  # noqa: PLR0913 - a horizon has ten fields and a test chooses any of them
    predictor: str,
    split: str,
    *,
    learned: bool = True,
    steps: float | None = 25.0,
    completed: int = 4,
    diverged: bool = False,
    failed: bool = False,
) -> Horizon:
    """One predictor on one split, with everything the page reads chosen directly."""
    return Horizon(
        predictor=predictor,
        split=split,
        learned=learned,
        steps=steps,
        rollout_steps=ROLLOUT_STEPS,
        whole_rollout=False,
        rollouts=4,
        completed=completed,
        diverged=diverged,
        failed=failed,
    )


def _card(  # noqa: PLR0913 - a run card has a field per column of the register
    *,
    run_id: str = "0123456789abcdef",
    directory: str = "fluid-0123456789abcdef",
    system: str = "fluid",
    created: str = "2026-01-01T00:00:00+00:00",
    model: str | None = "convolution",
    parameters: int | None = 1234,
    epochs: int | None = 7,
    fault: bool = False,
    horizons: tuple[Horizon, ...] | None = None,
    verdict: RunVerdict | None = None,
    cost: CostLadder | None = None,
) -> RunCard:
    """One run, as the register shows it."""
    return RunCard(
        run_id=run_id,
        name="example",
        directory=directory,
        system=system,
        created=created,
        commit="a" * 40,
        model=model,
        parameters=parameters,
        epochs=epochs,
        fault=fault,
        rollout_steps=ROLLOUT_STEPS,
        horizons=horizons
        if horizons is not None
        else (
            _horizon(PERSISTENCE, TRAINED_ON_SPLIT, learned=False, steps=10.0),
            _horizon("convolution", TRAINED_ON_SPLIT),
        ),
        verdict=verdict if verdict is not None else RunVerdict(VerdictKind.STABLE, "stable"),
        cost=cost,
    )


def _matched(*, speedup: float = 0.5, break_even: float | None = None) -> MatchedCost:
    """One surrogate against the cheapest solver setting as accurate as it."""
    return MatchedCost(
        predictor="ensemble",
        speedup=speedup,
        slowdown=1.0 / speedup if speedup < 1.0 else None,
        seconds_per_step=0.002,
        matched_seconds_per_step=0.001,
        matched_substeps=1,
        bracketed=True,
        break_even_rollouts=break_even,
    )


def _ladder(*matched: MatchedCost) -> CostLadder:
    """A benchmark carrying nothing but the comparisons a test needs."""
    return CostLadder(
        system="fluid",
        split=TRAINED_ON_SPLIT,
        threads=1,
        trials=5,
        solver=(),
        surrogates=(),
        matched=matched,
    )


def _headlines() -> Headlines:
    """The three hero figures, chosen directly rather than selected from runs."""
    return Headlines(
        usable_steps=11.5,
        usable_of=63,
        usable_predictor="convolution",
        usable_run_id="0123456789abcdef",
        held_out_completed=0,
        held_out_of=4,
        held_out_predictor="operator",
        held_out_run_id="0123456789abcdef",
        slowdown=17.25,
        slowdown_predictor="graph",
        slowdown_run_id="0123456789abcdef",
    )


def _score(source: str, ranks: tuple[FaultRank, ...]) -> DiagnoserScore:
    """One diagnoser over the faults handed to it."""
    top1 = sum(1 for entry in ranks if entry.rank == 1)
    top3 = sum(1 for entry in ranks if entry.rank is not None and entry.rank <= 3)
    return DiagnoserScore(
        source=source,
        model=source,
        faults=len(ranks),
        top1=top1 / len(ranks),
        top3=top3 / len(ranks),
        top1_count=top1,
        top3_count=top3,
        ranks=ranks,
    )


def _diagnosis() -> Diagnosis:
    """Both diagnosers, one fault each way."""
    ranks = (
        FaultRank(fault="wrong_regime", true_cause="training_regime", rank=1),
        FaultRank(fault="no_optimiser_state", true_cause="optimiser_state", rank=3),
        FaultRank(fault="invented_fault", true_cause="something_new", rank=None),
    )
    return Diagnosis(
        agent=_score("agent", ranks),
        baseline=_score("rule_based", ranks[:1]),
    )


def _frame(
    system: str = "fluid",
    predictor: str = "convolution",
    split: str = TRAINED_ON_SPLIT,
) -> DriftFrame:
    """One combination the drift viewer offers."""
    return DriftFrame(
        system=system,
        predictor=predictor,
        split=split,
        image=f"{system}-0123456789abcdef/plots/{split}-state-{predictor}.png",
        run_id="0123456789abcdef",
    )


def _viewer(*combinations: tuple[str, str]) -> DriftViewer:
    """A viewer offering the predictors handed to it, on both splits."""
    chosen = combinations if combinations else (("fluid", "convolution"), ("nbody", "graph"))
    return DriftViewer(
        frames=tuple(
            _frame(system, predictor, split)
            for system, predictor in chosen
            for split in (TRAINED_ON_SPLIT, HELD_OUT_SPLIT)
        )
    )


def _model(
    *runs: RunCard,
    headlines: Headlines | None = None,
    diagnosis: Diagnosis | None = None,
    drift: DriftViewer | None = None,
) -> PageModel:
    """A page model carrying the runs a test needs and nothing else."""
    return PageModel(
        runs=runs if runs else (_card(),),
        headlines=headlines if headlines is not None else _headlines(),
        diagnosis=diagnosis,
        drift=drift,
    )


def _body(page: str) -> str:
    """The page without its stylesheet or its script, which are not prose."""
    return page[page.index("</style>") : page.index("<script>")]


class TestFrame:
    def test_the_page_is_one_html_document(self) -> None:
        page = render_landing(_model())

        assert page.startswith("<!DOCTYPE html>")
        assert page.endswith("</html>\n")

    def test_the_same_model_gives_the_same_page(self) -> None:
        model = _model()

        assert render_landing(model) == render_landing(model)

    def test_the_page_references_no_host(self) -> None:
        page = render_landing(_model(diagnosis=_diagnosis()))

        assert "http://" not in page
        assert "https://" not in page
        assert "//" not in _body(page)

    def test_every_navigation_link_reaches_a_section(self) -> None:
        page = render_landing(_model(diagnosis=_diagnosis()))
        targets = set(re.findall(r'<section id="([^"]+)"', page))

        assert set(re.findall(r'<a href="#([^"]+)"', page)) <= targets

    def test_the_theme_button_starts_on_the_system_preference(self) -> None:
        page = render_landing(_model())

        assert 'matchMedia("(prefers-color-scheme: dark)")' in page
        assert 'id="theme"' in page

    def test_no_field_is_left_unfilled(self) -> None:
        # A sentence carrying a named field the builder never fills would otherwise reach
        # the reader with a brace in it.
        assert not re.search(r"\{[a-z_]+\}", _body(render_landing(_model(diagnosis=_diagnosis()))))


class TestHero:
    def test_the_three_figures_come_from_the_model(self) -> None:
        page = render_landing(_model())

        assert "11.5 of 63" in page
        assert "0 of 4" in page
        assert f"17{TIMES} slower" in page

    def test_a_figure_is_coloured_by_the_system_it_came_from(self) -> None:
        page = render_landing(_model(_card(system="nbody")))

        assert 'class="fig-big f-nbody"' in page

    def test_a_system_with_no_colour_of_its_own_is_drawn_neutral(self) -> None:
        page = render_landing(_model(_card(system="plasma")))

        assert 'class="fig-big f-muted"' in page

    def test_an_unseen_figure_of_nothing_surviving_is_drawn_as_a_failure(self) -> None:
        page = render_landing(_model())

        assert 'class="fig-big f-fail"' in page

    def test_a_surviving_unseen_figure_is_not_drawn_as_a_failure(self) -> None:
        headlines = Headlines(
            usable_steps=11.5,
            usable_of=63,
            usable_predictor="convolution",
            usable_run_id="0123456789abcdef",
            held_out_completed=3,
            held_out_of=4,
            held_out_predictor="convolution",
            held_out_run_id="0123456789abcdef",
            slowdown=2.0,
            slowdown_predictor="convolution",
            slowdown_run_id="0123456789abcdef",
        )
        page = render_landing(_model(headlines=headlines))

        assert "3 of 4" in page
        assert page.count('class="fig-big f-fail"') == 0

    def test_the_note_about_the_agent_appears_only_with_the_scores(self) -> None:
        with_scores = render_landing(_model(diagnosis=_diagnosis()))
        without = render_landing(_model())

        assert 'class="aside-link"' in with_scores
        assert 'class="aside-link"' not in without


class TestSections:
    def test_the_premise_carries_its_schematic(self) -> None:
        page = render_landing(_model())

        assert prose.SCHEMATIC.title in page
        assert html.escape(prose.SCHEMATIC.alt, quote=True) in page
        assert "<svg" in page

    def test_the_charts_are_placed_in_their_sections(self) -> None:
        page = render_landing(_model(diagnosis=_diagnosis()))

        assert prose.TRUST_CHART.title in page
        assert prose.DIAGNOSIS_CHART.title.format(faults=3) in page

    def test_a_benchmark_the_cost_chart_cannot_draw_is_still_stated(self) -> None:
        page = render_landing(_model(_card(cost=_ladder(_matched()))))

        assert prose.COST_ASIDE_TITLE in page
        assert prose.model_label("ensemble") in page

    def test_a_chart_carries_its_legend(self) -> None:
        page = render_landing(_model(diagnosis=_diagnosis()))

        assert 'class="legend"' in page
        assert 'class="swatch" style="background:var(--fluid)"' in page

    def test_a_section_whose_chart_cannot_be_drawn_keeps_its_prose(self) -> None:
        page = render_landing(_model())

        assert prose.COST.heading in page
        assert prose.COST_CHART.title.format(system="") not in page

    def test_the_findings_are_all_shown(self) -> None:
        page = render_landing(_model())

        for finding in prose.FINDING_LIST:
            assert finding.what in page

    def test_a_page_without_scores_has_no_diagnosis_section(self) -> None:
        page = render_landing(_model())

        assert 'id="diagnosis"' not in page
        assert 'href="#diagnosis"' not in page


class TestDiagnosisTable:
    def test_one_row_per_fault(self) -> None:
        diagnosis = _diagnosis()
        page = render_landing(_model(diagnosis=diagnosis))
        rows = page[page.index("<tbody>") : page.index("</tbody>")]

        assert rows.count("<tr>") == len(diagnosis.agent.ranks)

    def test_a_rank_is_written_as_a_position(self) -> None:
        page = render_landing(_model(diagnosis=_diagnosis()))

        assert ">1st<" in page
        assert ">3rd<" in page

    def test_a_cause_the_diagnoser_never_named_says_so(self) -> None:
        page = render_landing(_model(diagnosis=_diagnosis()))

        assert prose.UNKNOWN_RANK in page

    def test_a_fault_with_no_curated_description_falls_back_to_its_identifiers(self) -> None:
        page = render_landing(_model(diagnosis=_diagnosis()))

        assert "Invented fault" in page
        assert "Something new" in page


class TestRegister:
    def test_one_row_per_reported_run(self) -> None:
        model = _model(_card(), _card(run_id="f" * 16, directory="fluid-ffff"))
        page = render_landing(model)

        assert page.count('class="run"') == 2

    def test_a_fault_run_is_not_reported(self) -> None:
        model = _model(_card(), _card(run_id="f" * 16, directory="fluid-fault-x-ffff", fault=True))
        page = render_landing(model)

        assert page.count('class="run"') == 1

    def test_a_row_links_to_the_report_beside_it(self) -> None:
        page = render_landing(_model())

        assert 'href="fluid-0123456789abcdef/report.html"' in page

    def test_rows_are_grouped_by_system(self) -> None:
        model = _model(
            _card(run_id="a" * 16, directory="nbody-a", system="nbody"),
            _card(run_id="b" * 16, directory="fluid-b", system="fluid"),
        )
        page = render_landing(model)

        assert page.index('href="fluid-b') < page.index('href="nbody-a')

    def test_a_trained_run_shows_what_it_cost_to_train(self) -> None:
        page = render_landing(_model(_card(parameters=148889, epochs=40)))

        assert "148,889 parameters, 40 epochs" in page

    def test_a_benchmark_run_shows_a_speed_instead_of_a_stretch(self) -> None:
        card = _card(
            model=None,
            parameters=None,
            epochs=None,
            verdict=RunVerdict(VerdictKind.COST_BENCHMARK, "never pays back"),
            cost=_ladder(_matched(speedup=0.06), _matched(speedup=1.3)),
        )
        page = render_landing(_model(card))

        assert f"0.06 to 1.3{TIMES}" in page
        assert "speed benchmark, 1 thread, 5 timed trials" in page

    def test_a_run_that_trained_nothing_says_so(self) -> None:
        card = _card(
            model=None,
            parameters=None,
            epochs=None,
            horizons=(_horizon(PERSISTENCE, TRAINED_ON_SPLIT, learned=False),),
            verdict=RunVerdict(VerdictKind.HARNESS_CHECK, "harness check"),
        )
        page = render_landing(_model(card))

        assert prose.NO_MODEL in page
        assert prose.HARNESS_ROLE in page


class TestVerdictColour:
    def test_a_harness_check_is_stated_neutrally(self) -> None:
        card = _card(verdict=RunVerdict(VerdictKind.HARNESS_CHECK, "harness check"))

        assert 'class="status s-none"' in render_landing(_model(card))

    @pytest.mark.parametrize("kind", [VerdictKind.DIVERGES, VerdictKind.CANNOT_RUN_UNSEEN])
    def test_a_failure_is_stated_as_one(self, kind: VerdictKind) -> None:
        card = _card(verdict=RunVerdict(kind, "diverges"))

        assert 'class="status s-fail"' in render_landing(_model(card))

    def test_a_benchmark_that_never_pays_back_is_a_failure(self) -> None:
        card = _card(
            verdict=RunVerdict(VerdictKind.COST_BENCHMARK, "never pays back"),
            cost=_ladder(_matched(break_even=None)),
        )

        assert 'class="status s-fail"' in render_landing(_model(card))

    def test_a_benchmark_that_pays_back_is_qualified_rather_than_failed(self) -> None:
        card = _card(
            verdict=RunVerdict(VerdictKind.COST_BENCHMARK, "pays back eventually"),
            cost=_ladder(_matched(break_even=1000.0)),
        )

        assert 'class="status s-warn"' in render_landing(_model(card))

    def test_a_model_that_clears_the_free_baseline_is_stated_as_a_result(self) -> None:
        card = _card(
            horizons=(
                _horizon(PERSISTENCE, TRAINED_ON_SPLIT, learned=False, steps=10.0),
                _horizon("convolution", TRAINED_ON_SPLIT, steps=25.0),
            )
        )

        assert 'class="status s-good"' in render_landing(_model(card))

    def test_a_model_that_does_not_clear_it_is_qualified(self) -> None:
        card = _card(
            horizons=(
                _horizon(PERSISTENCE, TRAINED_ON_SPLIT, learned=False, steps=25.0),
                _horizon("convolution", TRAINED_ON_SPLIT, steps=10.0),
            )
        )

        assert 'class="status s-warn"' in render_landing(_model(card))


class TestDriftViewer:
    def test_the_first_combination_is_rendered_without_the_script(self) -> None:
        # A reader whose browser ran none of the script still gets a figure and its notes.
        page = _body(render_landing(_model(drift=_viewer())))

        assert 'src="fluid-0123456789abcdef/plots/test-state-convolution.png"' in page
        assert str(prose.drift_looking("fluid")) in html.unescape(page)
        assert str(prose.drift_meaning("fluid", "convolution", "test")) in html.unescape(page)

    def test_the_image_says_what_it_shows(self) -> None:
        page = render_landing(_model(drift=_viewer()))

        assert 'alt="The convolutional network prediction against the true Fluid' in page

    def test_every_control_group_reports_one_pressed_button(self) -> None:
        page = render_landing(_model(drift=_viewer()))

        for group in ("system", "predictor", "split"):
            pressed = re.findall(rf'data-{group}="([^"]+)" aria-pressed="true"', page)
            assert pressed == [
                {"system": "fluid", "predictor": "convolution", "split": "test"}[group]
            ]

    def test_the_predictors_of_the_other_system_are_written_out_and_hidden(self) -> None:
        page = render_landing(_model(drift=_viewer()))

        assert 'data-predictors="fluid">' in page
        assert 'data-predictors="nbody" hidden>' in page
        assert page.count('aria-labelledby="ctrl-predictor"') == 2
        assert 'data-predictor="graph"' in page

    def test_every_control_is_a_button_the_keyboard_reaches(self) -> None:
        page = render_landing(_model(drift=_viewer()))
        section = page[page.index('id="drift-viewer"') : page.index("</figcaption>")]

        assert section.count("<button") == section.count('type="button"')
        assert section.count("<button") == section.count("aria-pressed=")

    def test_the_table_carries_every_combination_the_controls_offer(self) -> None:
        viewer = _viewer()
        page = render_landing(_model(drift=viewer))
        table = json.loads(
            re.search(r'id="drift-data">(.*?)</script>', page, re.DOTALL).group(1)  # type: ignore[union-attr]
        )

        assert set(table["frames"]) == {
            f"{frame.system}|{frame.predictor}|{frame.split}" for frame in viewer.frames
        }
        assert table["first"] == ["fluid", "convolution", "test"]
        assert table["predictors"] == {"fluid": ["convolution"], "nbody": ["graph"]}

    def test_a_combination_nobody_has_written_about_refuses_to_render(self) -> None:
        # An empty box under the image would read as a combination with nothing to say.
        viewer = DriftViewer(frames=(_frame(system="plasma", predictor="graph"),))

        with pytest.raises(ConfigurationError, match="plasma"):
            render_landing(_model(drift=viewer))

    def test_markup_in_a_path_cannot_end_the_table_early(self) -> None:
        viewer = DriftViewer(frames=(replace(_frame(), image="</script><b>"),))
        page = render_landing(_model(drift=viewer))

        assert "</script><b>" not in _body(page)
        assert "\\u003c/script>" in page

    def test_a_page_without_a_viewer_drops_the_section_and_its_link(self) -> None:
        page = render_landing(_model())

        assert 'id="drift"' not in page
        assert 'href="#drift"' not in page
        assert "drift-viewer" not in page


class TestEscaping:
    def test_a_run_name_containing_markup_reaches_the_page_as_text(self) -> None:
        card = _card(directory="<script>alert(1)</script>", system='fluid" onclick="x')
        page = render_landing(_model(card))

        assert "<script>alert(1)</script>" not in _body(page)
        assert "&lt;script&gt;" in page
        assert 'onclick="x' not in page

    def test_a_predictor_name_containing_markup_is_escaped(self) -> None:
        card = _card(
            model="<b>graph</b>",
            horizons=(_horizon("<b>graph</b>", TRAINED_ON_SPLIT),),
        )
        page = render_landing(_model(card))

        assert "<b>graph</b>" not in page
        assert "&lt;b&gt;graph&lt;/b&gt;" in page

    def test_curated_emphasis_still_becomes_markup(self) -> None:
        page = render_landing(_model())

        assert "<strong>not yet</strong>" in page


def _predictor(predictor: str, split: str, *, horizon: float) -> PredictorResult:
    """One predictor on one split of a fixture run."""
    return PredictorResult(
        predictor=predictor,
        spec=predictor,
        split=split,
        regimes=("hot",),
        rollouts=(
            RolloutRecord(
                trajectory="t0",
                regime="hot",
                split=split,
                steps_requested=ROLLOUT_STEPS,
                steps_completed=ROLLOUT_STEPS,
                stop_reason="completed",
                seconds=0.01,
            ),
        ),
        metrics=(
            MetricRecord(
                name="rollout_error",
                scalars={f"horizon.{USABLE_THRESHOLD:g}": horizon},
            ),
        ),
        seconds_per_step=0.002,
        completed=True,
    )


def _benchmark() -> SpeedReport:
    """A benchmark whose surrogate loses to the solver.

    Every page needs one, because the third hero figure is a slowdown and a model with no
    slowdown to report refuses to build.
    """
    solver = SpeedPoint(
        label="reference:substeps=1",
        predictor="reference",
        kind="solver",
        substeps=1,
        error=0.5,
        seconds_per_step=0.001,
        iqr=0.0001,
        relative_spread=0.1,
        stable=True,
        completed=True,
    )
    surrogate = SpeedPoint(
        label="convolution",
        predictor="convolution",
        kind="surrogate",
        substeps=0,
        error=0.5,
        seconds_per_step=0.002,
        iqr=0.0002,
        relative_spread=0.1,
        stable=True,
        completed=True,
    )
    return SpeedReport(
        system="fluid",
        split=TRAINED_ON_SPLIT,
        steps=ROLLOUT_STEPS,
        n_initial_conditions=1,
        threads=1,
        trials=5,
        warmup=1,
        steps_per_trial=4,
        dataset_substeps=1,
        ladder=(solver,),
        surrogates=(surrogate,),
        matched=(matched_speedup(surrogate, (solver,)),),
        costs=(),
    )


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    """A runs root holding one readable run, written the way a real run writes itself."""
    record = RunRecord(
        run_id="0123456789abcdef",
        name="example",
        created="2026-01-01T00:00:00+00:00",
        code_version="0.1.0",
        commit="a" * 40,
        config=RunConfig.model_validate(CONFIG),
        environment=ENVIRONMENT,
        timings={"evaluation": 1.5},
        evaluation=SuiteResult(
            code_version="0.1.0",
            run_id="0123456789abcdef",
            dataset_id="fedcba9876543210",
            system="fluid",
            seed=5,
            settings=SuiteSettings(
                name="standard",
                metrics=("rollout_error",),
                rollout_steps=ROLLOUT_STEPS,
                n_initial_conditions=1,
                error_thresholds=(USABLE_THRESHOLD,),
                symmetry_steps=4,
                distribution_window=0.25,
                divergence_factor=1000.0,
            ),
            invariants={},
            results=(
                _predictor(PERSISTENCE, TRAINED_ON_SPLIT, horizon=0.1),
                _predictor("convolution", TRAINED_ON_SPLIT, horizon=0.25),
                _predictor("convolution", HELD_OUT_SPLIT, horizon=0.05),
            ),
        ),
        benchmark=_benchmark(),
    )
    directory = tmp_path / "fluid-0123456789abcdef"
    directory.mkdir()
    write_record(directory / RECORD_NAME, record)
    return tmp_path


class TestFromARunsRoot:
    def test_the_page_renders_from_a_runs_root(self, runs_root: Path) -> None:
        page = render_landing(build_page(runs_root))

        assert prose.TITLE in page
        assert 'href="fluid-0123456789abcdef/report.html"' in page

    def test_the_page_is_written_beside_the_runs_it_links_to(self, runs_root: Path) -> None:
        # The links are relative to the runs root, so the file belongs in it.
        target = runs_root / LANDING_NAME
        target.write_text(render_landing(build_page(runs_root)), encoding="utf-8")

        assert (runs_root / "fluid-0123456789abcdef").is_dir()
        assert target.is_file()


class TestAccessibility:
    """The page is navigated by structure and driven from a keyboard as well as read.

    These assert on the markup because that is where the property lives. Contrast is
    checked in the theme tests, against the tokens rather than against the page.
    """

    def test_the_first_thing_in_the_tab_order_skips_the_navigation(self) -> None:
        page = render_landing(_model())
        body = page[page.index("<body>") :]

        assert body.index('class="skip"') < body.index("<nav")
        assert f'href="#{CONTENT_ID}"' in page
        assert f'<main id="{CONTENT_ID}">' in page

    def test_every_section_is_named_by_its_own_heading(self) -> None:
        page = render_landing(_model(diagnosis=_diagnosis()))
        named = re.findall(r'<section id="([^"]+)" aria-labelledby="([^"]+)"', page)

        assert named
        for anchor, heading in named:
            assert heading == f"{anchor}-heading"
            assert f'<h2 id="{heading}">' in page

    def test_the_headings_descend_without_skipping_a_level(self) -> None:
        page = render_landing(_model(diagnosis=_diagnosis()))
        levels = [int(level) for level in re.findall(r"<h([1-6])", page)]

        assert levels[0] == 1
        assert levels.count(1) == 1
        for previous, level in pairwise(levels):
            assert level <= previous + 1

    def test_each_negative_result_is_a_heading(self) -> None:
        """Six results in a list of spans cannot be reached by structure."""
        page = render_landing(_model())

        assert page.count('<h3 class="what">') == len(prose.FINDING_LIST)

    def test_the_theme_button_says_what_it_does(self) -> None:
        page = render_landing(_model())
        light, dark = prose.THEME_BUTTON

        assert f'aria-label="{prose.THEME_LABEL.format(theme=light)}"' in page
        # The script has to correct both words when it follows the reader's system.
        assert json.dumps(prose.THEME_LABEL.format(theme=dark)) in page

    def test_a_box_that_scrolls_can_be_reached_from_the_keyboard(self) -> None:
        page = render_landing(_model(diagnosis=_diagnosis()))
        boxes = re.findall(r'<div class="scroller"([^>]*)>', page)

        assert boxes
        for box in boxes:
            assert 'role="region"' in box
            assert 'tabindex="0"' in box
            assert "aria-label" in box

    def test_every_chart_drawing_sits_in_a_named_box(self) -> None:
        page = render_landing(_model(diagnosis=_diagnosis()))

        # Every drawing opens immediately inside a box, so a chart cannot be added to a
        # section without the box that lets a narrow screen scroll it.
        assert page.count("<svg") == page.count('"><svg')
        assert f'aria-label="{prose.TRUST_CHART.title}"' in page

    def test_the_notes_under_the_drift_image_announce_themselves(self) -> None:
        """A picture that swaps says nothing to a reader who cannot see it.

        The words beside it have to announce that they changed.
        """
        page = render_landing(_model(drift=_viewer()))

        assert 'class="strip-note" aria-live="polite"' in page

    def test_the_navigation_is_named(self) -> None:
        page = render_landing(_model())

        assert f'<nav aria-label="{prose.NAV_LABEL}"' in page

    def test_a_control_carries_a_visible_focus_ring(self) -> None:
        page = render_landing(_model(drift=_viewer()))

        assert "button:focus-visible" in page
        assert "a:focus-visible" in page

    def test_motion_is_asked_for_rather_than_assumed(self) -> None:
        page = render_landing(_model())

        assert "@media (prefers-reduced-motion: no-preference)" in page
        assert page.count("scroll-behavior") == 1
