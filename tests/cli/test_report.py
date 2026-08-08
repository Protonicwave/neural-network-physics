from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from nnphysics.cli.app import app
from nnphysics.core.config import RunConfig
from nnphysics.data.build import build_dataset
from nnphysics.evals.result import (
    MetricRecord,
    PredictorResult,
    RolloutRecord,
    SuiteResult,
    SuiteSettings,
)
from nnphysics.evals.snapshots import read_snapshots
from nnphysics.evals.speed import CostAccounting, MatchedSpeedup, SpeedPoint, SpeedReport
from nnphysics.reporting.environment import EnvironmentRecord
from nnphysics.reporting.landing import LANDING_NAME
from nnphysics.reporting.layout import (
    HTML_NAME,
    MARKDOWN_NAME,
    PLOTS_DIR,
    RECORD_NAME,
    run_paths,
    state_plot_name,
)
from nnphysics.reporting.page import HELD_OUT_SPLIT, PERSISTENCE, TRAINED_ON_SPLIT, USABLE_THRESHOLD
from nnphysics.reporting.record import RunRecord, read_record, write_record

runner = CliRunner()

EXTERNAL = re.compile(r"""(?:src|href)\s*=\s*["'](?!data:)""", re.IGNORECASE)

REFERENCED = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
"""Whatever the landing page points at. The page references its plots and reports rather
than embedding them, so what matters is that each path is there beside it."""

HOSTED = re.compile(r"""(?:src|href)\s*=\s*["'](?:[a-z]+:)?//""", re.IGNORECASE)
"""A reference to somewhere else. The page is served from a directory, not from a site."""

LEARNED = "graph"
"""A predictor name the harness does not supply, which is what makes it a surrogate."""

CONFIG: dict[str, Any] = {
    "name": "cli-report",
    "seed": 2,
    "system": {"name": "nbody", "parameters": {"softening": 0.05}},
    "data": {
        "n_trajectories": 4,
        "n_steps": 6,
        "dt": 0.02,
        "substeps": 2,
        "regimes": ["cold_collapse", "virialised_cluster"],
        "held_out_regimes": ["hierarchical_pair"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "workers": 1,
        "shard_trajectories": 2,
    },
    "model": {"name": "placeholder"},
    "evaluation": {
        "name": "cli",
        "metrics": ["one_step_error", "rollout_error", "invariant_drift"],
        "predictors": ["reference", "persistence"],
        "rollout_steps": 4,
        "n_initial_conditions": 2,
        "symmetry_steps": 2,
    },
}


def write_config(tmp_path: Path, **overrides: Any) -> tuple[Path, RunConfig]:
    """A configuration file, its dataset already generated."""
    raw = {**CONFIG, **overrides}
    raw["data"] = {**CONFIG["data"], "root": str(tmp_path / "data")}
    raw["output_dir"] = str(tmp_path / "runs")
    path = tmp_path / f"{raw['name']}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = RunConfig.model_validate(raw)
    build_dataset(config)
    return path, config


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, RunConfig, Path]:
    """One evaluated run: the configuration, the resolved config and the runs root."""
    tmp_path = tmp_path_factory.mktemp("cli-report")
    path, config = write_config(tmp_path)
    result = runner.invoke(app, ["eval", "run", "-c", str(path)])
    assert result.exit_code == 0, result.output
    return path, config, tmp_path / "runs"


@pytest.fixture(scope="module")
def rendered(evaluated: tuple[Path, RunConfig, Path]) -> tuple[RunConfig, Path]:
    """The same run, with its reports rendered."""
    path, config, root = evaluated
    result = runner.invoke(app, ["report", "render", "-c", str(path)])
    assert result.exit_code == 0, result.output
    return config, root


class TestEvalWritesARecord:
    def test_a_record_is_written_beside_the_result(
        self, evaluated: tuple[Path, RunConfig, Path]
    ) -> None:
        _, config, _ = evaluated

        assert run_paths(config).record.is_file()

    def test_the_record_names_the_artefacts_it_refers_to(
        self, evaluated: tuple[Path, RunConfig, Path]
    ) -> None:
        _, config, _ = evaluated

        record = read_record(run_paths(config).record)

        assert record.artefacts["snapshots"] == "snapshots.npz"
        assert (run_paths(config).root / record.artefacts["result"]).is_file()

    def test_the_record_carries_the_configuration_and_the_environment(
        self, evaluated: tuple[Path, RunConfig, Path]
    ) -> None:
        _, config, _ = evaluated

        record = read_record(run_paths(config).record)

        assert record.config == config
        assert record.environment.cpu_count >= 1
        assert record.timings["evaluation"] > 0.0

    def test_the_states_kept_for_the_plot_are_written(
        self, evaluated: tuple[Path, RunConfig, Path]
    ) -> None:
        _, config, _ = evaluated

        assert run_paths(config).snapshots.is_file()

    def test_snapshots_can_be_turned_off(self, tmp_path: Path) -> None:
        path, config = write_config(tmp_path, name="no-snapshots")

        result = runner.invoke(app, ["eval", "run", "-c", str(path), "--no-snapshots"])

        assert result.exit_code == 0, result.output
        # The file is still written, so that a reader never has to tell absent from empty.
        assert len(read_snapshots(run_paths(config).snapshots)) == 0


class TestRender:
    def test_it_writes_both_reports(self, rendered: tuple[RunConfig, Path]) -> None:
        config, _ = rendered

        assert run_paths(config).markdown.is_file()
        assert run_paths(config).html.is_file()

    def test_it_writes_the_plots_the_reports_refer_to(
        self, rendered: tuple[RunConfig, Path]
    ) -> None:
        config, _ = rendered
        paths = run_paths(config)

        drawn = sorted(path.name for path in paths.plots.glob("*.png"))

        assert drawn
        for name in drawn:
            assert name in paths.markdown.read_text(encoding="utf-8")

    def test_the_html_references_nothing_outside_itself(
        self, rendered: tuple[RunConfig, Path]
    ) -> None:
        text = run_paths(rendered[0]).html.read_text(encoding="utf-8")

        assert not EXTERNAL.search(text)
        assert "http" not in text.replace("http-equiv", "")

    def test_rendering_twice_gives_byte_identical_markdown(
        self, rendered: tuple[RunConfig, Path], evaluated: tuple[Path, RunConfig, Path]
    ) -> None:
        config, _ = rendered
        first = run_paths(config).markdown.read_bytes()

        result = runner.invoke(app, ["report", "render", "-c", str(evaluated[0])])

        assert result.exit_code == 0, result.output
        assert run_paths(config).markdown.read_bytes() == first

    def test_a_run_can_be_rendered_by_its_identifier(
        self, rendered: tuple[RunConfig, Path]
    ) -> None:
        config, root = rendered

        result = runner.invoke(
            app, ["report", "render", "--run", config.run_id, "--root", str(root)]
        )

        assert result.exit_code == 0, result.output

    def test_it_says_which_run_it_rendered(self, rendered: tuple[RunConfig, Path]) -> None:
        config, root = rendered

        result = runner.invoke(
            app, ["report", "render", "--run", config.run_id, "--root", str(root)]
        )

        assert config.run_id in result.output

    def test_saying_nothing_about_which_run_is_a_usage_error(self) -> None:
        result = runner.invoke(app, ["report", "render"])

        assert result.exit_code == 2

    def test_an_unknown_run_is_a_failure_not_a_traceback(
        self, rendered: tuple[RunConfig, Path]
    ) -> None:
        result = runner.invoke(
            app, ["report", "render", "--run", "nope", "--root", str(rendered[1])]
        )

        assert result.exit_code == 1
        assert "no run under" in result.output


class TestList:
    def test_it_lists_the_run(self, rendered: tuple[RunConfig, Path]) -> None:
        config, root = rendered

        result = runner.invoke(app, ["report", "list", "--root", str(root)])

        assert result.exit_code == 0, result.output
        assert config.run_id in result.output
        assert "rollout_error.error.final" in result.output

    def test_it_can_be_narrowed_to_one_predictor(self, rendered: tuple[RunConfig, Path]) -> None:
        _, root = rendered

        result = runner.invoke(app, ["report", "list", "--root", str(root), "-p", "reference"])

        assert "persistence" not in result.output

    def test_an_empty_root_says_so(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["report", "list", "--root", str(tmp_path / "none")])

        assert result.exit_code == 0
        assert "No runs" in result.output


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[RunConfig]]:
    """Two evaluated runs of the same system, differing only in their seed."""
    tmp_path = tmp_path_factory.mktemp("cli-compare")
    configs = []
    for seed in (2, 3):
        path, config = write_config(tmp_path, name=f"run-{seed}", seed=seed)
        result = runner.invoke(app, ["eval", "run", "-c", str(path), "--no-snapshots"])
        assert result.exit_code == 0, result.output
        configs.append(config)
    return tmp_path / "runs", configs


class TestCompare:
    def test_it_compares_two_runs_and_writes_a_report(
        self, two_runs: tuple[Path, list[RunConfig]], tmp_path: Path
    ) -> None:
        root, configs = two_runs

        result = runner.invoke(
            app,
            [
                "report",
                "compare",
                configs[0].run_id,
                configs[1].run_id,
                "--root",
                str(root),
                "-o",
                str(tmp_path / "comparison"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "comparison" / MARKDOWN_NAME).is_file()
        assert (tmp_path / "comparison" / HTML_NAME).is_file()

    def test_the_comparison_states_a_verdict_for_every_run(
        self, two_runs: tuple[Path, list[RunConfig]], tmp_path: Path
    ) -> None:
        root, configs = two_runs

        result = runner.invoke(
            app,
            [
                "report",
                "compare",
                configs[0].run_id,
                configs[1].run_id,
                "--root",
                str(root),
                "-o",
                str(tmp_path / "c"),
            ],
        )

        assert "improved" in result.output
        assert "regressed" in result.output

    def test_comparing_a_run_with_itself_finds_no_change(
        self, two_runs: tuple[Path, list[RunConfig]], tmp_path: Path
    ) -> None:
        root, configs = two_runs

        result = runner.invoke(
            app,
            [
                "report",
                "compare",
                configs[0].run_id,
                configs[0].run_id,
                "--root",
                str(root),
                "-o",
                str(tmp_path / "c"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "0 improved, 0 regressed" in result.output

    def test_one_run_is_a_usage_error(self, two_runs: tuple[Path, list[RunConfig]]) -> None:
        root, configs = two_runs

        result = runner.invoke(app, ["report", "compare", configs[0].run_id, "--root", str(root)])

        assert result.exit_code == 2

    def test_a_run_that_does_not_exist_is_a_failure_not_a_traceback(
        self, two_runs: tuple[Path, list[RunConfig]]
    ) -> None:
        root, configs = two_runs

        result = runner.invoke(
            app, ["report", "compare", configs[0].run_id, "absent", "--root", str(root)]
        )

        assert result.exit_code == 1
        assert "no run under" in result.output


def test_the_group_is_attached_to_the_application() -> None:
    result = runner.invoke(app, ["report", "--help"])

    assert result.exit_code == 0
    for command in ("render", "page", "compare", "list"):
        assert command in result.output


def test_a_record_is_where_the_layout_says_it_is(evaluated: tuple[Path, RunConfig, Path]) -> None:
    _, config, root = evaluated

    assert (root / run_paths(config).root.name / RECORD_NAME).is_file()


class TestRenderingABenchmarkedRun:
    """The last link in the chain.

    `nnp benchmark` writes into the record, and `nnp report render` reads it back out
    into a section and a figure.
    """

    @pytest.fixture
    def benchmarked(self, tmp_path: Path) -> tuple[Path, RunConfig]:
        path, config = write_config(tmp_path, name="cli-report-speed")
        assert runner.invoke(app, ["eval", "run", "-c", str(path)]).exit_code == 0
        result = runner.invoke(
            app,
            ["benchmark", "-c", str(path), "--trials", "2", "--warmup", "0", "--threads", "1"],
        )
        assert result.exit_code == 0, result.output
        return path, config

    def test_the_speed_section_and_its_figure_reach_the_report(
        self, benchmarked: tuple[Path, RunConfig]
    ) -> None:
        path, config = benchmarked

        result = runner.invoke(app, ["report", "render", "-c", str(path)])

        assert result.exit_code == 0, result.output
        paths = run_paths(config)
        assert "Speed at matched accuracy" in paths.markdown.read_text(encoding="utf-8")
        assert (paths.plots / "speed.png").is_file()

    def test_the_html_still_references_nothing_outside_itself(
        self, benchmarked: tuple[Path, RunConfig]
    ) -> None:
        path, config = benchmarked
        assert runner.invoke(app, ["report", "render", "-c", str(path)]).exit_code == 0

        html = run_paths(config).html.read_text(encoding="utf-8")

        assert not EXTERNAL.search(html)


def rollouts(count: int = 2) -> tuple[RolloutRecord, ...]:
    """Rollouts that all ran to the horizon they asked for."""
    return tuple(
        RolloutRecord(
            trajectory=f"t{index}",
            regime="cold_collapse",
            split="test",
            steps_requested=100,
            steps_completed=100,
            stop_reason="completed",
            seconds=0.01,
        )
        for index in range(count)
    )


def predictor(name: str, split: str, horizon: float) -> PredictorResult:
    """One predictor on one split, with its usable stretch chosen directly."""
    return PredictorResult(
        predictor=name,
        spec=name,
        split=split,
        regimes=("cold_collapse",),
        rollouts=rollouts(),
        metrics=(
            MetricRecord(
                name="rollout_error",
                scalars={"error.final": 1.0, f"horizon.{USABLE_THRESHOLD:g}": horizon},
            ),
        ),
        seconds_per_step=0.002,
        completed=True,
    )


def benchmark() -> SpeedReport:
    """A benchmark whose surrogate is slower than the solver setting it matches."""
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
        label=LEARNED,
        predictor=LEARNED,
        kind="surrogate",
        substeps=0,
        error=0.5,
        seconds_per_step=0.002,
        iqr=0.0002,
        relative_spread=0.2,
        stable=True,
        completed=True,
    )
    return SpeedReport(
        system="nbody",
        split="test",
        steps=100,
        n_initial_conditions=2,
        threads=1,
        trials=5,
        warmup=1,
        steps_per_trial=4,
        dataset_substeps=1,
        ladder=(solver,),
        surrogates=(surrogate,),
        matched=(
            MatchedSpeedup(
                predictor=LEARNED,
                error=0.5,
                seconds_per_step=0.002,
                matched_substeps=1,
                matched_error=0.5,
                matched_seconds_per_step=0.001,
                speedup=0.5,
                bracketed=True,
            ),
        ),
        costs=(
            CostAccounting(
                predictor=LEARNED,
                training_seconds=10.0,
                generation_seconds=1.0,
                steps_per_rollout=100,
                saving_per_rollout=-0.1,
                break_even_rollouts=-1.0,
            ),
        ),
    )


def fixture_record() -> RunRecord:
    """A run a landing page can be built from: a surrogate, a baseline and a benchmark."""
    return RunRecord(
        run_id="0123456789abcdef",
        name="cli-page",
        created="2026-01-01T00:00:00+00:00",
        code_version="0.1.0",
        commit="a" * 40,
        config=RunConfig.model_validate(CONFIG),
        environment=EnvironmentRecord(
            python="3.12.0",
            implementation="CPython",
            platform="Linux 6.1",
            machine="x86_64",
            cpu_count=8,
            packages={"numpy": "2.1.0"},
        ),
        timings={"evaluation": 1.5},
        evaluation=SuiteResult(
            code_version="0.1.0",
            run_id="0123456789abcdef",
            dataset_id="fedcba9876543210",
            system="nbody",
            seed=2,
            settings=SuiteSettings(
                name="cli",
                metrics=("rollout_error",),
                rollout_steps=100,
                n_initial_conditions=2,
                error_thresholds=(0.01, USABLE_THRESHOLD, 1.0),
                symmetry_steps=2,
                distribution_window=0.25,
                divergence_factor=1000.0,
            ),
            invariants={},
            results=(
                predictor(PERSISTENCE, TRAINED_ON_SPLIT, 0.1),
                predictor(PERSISTENCE, HELD_OUT_SPLIT, 0.1),
                predictor(LEARNED, TRAINED_ON_SPLIT, 0.25),
                predictor(LEARNED, HELD_OUT_SPLIT, 0.2),
            ),
        ),
        benchmark=benchmark(),
    )


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    """A runs root holding one run, with the report and the plots the page links to."""
    root = tmp_path / "runs"
    run = root / "cli-page-0123456789abcdef"
    (run / PLOTS_DIR).mkdir(parents=True)
    write_record(run / RECORD_NAME, fixture_record())
    (run / HTML_NAME).write_text("<!DOCTYPE html>\n", encoding="utf-8")
    for split in (TRAINED_ON_SPLIT, HELD_OUT_SPLIT):
        for name in (LEARNED, PERSISTENCE):
            (run / PLOTS_DIR / state_plot_name(split, name)).write_bytes(b"")
    return root


class TestPage:
    def test_it_writes_the_page_into_the_runs_root(self, runs_root: Path) -> None:
        result = runner.invoke(app, ["report", "page", "--root", str(runs_root)])

        assert result.exit_code == 0, result.output
        assert (runs_root / LANDING_NAME).is_file()

    def test_every_link_resolves(self, runs_root: Path) -> None:
        assert runner.invoke(app, ["report", "page", "--root", str(runs_root)]).exit_code == 0

        page = (runs_root / LANDING_NAME).read_text(encoding="utf-8")

        targets = [
            target for target in REFERENCED.findall(page) if not target.startswith(("#", "data:"))
        ]
        assert targets
        assert [target for target in targets if not (runs_root / target).exists()] == []

    def test_it_references_no_host(self, runs_root: Path) -> None:
        assert runner.invoke(app, ["report", "page", "--root", str(runs_root)]).exit_code == 0

        assert not HOSTED.search((runs_root / LANDING_NAME).read_text(encoding="utf-8"))

    def test_building_it_twice_gives_byte_identical_output(self, runs_root: Path) -> None:
        assert runner.invoke(app, ["report", "page", "--root", str(runs_root)]).exit_code == 0
        first = (runs_root / LANDING_NAME).read_bytes()

        assert runner.invoke(app, ["report", "page", "--root", str(runs_root)]).exit_code == 0

        assert (runs_root / LANDING_NAME).read_bytes() == first

    def test_it_can_be_written_somewhere_else(self, runs_root: Path, tmp_path: Path) -> None:
        output = tmp_path / "elsewhere" / "landing.html"

        result = runner.invoke(app, ["report", "page", "--root", str(runs_root), "-o", str(output)])

        assert result.exit_code == 0, result.output
        assert output.is_file()
        assert not (runs_root / LANDING_NAME).exists()

    def test_a_root_with_no_records_names_the_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "nothing"

        result = runner.invoke(app, ["report", "page", "--root", str(empty)])

        assert result.exit_code == 1
        assert str(empty) in result.output

    def test_a_run_with_no_surrogate_says_so_rather_than_writing_a_blank_hero(
        self, rendered: tuple[RunConfig, Path]
    ) -> None:
        _, root = rendered

        result = runner.invoke(app, ["report", "page", "--root", str(root)])

        assert result.exit_code == 1
        assert "surrogate" in result.output
