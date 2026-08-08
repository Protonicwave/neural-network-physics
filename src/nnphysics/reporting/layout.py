"""Where a run's artefacts live.

One directory per run, named after the run and the identifier its configuration hashes
to, holding the record, the snapshots, the plots and the two reports. The names are fixed
so that a reader, a comparison or the phase 10 agent can find a run's parts without being
told where they are.

None of it is committed. Everything under the runs root is derived from a configuration
and a seed, and a repository that carried it would be carrying a copy of its own output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nnphysics.core.errors import ValidationError

if TYPE_CHECKING:
    from nnphysics.core.config import RunConfig

__all__ = [
    "BENCHMARK_NAME",
    "ENSEMBLE_LABEL",
    "FAULT_LABEL",
    "HISTORY_NAME",
    "HTML_NAME",
    "MARKDOWN_NAME",
    "PLOTS_DIR",
    "RECORD_NAME",
    "SNAPSHOTS_NAME",
    "RunPaths",
    "ensemble_paths",
    "fault_paths",
    "find_record",
    "find_records",
    "result_name",
    "run_paths",
    "state_plot_name",
]

RECORD_NAME = "record.json"
"""The run record. A directory without one is not a run directory."""

SNAPSHOTS_NAME = "snapshots.npz"
"""States kept for the qualitative plot."""

MARKDOWN_NAME = "report.md"
"""The rendered report, for reading in a terminal or a repository browser."""

HTML_NAME = "report.html"
"""The rendered report as a single self contained file, for sending to someone."""

PLOTS_DIR = "plots"
"""Rendered figures. The HTML report embeds them, so it stays readable without them."""

HISTORY_NAME = "training.json"
"""What training recorded about itself. Written on its own by a run that trains without
evaluating, so what it cost is not lost."""

BENCHMARK_NAME = "benchmark.json"
"""What the timings measured. Written beside the record as well as into it, so a
benchmark can be read without parsing a run."""

ENSEMBLE_LABEL = "ensemble"
"""Marks the directory a deep ensemble writes to, so it cannot land on top of the
directory of the member that happens to share its configuration."""

FAULT_LABEL = "fault"
"""Marks a directory the fault suite writes to. Fault runs are derived rather than asked
for, and a runs root that mixed them in with real runs would report a deliberately broken
model in the history as though someone had meant it."""


def result_name(suite: str) -> str:
    """Name the evaluation result file of one suite.

    Args:
        suite: Suite name.

    Returns:
        The file name.
    """
    return f"evaluation-{suite}.json"


def state_plot_name(split: str, predictor: str) -> str:
    """Name the plot comparing one predictor's states with the true ones on one split.

    Named here rather than where it is drawn, because the landing page links to the file
    without drawing it and a second copy of the pattern would be a broken link the first
    time either changed.

    Args:
        split: Split name.
        predictor: Registered predictor name.

    Returns:
        The file name, inside the run's plot directory.
    """
    return f"{split}-state-{predictor}.png"


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Every path belonging to one run.

    Attributes:
        root: The run directory, which may not exist yet.
    """

    root: Path

    @property
    def record(self) -> Path:
        """The run record."""
        return self.root / RECORD_NAME

    @property
    def snapshots(self) -> Path:
        """The kept states."""
        return self.root / SNAPSHOTS_NAME

    @property
    def markdown(self) -> Path:
        """The Markdown report."""
        return self.root / MARKDOWN_NAME

    @property
    def html(self) -> Path:
        """The HTML report."""
        return self.root / HTML_NAME

    @property
    def plots(self) -> Path:
        """The plot directory."""
        return self.root / PLOTS_DIR

    @property
    def history(self) -> Path:
        """What training recorded about itself."""
        return self.root / HISTORY_NAME

    @property
    def benchmark(self) -> Path:
        """What the timings measured."""
        return self.root / BENCHMARK_NAME

    def result(self, suite: str) -> Path:
        """The evaluation result file of one suite.

        Args:
            suite: Suite name.

        Returns:
            The path.
        """
        return self.root / result_name(suite)

    def relative(self, path: Path) -> str:
        """Express a path inside this run as a POSIX path relative to the run directory.

        Recorded rather than the absolute path, so that a run directory can be moved or
        copied to another machine and still describe itself.

        Args:
            path: A path inside the run directory.

        Returns:
            The relative path, with forward slashes on every platform.

        Raises:
            ValidationError: If the path is not inside the run directory.
        """
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValidationError(f"{path} is not inside the run directory {self.root}") from error

    def ensure(self) -> None:
        """Create the run directory and its plot directory."""
        self.plots.mkdir(parents=True, exist_ok=True)


def run_paths(config: RunConfig) -> RunPaths:
    """Where the artefacts of a configured run belong.

    Args:
        config: The resolved run configuration.

    Returns:
        The paths, which may not exist yet.
    """
    return RunPaths(config.run_dir)


def ensemble_paths(config: RunConfig) -> RunPaths:
    """Where a deep ensemble of a configuration writes its own artefacts.

    Its own directory rather than a member's. Member zero is the plain run and hashes to
    the same identifier the ensemble would, so without a label of its own the ensemble's
    record would overwrite the record of one of the models it is made of.

    Args:
        config: The resolved run configuration, at any member index.

    Returns:
        The paths, which may not exist yet.
    """
    base = config.for_member(0)
    return RunPaths(base.output_dir / f"{base.name}-{ENSEMBLE_LABEL}-{base.run_id}")


def fault_paths(config: RunConfig, label: str) -> RunPaths:
    """Where one run of the fault suite writes its artefacts.

    Labelled as well as hashed, for the same reason an ensemble is. Two of the injected
    faults change nothing in the configuration, so they hash to the identifier of the
    known good run they were injected into, and without a label of their own they would
    each overwrite the baseline they are supposed to be compared against.

    Args:
        config: The resolved configuration of that run, faulty or not.
        label: What to call it, normally the fault name.

    Returns:
        The paths, which may not exist yet.

    Raises:
        ValidationError: If the label is empty, which would put the run back on top of
            the directory the plain run uses.
    """
    if not label:
        raise ValidationError("a fault run needs a label to keep it off the plain run's directory")
    return RunPaths(config.output_dir / f"{config.name}-{FAULT_LABEL}-{label}-{config.run_id}")


def find_records(root: Path) -> tuple[Path, ...]:
    """Every run record under a runs root, in directory order.

    Args:
        root: The runs root.

    Returns:
        Paths to each record found. Empty if the root does not exist, because a machine
        that has never run anything is not a failure.
    """
    if not root.is_dir():
        return ()
    return tuple(
        sorted(path / RECORD_NAME for path in root.iterdir() if (path / RECORD_NAME).is_file())
    )


def find_record(root: Path, run_id: str) -> Path:
    """The record of one run, found by its identifier.

    Args:
        root: The runs root.
        run_id: The run identifier, or the full directory name.

    Returns:
        The path to that run's record.

    Raises:
        ValidationError: If no run matches, or more than one does.
    """
    matches = [
        path
        for path in find_records(root)
        if path.parent.name == run_id or path.parent.name.endswith(f"-{run_id}")
    ]
    if not matches:
        raise ValidationError(f"no run under {root} has the identifier {run_id!r}")
    if len(matches) > 1:
        raise ValidationError(
            f"{len(matches)} runs under {root} match {run_id!r}: "
            f"{sorted(path.parent.name for path in matches)}"
        )
    return matches[0]
