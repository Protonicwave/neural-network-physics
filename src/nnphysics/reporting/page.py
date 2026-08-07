"""What the landing page shows, derived from a runs root.

The landing page answers one question for a reader who knows none of the vocabulary, and
every number on it has to come from a stored record rather than from a sentence somebody
typed. This module does the deriving and stops there. It holds no HTML, no SVG and no CSS,
so the numbers can be tested without parsing a document.

Two sources, and only two. The run records under the runs root carry everything about
accuracy, cost and divergence. The committed fault scores file carries what the two
diagnosers scored. Prose that is genuinely editorial belongs with the builders, not here.

The fault scores are read through a minimal local schema rather than through
`nnphysics.agent`, because reporting may not import the agent layer: the agent reads
reports, so the dependency runs the other way and a test enforces it. Only the fields the
page shows are declared, and the rest of the committed file is ignored deliberately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nnphysics.core.errors import ConfigurationError
from nnphysics.data.manifest import Split
from nnphysics.evals.metrics.error import NEVER_REACHED
from nnphysics.evals.predictors import BROKEN_PREDICTORS, REFERENCE_NAME, UNCERTAIN_PREDICTORS
from nnphysics.evals.rollout import StopReason
from nnphysics.evals.speed import NEVER_PAYS
from nnphysics.reporting.layout import (
    FAULT_LABEL,
    HTML_NAME,
    PLOTS_DIR,
    find_records,
)
from nnphysics.reporting.record import read_record

if TYPE_CHECKING:
    from pathlib import Path

    from nnphysics.evals.result import PredictorResult
    from nnphysics.evals.speed import SpeedPoint, SpeedReport
    from nnphysics.reporting.record import RunRecord

__all__ = [
    "HARNESS_PREDICTORS",
    "HELD_OUT_SPLIT",
    "PERSISTENCE",
    "TOP_ONE",
    "TOP_THREE",
    "TRAINED_ON_SPLIT",
    "USABLE_THRESHOLD",
    "CostLadder",
    "CostPoint",
    "DiagnoserScore",
    "Diagnosis",
    "FaultRank",
    "Headlines",
    "Horizon",
    "MatchedCost",
    "PageModel",
    "RunCard",
    "RunVerdict",
    "VerdictKind",
    "build_page",
    "read_fault_scores",
    "summarise_run",
]

USABLE_THRESHOLD = 0.1
"""Normalised error a rollout stops being usable at. A tenth of the size of the state
means the mistake is a tenth of the thing being predicted, which is the last point a
reader can be told the prediction still means something."""

TOP_ONE = 1
TOP_THREE = 3
"""The two positions the page reports the diagnosers at. Naming the true cause first is
the useful result; having it in the first three is what a person would actually be handed
to check."""

TRAINED_ON_SPLIT = Split.TEST
"""The split drawn from the regimes the model trained on. Held back from training, so it
measures accuracy rather than recall, but not a change of setup."""

HELD_OUT_SPLIT = Split.HELD_OUT
"""The split drawn from a regime no training trajectory came from."""

PERSISTENCE = "persistence"
"""The free baseline: return the input unchanged. The page draws it beside every
surrogate because a surrogate that cannot beat doing nothing has not earned its training
cost. A test pins this to the predictor's own name so the two cannot drift."""

HARNESS_PREDICTORS = frozenset({REFERENCE_NAME, *BROKEN_PREDICTORS, *UNCERTAIN_PREDICTORS})
"""Everything the evaluation harness supplies itself: the reference solver, the baselines,
the deliberately broken sentinels and the uncertainty controls. Read from the registry
rather than listed here, so a predictor added to the harness does not silently become a
surrogate on the page. Anything a run evaluated that is not in this set is a learned
model."""


class VerdictKind(StrEnum):
    """What kind of answer a run gives, before it is put into words.

    The page styles a verdict by kind and prints it by phrase. Keeping the two apart is
    what stops a colour from being decided in this module.
    """

    HARNESS_CHECK = "harness_check"
    COST_BENCHMARK = "cost_benchmark"
    CANNOT_RUN_UNSEEN = "cannot_run_unseen"
    DIVERGES = "diverges"
    STABLE = "stable"


@dataclass(frozen=True, slots=True)
class RunVerdict:
    """One run's answer, in one short phrase.

    Attributes:
        kind: Which rule produced it.
        phrase: The phrase the page prints.
    """

    kind: VerdictKind
    phrase: str


@dataclass(frozen=True, slots=True)
class Horizon:
    """How long one predictor stayed usable on one split.

    The harness reports a horizon in simulated time, because two predictors stepping at
    different intervals are otherwise not comparable. The page states it in stored steps,
    because a step is a thing a reader can picture.

    Attributes:
        predictor: Registered predictor name.
        split: Split the rollouts were drawn from.
        learned: Whether this is a learned model rather than part of the harness.
        steps: Stored steps the prediction stayed usable for, or `None` when there is
            nothing to report: the predictor never ran, or the suite never measured
            error growth.
        rollout_steps: Steps the rollout asked for, which is what `steps` is out of.
        whole_rollout: Whether the error never crossed the threshold, in which case
            `steps` is the whole rollout rather than a measured crossing.
        rollouts: Rollouts attempted.
        completed: Rollouts that ran to the requested horizon.
        diverged: Whether any rollout left the physical range.
        failed: Whether every rollout stopped by raising, which is how a model that
            refuses to build on a split reports itself.
    """

    predictor: str
    split: str
    learned: bool
    steps: float | None
    rollout_steps: int
    whole_rollout: bool
    rollouts: int
    completed: int
    diverged: bool
    failed: bool


@dataclass(frozen=True, slots=True)
class CostPoint:
    """One point on the accuracy against wall clock curve.

    Attributes:
        label: How the benchmark named it.
        predictor: Registered predictor name.
        solver: Whether this is a setting of the reference solver rather than a surrogate.
        substeps: Solver steps per stored interval, zero for a surrogate.
        error: Worst error over the rollout.
        seconds_per_step: Median wall clock per stored step.
        relative_spread: Interquartile range of that time as a fraction of the median.
    """

    label: str
    predictor: str
    solver: bool
    substeps: int
    error: float
    seconds_per_step: float
    relative_spread: float


@dataclass(frozen=True, slots=True)
class MatchedCost:
    """What one surrogate is worth against the cheapest solver setting that is as accurate.

    Attributes:
        predictor: Registered predictor name.
        speedup: Solver time divided by surrogate time. Below one means slower.
        slowdown: The reciprocal, or `None` when the surrogate was not slower. Carried
            because the page states the losing case as a slowdown, and a reader should
            not be asked to invert a number in their head.
        seconds_per_step: What the surrogate cost.
        matched_seconds_per_step: What the matched solver setting cost.
        matched_substeps: Substeps of that setting.
        bracketed: Whether the crossing was caught between two measured settings. Not
            bracketed means no solver setting ran slower and less accurately, so the
            comparison is a bound rather than a measurement.
        break_even_rollouts: Rollouts needed to repay training and data generation, or
            `None` when the surrogate never pays back at any number.
    """

    predictor: str
    speedup: float
    slowdown: float | None
    seconds_per_step: float
    matched_seconds_per_step: float
    matched_substeps: int
    bracketed: bool
    break_even_rollouts: float | None


@dataclass(frozen=True, slots=True)
class CostLadder:
    """One benchmark, as the cost chart draws it.

    Attributes:
        system: System benchmarked.
        split: Split the initial conditions came from.
        threads: Threads the timings were taken on.
        trials: Timed trials per point.
        solver: The reference solver at each setting of its quality dial, cheapest first.
        surrogates: The learned models, as points.
        matched: One comparison per surrogate.
    """

    system: str
    split: str
    threads: int
    trials: int
    solver: tuple[CostPoint, ...]
    surrogates: tuple[CostPoint, ...]
    matched: tuple[MatchedCost, ...]


@dataclass(frozen=True, slots=True)
class RunCard:
    """One run, as the register shows it.

    Attributes:
        run_id: Identifier derived from the resolved configuration and the seed.
        name: Human readable label from the configuration.
        directory: Name of the run directory under the runs root.
        system: System evaluated.
        created: When the run finished, as an ISO 8601 string.
        commit: Commit of the working tree that produced it, empty when unknown.
        model: The learned model the run trained, or `None` for a run that trained none.
        parameters: How many parameters that model had, or `None`.
        epochs: How many epochs it trained for, or `None`.
        fault: Whether this is an injected fault run. Fault runs are derived rather than
            asked for, and a page that mixed them in would report a deliberately broken
            model as though somebody had meant it.
        rollout_steps: Steps every rollout in this run asked for.
        horizons: One per predictor and split, in the order they were evaluated.
        verdict: The run's answer, in one phrase.
        cost: What the benchmark measured, or `None` for a run that was never benchmarked.
    """

    run_id: str
    name: str
    directory: str
    system: str
    created: str
    commit: str
    model: str | None
    parameters: int | None
    epochs: int | None
    fault: bool
    rollout_steps: int
    horizons: tuple[Horizon, ...]
    verdict: RunVerdict
    cost: CostLadder | None

    @property
    def report(self) -> str:
        """Path to this run's report, relative to the runs root."""
        return f"{self.directory}/{HTML_NAME}"

    @property
    def plots(self) -> str:
        """Path to this run's plots, relative to the runs root."""
        return f"{self.directory}/{PLOTS_DIR}"

    def horizon(self, predictor: str, split: str) -> Horizon | None:
        """Look up one predictor's horizon on one split.

        Args:
            predictor: Registered predictor name.
            split: Split name.

        Returns:
            The horizon, or `None` if the run has none.
        """
        for entry in self.horizons:
            if entry.predictor == predictor and entry.split == split:
                return entry
        return None


@dataclass(frozen=True, slots=True)
class Headlines:
    """The three figures in the hero, each selected by a rule over the summaries.

    Attributes:
        usable_steps: Longest stretch any surrogate stayed usable on the split it trained
            on, counting only surrogates whose rollouts all completed there. A surrogate
            that diverged part way has not stayed usable, and averaging its horizon over
            rollouts that blew up would make the claim stronger than the record supports.
        usable_of: Steps that stretch is out of.
        usable_predictor: The surrogate it belongs to.
        usable_run_id: The run it came from.
        held_out_completed: Fewest rollouts any surrogate completed on the unseen split.
        held_out_of: Rollouts it attempted.
        held_out_predictor: The surrogate it belongs to.
        held_out_run_id: The run it came from.
        slowdown: Largest matched accuracy slowdown any benchmark measured.
        slowdown_predictor: The surrogate it belongs to.
        slowdown_run_id: The run it came from.
    """

    usable_steps: float
    usable_of: int
    usable_predictor: str
    usable_run_id: str
    held_out_completed: int
    held_out_of: int
    held_out_predictor: str
    held_out_run_id: str
    slowdown: float
    slowdown_predictor: str
    slowdown_run_id: str


@dataclass(frozen=True, slots=True)
class FaultRank:
    """Where one diagnoser ranked the true cause of one injected fault.

    Attributes:
        fault: Name of the fault.
        true_cause: What was actually broken.
        rank: One based position of the true cause, or `None` if it was never named.
    """

    fault: str
    true_cause: str
    rank: int | None


@dataclass(frozen=True, slots=True)
class DiagnoserScore:
    """One diagnoser over the whole fault set.

    Attributes:
        source: Which diagnoser, `agent` or `rule_based`.
        model: Model that answered, or the name of the rule.
        faults: Faults scored.
        top1: Share whose true cause was ranked first.
        top3: Share whose true cause was in the first three.
        top1_count: The same as a count, because the page states both.
        top3_count: The same as a count.
        ranks: One per fault, in the order the faults were run.
    """

    source: str
    model: str
    faults: int
    top1: float
    top3: float
    top1_count: int
    top3_count: int
    ranks: tuple[FaultRank, ...]


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """Both diagnosers, scored on the same faults.

    Attributes:
        agent: The language model agent.
        baseline: The conventional rule based diagnoser it is measured against.
    """

    agent: DiagnoserScore
    baseline: DiagnoserScore


@dataclass(frozen=True, slots=True)
class PageModel:
    """Everything the landing page shows.

    Attributes:
        runs: One per run under the runs root, oldest first and ties broken by identifier.
        headlines: The three figures in the hero.
        diagnosis: What the two diagnosers scored, or `None` when no scores were supplied.
    """

    runs: tuple[RunCard, ...]
    headlines: Headlines
    diagnosis: Diagnosis | None

    @property
    def reported(self) -> tuple[RunCard, ...]:
        """The runs the page reports on, which is every run that is not a fault run."""
        return tuple(run for run in self.runs if not run.fault)

    @property
    def benchmarks(self) -> tuple[CostLadder, ...]:
        """Every benchmark the page draws, in run order."""
        return tuple(run.cost for run in self.reported if run.cost is not None)


class _Outcome(BaseModel):
    """One scored fault, as much of it as the page shows."""

    # Ignores rather than forbids, because this reads a file another layer writes and the
    # page deliberately shows a subset of it. Forbidding would make the page fail on a
    # field the agent added for its own reasons.
    model_config = ConfigDict(frozen=True, extra="ignore")

    fault: str = Field(min_length=1)
    true_cause: str = Field(min_length=1)
    rank: int | None = None


class _Card(BaseModel):
    """One diagnoser's card, as much of it as the page shows."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    source: str = Field(min_length=1)
    model: str = Field(min_length=1)
    outcomes: tuple[_Outcome, ...] = Field(min_length=1)


class _Scores(BaseModel):
    """The committed fault scores file, as much of it as the page shows."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    cards: tuple[_Card, ...] = Field(min_length=1)


def _score(card: _Card) -> DiagnoserScore:
    """Turn one card into the rates and ranks the page states."""
    ranks = tuple(
        FaultRank(fault=entry.fault, true_cause=entry.true_cause, rank=entry.rank)
        for entry in card.outcomes
    )
    faults = len(ranks)
    top1 = sum(1 for entry in ranks if entry.rank == TOP_ONE)
    top3 = sum(1 for entry in ranks if entry.rank is not None and entry.rank <= TOP_THREE)
    return DiagnoserScore(
        source=card.source,
        model=card.model,
        faults=faults,
        top1=top1 / faults,
        top3=top3 / faults,
        top1_count=top1,
        top3_count=top3,
        ranks=ranks,
    )


def read_fault_scores(path: Path) -> Diagnosis:
    """Read the committed fault scores.

    Args:
        path: The scores file.

    Returns:
        Both diagnosers, scored.

    Raises:
        ConfigurationError: If the file is missing, is not valid JSON, is not an object,
            fails validation, or does not carry exactly one agent card and one baseline
            card. A page that quietly drew one diagnoser would be a page claiming the
            agent has nothing to be measured against.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"cannot read fault scores {path}: {error}") from error
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{path} must hold a JSON object, got {type(raw).__name__}")
    try:
        scores = _Scores.model_validate(raw)
    except ValidationError as error:
        raise ConfigurationError(f"invalid fault scores in {path}: {error}") from error
    cards = {card.source: card for card in scores.cards}
    missing = [source for source in ("agent", "rule_based") if source not in cards]
    if missing:
        raise ConfigurationError(f"{path} carries no {' and no '.join(missing)} card")
    return Diagnosis(agent=_score(cards["agent"]), baseline=_score(cards["rule_based"]))


def _horizon(entry: PredictorResult, dt: float, rollout_steps: int) -> Horizon:
    """Turn one predictor's result on one split into a usable stretch in stored steps."""
    rollouts = len(entry.rollouts)
    completed = sum(1 for rollout in entry.rollouts if rollout.stop_reason == StopReason.COMPLETED)
    diverged = any(rollout.stop_reason == StopReason.DIVERGED for rollout in entry.rollouts)
    failed = all(rollout.stop_reason == StopReason.FAILED for rollout in entry.rollouts)
    try:
        crossing: float | None = entry.scalar("rollout_error", f"horizon.{USABLE_THRESHOLD:g}")
    except KeyError:
        # A suite that never measured error growth cannot say how long anything stayed
        # usable, which is not the same as saying it stayed usable throughout.
        crossing = None
    # Never crossing the threshold and never running produce the same sentinel, and they
    # are opposite answers: the reference solver never crosses because it is right the
    # whole way, and a model that refuses to build never crosses because it never ran.
    whole_rollout = crossing == NEVER_REACHED and not failed
    steps: float | None
    if failed or crossing is None:
        steps = None
    elif whole_rollout:
        steps = float(rollout_steps)
    else:
        steps = crossing / dt
    return Horizon(
        predictor=entry.predictor,
        split=entry.split,
        learned=entry.predictor not in HARNESS_PREDICTORS,
        steps=steps,
        rollout_steps=rollout_steps,
        whole_rollout=whole_rollout,
        rollouts=rollouts,
        completed=completed,
        diverged=diverged,
        failed=failed,
    )


def _ladder(report: SpeedReport) -> CostLadder:
    """Turn a benchmark into the points and comparisons the cost chart draws."""
    break_even = {
        entry.predictor: entry.break_even_rollouts
        for entry in report.costs
        if entry.break_even_rollouts != NEVER_PAYS
    }
    return CostLadder(
        system=report.system,
        split=report.split,
        threads=report.threads,
        trials=report.trials,
        solver=tuple(_point(point, solver=True) for point in report.ladder),
        surrogates=tuple(_point(point, solver=False) for point in report.surrogates),
        matched=tuple(
            MatchedCost(
                predictor=entry.predictor,
                speedup=entry.speedup,
                slowdown=1.0 / entry.speedup if 0.0 < entry.speedup < 1.0 else None,
                seconds_per_step=entry.seconds_per_step,
                matched_seconds_per_step=entry.matched_seconds_per_step,
                matched_substeps=entry.matched_substeps,
                bracketed=entry.bracketed,
                break_even_rollouts=break_even.get(entry.predictor),
            )
            for entry in report.matched
        ),
    )


def _point(point: SpeedPoint, *, solver: bool) -> CostPoint:
    """Turn one measured point into the form the chart draws.

    Whether a point is a solver setting is passed in rather than read off the record,
    because it is decided by which list of the benchmark the point arrived in.
    """
    return CostPoint(
        label=point.label,
        predictor=point.predictor,
        solver=solver,
        substeps=point.substeps,
        error=point.error,
        seconds_per_step=point.seconds_per_step,
        relative_spread=point.relative_spread,
    )


def _cost_verdict(cost: CostLadder) -> RunVerdict:
    """Read a cost benchmark's answer off its matched comparisons."""
    if not cost.matched:
        return RunVerdict(kind=VerdictKind.COST_BENCHMARK, phrase="not measured")
    if any(not entry.bracketed for entry in cost.matched):
        return RunVerdict(kind=VerdictKind.COST_BENCHMARK, phrase="undecided")
    if all(entry.break_even_rollouts is None for entry in cost.matched):
        return RunVerdict(kind=VerdictKind.COST_BENCHMARK, phrase="never pays back")
    return RunVerdict(kind=VerdictKind.COST_BENCHMARK, phrase="pays back eventually")


def _verdict(
    record: RunRecord, horizons: tuple[Horizon, ...], cost: CostLadder | None
) -> RunVerdict:
    """Work out what one run says, in one phrase.

    The rules are tried in order and the first that fits wins, because a run that trained
    nothing cannot also be a run whose model diverged.
    """
    learned = tuple(entry for entry in horizons if entry.learned)
    if not learned:
        return RunVerdict(kind=VerdictKind.HARNESS_CHECK, phrase="harness check")
    # A run that benchmarks predictors it did not train is asking a cost question, not an
    # accuracy one, so its answer is read off the speedup rather than off the rollouts.
    if cost is not None and record.training is None:
        return _cost_verdict(cost)
    held_out = tuple(entry for entry in learned if entry.split == HELD_OUT_SPLIT)
    if held_out and all(entry.failed for entry in held_out):
        return RunVerdict(kind=VerdictKind.CANNOT_RUN_UNSEEN, phrase="cannot run unseen")
    diverged = tuple(entry for entry in learned if entry.diverged)
    if diverged:
        if all(entry.split == HELD_OUT_SPLIT for entry in diverged):
            return RunVerdict(kind=VerdictKind.DIVERGES, phrase="diverges when unseen")
        return RunVerdict(kind=VerdictKind.DIVERGES, phrase="diverges")
    return RunVerdict(kind=VerdictKind.STABLE, phrase=_stable_phrase(horizons, learned))


def _stable_phrase(horizons: tuple[Horizon, ...], learned: tuple[Horizon, ...]) -> str:
    """Qualify a stable run by whether it clears the free baseline."""
    baseline = next(
        (
            entry
            for entry in horizons
            if entry.predictor == PERSISTENCE and entry.split == TRAINED_ON_SPLIT
        ),
        None,
    )
    model = next((entry for entry in learned if entry.split == TRAINED_ON_SPLIT), None)
    if baseline is None or model is None or baseline.steps is None or model.steps is None:
        return "stable"
    if model.steps > baseline.steps:
        return "stable, beats persistence"
    return "stable, inaccurate"


def summarise_run(record: RunRecord, directory: str) -> RunCard:
    """Summarise one run.

    Args:
        record: The run record.
        directory: Name of the run directory under the runs root.

    Returns:
        The run as the register shows it.
    """
    rollout_steps = record.evaluation.settings.rollout_steps
    dt = record.config.data.dt
    horizons = tuple(_horizon(entry, dt, rollout_steps) for entry in record.evaluation.results)
    cost = _ladder(record.benchmark) if record.benchmark is not None else None
    training = record.training
    return RunCard(
        run_id=record.run_id,
        name=record.name,
        directory=directory,
        system=record.evaluation.system,
        created=record.created,
        commit=record.commit,
        model=training.model if training is not None else None,
        parameters=training.n_parameters if training is not None else None,
        epochs=len(training.epochs) if training is not None else None,
        # A fault run labels its own directory, which is the only place the fact survives:
        # two of the injected faults change nothing in the configuration, so the record
        # itself is indistinguishable from the healthy run they were injected into.
        fault=f"-{FAULT_LABEL}-" in directory,
        rollout_steps=rollout_steps,
        horizons=horizons,
        verdict=_verdict(record, horizons, cost),
        cost=cost,
    )


def _headlines(runs: tuple[RunCard, ...]) -> Headlines:
    """Select the three hero figures by rule over the summaries.

    Fault runs are left out of all three. A deliberately broken run reached further on its
    trained on split than any real one did, and a headline that quoted it would be quoting
    a model nobody meant to train.

    Raises:
        ConfigurationError: If no run supports a figure. A hero with a blank number is
            worse than an error naming what is missing.
    """
    reported = tuple(run for run in runs if not run.fault)
    usable = [
        (run, entry)
        for run in reported
        for entry in run.horizons
        if entry.learned
        and entry.split == TRAINED_ON_SPLIT
        and entry.steps is not None
        and entry.completed == entry.rollouts
    ]
    if not usable:
        raise ConfigurationError("no run has a surrogate that completed its trained on split")
    best_run, best = max(usable, key=lambda pair: (pair[1].steps, pair[0].run_id))
    held_out = [
        (run, entry)
        for run in reported
        for entry in run.horizons
        if entry.learned and entry.split == HELD_OUT_SPLIT
    ]
    if not held_out:
        raise ConfigurationError("no run evaluated a surrogate on a held out split")
    worst_run, worst = min(held_out, key=lambda pair: (pair[1].completed, pair[0].run_id))
    slowdowns = [
        (run, entry)
        for run in reported
        if run.cost is not None
        for entry in run.cost.matched
        if entry.slowdown is not None
    ]
    if not slowdowns:
        raise ConfigurationError("no benchmark measured a surrogate slower than the solver")
    slow_run, slow = max(slowdowns, key=lambda pair: (pair[1].slowdown, pair[0].run_id))
    return Headlines(
        usable_steps=best.steps if best.steps is not None else 0.0,
        usable_of=best.rollout_steps,
        usable_predictor=best.predictor,
        usable_run_id=best_run.run_id,
        held_out_completed=worst.completed,
        held_out_of=worst.rollouts,
        held_out_predictor=worst.predictor,
        held_out_run_id=worst_run.run_id,
        slowdown=slow.slowdown if slow.slowdown is not None else 0.0,
        slowdown_predictor=slow.predictor,
        slowdown_run_id=slow_run.run_id,
    )


def build_page(root: Path, *, scores: Path | None = None) -> PageModel:
    """Derive everything the landing page shows from a runs root.

    Args:
        root: The runs root.
        scores: The committed fault scores file, or `None` to build a page without the
            diagnosis section.

    Returns:
        The model, with runs oldest first and ties broken by identifier so that the same
        inputs give the same page whatever order the filesystem returns directories in.

    Raises:
        ConfigurationError: If the root holds no readable record, if a record cannot be
            read, or if no run supports one of the headline figures.
    """
    cards = [summarise_run(read_record(path), path.parent.name) for path in find_records(root)]
    if not cards:
        raise ConfigurationError(f"{root} holds no run record to build a page from")
    runs = tuple(sorted(cards, key=lambda card: (card.created, card.run_id)))
    return PageModel(
        runs=runs,
        headlines=_headlines(runs),
        diagnosis=read_fault_scores(scores) if scores is not None else None,
    )
