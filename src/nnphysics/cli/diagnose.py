"""The `nnp diagnose` commands.

Two of them. `nnp diagnose` explains one comparison, which is the thing someone actually
wants when a run got worse and they do not know why. `nnp diagnose score` runs the whole
fault suite, asks both diagnosers about every fault, marks them against the causes that
were written down first, and writes the table.

Both can be run with `--rule-based` alone, which needs no credential and no network. That
is not a convenience: the rule based number is what the agent's number has to be read
against, and being able to produce it on its own is what stops the comparison from being
available only to whoever is holding an API key.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from nnphysics.agent.client import AgentConfig, AnthropicClient, load_agent_config
from nnphysics.agent.context import build_context
from nnphysics.agent.diagnose import AGENT_SOURCE, RULE_SOURCE, diagnose, rule_based_diagnosis
from nnphysics.agent.faults import FAULTS, Fault, fault
from nnphysics.agent.scoring import SuiteReport, render_report, score_card, score_fault
from nnphysics.cli.faultrun import BASELINE_LABEL, execute
from nnphysics.core.config import RunConfig, load_run_config
from nnphysics.core.errors import NNPhysicsError
from nnphysics.reporting.compare import DEFAULT_THRESHOLD
from nnphysics.reporting.layout import fault_paths, find_record
from nnphysics.reporting.record import read_record

if TYPE_CHECKING:
    from nnphysics.agent.client import Client
    from nnphysics.agent.context import DiagnosisContext
    from nnphysics.agent.diagnose import Diagnosis
    from nnphysics.agent.scoring import FaultOutcome, ScoreCard
    from nnphysics.reporting.record import RunRecord

__all__ = ["app"]

app = typer.Typer(
    name="diagnose",
    help="Explain a regression by reading two run reports, and score that explanation.",
    invoke_without_command=True,
)

_EXIT_FAILURE = 1
_EXIT_USAGE = 2

_DEFAULT_ROOT = Path("runs")
_DEFAULT_AGENT_CONFIG = Path("configs/agent.yaml")
_REPORT_NAME = "diagnosis.md"
_SCORES_NAME = "fault-scores.json"

RootOption = Annotated[
    Path, typer.Option("--root", help="Directory holding one subdirectory per run.")
]
AgentConfigOption = Annotated[
    Path,
    typer.Option("--agent-config", help="YAML file naming the model and its token limits."),
]
RuleBasedOption = Annotated[
    bool,
    typer.Option(
        "--rule-based/--agent",
        help="Use only the rule based diagnoser, which calls nothing and costs nothing.",
    ),
]


@app.callback()
def main(  # noqa: PLR0913, PLR0917
    # Two runs to compare and five switches. Typer builds the option list from the
    # signature, so a command with several options has several parameters.
    ctx: typer.Context,
    baseline: Annotated[
        str | None,
        typer.Option("--baseline", help="Run identifier believed to be good.", show_default=False),
    ] = None,
    candidate: Annotated[
        str | None,
        typer.Option("--candidate", help="Run identifier to explain.", show_default=False),
    ] = None,
    root: RootOption = _DEFAULT_ROOT,
    agent_config: AgentConfigOption = _DEFAULT_AGENT_CONFIG,
    rule_based: RuleBasedOption = False,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Relative change below which a scalar is unchanged."),
    ] = DEFAULT_THRESHOLD,
    context_only: Annotated[
        bool,
        typer.Option(
            "--context-only",
            help="Print the summary that would be sent and stop, without calling anything.",
        ),
    ] = False,
) -> None:
    """Explain what changed between two runs and why."""
    if ctx.invoked_subcommand is not None:
        return
    if baseline is None or candidate is None:
        typer.echo("pass --baseline and --candidate, or use the `score` subcommand", err=True)
        raise typer.Exit(code=_EXIT_USAGE)

    try:
        left = read_record(find_record(root, baseline))
        right = read_record(find_record(root, candidate))
        context = build_context(left, right, threshold=threshold)
    except NNPhysicsError as error:
        _fail(str(error))

    if context_only:
        typer.echo(context.render())
        return

    try:
        settings = None if rule_based else _agent_config(agent_config)
        result = _diagnose(context, settings)
    except NNPhysicsError as error:
        _fail(str(error))
    _report(result)


@app.command()
def score(  # noqa: PLR0913, PLR0917
    # One required option and five switches, each turning off or narrowing a stage.
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="YAML run configuration.", show_default=False),
    ],
    agent_config: AgentConfigOption = _DEFAULT_AGENT_CONFIG,
    rule_based: RuleBasedOption = False,
    faults: Annotated[
        list[str] | None,
        typer.Option("--fault", help="Run only these faults.", show_default=False),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory the scored table is written to."),
    ] = Path("docs/results"),
    reuse: Annotated[
        bool,
        typer.Option(
            "--reuse/--rerun",
            help="Reuse a fault run whose record already exists rather than training again.",
        ),
    ] = True,
) -> None:
    """Inject every fault, diagnose each one, and score the answers against the truth."""
    resolved = _resolve(config)
    selected = _selected(faults)
    try:
        settings = None if rule_based else _agent_config(agent_config)
        baseline = _run(resolved, None, reuse=reuse)
        agent_outcomes: list[FaultOutcome] = []
        rule_outcomes: list[FaultOutcome] = []
        client = _client(settings) if settings is not None else None
        for injected in selected:
            record = _run(resolved, injected, reuse=reuse)
            context = build_context(baseline, record)
            rule = rule_based_diagnosis(context)
            rule_outcomes.append(score_fault(injected, rule))
            typer.echo(f"  rule based: {', '.join(cause.value for cause in rule.causes)}")
            if client is not None and settings is not None:
                answer = diagnose(context, client, settings)
                agent_outcomes.append(score_fault(injected, answer))
                typer.echo(f"  agent: {', '.join(cause.value for cause in answer.causes)}")

        cards: list[ScoreCard] = []
        if agent_outcomes and client is not None:
            cards.append(score_card(AGENT_SOURCE, client.model, agent_outcomes))
        cards.append(score_card(RULE_SOURCE, "worst regressed metric", rule_outcomes))
        report = SuiteReport(
            created=datetime.now(UTC).isoformat(timespec="seconds"),
            system=resolved.system.name,
            baseline_run=baseline.run_id,
            agent=settings,
            provenance=(
                f"Produced by `nnp diagnose score --config {config}`"
                + ("" if settings is None else f" against `{settings.model}`")
                + ". Every number in it comes from that command."
            ),
            cards=tuple(cards),
        )
    except NNPhysicsError as error:
        _fail(str(error))

    output.mkdir(parents=True, exist_ok=True)
    (output / _REPORT_NAME).write_text(render_report(report), encoding="utf-8")
    (output / _SCORES_NAME).write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    for card in report.cards:
        typer.echo(
            f"{card.source}: top 1 {card.accuracy(1):.0%}, top 3 {card.accuracy(3):.0%}, "
            f"${card.dollars_per_diagnosis:.4f} per diagnosis."
        )
    typer.echo(f"Wrote {output / _REPORT_NAME} and {output / _SCORES_NAME}.")


def _run(config: RunConfig, injected: Fault | None, *, reuse: bool) -> RunRecord:
    """Execute one fault run, or read back the record of one already done."""
    label = injected.name if injected is not None else BASELINE_LABEL
    resolved = injected.apply(config) if injected is not None else config
    record_path = fault_paths(resolved, label).record
    if reuse and record_path.is_file():
        typer.echo(f"[{label}] reusing {record_path}.")
        return read_record(record_path)
    return execute(config, injected, echo=typer.echo).record


def _selected(names: list[str] | None) -> tuple[Fault, ...]:
    """The faults to run, all of them unless some were named."""
    if not names:
        return FAULTS
    try:
        return tuple(fault(name) for name in names)
    except NNPhysicsError as error:
        _fail(str(error))


def _diagnose(context: DiagnosisContext, settings: AgentConfig | None) -> Diagnosis:
    """Ask whichever diagnoser was chosen."""
    if settings is None:
        return rule_based_diagnosis(context)
    return diagnose(context, _client(settings), settings)


def _client(settings: AgentConfig) -> Client:
    """Build a client against the real API."""
    return AnthropicClient(settings, os.environ)


def _agent_config(path: Path) -> AgentConfig:
    """Load the agent settings, saying plainly when the file is simply not there."""
    if not path.is_file():
        _fail(
            f"no agent configuration at {path}. Write one naming the model, or pass "
            f"--rule-based to use the diagnoser that calls nothing."
        )
    return load_agent_config(path, os.environ)


def _report(diagnosis: Diagnosis) -> None:
    """Print a diagnosis."""
    typer.echo(f"Diagnosed by {diagnosis.source} ({diagnosis.model}).")
    if diagnosis.regressed_metrics:
        typer.echo(f"Regressed: {', '.join(diagnosis.regressed_metrics)}.")
    for position, candidate in enumerate(diagnosis.candidates, start=1):
        typer.echo(f"  {position}. {candidate.cause.value} ({candidate.confidence:.0%})")
        if candidate.reasoning:
            typer.echo(f"     {candidate.reasoning}")
    if diagnosis.next_check:
        typer.echo(f"Next check: {diagnosis.next_check}")
    cost = diagnosis.cost
    if cost.input_tokens or cost.output_tokens:
        typer.echo(
            f"Cost: {cost.input_tokens} input and {cost.output_tokens} output tokens, "
            f"${cost.dollars:.4f}, {cost.attempts} attempt(s)."
        )


def _resolve(config: Path) -> RunConfig:
    """Load a configuration, reporting failures as a clean exit."""
    try:
        return load_run_config(config, os.environ)
    except NNPhysicsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=_EXIT_USAGE) from error


def _fail(message: str) -> NoReturn:
    """Report a failure and exit."""
    typer.echo(message, err=True)
    raise typer.Exit(code=_EXIT_FAILURE)
