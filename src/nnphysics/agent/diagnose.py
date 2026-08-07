"""Asking what went wrong, and answering it without a model as a control.

The agent is handed the reduced context and must return a ranked list of causes drawn
from a fixed vocabulary, a confidence for each and one suggested next check. The structure
is enforced by the API through a strict tool schema, so a reply that is the right words in
the wrong shape never becomes a diagnosis. Everything that comes back is validated again
here, because a schema constrains the shape and not the content: a confidence outside zero
to one and the same cause named twice both fit the schema and neither is an answer.

Beside it is a rule based diagnoser that reads no prose and calls nothing. It finds the
metric that regressed most and names the cause that metric is usually about. It is the
number the agent has to beat, and it is given every advantage: the same vocabulary, and a
metric to cause table written with the fault set already in hand. If the agent does not
beat that, the finding is that the agent is not earning its cost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from nnphysics.agent.causes import Cause, cause_catalogue
from nnphysics.agent.client import AgentConfig, ToolSchema
from nnphysics.core.errors import ValidationError

if TYPE_CHECKING:
    from nnphysics.agent.client import Client
    from nnphysics.agent.context import DiagnosisContext, ScalarChange

__all__ = [
    "AGENT_SOURCE",
    "DIAGNOSIS_TOOL",
    "MAX_CANDIDATES",
    "METRIC_CAUSES",
    "RULE_SOURCE",
    "SYSTEM_PROMPT",
    "Candidate",
    "Diagnosis",
    "DiagnosisCost",
    "diagnose",
    "rule_based_diagnosis",
]

MAX_CANDIDATES = 5
"""Causes a diagnosis may rank. Five so that a top three accuracy is measured against a
list that could have been longer, and short enough that ranking still means something."""

AGENT_SOURCE = "agent"
RULE_SOURCE = "rule_based"

SYSTEM_PROMPT = """\
You diagnose regressions in a neural surrogate for physical simulation.

You are given two runs of the same pipeline: a baseline believed to be good, and a
candidate that may be worse. You are shown the configuration differences between them, the
scalars that moved and in which direction each one is supposed to move, the shape of the
error curves, what training did, and which rollouts failed to finish.

Rank the causes that best explain the candidate, most likely first, choosing only from
this list:

{catalogue}

Rules.

Name a cause only if something in the evidence points at it. A configuration difference is
strong evidence; a regression with no configuration difference behind it means the fault
is in how the run was executed rather than in how it was configured.

Prefer the cause that explains the most. A single fault usually moves several metrics at
once, and a list of one cause per metric is a list that has explained nothing.

Confidences are your own belief that each cause is the real one. They do not have to sum
to one, and a confident wrong answer is worse than an uncertain right one.

The timing scalars are wall clock on a shared machine and move by a factor of two between
two runs of identical code, so a timing regression on its own is evidence of nothing.

Suggest one next check: the single cheapest thing someone could look at that would confirm
or rule out your first candidate.

If nothing in the evidence is a regression, say so with the no_regression cause rather
than picking the least implausible fault.
"""

METRIC_CAUSES: Mapping[str, Cause] = {
    "symmetry_violation": Cause.MODEL_SYMMETRY,
    "invariant_drift": Cause.INTEGRATOR_STEP_SIZE,
    "one_step_error": Cause.LEARNING_RATE,
    "rollout_error": Cause.ROLLOUT_CURRICULUM,
    "distribution_drift": Cause.TRAINING_REGIME,
    "calibration": Cause.NORMALISATION_STATISTICS,
    "resolution_generalisation": Cause.MODEL_CAPACITY,
}
"""What the rule based diagnoser guesses from each metric. A hand written table, and one
written knowing which faults were going to be injected, which is deliberate: the baseline
is worth reporting only if it is the strongest trivial diagnoser rather than a straw one.

`timing` is absent on the same principle. It is wall clock, and phase 09 established that
wall clock on this machine moves by a factor of two between two runs of the same code, so
a rule that let it decide the ranking would be a rule losing to noise rather than to the
agent."""


class DiagnosisCost(BaseModel):
    """What one diagnosis cost.

    Attributes:
        input_tokens: Tokens the prompt occupied.
        output_tokens: Tokens the reply occupied, thinking included.
        dollars: Cost at the configured prices, zero when none are configured.
        attempts: Requests made, the successful one included.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    dollars: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=0, ge=0)


class Candidate(BaseModel):
    """One cause a diagnosis names.

    Attributes:
        cause: The cause, from the fixed vocabulary.
        confidence: Belief that this is the real one, between zero and one.
        reasoning: Why, in one or two sentences.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cause: Cause
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class Diagnosis(BaseModel):
    """What a diagnoser made of two runs.

    Attributes:
        source: Which diagnoser produced it, `agent` or `rule_based`.
        model: Model that answered, or the name of the rule.
        regressed_metrics: Metrics the diagnoser says regressed, most affected first.
        candidates: Causes, most likely first.
        next_check: The one thing to look at next.
        cost: What it cost.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    model: str = Field(min_length=1)
    regressed_metrics: tuple[str, ...] = ()
    candidates: tuple[Candidate, ...] = Field(min_length=1)
    next_check: str = ""
    cost: DiagnosisCost = DiagnosisCost()

    @property
    def causes(self) -> tuple[Cause, ...]:
        """The ranked causes alone, most likely first."""
        return tuple(candidate.cause for candidate in self.candidates)

    def rank_of(self, cause: Cause) -> int | None:
        """Where a cause appears in the ranking.

        Args:
            cause: The cause to look for.

        Returns:
            Its one based position, or `None` if it was not named.
        """
        causes = self.causes
        return causes.index(cause) + 1 if cause in causes else None


DIAGNOSIS_TOOL = ToolSchema(
    name="report_diagnosis",
    description=(
        "Report which metrics regressed, a ranked list of candidate causes with a "
        "confidence for each, and one suggested next check."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "regressed_metrics": {
                "type": "array",
                "description": "Metric names that regressed, most affected first.",
                "items": {"type": "string"},
            },
            "candidates": {
                "type": "array",
                "description": (f"Candidate causes, most likely first, at most {MAX_CANDIDATES}."),
                "items": {
                    "type": "object",
                    "properties": {
                        "cause": {
                            "type": "string",
                            "description": "The cause, from the fixed vocabulary.",
                            "enum": [cause.value for cause in Cause],
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Belief that this is the real cause, 0 to 1.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why, in one or two sentences.",
                        },
                    },
                    "required": ["cause", "confidence", "reasoning"],
                    "additionalProperties": False,
                },
            },
            "next_check": {
                "type": "string",
                "description": (
                    "The cheapest single check that would confirm or rule out the first candidate."
                ),
            },
        },
        "required": ["regressed_metrics", "candidates", "next_check"],
        "additionalProperties": False,
    },
)
"""The shape a diagnosis has to arrive in. Strict, so the API rejects a reply that does not
fit rather than this package parsing one out of prose."""


def diagnose(context: DiagnosisContext, client: Client, config: AgentConfig) -> Diagnosis:
    """Ask the model what went wrong.

    Args:
        context: The reduced comparison.
        client: Something that can answer one structured question.
        config: The settings the call is made under, which is also where the prices the
            cost is computed from live.

    Returns:
        The diagnosis.

    Raises:
        AgentError: If the model could not be reached or answered without calling the
            tool.
        ValidationError: If the reply fits the schema but is not a usable answer.
    """
    reply = client.call(
        system=SYSTEM_PROMPT.format(catalogue=cause_catalogue()),
        prompt=context.render(),
        tool=DIAGNOSIS_TOOL,
    )
    return Diagnosis(
        source=AGENT_SOURCE,
        model=reply.model,
        regressed_metrics=_metrics(reply.arguments.get("regressed_metrics")),
        candidates=_candidates(reply.arguments.get("candidates")),
        next_check=str(reply.arguments.get("next_check", "")),
        cost=DiagnosisCost(
            input_tokens=reply.usage.input_tokens,
            output_tokens=reply.usage.output_tokens,
            dollars=reply.usage.cost(config),
            attempts=reply.attempts,
        ),
    )


def rule_based_diagnosis(context: DiagnosisContext) -> Diagnosis:
    """Name whichever metric regressed most, and the cause that metric is usually about.

    No list is padded out to three. A diagnoser that saw one regression has one thing to
    say, and padding it with causes nobody had evidence for would buy a top three accuracy
    that the rule did not earn.

    Args:
        context: The reduced comparison.

    Returns:
        The diagnosis, costing nothing.
    """
    weights: dict[Cause, float] = {}
    metrics: list[str] = []
    for metric in context.regressed_metrics:
        cause = METRIC_CAUSES.get(metric)
        if cause is None:
            continue
        metrics.append(metric)
        worst = max(
            (
                abs(change.relative)
                for change in context.regressions
                if _metric_of(change) == metric
            ),
            default=0.0,
        )
        weights[cause] = max(weights.get(cause, 0.0), worst)

    if not weights:
        return Diagnosis(
            source=RULE_SOURCE,
            model="worst regressed metric",
            regressed_metrics=(),
            candidates=(
                Candidate(
                    cause=Cause.NO_REGRESSION,
                    confidence=1.0,
                    reasoning="No metric with a known cause regressed beyond the threshold.",
                ),
            ),
            next_check="Nothing to check: no metric regressed.",
        )

    ranked = sorted(weights, key=lambda cause: -weights[cause])[:MAX_CANDIDATES]
    total = sum(weights[cause] for cause in ranked) or 1.0
    return Diagnosis(
        source=RULE_SOURCE,
        model="worst regressed metric",
        regressed_metrics=tuple(metrics),
        candidates=tuple(
            Candidate(
                cause=cause,
                confidence=min(weights[cause] / total, 1.0),
                reasoning=f"The metric this cause is usually about regressed by "
                f"{weights[cause]:.1%}.",
            )
            for cause in ranked
        ),
        next_check=f"Look at the {metrics[0]} metric, which regressed most.",
    )


def _metric_of(change: ScalarChange) -> str:
    """The metric part of a scalar name."""
    return change.scalar.split(".", 1)[0]


def _metrics(raw: Any) -> tuple[str, ...]:  # noqa: ANN401 - straight off a JSON payload
    """Read the regressed metric names out of a reply."""
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValidationError(
            f"regressed_metrics must be a list of names, got {type(raw).__name__}"
        )
    return tuple(str(item) for item in raw)


def _candidates(raw: Any) -> tuple[Candidate, ...]:  # noqa: ANN401 - straight off a JSON payload
    """Read the ranked causes out of a reply, refusing one that cannot be scored."""
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes) or not raw:
        raise ValidationError("a diagnosis must name at least one candidate cause")
    candidates: list[Candidate] = []
    seen: set[Cause] = set()
    for index, entry in enumerate(raw[:MAX_CANDIDATES], start=1):
        if not isinstance(entry, Mapping):
            raise ValidationError(f"candidate {index} is not an object")
        try:
            cause = Cause(str(entry.get("cause")))
        except ValueError as error:
            raise ValidationError(
                f"candidate {index} names {entry.get('cause')!r}, which is not one of the "
                f"causes a diagnosis may choose from"
            ) from error
        # A ranking that names the same cause twice has two positions for one answer, and
        # a top three accuracy computed over it would be measuring the repetition.
        if cause in seen:
            raise ValidationError(f"the diagnosis names {cause.value!r} more than once")
        seen.add(cause)
        confidence = entry.get("confidence")
        if not isinstance(confidence, int | float) or isinstance(confidence, bool):
            raise ValidationError(f"candidate {index} has a non numeric confidence")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValidationError(
                f"candidate {index} states a confidence of {confidence}, which is not a probability"
            )
        candidates.append(
            Candidate(
                cause=cause,
                confidence=float(confidence),
                reasoning=str(entry.get("reasoning", "")),
            )
        )
    return tuple(candidates)
