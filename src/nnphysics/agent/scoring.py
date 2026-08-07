"""Marking a diagnosis against the fault that was actually injected.

The score is a rank. For each fault the true cause is known, the diagnoser returned a
ranked list, and the only question is where in that list the true cause appears. Top one
accuracy is the share of faults it was first; top three is the share it was in the first
three. Both are reported because they answer different questions: top one is whether the
agent can be trusted to act on, top three is whether it is worth reading.

A cause that was never named has no rank, and it is counted as not found rather than as a
rank past the end of the list. Averaging a missing rank in as some large number would make
a diagnoser that returns nothing look better than one that returns three wrong answers.

Nothing here knows how a diagnosis was produced. The agent and the rule based baseline are
scored by the same function against the same faults, which is what makes the comparison
between them mean anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from nnphysics.agent.causes import Cause
from nnphysics.agent.client import AgentConfig
from nnphysics.agent.diagnose import RULE_SOURCE, DiagnosisCost

if TYPE_CHECKING:
    from nnphysics.agent.diagnose import Diagnosis
    from nnphysics.agent.faults import Fault

__all__ = [
    "TOP_K",
    "FaultOutcome",
    "ScoreCard",
    "SuiteReport",
    "render_report",
    "score_card",
    "score_fault",
]

TOP_K = (1, 3)
"""The positions accuracy is reported at."""


class FaultOutcome(BaseModel):
    """One diagnoser's answer to one injected fault.

    Attributes:
        fault: Name of the fault.
        true_cause: What was actually broken.
        named: Causes the diagnoser ranked, most likely first.
        rank: One based position of the true cause, or `None` if it was never named.
        confidence: Confidence stated for the true cause, or zero if it was never named.
        top_confidence: Confidence stated for whatever was ranked first.
        next_check: What the diagnoser suggested looking at next.
        cost: What this diagnosis cost.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fault: str = Field(min_length=1)
    true_cause: Cause
    named: tuple[Cause, ...] = Field(min_length=1)
    rank: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    top_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    next_check: str = ""
    cost: DiagnosisCost = DiagnosisCost()

    @property
    def correct(self) -> bool:
        """Whether the true cause was ranked first."""
        return self.rank == 1

    def within(self, k: int) -> bool:
        """Whether the true cause was in the first `k` positions.

        Args:
            k: How far down the ranking to look.

        Returns:
            Whether it was found that high.
        """
        return self.rank is not None and self.rank <= k


class ScoreCard(BaseModel):
    """One diagnoser over the whole fault set.

    Attributes:
        source: Which diagnoser, `agent` or `rule_based`.
        model: Model that answered, or the name of the rule.
        outcomes: One per fault, in the order the faults were run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    model: str = Field(min_length=1)
    outcomes: tuple[FaultOutcome, ...] = Field(min_length=1)

    def accuracy(self, k: int) -> float:
        """Share of faults whose true cause was in the first `k` positions.

        Args:
            k: How far down each ranking to look.

        Returns:
            The share, between zero and one.
        """
        return sum(1 for outcome in self.outcomes if outcome.within(k)) / len(self.outcomes)

    @property
    def found(self) -> int:
        """Faults whose true cause was named anywhere in the ranking."""
        return sum(1 for outcome in self.outcomes if outcome.rank is not None)

    @property
    def mean_rank(self) -> float | None:
        """Mean position of the true cause, over the faults where it was named at all.

        Returns:
            The mean, or `None` if it was never named. Deliberately not averaged over the
            misses: a rank invented for a cause nobody offered would flatter the diagnoser
            that offered fewest.
        """
        ranks = [outcome.rank for outcome in self.outcomes if outcome.rank is not None]
        return sum(ranks) / len(ranks) if ranks else None

    @property
    def cost(self) -> DiagnosisCost:
        """What the whole card cost, summed over its diagnoses."""
        return DiagnosisCost(
            input_tokens=sum(outcome.cost.input_tokens for outcome in self.outcomes),
            output_tokens=sum(outcome.cost.output_tokens for outcome in self.outcomes),
            dollars=sum(outcome.cost.dollars for outcome in self.outcomes),
            attempts=sum(outcome.cost.attempts for outcome in self.outcomes),
        )

    @property
    def dollars_per_diagnosis(self) -> float:
        """Mean cost of one diagnosis."""
        return self.cost.dollars / len(self.outcomes)


class SuiteReport(BaseModel):
    """The scored fault suite, as it is committed.

    Attributes:
        created: When the suite finished, as an ISO 8601 string. Supplied by the caller,
            so that everything except this field is reproducible.
        system: System the faults were injected into.
        baseline_run: Run identifier of the known good run every fault was compared to.
        agent: The settings the agent was called under, or `None` if it was not called.
        provenance: How this report was produced, in the reader's own terms. Required,
            because a committed table of accuracies that does not say what produced it is
            a number a reader has to take on trust.
        cards: One per diagnoser scored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    created: str = Field(min_length=1)
    system: str = Field(min_length=1)
    baseline_run: str = Field(min_length=1)
    agent: AgentConfig | None = None
    provenance: str = Field(min_length=1)
    cards: tuple[ScoreCard, ...] = Field(min_length=1)


def score_fault(injected: Fault, diagnosis: Diagnosis) -> FaultOutcome:
    """Mark one diagnosis against the fault that produced it.

    Args:
        injected: The fault that was injected, which carries the true cause.
        diagnosis: What the diagnoser said.

    Returns:
        The outcome.
    """
    rank = diagnosis.rank_of(injected.cause)
    confidence = next(
        (
            candidate.confidence
            for candidate in diagnosis.candidates
            if candidate.cause is injected.cause
        ),
        0.0,
    )
    return FaultOutcome(
        fault=injected.name,
        true_cause=injected.cause,
        named=diagnosis.causes,
        rank=rank,
        confidence=confidence,
        top_confidence=diagnosis.candidates[0].confidence,
        next_check=diagnosis.next_check,
        cost=diagnosis.cost,
    )


def score_card(source: str, model: str, outcomes: Sequence[FaultOutcome]) -> ScoreCard:
    """Collect one diagnoser's outcomes.

    Args:
        source: Which diagnoser, `agent` or `rule_based`.
        model: Model that answered, or the name of the rule.
        outcomes: One per fault.

    Returns:
        The card.
    """
    return ScoreCard(source=source, model=model, outcomes=tuple(outcomes))


def render_report(report: SuiteReport) -> str:
    """Render the scored suite as Markdown.

    Args:
        report: The scored suite.

    Returns:
        The document, ending in a newline. Deterministic apart from the timestamp the
        caller supplied, so rendering the same report twice gives the same bytes.
    """
    lines = [
        "# Diagnosis agent: scored fault injection",
        "",
        f"System `{report.system}`, baseline run `{report.baseline_run}`, "
        f"{len(report.cards[0].outcomes)} faults, generated {report.created}.",
        "",
        "Every fault is injected into a copy of the same known good run, the resulting "
        "record is compared against the baseline, and each diagnoser is asked to rank the "
        "causes. The true cause is known before the question is asked. Both diagnosers "
        "choose from the same twelve cause labels, five of which no fault uses.",
        "",
        "## How this was produced",
        "",
        report.provenance,
        "",
        "## Accuracy",
        "",
        "| Diagnoser | model | top 1 | top 3 | named at all | mean rank when named | "
        "cost per diagnosis |",
        "|---|---|---|---|---|---|---|",
    ]
    for card in report.cards:
        mean = card.mean_rank
        lines.append(
            f"| `{card.source}` | {card.model} | {card.accuracy(1):.0%} | "
            f"{card.accuracy(3):.0%} | {card.found} of {len(card.outcomes)} | "
            f"{'n/a' if mean is None else f'{mean:.2f}'} | "
            f"{_cost(card)} |"
        )
    lines += ["", "## Per fault", ""]
    for card in report.cards:
        lines += [
            f"### `{card.source}`",
            "",
            "| Fault | true cause | rank | first named | confidence in the truth |",
            "|---|---|---|---|---|",
        ]
        for outcome in card.outcomes:
            rank = "not named" if outcome.rank is None else str(outcome.rank)
            lines.append(
                f"| `{outcome.fault}` | `{outcome.true_cause.value}` | {rank} | "
                f"`{outcome.named[0].value}` | {outcome.confidence:.2f} |"
            )
        lines.append("")
    lines += ["## Cost", ""]
    for card in report.cards:
        total = card.cost
        lines.append(
            f"- `{card.source}`: {total.input_tokens} input and {total.output_tokens} "
            f"output tokens over the whole suite, {_cost(card)} per diagnosis."
        )
    lines += [
        "",
    ]
    return "\n".join(lines) + "\n"


def _cost(card: ScoreCard) -> str:
    """What one diagnosis cost, or that nobody measured it.

    A diagnoser that reported no tokens at all was not free, it was unmetered. The rule
    based one genuinely is free, and it says so in its own row: it is the zero token case
    that has to be distinguished from a zero dollar one, and `not measured` is what
    distinguishes them.
    """
    total = card.cost
    if not total.input_tokens and not total.output_tokens:
        return "free" if card.source == RULE_SOURCE else "not measured"
    return f"${card.dollars_per_diagnosis:.4f}"
