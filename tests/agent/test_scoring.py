from __future__ import annotations

from collections.abc import Sequence

import pytest

from nnphysics.agent.causes import Cause
from nnphysics.agent.client import AgentConfig
from nnphysics.agent.diagnose import RULE_SOURCE, Candidate, Diagnosis, DiagnosisCost
from nnphysics.agent.faults import FAULTS, fault
from nnphysics.agent.scoring import SuiteReport, render_report, score_card, score_fault


def _diagnosis(causes: Sequence[Cause], *, cost: DiagnosisCost | None = None) -> Diagnosis:
    return Diagnosis(
        source=RULE_SOURCE,
        model="test",
        regressed_metrics=("rollout_error",),
        candidates=tuple(
            Candidate(cause=cause, confidence=1.0 / (index + 1), reasoning="because")
            for index, cause in enumerate(causes)
        ),
        next_check="look at it",
        cost=cost or DiagnosisCost(),
    )


class TestScoreFault:
    def test_the_true_cause_first_is_rank_one(self) -> None:
        injected = fault("no_curriculum")

        outcome = score_fault(injected, _diagnosis([injected.cause, Cause.LEARNING_RATE]))

        assert outcome.rank == 1
        assert outcome.correct

    def test_the_true_cause_third_is_rank_three(self) -> None:
        injected = fault("no_curriculum")

        outcome = score_fault(
            injected, _diagnosis([Cause.LEARNING_RATE, Cause.MODEL_CAPACITY, injected.cause])
        )

        assert outcome.rank == 3
        assert not outcome.correct
        assert outcome.within(3)

    def test_a_cause_never_named_has_no_rank(self) -> None:
        """Counted as not found rather than as a rank past the end of the list."""
        injected = fault("no_curriculum")

        outcome = score_fault(injected, _diagnosis([Cause.LEARNING_RATE]))

        assert outcome.rank is None
        assert not outcome.within(3)
        assert outcome.confidence == 0.0

    def test_the_confidence_in_the_truth_is_kept_beside_the_rank(self) -> None:
        injected = fault("no_curriculum")

        outcome = score_fault(injected, _diagnosis([Cause.LEARNING_RATE, injected.cause]))

        assert outcome.confidence == pytest.approx(0.5)
        assert outcome.top_confidence == pytest.approx(1.0)


class TestScoreCard:
    def test_a_diagnoser_that_is_always_right_scores_one(self) -> None:
        card = score_card(
            "perfect",
            "test",
            [score_fault(entry, _diagnosis([entry.cause])) for entry in FAULTS],
        )

        assert card.accuracy(1) == 1.0
        assert card.accuracy(3) == 1.0
        assert card.mean_rank == 1.0

    def test_a_diagnoser_that_is_always_wrong_scores_zero(self) -> None:
        """The sentinel. A score that cannot reach zero is not a score."""
        wrong = Cause.NO_REGRESSION
        card = score_card(
            "useless",
            "test",
            [
                score_fault(entry, _diagnosis([wrong]))
                for entry in FAULTS
                if entry.cause is not wrong
            ],
        )

        assert card.accuracy(1) == 0.0
        assert card.accuracy(3) == 0.0
        assert card.mean_rank is None
        assert card.found == 0

    def test_naming_a_fixed_list_cannot_score_well(self) -> None:
        """The reason the vocabulary has more members than the fault set.

        A diagnoser that ignores the evidence and always returns the same three causes gets at most
        three of the seven, because no two faults share a cause.
        """
        fixed = [Cause.LEARNING_RATE, Cause.MODEL_SYMMETRY, Cause.DATASET_SIZE]
        card = score_card(
            "fixed", "test", [score_fault(entry, _diagnosis(fixed)) for entry in FAULTS]
        )

        assert card.accuracy(1) <= 1 / len(FAULTS)
        assert card.accuracy(3) <= 3 / len(FAULTS)

    def test_the_mean_rank_ignores_the_misses(self) -> None:
        """An invented rank would flatter the diagnoser that offered fewest causes."""
        outcomes = [
            score_fault(FAULTS[0], _diagnosis([FAULTS[0].cause])),
            score_fault(FAULTS[1], _diagnosis([Cause.DATASET_SIZE])),
        ]

        card = score_card("partial", "test", outcomes)

        assert card.mean_rank == 1.0
        assert card.found == 1

    def test_cost_is_summed_and_averaged(self) -> None:
        cost = DiagnosisCost(input_tokens=100, output_tokens=20, dollars=0.001, attempts=1)
        card = score_card(
            "priced",
            "test",
            [score_fault(entry, _diagnosis([entry.cause], cost=cost)) for entry in FAULTS],
        )

        assert card.cost.input_tokens == 100 * len(FAULTS)
        assert card.dollars_per_diagnosis == pytest.approx(0.001)


class TestRenderReport:
    def _report(self) -> SuiteReport:
        card = score_card(
            "rule_based",
            "worst regressed metric",
            [score_fault(entry, _diagnosis([entry.cause])) for entry in FAULTS],
        )
        return SuiteReport(
            created="2026-01-01T00:00:00+00:00",
            system="nbody",
            baseline_run="0123456789abcdef",
            agent=AgentConfig(model="claude-recorded-1"),
            cards=(card,),
        )

    def test_rendering_twice_gives_the_same_bytes(self) -> None:
        report = self._report()

        assert render_report(report) == render_report(report)

    def test_every_fault_has_a_row(self) -> None:
        text = render_report(self._report())

        for entry in FAULTS:
            assert f"`{entry.name}`" in text

    def test_the_accuracies_are_stated(self) -> None:
        text = render_report(self._report())

        assert "top 1" in text
        assert "top 3" in text

    def test_the_cost_per_diagnosis_is_stated(self) -> None:
        """A table without it cannot say whether the agent is worth what it costs."""
        assert "cost per diagnosis" in render_report(self._report())

    def test_the_report_is_json_serialisable(self) -> None:
        payload = self._report().model_dump_json()

        assert "rule_based" in payload
