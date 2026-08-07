from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from nnphysics.agent.causes import Cause, cause_catalogue
from nnphysics.agent.client import AgentConfig, RecordedClient, Reply, Usage
from nnphysics.agent.context import build_context
from nnphysics.agent.diagnose import (
    AGENT_SOURCE,
    DIAGNOSIS_TOOL,
    MAX_CANDIDATES,
    METRIC_CAUSES,
    RULE_SOURCE,
    SYSTEM_PROMPT,
    diagnose,
    rule_based_diagnosis,
)
from nnphysics.core.errors import ValidationError
from nnphysics.reporting.record import RunRecord

CONFIG = AgentConfig(model="claude-recorded-1", input_price=5.0, output_price=25.0)


def _reply(arguments: dict[str, Any], *, tokens: tuple[int, int] = (100, 50)) -> Reply:
    return Reply(
        arguments=arguments,
        usage=Usage(input_tokens=tokens[0], output_tokens=tokens[1]),
        model="claude-recorded-1",
        attempts=1,
    )


def _candidate(cause: str, confidence: float = 0.5) -> dict[str, Any]:
    return {"cause": cause, "confidence": confidence, "reasoning": "because"}


class TestTool:
    def test_the_schema_is_strict(self) -> None:
        """Strict is what makes the API reject a reply of the wrong shape."""
        schema = DIAGNOSIS_TOOL.input_schema

        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"regressed_metrics", "candidates", "next_check"}

    def test_the_cause_field_only_accepts_the_vocabulary(self) -> None:
        allowed = DIAGNOSIS_TOOL.input_schema["properties"]["candidates"]["items"]["properties"][
            "cause"
        ]["enum"]

        assert set(allowed) == {cause.value for cause in Cause}

    def test_every_cause_appears_in_the_system_prompt(self) -> None:
        """A vocabulary the model is scored against and never shown is a trick question."""
        prompt = SYSTEM_PROMPT.format(catalogue=cause_catalogue())

        for cause in Cause:
            assert cause.value in prompt


class TestDiagnose:
    def test_a_recorded_reply_becomes_a_ranked_diagnosis(
        self, make_record: Callable[..., RunRecord], recorded_reply: Callable[..., Reply]
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))
        client = RecordedClient([recorded_reply()])

        result = diagnose(context, client, CONFIG)

        assert result.source == AGENT_SOURCE
        assert result.causes[0] is Cause.ROLLOUT_CURRICULUM
        assert result.next_check

    def test_the_cost_is_measured_and_recorded(
        self, make_record: Callable[..., RunRecord], recorded_reply: Callable[..., Reply]
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))

        result = diagnose(context, RecordedClient([recorded_reply()]), CONFIG)

        assert result.cost.input_tokens == 2431
        assert result.cost.output_tokens == 318
        assert result.cost.dollars == pytest.approx(2431 * 5.0e-6 + 318 * 25.0e-6)

    def test_the_rendered_context_is_what_is_sent(
        self, make_record: Callable[..., RunRecord], recorded_reply: Callable[..., Reply]
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))
        client = RecordedClient([recorded_reply()])

        diagnose(context, client, CONFIG)

        assert client.calls[0][1] == context.render()

    def test_an_unknown_cause_is_refused(self, make_record: Callable[..., RunRecord]) -> None:
        """The score is an exact match against a fixed vocabulary.

        A cause outside it could never be marked right or wrong.
        """
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))
        client = RecordedClient(
            [
                _reply(
                    {
                        "candidates": [_candidate("gremlins")],
                        "next_check": "",
                        "regressed_metrics": [],
                    }
                )
            ]
        )

        with pytest.raises(ValidationError, match="not one of the causes"):
            diagnose(context, client, CONFIG)

    def test_a_repeated_cause_is_refused(self, make_record: Callable[..., RunRecord]) -> None:
        """Two positions for one answer would make a top three accuracy measure repetition."""
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))
        client = RecordedClient(
            [
                _reply(
                    {
                        "candidates": [_candidate("learning_rate"), _candidate("learning_rate")],
                        "next_check": "",
                        "regressed_metrics": [],
                    }
                )
            ]
        )

        with pytest.raises(ValidationError, match="more than once"):
            diagnose(context, client, CONFIG)

    @pytest.mark.parametrize("confidence", [-0.1, 1.5])
    def test_a_confidence_that_is_not_a_probability_is_refused(
        self, make_record: Callable[..., RunRecord], confidence: float
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))
        client = RecordedClient(
            [
                _reply(
                    {
                        "candidates": [_candidate("learning_rate", confidence)],
                        "next_check": "",
                        "regressed_metrics": [],
                    }
                )
            ]
        )

        with pytest.raises(ValidationError, match="probability"):
            diagnose(context, client, CONFIG)

    def test_an_empty_ranking_is_refused(self, make_record: Callable[..., RunRecord]) -> None:
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))
        client = RecordedClient(
            [_reply({"candidates": [], "next_check": "", "regressed_metrics": []})]
        )

        with pytest.raises(ValidationError, match="at least one candidate"):
            diagnose(context, client, CONFIG)

    def test_a_longer_ranking_than_allowed_is_truncated(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        """A strict schema cannot bound an array's length, so the bound is applied here."""
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))
        client = RecordedClient(
            [
                _reply(
                    {
                        "candidates": [_candidate(cause.value) for cause in list(Cause)],
                        "next_check": "",
                        "regressed_metrics": [],
                    }
                )
            ]
        )

        assert len(diagnose(context, client, CONFIG).candidates) == MAX_CANDIDATES


class TestRuleBased:
    def test_it_names_the_cause_of_the_metric_that_regressed_most(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))

        result = rule_based_diagnosis(context)

        assert result.source == RULE_SOURCE
        assert result.causes[0] is METRIC_CAUSES[context.regressed_metrics[0]]

    def test_it_costs_nothing_and_calls_nothing(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        """Which is why the comparison is available to someone with no credential."""
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))

        result = rule_based_diagnosis(context)

        assert result.cost.dollars == 0.0
        assert result.cost.attempts == 0

    def test_nothing_regressed_is_answered_rather_than_guessed(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1"))

        result = rule_based_diagnosis(context)

        assert result.causes == (Cause.NO_REGRESSION,)

    def test_the_ranking_is_not_padded_out_to_three(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        """Padding would buy a top three accuracy the rule did not earn."""
        context = build_context(make_record(), make_record(run_id="1"))

        assert len(rule_based_diagnosis(context).candidates) == 1

    def test_wall_clock_is_not_allowed_to_decide_the_ranking(self) -> None:
        """Wall clock moves by a factor of two between runs, so it must not rank anything."""
        assert "timing" not in METRIC_CAUSES
