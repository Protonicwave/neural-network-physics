"""The one test that talks to the real API.

Everything else about the agent is tested against a recorded response, so the suite passes
with no network and no credential. This is the test that would notice the thing a
recording cannot: that the request shape is still one the API accepts, that a strict tool
schema is still enforced the way this package assumes, and that the model still answers
with a cause from the vocabulary rather than a paragraph.

Marked `integration`, so it is excluded from the default run and from CI. It costs money
to run, which is the other reason it is not in the default run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nnphysics.agent.causes import Cause
from nnphysics.agent.client import API_KEY_VARIABLES, AnthropicClient, load_agent_config
from nnphysics.agent.context import (
    ConfigDifference,
    CurveShape,
    DiagnosisContext,
    ScalarChange,
    TrainingShape,
)
from nnphysics.agent.diagnose import diagnose

pytestmark = pytest.mark.integration

AGENT_CONFIG = Path("configs/agent.yaml")


def _context() -> DiagnosisContext:
    """A curriculum that was switched off, described the way the reduction describes one.

    Built here rather than from a run record so that the test needs no dataset and no
    trained model. What it is exercising is the call, not the reduction, and the reduction
    has its own tests.
    """
    return DiagnosisContext(
        system="nbody",
        suite="standard",
        baseline_id="0123456789abcdef",
        candidate_id="fedcba9876543210",
        baseline_name="nbody",
        candidate_name="nbody",
        threshold=0.05,
        config_differences=(
            ConfigDifference(path="training.curriculum", baseline="[1, 4, 8]", candidate="[1]"),
            ConfigDifference(
                path="training.curriculum_epochs", baseline="[0, 3, 6]", candidate="[0]"
            ),
        ),
        config_differences_dropped=0,
        regressions=(
            ScalarChange(
                split="test",
                predictor="graph",
                scalar="rollout_error.error.final",
                baseline=0.42,
                candidate=8.9,
                relative=20.2,
                direction="lower is better",
            ),
            ScalarChange(
                split="test",
                predictor="graph",
                scalar="rollout_error.error.max",
                baseline=0.44,
                candidate=9.1,
                relative=19.7,
                direction="lower is better",
            ),
            ScalarChange(
                split="held_out",
                predictor="graph",
                scalar="invariant_drift.worst_violation",
                baseline=1.2e9,
                candidate=4.5e10,
                relative=36.5,
                direction="lower is better",
            ),
        ),
        regressions_dropped=0,
        improvements=(),
        unchanged=120,
        undirected=64,
        curves=(
            CurveShape(
                split="test",
                predictor="graph",
                first=0.05,
                final=8.9,
                worst=8.9,
                worst_position=1.0,
                rising_fraction=0.98,
                growth=178.0,
            ),
        ),
        baseline_training=TrainingShape(
            model="graph",
            n_parameters=62849,
            epochs=40,
            best_epoch=38,
            best_validation_error=0.0785,
            stopped_early=False,
            first_loss=0.31,
            final_loss=0.041,
            first_validation_error=0.1388,
            final_validation_error=0.0785,
            peak_learning_rate=0.002,
            max_gradient_norm=1.4,
            curriculum_stages=(1, 4, 8),
            non_finite_epochs=0,
        ),
        candidate_training=TrainingShape(
            model="graph",
            n_parameters=62849,
            epochs=40,
            best_epoch=37,
            best_validation_error=0.1301,
            stopped_early=False,
            first_loss=0.31,
            final_loss=0.038,
            first_validation_error=0.1388,
            final_validation_error=0.1301,
            peak_learning_rate=0.002,
            max_gradient_norm=1.3,
            curriculum_stages=(1,),
            non_finite_epochs=0,
        ),
        rollouts=(),
    )


@pytest.fixture(scope="module")
def credential() -> None:
    if not any(os.environ.get(name) for name in API_KEY_VARIABLES):
        pytest.skip(f"set one of {', '.join(API_KEY_VARIABLES)} to run the live API test")
    if not AGENT_CONFIG.is_file():
        pytest.skip(f"no agent configuration at {AGENT_CONFIG}")


def test_the_live_api_returns_a_diagnosis_this_package_can_read(credential: None) -> None:  # noqa: ARG001
    """The request is accepted, the tool is called, and the answer validates."""
    config = load_agent_config(AGENT_CONFIG, os.environ)
    client = AnthropicClient(config, os.environ)

    result = diagnose(_context(), client, config)

    assert result.candidates
    assert all(candidate.cause in set(Cause) for candidate in result.candidates)
    assert result.next_check
    assert result.cost.input_tokens > 0
    assert result.cost.output_tokens > 0


def test_the_live_api_finds_the_cause_this_context_was_built_around(credential: None) -> None:  # noqa: ARG001
    """A weak assertion on purpose.

    It says the call works end to end on an obvious case, not that the agent is accurate: `nnp
    diagnose score` is what measures accuracy, and a test that asserted a score would be an
    evaluation pretending to be a test.
    """
    config = load_agent_config(AGENT_CONFIG, os.environ)

    result = diagnose(_context(), AnthropicClient(config, os.environ), config)

    assert Cause.ROLLOUT_CURRICULUM in result.causes
