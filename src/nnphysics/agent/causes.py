"""The closed vocabulary of things that can be wrong with a run.

Scoring a diagnosis means deciding whether the agent named the right cause, and deciding
that from free text is a judgement call the scorer would have to make. So the causes are
an enumeration, the tool schema only accepts a member of it, and the score is an exact
match. That is a real simplification of the problem and it is stated in the results rather
than buried: an agent choosing from twelve labels is doing an easier job than one writing
a paragraph, and the rule based baseline is given exactly the same twelve.

Five of the twelve are distractors, named by nothing in the fault set. Without them the
list would be the answer key and a diagnoser that guessed uniformly would score well
enough to be indistinguishable from one that read the numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

__all__ = ["CAUSE_DESCRIPTIONS", "Cause", "cause_catalogue", "describe_cause"]


class Cause(StrEnum):
    """Something that can be wrong with a run, as a diagnosis may name it."""

    NORMALISATION_STATISTICS = "normalisation_statistics"
    MODEL_SYMMETRY = "model_symmetry"
    LEARNING_RATE = "learning_rate"
    TRAINING_REGIME = "training_regime"
    ROLLOUT_CURRICULUM = "rollout_curriculum"
    INTEGRATOR_STEP_SIZE = "integrator_step_size"
    OPTIMISER_STATE = "optimiser_state"
    MODEL_CAPACITY = "model_capacity"
    DATASET_SIZE = "dataset_size"
    EVALUATION_HORIZON = "evaluation_horizon"
    RANDOM_SEED = "random_seed"
    NO_REGRESSION = "no_regression"


CAUSE_DESCRIPTIONS: Mapping[Cause, str] = {
    Cause.NORMALISATION_STATISTICS: (
        "The model was trained or run with normalisation statistics that do not describe "
        "its data, so every input it sees is centred or scaled wrongly."
    ),
    Cause.MODEL_SYMMETRY: (
        "The model no longer commutes with a transformation the system declares, so "
        "transforming the initial condition and predicting give different answers from "
        "predicting and then transforming."
    ),
    Cause.LEARNING_RATE: (
        "The learning rate is far too high, so the optimiser steps past the minimum and "
        "the training loss stalls, oscillates or diverges."
    ),
    Cause.TRAINING_REGIME: (
        "The training data was drawn from a different regime from the one being "
        "evaluated, so the model is being asked about states it never saw."
    ),
    Cause.ROLLOUT_CURRICULUM: (
        "The rollout curriculum was disabled, so the model was only ever trained to "
        "predict one step and nothing taught it to stay stable over many."
    ),
    Cause.INTEGRATOR_STEP_SIZE: (
        "The reference solver takes too few substeps per stored interval, so the ground "
        "truth it produces is itself inaccurate or unstable."
    ),
    Cause.OPTIMISER_STATE: (
        "Training resumed from a checkpoint without the optimiser state, so the moment "
        "estimates were lost and the run took different steps from an uninterrupted one."
    ),
    Cause.MODEL_CAPACITY: (
        "The model has too few or too many parameters for the problem, so it underfits "
        "or overfits regardless of how it was trained."
    ),
    Cause.DATASET_SIZE: (
        "There are too few trajectories or too few states per trajectory for the model to "
        "learn the dynamics from."
    ),
    Cause.EVALUATION_HORIZON: (
        "The evaluation rolls out over a different horizon or from a different number of "
        "initial conditions, so the two runs are not answering the same question."
    ),
    Cause.RANDOM_SEED: (
        "Nothing is wrong beyond the run seed, so the difference is initialisation and "
        "shuffling rather than a fault."
    ),
    Cause.NO_REGRESSION: (
        "Nothing regressed. The two runs differ only in ways that do not make the second one worse."
    ),
}
"""One line per cause, shown to the diagnoser so that it chooses from descriptions rather
than from bare identifiers."""


def describe_cause(cause: Cause) -> str:
    """Say in one line what a cause means.

    Args:
        cause: The cause.

    Returns:
        The description.
    """
    return CAUSE_DESCRIPTIONS[cause]


def cause_catalogue() -> str:
    """The whole vocabulary as a block of text, for a prompt.

    Returns:
        One line per cause, in declaration order.
    """
    return "\n".join(f"- {cause.value}: {CAUSE_DESCRIPTIONS[cause]}" for cause in Cause)
