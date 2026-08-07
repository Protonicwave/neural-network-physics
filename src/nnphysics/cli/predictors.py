"""Where a trained predictor meets the harness.

The evaluation layer may not import the models layer, and it does not. It is handed a
factory instead, and building that factory is a command line concern because the command
line is the only place that knows a run has weights on disk. Three commands need the same
adapter, so it lives here rather than in whichever of them was written first.

Nothing is added to any registry. One run's weights cannot become visible to the next.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nnphysics.core.errors import ValidationError
from nnphysics.models import load_model
from nnphysics.reporting.layout import run_paths

if TYPE_CHECKING:
    from pathlib import Path

    from nnphysics.core.config import RunConfig
    from nnphysics.core.protocols import Predictor
    from nnphysics.evals.predictors import PredictorContext, PredictorParameters
    from nnphysics.models import SurrogateModel

__all__ = [
    "CHECKPOINT_DIR",
    "ModelFactory",
    "member_checkpoint",
    "trained_members",
]

CHECKPOINT_DIR = "checkpoints"
"""Where the best and last checkpoints of a run live, inside its run directory."""

_BEST = "best.pt"
"""The checkpoint a run is scored on: the epoch with the lowest validation rollout."""

_INTERVAL_ATOL = 1e-9
"""Relative agreement required between a predictor's interval and the stored one."""


class ModelFactory:
    """Hands the harness one already loaded predictor, whatever case it asks for.

    A predictor factory normally builds something from the context. This one ignores the
    context apart from checking that the predictor and the reference solver step by the
    same interval, because a surrogate scored over a different interval from the truth it
    is compared against would be measured against the wrong states.

    Args:
        predictor: The trained model, or an ensemble of them.
    """

    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor

    @property
    def name(self) -> str:
        """What the harness will report the predictor under."""
        return self._predictor.name

    def __call__(self, context: PredictorContext, parameters: PredictorParameters) -> Predictor:
        """Return the predictor, once the interval has been checked.

        Args:
            context: What the harness offers a factory.
            parameters: None accepted: a trained predictor's settings are in its
                checkpoint.

        Returns:
            The predictor.

        Raises:
            ValidationError: If a parameter is given, or the intervals disagree.
        """
        if parameters:
            raise ValidationError(
                f"the trained predictor {self._predictor.name!r} takes no parameters, got "
                f"{sorted(parameters)}"
            )
        if abs(self._predictor.dt - context.reference.dt) > _INTERVAL_ATOL * context.reference.dt:
            raise ValidationError(
                f"predictor {self._predictor.name!r} steps by {self._predictor.dt:g} but "
                f"the dataset stores every {context.reference.dt:g}"
            )
        return self._predictor


def member_checkpoint(config: RunConfig) -> Path:
    """The best checkpoint of one run, which for an ensemble is one member.

    Args:
        config: The resolved configuration of that run, at its own member index.

    Returns:
        The path, which may not exist yet.
    """
    return run_paths(config).root / CHECKPOINT_DIR / _BEST


def trained_members(config: RunConfig) -> tuple[SurrogateModel, ...]:
    """Load every ensemble member of a configuration that has been trained.

    Returned in member order and without the ones that are missing, so a benchmark run
    partway through training an ensemble reports what exists rather than refusing. How
    many were found is the caller's to state.

    Args:
        config: The resolved run configuration, at any member index.

    Returns:
        The loaded members.

    Raises:
        ConfigurationError: If a checkpoint that exists cannot be read.
    """
    return tuple(
        load_model(path)
        for index in range(config.ensemble.members)
        if (path := member_checkpoint(config.for_member(index))).is_file()
    )
