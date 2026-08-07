"""What a metric is built from, and the registry that names them.

A metric scores a rollout, and a rollout carries only two trajectories. Anything else a
metric needs, the invariants a system declares, the symmetries it declares, and the
predictor itself for the metric that has to run it again, arrives at construction time in
a context the runner assembles. That is what keeps the metrics free of any knowledge of
which system they are looking at: the system declares, the runner passes on, the metric
reads.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.core.registry import Registry

if TYPE_CHECKING:
    from nnphysics.core.protocols import Invariant, Metric, Predictor, Refinement, Symmetry
    from nnphysics.core.types import FloatArray, Trajectory

__all__ = [
    "DEFAULT_ERROR_THRESHOLDS",
    "METRICS",
    "MetricContext",
    "MetricFactory",
    "build_metrics",
    "relative_error",
]

DEFAULT_ERROR_THRESHOLDS = (0.01, 0.1, 1.0)
"""Normalised error levels a horizon is reported for: one per cent, ten per cent, and
error as large as the signal itself."""

_NORM_FLOOR = 1.0e-12
"""Relative to a trajectory's own largest norm, below which a reference state is treated
as carrying no scale to normalise by."""


@dataclass(frozen=True, slots=True)
class MetricContext:
    """Everything a metric factory is allowed to know.

    Attributes:
        invariants: Quantities the system declares for the regime being evaluated.
        symmetries: Transformations the system declares its dynamics commute with.
        refinements: Finer resolutions the system declares its states can be expressed
            at. Empty for a system whose state is not a discretisation of anything.
        predictor: The predictor being scored, for metrics that must roll it out again.
        thresholds: Normalised error levels a horizon is reported for.
        symmetry_steps: Steps rolled out when testing equivariance. Shorter than the main
            rollout by default, because each symmetry costs another rollout and the
            violation shows up early or not at all.
        resolution_steps: Steps rolled out when testing resolution generalisation.
            Shorter again: a refined rollout works on a state several times the size, and
            a predictor that has stopped meaning the same thing on a finer grid shows it
            within a few steps.
        distribution_window: Fraction of the rollout, taken from its end, that
            distributional statistics are computed over.
        divergence_factor: Passed to any rollout a metric drives itself, so that a metric
            cannot produce a trajectory the main driver would have abandoned.
    """

    invariants: tuple[Invariant, ...] = ()
    symmetries: tuple[Symmetry, ...] = ()
    refinements: tuple[Refinement, ...] = ()
    predictor: Predictor | None = None
    thresholds: tuple[float, ...] = DEFAULT_ERROR_THRESHOLDS
    symmetry_steps: int = 32
    resolution_steps: int = 16
    distribution_window: float = 0.25
    divergence_factor: float = 1.0e3

    def __post_init__(self) -> None:
        if any(threshold <= 0.0 for threshold in self.thresholds):
            raise ValidationError(f"error thresholds must be positive, got {self.thresholds}")
        if self.symmetry_steps < 1:
            raise ValidationError(
                f"a symmetry test needs at least one step, got {self.symmetry_steps}"
            )
        if self.resolution_steps < 1:
            raise ValidationError(
                f"a resolution test needs at least one step, got {self.resolution_steps}"
            )
        if not 0.0 < self.distribution_window <= 1.0:
            raise ValidationError(
                f"the distribution window must be a fraction in (0, 1], got "
                f"{self.distribution_window}"
            )

    def require_predictor(self, metric: str) -> Predictor:
        """Return the predictor being scored, failing loudly if there is none.

        Args:
            metric: Name of the metric asking, used in the error message.

        Returns:
            The predictor.

        Raises:
            ValidationError: If the context carries no predictor.
        """
        if self.predictor is None:
            raise ValidationError(f"metric {metric!r} needs the predictor it is scoring")
        return self.predictor


type MetricFactory = Callable[[MetricContext], Metric]
"""Builds one metric from the context the runner assembled."""

METRICS: Registry[MetricFactory] = Registry("metric")
"""Every registered metric factory. Importing `nnphysics.evals` fills it."""


def build_metrics(names: Sequence[str], context: MetricContext) -> tuple[Metric, ...]:
    """Resolve metric names against the registry and build them.

    Args:
        names: Registered metric names, in the order they should be reported.
        context: What the factories are allowed to know.

    Returns:
        The metrics.

    Raises:
        UnknownNameError: If a name is not registered.
        ValidationError: If no names were given or a name appears twice.
    """
    if not names:
        raise ValidationError("a suite must name at least one metric")
    duplicates = sorted({name for name in names if list(names).count(name) > 1})
    if duplicates:
        raise ValidationError(f"a suite names the same metric more than once: {duplicates}")
    return tuple(METRICS.get(name)(context) for name in names)


def relative_error(
    predicted: Trajectory, reference: Trajectory
) -> tuple[dict[str, FloatArray], FloatArray]:
    """Normalised error between two trajectories, per field and aggregated.

    The error at each step is normalised by the size of the reference state at that step,
    so a value of one means the predictor is as wrong as the signal is large, whatever
    the field's units are. Fields are combined as a root mean square, which weights them
    equally: nothing here knows which of them matters most, and a weighting would be
    exactly the kind of system knowledge this layer must not carry.

    Args:
        predicted: The predictor's trajectory.
        reference: Ground truth over the same steps.

    Returns:
        Field name to its normalised error per step, and the aggregate per step. Both
        have one entry per step, the initial state included.

    Raises:
        ValidationError: If the trajectories differ in length or in fields.
    """
    if len(predicted) != len(reference):
        raise ValidationError(
            f"cannot compare trajectories of {len(predicted)} and {len(reference)} steps"
        )
    if set(predicted.names) != set(reference.names):
        raise ValidationError(
            f"cannot compare trajectories with fields {predicted.names} and {reference.names}"
        )
    per_field = {
        name: _field_error(predicted.fields[name], reference.fields[name])
        for name in sorted(reference.names)
    }
    stacked = np.stack([per_field[name] for name in sorted(per_field)])
    aggregate: FloatArray = np.sqrt(np.mean(stacked**2, axis=0))
    return per_field, aggregate


def _field_error(predicted: FloatArray, reference: FloatArray) -> FloatArray:
    """Normalised error of one field at every step."""
    axes = tuple(range(1, reference.ndim))
    difference: FloatArray = np.sqrt(np.sum((predicted - reference) ** 2, axis=axes))
    magnitude: FloatArray = np.sqrt(np.sum(reference**2, axis=axes))
    # A single floor across the whole trajectory rather than one per step: normalising a
    # step whose reference happens to pass through zero by that step's own size would
    # report a spike where the physics has none.
    floor = _NORM_FLOOR * max(float(np.max(magnitude)), 1.0)
    error: FloatArray = difference / np.maximum(magnitude, floor)
    return error
