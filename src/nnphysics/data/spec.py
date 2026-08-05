"""What a trajectory is asked for, and how one is named and seeded.

The distinction this module exists to carry is between the step the solver takes and
the interval that is stored. A spectral solver needs a step small enough to stay stable
and a symplectic one needs a step small enough to stay accurate, but a surrogate is not
obliged to inherit either. It learns the stored interval, and the ratio between the two
is where its speedup comes from, so both are recorded rather than one being inferred.
"""

from __future__ import annotations

from dataclasses import dataclass

from nnphysics.core.errors import ValidationError

__all__ = [
    "SEED_STREAM_TEMPLATE",
    "TrajectorySpec",
    "trajectory_id",
    "trajectory_stream",
]

SEED_STREAM_TEMPLATE = "data.trajectory.{system}.{regime}.{index}"
"""How a trajectory's seed stream is derived. Recorded in the manifest so that a reader
can re-derive any trajectory without reading this source."""


@dataclass(frozen=True, slots=True)
class TrajectorySpec:
    """How long a trajectory runs and how finely it is stored.

    Attributes:
        n_steps: States recorded, the initial one included.
        dt: Time between recorded states.
        substeps: Solver steps taken between one recorded state and the next.
    """

    n_steps: int
    dt: float
    substeps: int = 1

    def __post_init__(self) -> None:
        if self.n_steps < 2:
            raise ValidationError(
                f"a trajectory needs at least two recorded states, got {self.n_steps}"
            )
        if not self.dt > 0.0:
            raise ValidationError(f"the recorded interval must be positive, got {self.dt}")
        if self.substeps < 1:
            raise ValidationError(f"a recorded interval needs at least one solver step, got "
                                  f"{self.substeps}")

    @property
    def solver_dt(self) -> float:
        """Step the reference solver takes."""
        return self.dt / self.substeps

    @property
    def duration(self) -> float:
        """Simulation time from the first recorded state to the last."""
        return (self.n_steps - 1) * self.dt

    @property
    def solver_steps(self) -> int:
        """Total steps the solver takes to produce one trajectory."""
        return (self.n_steps - 1) * self.substeps


def trajectory_id(regime: str, index: int) -> str:
    """Name a trajectory.

    The index is zero padded so that identifiers sort in generation order, which is what
    makes a split reproducible from a sorted list rather than from an insertion order.

    Args:
        regime: Name of the regime the trajectory was drawn from.
        index: Position within that regime.

    Returns:
        The identifier, for example `cold_collapse/00007`.

    Raises:
        ValidationError: If the regime name is empty or the index is negative.
    """
    if not regime:
        raise ValidationError("a trajectory identifier needs a regime name")
    if index < 0:
        raise ValidationError(f"a trajectory index must not be negative, got {index}")
    return f"{regime}/{index:05d}"


def trajectory_stream(system: str, regime: str, index: int) -> str:
    """Name the seed stream one trajectory is drawn from.

    The stream depends only on the system, the regime and the index, so which worker
    generated a trajectory, and how many trajectories were asked for, cannot change it.

    Args:
        system: Name of the system.
        regime: Name of the regime.
        index: Position within that regime.

    Returns:
        The stream name, to be passed to `nnphysics.core.seeding`.
    """
    return SEED_STREAM_TEMPLATE.format(system=system, regime=regime, index=index)
