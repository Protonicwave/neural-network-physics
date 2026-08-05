"""Producing trajectories with a system's reference solver.

Two properties are worth more here than speed. A trajectory's contents depend only on
the system, the regime, the index and the root seed, never on how the work was divided,
so the number of workers cannot change the data. And the recorded times are exact
multiples of the recorded interval rather than a running sum of solver steps, so a
trajectory generated with ten substeps lines up with one generated with a hundred.

Work is submitted a shard at a time so that memory is bounded by one shard rather than
by the whole dataset, which a fluid dataset would otherwise exceed.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import UnknownNameError, ValidationError
from nnphysics.core.seeding import numpy_generator
from nnphysics.core.types import Trajectory
from nnphysics.data.spec import TrajectorySpec, trajectory_stream
from nnphysics.systems import build_system

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from nnphysics.core.protocols import System
    from nnphysics.core.types import FloatArray, Regime
    from nnphysics.systems.base import SystemParameters

__all__ = [
    "TrajectoryRequest",
    "batched",
    "default_workers",
    "find_regime",
    "generate_many",
    "generate_trajectory",
    "iter_requests",
]


type _Result = tuple[dict[str, FloatArray], FloatArray]
"""What a worker sends back: the stacked fields and the recorded times."""


@dataclass(frozen=True, slots=True)
class TrajectoryRequest:
    """One trajectory to generate, in a form a worker process can be handed.

    Only names and numbers, because a built system holds arrays and a `State` holds a
    read only mapping, neither of which pickles.

    Attributes:
        system: Registered system name.
        parameters: System level parameters.
        regime: Regime name, resolved against the built system in the worker.
        index: Position within the regime, which fixes the seed stream.
        spec: The trajectory specification.
        seed: Root seed of the run.
    """

    system: str
    parameters: tuple[tuple[str, float | int | bool | str], ...]
    regime: str
    index: int
    spec: TrajectorySpec
    seed: int


def default_workers() -> int:
    """Processes to generate with when a configuration does not say.

    Returns:
        Cores minus one, leaving one for everything else, and at least one.
    """
    return max(1, (os.cpu_count() or 2) - 1)


def find_regime(system: System, name: str) -> Regime:
    """Resolve a regime name against a system's declared regimes.

    Args:
        system: The system.
        name: The regime name.

    Returns:
        The regime.

    Raises:
        UnknownNameError: If the system declares no regime by that name.
    """
    for regime in system.regimes:
        if regime.name == name:
            return regime
    known = ", ".join(sorted(regime.name for regime in system.regimes))
    raise UnknownNameError(f"unknown {system.name} regime {name!r}, declared: {known}")


def generate_trajectory(
    system: System, regime: Regime, spec: TrajectorySpec, seed: int, index: int
) -> Trajectory:
    """Generate one trajectory with the system's reference solver.

    Args:
        system: The system.
        regime: The regime to draw the initial condition from.
        spec: How long the trajectory runs and how finely it is stored.
        seed: Root seed of the run.
        index: Position within the regime, which fixes the seed stream.

    Returns:
        The trajectory, sampled at the recorded interval.

    Raises:
        NumericalError: If the solver diverges or breaks its own stability condition.
        ValidationError: If the initial condition does not match the system.
    """
    rng = numpy_generator(seed, trajectory_stream(system.name, regime.name, index))
    state = system.initial_state(regime, rng)
    system.state_spec.validate(state)
    predictor = system.reference_predictor(regime, spec.solver_dt)

    samples: list[dict[str, FloatArray]] = [dict(state.fields)]
    for _ in range(spec.n_steps - 1):
        for _ in range(spec.substeps):
            state = predictor.step(state)
        state.require_finite()
        samples.append(dict(state.fields))
    names = tuple(samples[0])
    return Trajectory(
        fields={name: np.stack([sample[name] for sample in samples]) for name in names},
        times=np.arange(spec.n_steps, dtype=np.float64) * spec.dt,
    )


def generate_many(
    requests: Sequence[TrajectoryRequest], *, workers: int
) -> list[Trajectory]:
    """Generate a batch of trajectories, in the order they were asked for.

    Args:
        requests: The trajectories to generate.
        workers: Processes to spread the work over. One generates in this process, which
            is what tests and small datasets want.

    Returns:
        The trajectories, in request order regardless of completion order.

    Raises:
        ValidationError: If no requests were given or the worker count is not positive.
    """
    if not requests:
        raise ValidationError("cannot generate an empty batch of trajectories")
    if workers < 1:
        raise ValidationError(f"worker count must be positive, got {workers}")
    if workers == 1:
        return [_rebuild(_generate(request)) for request in requests]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [_rebuild(result) for result in pool.map(_generate, requests)]


def iter_requests(
    system: str,
    parameters: SystemParameters,
    regime: str,
    count: int,
    spec: TrajectorySpec,
    seed: int,
) -> Iterator[TrajectoryRequest]:
    """Enumerate the requests for one regime.

    Args:
        system: Registered system name.
        parameters: System level parameters.
        regime: Regime name.
        count: Trajectories to generate.
        spec: The trajectory specification.
        seed: Root seed of the run.

    Yields:
        One request per trajectory, in index order.
    """
    frozen = _freeze_parameters(parameters)
    for index in range(count):
        yield TrajectoryRequest(system, frozen, regime, index, spec, seed)


def batched(requests: Iterable[TrajectoryRequest], size: int) -> Iterator[list[TrajectoryRequest]]:
    """Split requests into batches of at most `size`.

    Args:
        requests: The requests.
        size: Largest batch.

    Yields:
        Batches, in order.

    Raises:
        ValidationError: If the batch size is not positive.
    """
    if size < 1:
        raise ValidationError(f"batch size must be positive, got {size}")
    batch: list[TrajectoryRequest] = []
    for request in requests:
        batch.append(request)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _generate(request: TrajectoryRequest) -> _Result:
    """Build the system and generate one trajectory. Runs in a worker process.

    Plain arrays come back rather than a `Trajectory`, because a trajectory holds a read
    only mapping and a read only mapping does not pickle. Arrays are what crosses the
    boundary in any case, so nothing is lost by saying so.
    """
    system = build_system(request.system, dict(request.parameters))
    regime = find_regime(system, request.regime)
    trajectory = generate_trajectory(
        system, regime, request.spec, request.seed, request.index
    )
    return dict(trajectory.fields), trajectory.times


def _rebuild(result: _Result) -> Trajectory:
    """Put a worker's arrays back into a trajectory."""
    fields, times = result
    return Trajectory(fields=fields, times=times)


def _freeze_parameters(
    parameters: SystemParameters,
) -> tuple[tuple[str, float | int | bool | str], ...]:
    """Put system parameters in a picklable, order independent form.

    Raises:
        ValidationError: If a parameter is not a scalar a configuration can carry.
    """
    frozen: list[tuple[str, float | int | bool | str]] = []
    for name in sorted(parameters):
        value = parameters[name]
        if not isinstance(value, float | int | bool | str):
            raise ValidationError(
                f"system parameter {name!r} is a {type(value).__name__}, which cannot be "
                f"carried to a worker process"
            )
        frozen.append((name, value))
    return tuple(frozen)
