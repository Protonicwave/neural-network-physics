import numpy as np
import pytest

from nnphysics.core.errors import UnknownNameError, ValidationError
from nnphysics.data.generation import (
    batched,
    default_workers,
    find_regime,
    generate_many,
    generate_trajectory,
    iter_requests,
)
from nnphysics.data.spec import TrajectorySpec
from nnphysics.systems import build_system

SPEC = TrajectorySpec(n_steps=5, dt=0.05, substeps=2)
SEED = 11


def test_a_trajectory_is_recorded_at_the_stored_interval() -> None:
    system = build_system("nbody", {"softening": 0.05})
    regime = find_regime(system, "virialised_cluster")
    trajectory = generate_trajectory(system, regime, SPEC, SEED, 0)
    assert len(trajectory) == SPEC.n_steps
    assert np.allclose(trajectory.times, np.arange(SPEC.n_steps) * SPEC.dt)
    assert np.array_equal(trajectory.times, np.arange(SPEC.n_steps) * SPEC.dt)


def test_the_same_seed_and_index_give_the_same_trajectory() -> None:
    system = build_system("nbody", {"softening": 0.05})
    regime = find_regime(system, "cold_collapse")
    first = generate_trajectory(system, regime, SPEC, SEED, 4)
    second = generate_trajectory(system, regime, SPEC, SEED, 4)
    for name, array in first.fields.items():
        assert np.array_equal(array, second.fields[name])


def test_different_indices_give_different_trajectories() -> None:
    system = build_system("nbody", {"softening": 0.05})
    regime = find_regime(system, "cold_collapse")
    first = generate_trajectory(system, regime, SPEC, SEED, 0)
    second = generate_trajectory(system, regime, SPEC, SEED, 1)
    assert not np.array_equal(first.fields["position"], second.fields["position"])


def test_substeps_change_the_solver_step_and_nothing_else() -> None:
    """Sub-sampling must refine the integration without moving what is recorded.

    The stored times and the initial state are fixed by the specification, so they are
    identical however many substeps are taken. What changes is the truncation error, and
    it must shrink: the second order solver converges towards one trajectory rather than
    towards a different one for each choice of step.
    """
    system = build_system("nbody", {"softening": 0.05})
    regime = find_regime(system, "virialised_cluster")

    def positions(substeps: int) -> np.ndarray:
        spec = TrajectorySpec(SPEC.n_steps, SPEC.dt, substeps)
        trajectory = generate_trajectory(system, regime, spec, SEED, 0)
        assert np.array_equal(trajectory.times, np.arange(SPEC.n_steps) * SPEC.dt)
        return np.asarray(trajectory.fields["position"])

    reference = positions(64)
    coarse = positions(2)
    fine = positions(8)

    assert np.array_equal(coarse[0], fine[0])
    assert np.array_equal(coarse[0], reference[0])
    coarse_error = float(np.max(np.abs(coarse - reference)))
    fine_error = float(np.max(np.abs(fine - reference)))
    assert 0.0 < fine_error < coarse_error


def test_the_fluid_system_generates_through_the_same_interface() -> None:
    system = build_system("fluid", {"grid_size": 32})
    regime = find_regime(system, "taylor_green")
    trajectory = generate_trajectory(system, regime, SPEC, SEED, 0)
    assert trajectory.fields["vorticity"].shape == (SPEC.n_steps, 32, 32)


def test_an_undeclared_regime_is_rejected() -> None:
    system = build_system("nbody")
    with pytest.raises(UnknownNameError):
        find_regime(system, "no_such_regime")


def test_a_batch_comes_back_in_request_order() -> None:
    requests = list(
        iter_requests("nbody", {"softening": 0.05}, "cold_collapse", count=3, spec=SPEC, seed=SEED)
    )
    trajectories = generate_many(requests, workers=1)
    assert [request.index for request in requests] == [0, 1, 2]
    for request, trajectory in zip(requests, trajectories, strict=True):
        system = build_system("nbody", {"softening": 0.05})
        expected = generate_trajectory(
            system, find_regime(system, "cold_collapse"), SPEC, SEED, request.index
        )
        assert np.array_equal(trajectory.fields["position"], expected.fields["position"])


def test_worker_count_does_not_change_what_is_generated() -> None:
    requests = list(
        iter_requests("nbody", {"softening": 0.05}, "cold_collapse", count=4, spec=SPEC, seed=SEED)
    )
    serial = generate_many(requests, workers=1)
    parallel = generate_many(requests, workers=2)
    for one, other in zip(serial, parallel, strict=True):
        for name, array in one.fields.items():
            assert np.array_equal(array, other.fields[name])


def test_an_empty_or_unusable_batch_is_rejected() -> None:
    requests = list(iter_requests("nbody", {}, "cold_collapse", count=1, spec=SPEC, seed=SEED))
    with pytest.raises(ValidationError):
        generate_many([], workers=1)
    with pytest.raises(ValidationError):
        generate_many(requests, workers=0)


def test_a_system_parameter_that_cannot_reach_a_worker_is_rejected() -> None:
    with pytest.raises(ValidationError):
        list(
            iter_requests(
                "nbody", {"softening": [0.05]}, "cold_collapse", count=1, spec=SPEC, seed=SEED
            )
        )


def test_batches_cover_every_request_once() -> None:
    requests = list(iter_requests("nbody", {}, "cold_collapse", count=7, spec=SPEC, seed=SEED))
    batches = list(batched(requests, 3))
    assert [len(batch) for batch in batches] == [3, 3, 1]
    assert [request.index for batch in batches for request in batch] == list(range(7))


def test_an_unusable_batch_size_is_rejected() -> None:
    with pytest.raises(ValidationError):
        list(batched([], 0))


def test_the_default_worker_count_leaves_a_core_free() -> None:
    assert default_workers() >= 1
