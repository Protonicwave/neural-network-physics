import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.data.spec import TrajectorySpec, trajectory_id, trajectory_stream


def test_the_solver_step_is_the_stored_step_divided_by_the_substeps() -> None:
    spec = TrajectorySpec(n_steps=11, dt=0.1, substeps=5)
    assert spec.solver_dt == pytest.approx(0.02)
    assert spec.duration == pytest.approx(1.0)
    assert spec.solver_steps == 50


def test_one_substep_leaves_the_two_intervals_equal() -> None:
    spec = TrajectorySpec(n_steps=4, dt=0.25)
    assert spec.solver_dt == spec.dt
    assert spec.solver_steps == 3


@pytest.mark.parametrize(
    ("n_steps", "dt", "substeps"),
    [(1, 0.1, 1), (4, 0.0, 1), (4, -0.1, 1), (4, 0.1, 0)],
)
def test_an_unusable_specification_is_rejected(n_steps: int, dt: float, substeps: int) -> None:
    with pytest.raises(ValidationError):
        TrajectorySpec(n_steps=n_steps, dt=dt, substeps=substeps)


def test_identifiers_sort_in_generation_order() -> None:
    identifiers = [trajectory_id("regime", index) for index in (0, 2, 10, 100)]
    assert identifiers == sorted(identifiers)
    assert identifiers[0] == "regime/00000"


@pytest.mark.parametrize(("regime", "index"), [("", 0), ("regime", -1)])
def test_an_unusable_identifier_is_rejected(regime: str, index: int) -> None:
    with pytest.raises(ValidationError):
        trajectory_id(regime, index)


def test_a_seed_stream_depends_only_on_the_system_regime_and_index() -> None:
    assert trajectory_stream("nbody", "cold", 7) == trajectory_stream("nbody", "cold", 7)
    distinct = {
        trajectory_stream("nbody", "cold", 7),
        trajectory_stream("fluid", "cold", 7),
        trajectory_stream("nbody", "warm", 7),
        trajectory_stream("nbody", "cold", 8),
    }
    assert len(distinct) == 4
