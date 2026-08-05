import numpy as np
import pytest

from nnphysics.core.errors import NumericalError, ValidationError
from nnphysics.core.types import (
    FieldSpec,
    MetricResult,
    Regime,
    Rollout,
    State,
    StateSpec,
    Trajectory,
)
from nnphysics.core.units import LENGTH, VELOCITY


def make_state(time: float = 0.0, value: float = 1.0) -> State:
    return State(fields={"q": np.full((3, 2), float(value)), "p": np.zeros((3, 2))}, time=time)


def make_trajectory(n_steps: int = 4, offset: float = 0.0) -> Trajectory:
    return Trajectory.from_states(
        [make_state(time=step * 0.5, value=offset + step) for step in range(n_steps)]
    )


class TestState:
    def test_fields_are_immutable_after_construction(self) -> None:
        state = make_state()
        with pytest.raises(TypeError):
            state.fields["q"] = np.zeros((3, 2))  # type: ignore[index]

    def test_a_state_with_no_fields_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no fields"):
            State(fields={})

    def test_a_non_float64_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="expected float64"):
            State(fields={"q": np.zeros((3, 2), dtype=np.float32)})

    def test_time_is_coerced_to_float(self) -> None:
        assert isinstance(State(fields={"q": np.zeros(1)}, time=1).time, float)

    def test_require_finite_catches_a_blown_up_state(self) -> None:
        state = State(fields={"q": np.array([1.0, np.inf])}, time=2.0)
        with pytest.raises(NumericalError, match="not finite"):
            state.require_finite()

    def test_require_finite_passes_a_healthy_state(self) -> None:
        make_state().require_finite()


class TestStateSpec:
    spec = StateSpec(
        fields=(
            FieldSpec("q", (None, 2), LENGTH, "positions"),
            FieldSpec("p", (None, 2), VELOCITY, "velocities"),
        )
    )

    def test_a_conforming_state_validates(self) -> None:
        self.spec.validate(make_state())

    def test_free_extents_accept_any_size(self) -> None:
        self.spec.validate(State(fields={"q": np.zeros((9, 2)), "p": np.zeros((9, 2))}))

    def test_a_missing_field_is_caught(self) -> None:
        with pytest.raises(ValidationError, match="missing fields"):
            self.spec.validate(State(fields={"q": np.zeros((3, 2))}))

    def test_an_undeclared_field_is_caught(self) -> None:
        state = State(fields={"q": np.zeros((3, 2)), "p": np.zeros((3, 2)), "x": np.zeros(1)})
        with pytest.raises(ValidationError, match="undeclared fields"):
            self.spec.validate(state)

    def test_a_wrong_shape_is_caught(self) -> None:
        state = State(fields={"q": np.zeros((3, 3)), "p": np.zeros((3, 2))})
        with pytest.raises(ValidationError, match="expected"):
            self.spec.validate(state)

    def test_a_wrong_rank_is_caught(self) -> None:
        state = State(fields={"q": np.zeros(3), "p": np.zeros((3, 2))})
        with pytest.raises(ValidationError, match="expected"):
            self.spec.validate(state)

    def test_duplicate_field_names_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate field names"):
            StateSpec(fields=(FieldSpec("q", (1,)), FieldSpec("q", (1,))))

    def test_an_empty_specification_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one field"):
            StateSpec(fields=())


class TestTrajectory:
    def test_stacking_and_indexing_round_trip(self) -> None:
        states = [make_state(time=step * 0.5, value=step) for step in range(4)]
        trajectory = Trajectory.from_states(states)
        assert len(trajectory) == 4
        assert trajectory.names == ("q", "p")
        recovered = trajectory[2]
        assert recovered.time == pytest.approx(1.0)
        assert np.array_equal(recovered.fields["q"], states[2].fields["q"])

    def test_iteration_yields_every_state_in_order(self) -> None:
        trajectory = make_trajectory()
        assert [state.time for state in trajectory] == [0.0, 0.5, 1.0, 1.5]

    def test_times_must_increase(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            Trajectory(fields={"q": np.zeros((2, 3))}, times=np.array([1.0, 1.0]))

    def test_field_leading_extent_must_match_the_times(self) -> None:
        with pytest.raises(ValidationError, match="leading extent"):
            Trajectory(fields={"q": np.zeros((3, 3))}, times=np.array([0.0, 1.0]))

    def test_an_empty_trajectory_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no steps"):
            Trajectory(fields={"q": np.zeros((0, 3))}, times=np.array([]))

    def test_stacking_nothing_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty sequence"):
            Trajectory.from_states([])

    def test_stacking_does_not_care_what_order_the_fields_were_built_in(self) -> None:
        """A state is a mapping, so insertion order carries no physics.

        Two states of the same system can reach a trajectory from different code paths,
        a solver building one and a store reading the other, and rejecting the pair for
        disagreeing on an order neither of them meant would be a false alarm.
        """
        first = State(fields={"q": np.zeros(2), "p": np.ones(2)}, time=0.0)
        second = State(fields={"p": np.ones(2), "q": np.zeros(2)}, time=1.0)

        trajectory = Trajectory.from_states([first, second])

        assert len(trajectory) == 2
        assert np.array_equal(trajectory.fields["p"], np.ones((2, 2)))

    def test_stacking_inconsistent_states_is_rejected(self) -> None:
        states = [make_state(time=0.0), State(fields={"q": np.zeros((3, 2))}, time=1.0)]
        with pytest.raises(ValidationError, match="state 1 has fields"):
            Trajectory.from_states(states)


class TestRollout:
    def test_a_matched_pair_is_accepted(self) -> None:
        rollout = Rollout(make_trajectory(), make_trajectory(offset=1.0), "broken", "toy")
        assert len(rollout) == 4
        assert np.array_equal(rollout.times, make_trajectory().times)

    def test_lengths_must_agree(self) -> None:
        with pytest.raises(ValidationError, match="lengths differ"):
            Rollout(make_trajectory(3), make_trajectory(4), "broken", "toy")

    def test_fields_must_agree(self) -> None:
        thin = Trajectory(fields={"q": np.zeros((4, 3, 2))}, times=make_trajectory().times)
        with pytest.raises(ValidationError, match="fields differ"):
            Rollout(thin, make_trajectory(), "broken", "toy")

    def test_times_must_agree(self) -> None:
        shifted = Trajectory(
            fields=dict(make_trajectory().fields), times=make_trajectory().times + 1.0
        )
        with pytest.raises(ValidationError, match="times differ"):
            Rollout(shifted, make_trajectory(), "broken", "toy")


class TestMetricResult:
    def test_scalars_and_series_are_immutable_after_construction(self) -> None:
        result = MetricResult("drift", {"final": 0.5}, {"per_step": np.zeros(4)})
        with pytest.raises(TypeError):
            result.scalars["final"] = 1.0  # type: ignore[index]
        with pytest.raises(TypeError):
            result.series["per_step"] = np.ones(4)  # type: ignore[index]

    def test_series_are_optional(self) -> None:
        assert MetricResult("drift", {"final": 0.5}).series == {}

    def test_a_result_must_name_its_metric(self) -> None:
        with pytest.raises(ValidationError, match="metric name"):
            MetricResult("", {"final": 0.5})


class TestRegime:
    def test_parameters_are_immutable_after_construction(self) -> None:
        regime = Regime("dense", {"count": 8.0})
        with pytest.raises(TypeError):
            regime.parameters["count"] = 9.0  # type: ignore[index]

    def test_a_regime_must_be_named(self) -> None:
        with pytest.raises(ValidationError, match="must have a name"):
            Regime("", {})
