"""The protocols are proven implementable here and nowhere else.

The toy system below exists only to show that the five interfaces can be satisfied
together without any of them knowing what the others are made of. It must never grow
into anything resembling a real system, and it must never leave this file.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from nnphysics.core.protocols import (
    Conservation,
    Invariant,
    Metric,
    Predictor,
    Symmetry,
    System,
)
from nnphysics.core.types import (
    FieldSpec,
    MetricResult,
    Regime,
    Rollout,
    State,
    StateSpec,
    Trajectory,
)
from nnphysics.core.units import DIMENSIONLESS, Dimension

SPEC = StateSpec(fields=(FieldSpec("x", (2,), DIMENSIONLESS, "an abstract coordinate"),))
REGIME = Regime("toy", {"rate": 1.0})


@dataclass(frozen=True)
class Total:
    """Sum of the coordinate, which the toy drift below preserves."""

    name: str = "total"
    dimension: Dimension = DIMENSIONLESS
    conservation: Conservation = Conservation.EXACT
    rtol: float = 1e-9

    def evaluate(self, state: State) -> float:
        return float(state.fields["x"].sum())


@dataclass(frozen=True)
class Swap:
    """Exchanging the two coordinates, which the toy drift commutes with."""

    name: str = "swap"

    def apply(self, state: State) -> State:
        return State(fields={"x": state.fields["x"][::-1].copy()}, time=state.time)

    def apply_inverse(self, state: State) -> State:
        return self.apply(state)


@dataclass(frozen=True)
class Drift:
    """Adds equal and opposite amounts to the two coordinates."""

    dt: float = 0.1
    name: str = "drift"

    def step(self, state: State) -> State:
        return State(
            fields={"x": state.fields["x"] + np.array([self.dt, -self.dt])},
            time=state.time + self.dt,
        )


@dataclass(frozen=True)
class Stuck:
    """Advances time but never moves: the simplest predictor that is broken on purpose."""

    dt: float = 0.1
    name: str = "stuck"

    def step(self, state: State) -> State:
        return State(fields=dict(state.fields), time=state.time + self.dt)


@dataclass(frozen=True)
class FinalError:
    """Absolute error in the coordinate at the last step."""

    name: str = "final_error"

    def compute(self, rollout: Rollout) -> MetricResult:
        error = np.abs(rollout.predicted.fields["x"] - rollout.reference.fields["x"]).max(axis=-1)
        return MetricResult(self.name, {"final": float(error[-1])}, {"per_step": error})


@dataclass(frozen=True)
class Toy:
    """A system with one field, one invariant and one symmetry."""

    name: str = "toy"

    @property
    def state_spec(self) -> StateSpec:
        return SPEC

    @property
    def regimes(self) -> tuple[Regime, ...]:
        return (REGIME,)

    def invariants(self, regime: Regime) -> tuple[Invariant, ...]:
        return (Total(rtol=1e-9 * regime.parameters["rate"]),)

    @property
    def symmetries(self) -> tuple[Symmetry, ...]:
        return (Swap(),)

    def initial_state(self, regime: Regime, rng: np.random.Generator) -> State:
        return State(fields={"x": rng.standard_normal(2) * regime.parameters["rate"]}, time=0.0)

    def reference_predictor(self, regime: Regime, dt: float) -> Predictor:
        return Drift(dt=dt * regime.parameters["rate"])


def roll_out(predictor: Predictor, initial: State, n_steps: int) -> Trajectory:
    states = [initial]
    for _ in range(n_steps):
        states.append(predictor.step(states[-1]))
    return Trajectory.from_states(states)


def test_the_toy_implementations_satisfy_the_protocols() -> None:
    assert isinstance(Toy(), System)
    assert isinstance(Total(), Invariant)
    assert isinstance(Swap(), Symmetry)
    assert isinstance(Drift(), Predictor)
    assert isinstance(FinalError(), Metric)


def test_a_system_can_be_driven_through_the_protocols_alone() -> None:
    system: System = Toy()
    initial = system.initial_state(system.regimes[0], np.random.default_rng(0))
    system.state_spec.validate(initial)
    predictor = system.reference_predictor(system.regimes[0], dt=0.1)
    trajectory = roll_out(predictor, initial, n_steps=8)
    assert len(trajectory) == 9


def test_invariants_are_read_from_the_system_not_known_by_the_caller() -> None:
    system: System = Toy()
    regime = system.regimes[0]
    initial = system.initial_state(regime, np.random.default_rng(0))
    trajectory = roll_out(system.reference_predictor(regime, 0.1), initial, 8)
    for invariant in system.invariants(regime):
        values = np.array([invariant.evaluate(state) for state in trajectory])
        drift = np.abs(values - values[0]).max() / max(abs(values[0]), 1e-30)
        assert drift < invariant.rtol or invariant.conservation is Conservation.APPROXIMATE


def test_symmetries_can_be_applied_and_undone() -> None:
    system: System = Toy()
    initial = system.initial_state(system.regimes[0], np.random.default_rng(0))
    for symmetry in system.symmetries:
        restored = symmetry.apply_inverse(symmetry.apply(initial))
        assert np.allclose(restored.fields["x"], initial.fields["x"])


def test_a_metric_scores_a_rollout_without_knowing_the_system() -> None:
    system: System = Toy()
    initial = system.initial_state(system.regimes[0], np.random.default_rng(0))
    reference = roll_out(system.reference_predictor(system.regimes[0], 0.1), initial, 8)
    broken = roll_out(Stuck(dt=0.1), initial, 8)
    result = FinalError().compute(Rollout(broken, reference, "stuck", system.name))
    assert result.name == "final_error"
    assert result.scalars["final"] == pytest.approx(0.8)
    assert result.series["per_step"].shape == (9,)
    assert result.series["per_step"][0] == pytest.approx(0.0)
