from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import State
from nnphysics.data.normalisation import FieldStats, Normalisation
from nnphysics.models import ModelContext, build_model
from nnphysics.models.graph import NBodyGraphModel
from nnphysics.systems import build_system
from nnphysics.systems.nbody.dynamics import NBodyDynamics
from nnphysics.systems.nbody.integrators import VelocityVerlet
from nnphysics.systems.nbody.invariants import TotalEnergy
from nnphysics.systems.nbody.state import MASS_FIELD, POSITION_FIELD, VELOCITY_FIELD, make_state
from nnphysics.systems.nbody.symmetries import GalileanBoost, Rotation, Translation

Batcher = Callable[..., dict[str, torch.Tensor]]
Exciter = Callable[..., None]

SOFTENING = 0.05


class _TrueForce(NBodyGraphModel):
    """The graph model with the real force law in place of the learned one.

    Substituting it is what lets a test compare this model's time stepping against the
    solver from phase 02 directly. Everything else about the step is unchanged, so what
    the comparison measures is the integrator and nothing else.
    """

    def __init__(self, context: ModelContext, dynamics: NBodyDynamics) -> None:
        super().__init__(context)
        self._dynamics = dynamics

    def accelerations(self, position: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        stacked = [
            self._dynamics.accelerations(
                position[index].detach().numpy().astype(np.float64),
                mass[index].detach().numpy().astype(np.float64),
            )
            for index in range(position.shape[0])
        ]
        return torch.from_numpy(np.stack(stacked).astype(np.float32))


def test_the_step_is_the_symplectic_scheme_from_phase_02(
    context: ModelContext, state: State
) -> None:
    """The one claim the module docstring makes that a reader cannot check by eye."""
    dynamics = NBodyDynamics(softening=SOFTENING)
    model = _TrueForce(context, dynamics)
    solver = VelocityVerlet(dynamics, context.dt)

    expected = solver.step(state)
    actual = model.step(state)
    for field in (POSITION_FIELD, VELOCITY_FIELD):
        assert np.allclose(actual.fields[field], expected.fields[field], rtol=1e-4, atol=1e-6)


def test_the_carried_acceleration_is_the_one_the_last_step_ended_at(
    context: ModelContext, state: State
) -> None:
    """A rollout must cost one network evaluation per step, and give the same answer."""
    dynamics = NBodyDynamics(softening=SOFTENING)
    model = _TrueForce(context, dynamics)
    solver = VelocityVerlet(dynamics, context.dt)

    stepped = state
    expected = state
    for _ in range(4):
        stepped = model.step(stepped)
        expected = solver.step(expected)
    for field in (POSITION_FIELD, VELOCITY_FIELD):
        assert np.allclose(stepped.fields[field], expected.fields[field], rtol=1e-3, atol=1e-5)


def test_an_untrained_model_moves_in_a_straight_line(context: ModelContext, state: State) -> None:
    model = build_model("graph", context)
    stepped = model.step(state)
    drifted = state.fields[POSITION_FIELD] + state.fields[VELOCITY_FIELD] * context.dt
    assert np.allclose(stepped.fields[POSITION_FIELD], drifted, atol=1e-6)
    assert np.allclose(stepped.fields[VELOCITY_FIELD], state.fields[VELOCITY_FIELD], atol=1e-6)


@pytest.mark.parametrize(
    "symmetry",
    [Translation((1.5, -0.75)), Rotation(0.7), GalileanBoost((0.3, -0.2))],
    ids=["translation", "rotation", "galilean_boost"],
)
def test_every_declared_symmetry_is_exact_by_construction(
    symmetry: Translation | Rotation | GalileanBoost,
    context: ModelContext,
    state: State,
    excite: Exciter,
) -> None:
    """Equivariance here is a property of the design, so it holds at round off.

    The evaluation harness measures it anyway, on this model as on every other. What the
    harness reports for this one is therefore a check on the implementation rather than a
    discovery about what was learned, and that is worth saying out loud.
    """
    model = build_model("graph", context)
    excite(model)
    transformed = symmetry.apply_inverse(model.step(symmetry.apply(state)))
    direct = model.step(state)
    for field in (POSITION_FIELD, VELOCITY_FIELD):
        scale = float(np.max(np.abs(direct.fields[field])))
        assert np.allclose(transformed.fields[field], direct.fields[field], atol=1e-5 * scale)


def test_the_same_weights_run_on_a_different_number_of_bodies(
    context: ModelContext, state: State, excite: Exciter
) -> None:
    """The point of the graph: nothing in it is tied to the size it was built for."""
    model = build_model("graph", context)
    excite(model)
    system = build_system("nbody", {"softening": SOFTENING})
    smaller = system.initial_state(system.regimes[2], np.random.default_rng(0))
    assert smaller.fields[MASS_FIELD].shape != state.fields[MASS_FIELD].shape
    model.step(smaller).require_finite()


def test_data_that_is_not_an_nbody_system_is_refused(context: ModelContext) -> None:
    with pytest.raises(ValidationError, match="reads the N-body fields"):
        build_model(
            "graph",
            ModelContext(
                field_shapes={"vorticity": (8, 8), MASS_FIELD: (8,)},
                static_fields=(MASS_FIELD,),
                normalisation=Normalisation(
                    {
                        "vorticity": FieldStats(0.0, 1.0, 10),
                        MASS_FIELD: FieldStats(1.0, 0.1, 10),
                    }
                ),
                dt=context.dt,
                seed=0,
            ),
        )


def test_positions_must_be_two_dimensional(context: ModelContext) -> None:
    stats = context.normalisation.stats
    with pytest.raises(ValidationError, match=r"shape \(bodies, 2\)"):
        build_model(
            "graph",
            ModelContext(
                field_shapes={
                    POSITION_FIELD: (4, 3),
                    VELOCITY_FIELD: (4, 3),
                    MASS_FIELD: (4,),
                },
                static_fields=(MASS_FIELD,),
                normalisation=Normalisation(dict(stats)),
                dt=context.dt,
                seed=0,
            ),
        )


def test_the_energy_the_penalty_reads_is_the_one_the_system_declares(
    context: ModelContext, state: State, as_batch: Batcher
) -> None:
    dynamics = NBodyDynamics(softening=SOFTENING)
    model = build_model("graph", context, {"physical_softening": SOFTENING})
    assert isinstance(model, NBodyGraphModel)
    batch = as_batch(state)
    with torch.no_grad():
        energy = model._energy(batch[POSITION_FIELD], batch[VELOCITY_FIELD], batch[MASS_FIELD])
    assert float(energy[0]) == pytest.approx(TotalEnergy(dynamics).evaluate(state), rel=1e-4)


def test_the_penalty_punishes_a_step_that_changes_the_energy(
    context: ModelContext, state: State, as_batch: Batcher
) -> None:
    model = build_model("graph", context, {"physical_softening": SOFTENING})
    batch = as_batch(state)
    honest, _ = model.advance(batch, None)
    inflated = {
        POSITION_FIELD: honest[POSITION_FIELD],
        VELOCITY_FIELD: honest[VELOCITY_FIELD] * 1.5,
    }
    with torch.no_grad():
        assert float(model.physics_penalty(batch, inflated)) > 100.0 * float(
            model.physics_penalty(batch, honest)
        )


def test_the_penalty_is_zero_for_a_model_that_declares_none(
    context: ModelContext, batch: dict[str, torch.Tensor]
) -> None:
    model = build_model("mlp", context)
    with torch.no_grad():
        predicted, _ = model.advance(batch, None)
        assert float(model.physics_penalty(batch, predicted)) == 0.0


def test_the_model_records_the_settings_it_was_built_with(context: ModelContext) -> None:
    model = build_model("graph", context, {"hidden": 16, "rounds": 1})
    assert model.hyperparameters["hidden"] == 16
    assert model.hyperparameters["rounds"] == 1


def test_a_hyperparameter_the_model_does_not_understand_is_refused(
    context: ModelContext,
) -> None:
    with pytest.raises(ValidationError, match="unknown"):
        build_model("graph", context, {"layers": 3})


def test_a_state_of_one_body_accelerates_nowhere(context: ModelContext, excite: Exciter) -> None:
    model = build_model("graph", context)
    excite(model)
    alone = make_state(np.zeros((1, 2)), np.ones((1, 2)), np.ones(1))
    stepped = model.step(alone)
    assert np.allclose(stepped.fields[VELOCITY_FIELD], alone.fields[VELOCITY_FIELD], atol=1e-6)
