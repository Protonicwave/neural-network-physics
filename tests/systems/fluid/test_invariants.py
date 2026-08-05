import numpy as np
import pytest

from nnphysics.core.protocols import Conservation, Invariant
from nnphysics.core.types import FloatArray, Regime, Trajectory
from nnphysics.systems.fluid import (
    Enstrophy,
    FluidDynamics,
    FluidGrid,
    IntegratingFactorRK4,
    KineticEnergy,
    build_fluid,
    initial_state,
)

GRID = FluidGrid(48)

TURBULENCE = Regime("decaying_turbulence", {"reynolds": 100.0, "peak_wavenumber": 4.0})
"""Only the spectrum is read from this. The viscosity is set on the dynamics directly, so
that one flow can be run at several viscosities without changing anything else."""


def _rollout(viscosity: float, n_steps: int, dt: float = 0.002) -> tuple[FluidDynamics, Trajectory]:
    dynamics = FluidDynamics(GRID, viscosity)
    state = initial_state(TURBULENCE, np.random.default_rng(7), dynamics)
    predictor = IntegratingFactorRK4(dynamics, dt)
    states = [state]
    for _ in range(n_steps):
        states.append(predictor.step(states[-1]))
    return dynamics, Trajectory.from_states(states)


def _series(invariant: Invariant, trajectory: Trajectory) -> FloatArray:
    values: FloatArray = np.array([invariant.evaluate(state) for state in trajectory])
    return values


def _drift(invariant: Invariant, trajectory: Trajectory) -> float:
    values = _series(invariant, trajectory)
    return float(np.abs(values / values[0] - 1.0).max())


class TestDeclarations:
    def test_both_satisfy_the_protocol(self) -> None:
        dynamics = FluidDynamics(GRID, 0.01)
        assert isinstance(KineticEnergy(dynamics), Invariant)
        assert isinstance(Enstrophy(dynamics), Invariant)

    def test_they_are_named_and_dimensioned(self) -> None:
        dynamics = FluidDynamics(GRID, 0.01)
        energy, enstrophy = KineticEnergy(dynamics), Enstrophy(dynamics)

        assert (energy.name, enstrophy.name) == ("energy", "enstrophy")
        assert energy.dimension.symbol == "L^2 T^-2"
        assert enstrophy.dimension.symbol == "T^-2"

    def test_the_inviscid_limit_declares_them_conserved(self) -> None:
        dynamics = FluidDynamics(GRID, 0.0)
        assert KineticEnergy(dynamics).conservation is Conservation.APPROXIMATE
        assert Enstrophy(dynamics).conservation is Conservation.APPROXIMATE

    def test_any_viscosity_declares_them_draining(self) -> None:
        """Declaring a drained quantity as conserved would throw away a testable claim."""
        dynamics = FluidDynamics(GRID, 1e-6)
        assert KineticEnergy(dynamics).conservation is Conservation.DECAYING
        assert Enstrophy(dynamics).conservation is Conservation.DECAYING

    def test_they_read_the_same_numbers_as_the_dynamics(self) -> None:
        dynamics, trajectory = _rollout(viscosity=0.0, n_steps=1)
        vorticity = GRID.unpack(trajectory[0])

        assert KineticEnergy(dynamics).evaluate(trajectory[0]) == dynamics.energy(vorticity)
        assert Enstrophy(dynamics).evaluate(trajectory[0]) == dynamics.enstrophy(vorticity)


class TestInviscidConservation:
    """With no viscosity the truncated system conserves both to the solver's own error.

    This is the pair of tests that catches a dealiasing cutoff placed one mode too high.
    The drift it causes does not shrink when the step size does, so it stands out against
    the time stepping error that is supposed to be all there is.
    """

    def test_both_are_conserved_over_a_moderate_run(self) -> None:
        dynamics, trajectory = _rollout(viscosity=0.0, n_steps=500)

        for invariant in (KineticEnergy(dynamics), Enstrophy(dynamics)):
            assert _drift(invariant, trajectory) < invariant.rtol

    def test_the_drift_falls_steeply_with_the_step_size(self) -> None:
        coarse_dynamics, coarse = _rollout(viscosity=0.0, n_steps=100, dt=0.008)
        fine_dynamics, fine = _rollout(viscosity=0.0, n_steps=200, dt=0.004)

        coarse_drift = _drift(KineticEnergy(coarse_dynamics), coarse)
        fine_drift = _drift(KineticEnergy(fine_dynamics), fine)

        assert coarse_drift / fine_drift > 8.0


class TestViscousDecay:
    def test_the_energy_only_ever_decreases(self) -> None:
        dynamics, trajectory = _rollout(viscosity=0.01, n_steps=300)

        values = _series(KineticEnergy(dynamics), trajectory)

        assert np.all(np.diff(values) < 0.0)
        assert values[-1] < 0.9 * values[0]

    def test_the_enstrophy_decays_at_least_as_fast(self) -> None:
        """Dissipation weights `k^2`, and enstrophy is the invariant that lives up there."""
        dynamics, trajectory = _rollout(viscosity=0.01, n_steps=300)

        energy = _series(KineticEnergy(dynamics), trajectory)
        enstrophy = _series(Enstrophy(dynamics), trajectory)

        assert np.all(np.diff(enstrophy) < 0.0)
        assert enstrophy[-1] / enstrophy[0] < energy[-1] / energy[0]

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_the_reference_solver_obeys_what_each_regime_declares(self, index: int) -> None:
        """A declaration nothing checks is a comment. Every regime is held to its own."""
        system = build_fluid({"grid_size": 32})
        regime = system.regimes[index]
        state = system.initial_state(regime, np.random.default_rng(0))
        predictor = system.reference_predictor(regime, 0.005)
        states = [state]
        for _ in range(100):
            states.append(predictor.step(states[-1]))
        trajectory = Trajectory.from_states(states)

        for invariant in system.invariants(regime):
            values = _series(invariant, trajectory)
            assert invariant.conservation is Conservation.DECAYING
            assert float(np.diff(values).max()) <= invariant.rtol * values[0]
