"""Spectral refinement between a grid and an integer multiple of it.

The property the resolution metric rests on is that coarsening a refined state returns
it exactly. If it did not, every number that metric produces would be part
transformation error, and no amount of care in the metric would separate the two.
"""

from __future__ import annotations

import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.protocols import Refinement
from nnphysics.core.types import State
from nnphysics.systems.fluid.grid import VORTICITY_FIELD, FluidGrid
from nnphysics.systems.fluid.refinement import SpectralRefinement
from nnphysics.systems.fluid.system import build_fluid
from nnphysics.systems.nbody.system import build_nbody

ROUND_OFF = 1e-12

GRID = FluidGrid(size=32)


def smooth_state(grid: FluidGrid, seed: int = 0) -> State:
    """A band limited field, as every state this system produces is."""
    x, y = grid.coordinates()
    rng = np.random.default_rng(seed)
    field = np.zeros((grid.size, grid.size))
    for wavenumber in (1, 2, 3):
        field += rng.standard_normal() * np.sin(wavenumber * x) * np.cos(wavenumber * y)
    return grid.make_state(field - float(np.mean(field)))


class TestProtocol:
    def test_it_satisfies_the_refinement_protocol(self) -> None:
        assert isinstance(SpectralRefinement(GRID, 2), Refinement)

    def test_the_name_says_what_it_does(self) -> None:
        assert SpectralRefinement(GRID, 4).name == "grid_x4"

    @pytest.mark.parametrize("factor", [-1, 0, 1])
    def test_a_factor_that_does_not_refine_is_refused(self, factor: int) -> None:
        with pytest.raises(ValidationError, match="finer"):
            SpectralRefinement(GRID, factor)


class TestRoundTrip:
    @pytest.mark.parametrize("factor", [2, 4])
    def test_coarsening_a_refined_state_returns_it_exactly(self, factor: int) -> None:
        refinement = SpectralRefinement(GRID, factor)
        state = smooth_state(GRID)

        recovered = refinement.coarsen(refinement.refine(state))

        assert np.allclose(
            recovered.fields[VORTICITY_FIELD], state.fields[VORTICITY_FIELD], atol=ROUND_OFF
        )

    def test_the_refined_state_lives_on_the_finer_grid(self) -> None:
        refinement = SpectralRefinement(GRID, 2)

        refined = refinement.refine(smooth_state(GRID))

        assert refined.fields[VORTICITY_FIELD].shape == (64, 64)

    def test_the_time_is_carried_through_both_directions(self) -> None:
        refinement = SpectralRefinement(GRID, 2)
        state = State(fields=dict(smooth_state(GRID).fields), time=1.25)

        assert refinement.refine(state).time == 1.25
        assert refinement.coarsen(refinement.refine(state)).time == 1.25


class TestFaithfulness:
    def test_refining_samples_the_same_function(self) -> None:
        """Every second point of the refined field is a point of the coarse one."""
        refinement = SpectralRefinement(GRID, 2)
        state = smooth_state(GRID)

        refined = refinement.refine(state)

        assert np.allclose(
            refined.fields[VORTICITY_FIELD][::2, ::2],
            state.fields[VORTICITY_FIELD],
            atol=ROUND_OFF,
        )

    def test_the_declared_invariants_are_unchanged_by_refining(self) -> None:
        system = build_fluid({"grid_size": 32})
        regime = system.regimes[0]
        state = system.initial_state(regime, np.random.default_rng(0))
        (refinement,) = system.refinements
        fine = build_fluid({"grid_size": 64})

        refined = refinement.refine(state)

        for coarse_invariant, fine_invariant in zip(
            system.invariants(regime), fine.invariants(regime), strict=True
        ):
            assert fine_invariant.evaluate(refined) == pytest.approx(
                coarse_invariant.evaluate(state), rel=1e-10
            )


class TestLimits:
    def test_a_field_with_energy_at_the_nyquist_mode_is_refused(self) -> None:
        """Refining it would mean choosing how to split a mode between two wavenumbers."""
        refinement = SpectralRefinement(GRID, 2)
        x, _ = GRID.coordinates()
        alternating = np.tile(np.cos(GRID.size / 2 * x), (1, GRID.size))
        state = GRID.make_state(alternating - float(np.mean(alternating)))

        with pytest.raises(ValidationError, match="Nyquist"):
            refinement.refine(state)

    def test_a_state_on_the_wrong_grid_is_refused(self) -> None:
        refinement = SpectralRefinement(GRID, 2)

        with pytest.raises(ValidationError):
            refinement.coarsen(smooth_state(GRID))


class TestDeclaration:
    def test_the_fluid_declares_one_refinement_and_the_nbody_none(self) -> None:
        """A set of point masses is not a discretisation, so the question does not apply."""
        assert len(build_fluid({}).refinements) == 1
        assert build_nbody({}).refinements == ()
