"""The resolution generalisation metric, and the sentinels that give it meaning.

A metric that cannot fail is not a metric, and one that flags everything is worse. Both
halves are asserted here against predictors built for the purpose: one that is the same
operator on any grid, and one that is not.

The predictors are written in this file rather than registered, because they are not
things anyone would evaluate. They exist to bracket the metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from nnphysics.core.types import MetricResult, Rollout, State, Trajectory
from nnphysics.evals.metrics import MetricContext, ResolutionGeneralisation
from nnphysics.evals.rollout import roll_out
from nnphysics.systems.fluid.grid import VORTICITY_FIELD, FluidGrid
from nnphysics.systems.fluid.refinement import SpectralRefinement
from nnphysics.systems.fluid.system import build_fluid

if TYPE_CHECKING:
    from nnphysics.core.protocols import Predictor

SIZE = 32
DT = 0.05
STEPS = 4

GRID = FluidGrid(size=SIZE)
REFINEMENT = SpectralRefinement(GRID, 2)


@dataclass(frozen=True)
class ModeDecay:
    """Multiplies every mode by a factor of its wavenumber: resolution independent.

    It is a caricature of a viscous solver and it is written in wavenumbers, so it says
    the same thing whatever grid the field arrives on. The metric must not flag it.
    """

    dt: float = DT
    rate: float = 0.4
    name: str = "mode_decay"

    def step(self, state: State) -> State:
        field = state.fields[VORTICITY_FIELD]
        size = field.shape[0]
        grid = FluidGrid(size=size)
        wavenumber_x, wavenumber_y = grid.wavenumbers()
        squared = wavenumber_x**2 + wavenumber_y**2
        spectrum = grid.forward(field) * np.exp(-self.rate * squared * self.dt)
        return State(fields={VORTICITY_FIELD: grid.inverse(spectrum)}, time=state.time + self.dt)


@dataclass(frozen=True)
class NeighbourAverage:
    """Averages each point with its neighbours: a statement about cells, not distance.

    On a grid of half the spacing it smooths over half the domain it did before, so it is
    a different operator there. The metric must catch it.
    """

    dt: float = DT
    weight: float = 0.4
    name: str = "neighbour_average"

    def step(self, state: State) -> State:
        field = state.fields[VORTICITY_FIELD]
        neighbours = sum(np.roll(field, shift, axis=axis) for axis in (0, 1) for shift in (-1, 1))
        smoothed = field + self.weight * (neighbours / 4.0 - field)
        return State(fields={VORTICITY_FIELD: smoothed}, time=state.time + self.dt)


def initial() -> State:
    """A band limited flow, as every state this system produces is."""
    return build_fluid({"grid_size": SIZE}).initial_state(
        build_fluid({"grid_size": SIZE}).regimes[0], np.random.default_rng(0)
    )


def score(
    predictor: Predictor, *, refinements: tuple[SpectralRefinement, ...] = (REFINEMENT,)
) -> MetricResult:
    """Roll a predictor out and score it, against a reference of its own truth.

    The reference is a rollout of the same predictor, so `native_error` is zero and
    `degradation` reads as the refined path's error alone. That isolates what this test
    is about: nothing here is measuring accuracy.
    """
    result = roll_out(predictor, initial(), STEPS)
    rollout = Rollout(
        predicted=result.trajectory,
        reference=result.trajectory,
        predictor=result.predictor,
        system="fluid",
    )
    context = MetricContext(refinements=refinements, predictor=predictor, resolution_steps=STEPS)
    return ResolutionGeneralisation(context).compute(rollout)


class TestWhatItCatches:
    def test_a_predictor_written_in_cells_is_caught(self) -> None:
        """The sentinel. A neighbour average is a different operator on a finer grid.

        Four steps of it disagree with themselves by a tenth of the signal, which is the
        catch. The number that makes it a catch rather than a reading is the second
        assertion: the predictor written in wavenumbers scores five orders of magnitude
        lower on the same rollout.
        """
        scored = score(NeighbourAverage())

        assert scored.scalars["worst_consistency"] > 0.05
        assert (
            scored.scalars["worst_consistency"]
            > 1e4 * score(ModeDecay()).scalars["worst_consistency"]
        )

    def test_the_degradation_shows_it_too(self) -> None:
        scored = score(NeighbourAverage())

        assert scored.scalars[f"{REFINEMENT.name}.degradation"] > 0.05

    def test_a_stronger_fault_scores_worse(self) -> None:
        """A metric whose number does not order the faults it catches is a flag, not a metric."""
        mild = score(NeighbourAverage(weight=0.1)).scalars["worst_consistency"]
        severe = score(NeighbourAverage(weight=0.8)).scalars["worst_consistency"]

        assert severe > mild


class TestWhatItDoesNotCatch:
    def test_a_predictor_written_in_wavenumbers_is_not_flagged(self) -> None:
        """The other half of the sentinel: a metric that flags everything measures nothing."""
        scored = score(ModeDecay())

        assert scored.scalars["worst_consistency"] < 1e-6

    def test_it_is_not_lenient_because_the_predictor_does_nothing(self) -> None:
        """A predictor that returned its input would pass the test above for free."""
        result = roll_out(ModeDecay(), initial(), STEPS)
        moved = np.abs(
            result.trajectory.fields[VORTICITY_FIELD][-1]
            - result.trajectory.fields[VORTICITY_FIELD][0]
        ).max()

        assert moved > 0.1

    def test_the_degradation_of_a_resolution_independent_predictor_is_negligible(self) -> None:
        scored = score(ModeDecay())

        assert abs(scored.scalars[f"{REFINEMENT.name}.degradation"]) < 1e-6


class TestWhenItDoesNotApply:
    def test_a_system_declaring_no_refinement_is_not_tested(self) -> None:
        """Zero steps rather than a perfect score, which is what a reader must be told."""
        scored = score(ModeDecay(), refinements=())

        assert scored.scalars == {"steps": 0.0}

    def test_a_predictor_that_cannot_run_on_the_finer_grid_reports_no_steps(self) -> None:
        """A solver built for one grid is such a predictor, and that is worth reading."""
        system = build_fluid({"grid_size": SIZE})
        regime = system.regimes[0]
        solver = system.reference_predictor(regime, DT / 10)

        scored = score(_Substeps(solver, 10))

        assert scored.scalars[f"{REFINEMENT.name}.steps"] == 0.0
        assert "worst_consistency" not in scored.scalars

    def test_the_metric_needs_the_predictor_it_is_scoring(self) -> None:
        rollout_result = roll_out(ModeDecay(), initial(), STEPS)
        rollout = Rollout(
            predicted=rollout_result.trajectory,
            reference=rollout_result.trajectory,
            predictor="mode_decay",
            system="fluid",
        )
        metric = ResolutionGeneralisation(MetricContext(refinements=(REFINEMENT,)))

        with pytest.raises(Exception, match="needs the predictor"):
            metric.compute(rollout)


@dataclass(frozen=True)
class _Substeps:
    """Folds a solver's several steps into one stored interval."""

    solver: Predictor
    times: int

    @property
    def name(self) -> str:
        return self.solver.name

    @property
    def dt(self) -> float:
        return self.solver.dt * self.times

    def step(self, state: State) -> State:
        for _ in range(self.times):
            state = self.solver.step(state)
        return state


def test_the_series_it_returns_is_one_curve_per_refinement() -> None:
    scored = score(NeighbourAverage())

    assert set(scored.series) == {REFINEMENT.name}
    assert len(scored.series[REFINEMENT.name]) == STEPS + 1


def test_a_trajectory_of_one_state_is_not_scored() -> None:
    """Nothing to compare, and a zero here would read as a pass."""
    state = initial()
    single = Trajectory.from_states([state, ModeDecay().step(state)])
    rollout = Rollout(predicted=single, reference=single, predictor="mode_decay", system="fluid")
    metric = ResolutionGeneralisation(
        MetricContext(refinements=(REFINEMENT,), predictor=ModeDecay(), resolution_steps=1)
    )

    assert metric.compute(rollout).scalars["steps"] == 1.0
