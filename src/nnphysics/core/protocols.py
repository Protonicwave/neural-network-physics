"""The five protocols the whole design rests on.

Nothing here is implemented. Systems, models, metrics and the evaluation harness all
speak through these interfaces, which is what lets one suite score a reference solver,
a trained network and a deliberately broken baseline without knowing which is which.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from nnphysics.core.types import (
        MetricResult,
        Regime,
        Rollout,
        State,
        StateSpec,
    )
    from nnphysics.core.units import Dimension

__all__ = [
    "Conservation",
    "Invariant",
    "Metric",
    "Predictor",
    "Symmetry",
    "System",
]


class Conservation(StrEnum):
    """How an invariant is expected to behave under the reference solver."""

    EXACT = "exact"
    """Conserved to round off, for example momentum under a symmetric integrator."""

    APPROXIMATE = "approximate"
    """Conserved only within a tolerance, for example energy under any integrator."""

    DECAYING = "decaying"
    """Not conserved: it must decrease, for example enstrophy under viscosity.

    A quantity a dissipative system drains is still worth declaring, because the
    direction is as strong a statement as conservation is: a predictor that lets it
    grow is unphysical.
    """


@runtime_checkable
class Invariant(Protocol):
    """A named scalar a system conserves, with the tolerance it is conserved to.

    The harness reads invariants from the system rather than knowing any physics
    itself, so drift can be measured for any system that declares one.
    """

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        ...

    @property
    def dimension(self) -> Dimension:
        """Physical dimension of the scalar."""
        ...

    @property
    def conservation(self) -> Conservation:
        """Whether the quantity is conserved exactly or only approximately."""
        ...

    @property
    def rtol(self) -> float:
        """How far the quantity may stray from its declared behaviour.

        For a conserved quantity this is the relative drift over a reference rollout
        that is still acceptable. For a decaying one it is the largest relative
        increase that still counts as monotone.
        """
        ...

    def evaluate(self, state: State) -> float:
        """Compute the invariant for one state."""
        ...


@runtime_checkable
class Symmetry(Protocol):
    """A transformation a system's dynamics commute with.

    Equivariance is tested by transforming an input, predicting, undoing the
    transformation, and comparing against predicting from the untransformed input.
    """

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        ...

    def apply(self, state: State) -> State:
        """Transform a state."""
        ...

    def apply_inverse(self, state: State) -> State:
        """Undo `apply`."""
        ...


@runtime_checkable
class Predictor(Protocol):
    """Anything that advances a state by one step of fixed size.

    The reference solver, a trained network and a broken baseline are all predictors.
    The harness sees nothing else.
    """

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        ...

    @property
    def dt(self) -> float:
        """Size of the step this predictor advances by."""
        ...

    def step(self, state: State) -> State:
        """Advance one step and return the new state."""
        ...


@runtime_checkable
class Metric(Protocol):
    """A scoring rule over a rollout.

    Metrics are registered by name and selected by configuration. A metric that
    cannot fail on a deliberately broken predictor does not earn its place.
    """

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        ...

    def compute(self, rollout: Rollout) -> MetricResult:
        """Score a rollout."""
        ...


@runtime_checkable
class System(Protocol):
    """A physical system: what its states look like, what it conserves, how it evolves.

    Both supported systems implement this identically from the outside. Any layer above
    `systems` works through this interface only.
    """

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        ...

    @property
    def state_spec(self) -> StateSpec:
        """Shape and meaning of the fields a state carries."""
        ...

    @property
    def regimes(self) -> tuple[Regime, ...]:
        """Named regions of parameter space this system declares."""
        ...

    def invariants(self, regime: Regime) -> tuple[Invariant, ...]:
        """Quantities the dynamics conserve or drain, in a given regime.

        A regime can change what a quantity does, not only how fast. An inviscid
        fluid conserves its enstrophy and a viscous one destroys it, so a system that
        had to answer this without naming a regime could only answer it wrongly for
        one of them. The reference solver is selected the same way, for the same
        reason.

        Args:
            regime: The regime the invariants will be measured in.

        Returns:
            The invariants, in declaration order.
        """
        ...

    @property
    def symmetries(self) -> tuple[Symmetry, ...]:
        """Transformations the dynamics commute with."""
        ...

    def initial_state(self, regime: Regime, rng: np.random.Generator) -> State:
        """Draw one initial condition from a regime."""
        ...

    def reference_predictor(self, regime: Regime, dt: float) -> Predictor:
        """Build the solver that produces ground truth for a regime."""
        ...
